# DMU Analytics Platform v2 — Complete Technical Documentation

> Interview-prep deep dive. Everything end-to-end: what activates on each query, how
> RRF/embeddings/judge/eval work, why each design choice was made, and every
> exception/fallback path in the code.

---

## 0. The One-Paragraph Pitch

An AI-powered educational decision-intelligence system for the **Nilgiris district
(Tamil Nadu)**. A natural-language question hits a **FastAPI** backend, which runs a
**LangGraph multi-agent supervisor**. The supervisor routes to one of four specialists:
**SQL analytics** over a **DuckDB** student-marks database, **RAG** over policy/survey
PDFs/DOCX, **web search** via **Tavily**, or **visualization** via **Plotly**. Every LLM
call goes through **Groq** (`langchain-groq`). A **React** frontend renders answers,
judge scores, and charts (with on-bar value labels and an auto-generated summary). When
a query is too vague to route, the system asks the user to clarify
(**human-in-the-loop**) instead of guessing. The whole thing is observable
(structlog JSON + token/cost accounting + optional LangSmith tracing) and evaluated
(routing accuracy, SQL accuracy, retrieval Recall/MRR/nDCG, RAGAS).

**The headline USP:** it's not a single LLM call — it's a *self-correcting,
self-evaluating* agentic system. SQL has an LLM-as-Judge retry loop; RAG is full hybrid
retrieval (dense + sparse + RRF + cross-encoder rerank); routing is a two-layer
classifier with a deterministic regex fallback that **can't crash**, and a
human-in-the-loop clarification when even the regex can't route.

---

## 1. Tech Stack (and why)

| Layer | Choice | Why |
|---|---|---|
| API | FastAPI + uvicorn | async, pydantic validation, OpenAPI docs free |
| Orchestration | LangGraph | stateful multi-agent graph with checkpointing (memory across turns) |
| LLM provider | Groq (`langchain-groq`) | extremely fast inference; cheap; but **TPM rate-limited** → drives many design choices |
| Analytics DB | DuckDB | embedded OLAP, no server, fast aggregations, reads pandas directly |
| Embeddings | `BAAI/bge-small-en-v1.5` (HuggingFace, local) | strong small English embedder, runs locally → no API cost/latency |
| Dense index | FAISS (`IndexFlatIP`) | exact cosine search on normalized vectors |
| Sparse index | `rank_bm25` (BM25Okapi) | keyword/lexical match, complements dense |
| Web | Tavily | LLM-optimized search API |
| Charts | Plotly | interactive JSON charts the frontend renders |
| Eval | custom metrics + RAGAS 0.4.x | retrieval + generation quality |
| Observability | structlog (JSON) + in-memory MetricsStore + LangSmith | token/cost/latency tracking |
| Cache | In-memory LRU cache (256 entries, inside SQLEngine) | repeated SQL queries cost 0 tokens |
| Frontend | React | dashboard UI |

**Models by role** (`core/config.py`):
- `LLM_MODEL = llama-3.1-8b-instant` — fast: answer-gen, judge, RAG generation, web synthesis, supervisor, classifier, chart summary.
- `SQL_MODEL = openai/gpt-oss-120b` — **SQL generation only**. Deliberately a *different model on a separate Groq quota bucket* so heavy SQL generation doesn't burn the 8B TPM limit.
- `VIZ_MODEL = llama-3.3-70b-versatile` — chart-type + axis-column planning.
- `JUDGE_MODEL = llama-3.3-70b-versatile` — **declared but currently unused**; the engine actually judges with `LLM_MODEL` (8B). (Good "gotcha" to know: the config name suggests 70B but code uses 8B — done to relieve TPM pressure.)

---

## 2. Repository Layout & The "Two Copies" Trap

**Canonical package = `backend/app/`.** That's what `uvicorn app.api.main:app`,
`ingest.py`, `evaluate.py`, and the Makefile all import.

`backend/agents/`, `backend/api/`, `backend/core/`, `backend/rag/` etc. at the top
level are **stale duplicates** imported by nothing. There's also a junk dir literally
named `{backend,frontend}` (a shell brace-expansion that ran as a literal path) and
one-off scripts (`fix_serialization.py`, `patch_graph.py`). **Always edit
`backend/app/`.** (Interviewers love asking "why two copies?" — answer: legacy/refactor
artifact, canonical is `app/`.)

```
backend/app/
  api/main.py            FastAPI app + 5 endpoints + _safe() JSON coercion
  agents/graph.py        LangGraph supervisor + 4 tool executors + HITL clarify (the brain)
  agents/router.py       deterministic regex router (classify) + no-default matcher (matched_route) + LLM fallback
  sql/engine.py          text-to-SQL + validate + execute + judge + retry
  rag/pipeline.py        parent-child chunk + dense+sparse+RRF + rerank + answer
  visualization/agent.py Plotly chart selection (LLM plan + rule fallback)
  evaluation/evaluator.py 4 metric families + RAGAS + thresholds
  ingestion/excel_loader.py wide→long Excel normalization → DuckDB
  core/config.py         single pydantic Settings, lru_cached
  core/database.py       ONLY module importing duckdb; thread-safe singleton
  monitoring/telemetry.py structlog + token/cost + MetricsStore
data/
  24255.xlsx (year 2025), 2526.xlsx (year 2026), SLAS - 2025 Report.pdf
```

---

## 3. The Data Model (DuckDB)

Built by `ingestion/excel_loader.py`. **Schema-agnostic by design**: scans `DATA_DIR`
for *any* `.xlsx`, reads *every* sheet, normalizes wildly varying wide columns into a
**fixed long format**. Adding a new file "just works."

Two tables:

**`marks`** — *one row per student × subject* (long format). Columns: `rollno, name,
school_code, school_name, management, district, revname, sex, community, group_code,
total_marks, year, management_type, source_file, source_sheet, overall_result,
pass_flag, fail_flag, subject_name, marks, subject_result`.

**`students`** — *one row per student* (deduplicated by `rollno`, drops the subject-level
columns). Use for total counts, overall pass %, gender analysis.

### Ingestion mechanics (`_to_long`)
- **Column matching is case-insensitive against candidate lists** (`_normalize_col`).
  E.g. roll number could be `ROLLNO`/`ROLL_NO`/`ROLL NO`/`REGNO` — all map to `rollno`.
- **Subject slots:** up to 6 subjects per student via paired columns
  `(S1_DESC, MARK01) … (S6_DESC, MARK06)`. Each present pair becomes one long row.
- **Year inference** (`_infer_year`): pulls 4+ digit runs from the filename stem; a value
  starting `25` → 2026, starting `24` → 2025. So `2526.xlsx`→2026, `24255.xlsx`→2025.
- **Management type** (`_infer_management`): from sheet name keywords — `TW`/`TRIBAL`→TW,
  `GOVT`/`GOV`→GOVT, `AIDED`→AIDED, else UNKNOWN.
- **Business rules baked in here AND in the SQL prompt:**
  - `marks < 35` → that subject = **FAIL** (`subject_result`).
  - `NaN`/unparseable mark → **ABSENT**.
  - Mark strings get `"XXX"` stripped before float parse (dirty source data).
  - Overall: `pass_flag=1` if result code `P`; `fail_flag=1` if `F` (or, for 2026, any
    non-P/blank/absent code — different years encode results differently).
- Writes via `con.register(temp_df)` + `CREATE OR REPLACE TABLE … AS SELECT *`.

**Exceptions/fallbacks:** a file that fails to open is logged + skipped; a sheet that
fails is logged + skipped; if *zero* data loads it raises `RuntimeError`. District
defaults to `NILGIRIS` if missing.

---

## 4. End-to-End Request Flow

```
POST /api/query (api/main.py)
   │  validate (pydantic: 2–1000 chars), assign thread_id
   │  query_type = router.classify(query)   ← used only for the response label
   ▼
get_graph().invoke(initial_state, thread_id)   ← LangGraph, MemorySaver checkpointer
   ▼
supervisor_node  ──┐
   │  Layer 1: classify_intent() (8B, ~80ms, lru_cached)
   │  HITL gate: cold + low-confidence + no keyword → ask user to clarify (END)
   │  Layer 2: full supervisor LLM (only if ambiguous/low-confidence)
   ▼
tools_node_wrapper  ← actually executes analytics / rag / web / viz
   │  enriches state: sql_result / rag_result / web_result / chart_output / summary
   ▼
back to supervisor → finish OR chained hop (e.g. analytics→rag→viz)
   ▼
final tagged AIMessage  [ANALYTICS] / [RAG] / [WEB]
   ▼
api/main.py extracts answer, _safe()-serializes, returns QueryResponse
```

### 4.1 The Supervisor (two layers) — `supervisor_node`

**Layer 1 — fast classifier** `classify_intent()`:
- A tiny 8B Groq call (`max_tokens=80`, temp 0) with `CLASSIFIER_SYSTEM` prompt.
- Returns JSON `{"reasoning","intent","confidence"}`. Intents:
  `conversational | analytics | rag | web | viz | hybrid | ambiguous`.
- **`@lru_cache(maxsize=256)`** → identical repeated queries cost 0.
- Strips ```` ```json ```` fences before parsing; on any exception → `("ambiguous", 0.0)`.

If `confidence ≥ 0.65` (CONFIDENCE_THRESHOLD), the supervisor acts **without** the
heavier LLM:
- **conversational** → `get_conversational_reply()` (8B, friendly, threads in last 6
  turns so it remembers names etc.) → goes straight to END. No tools, ~no cost.
- **hybrid** → emits a `run_analytics` tool call **and queues `pending_hops=["rag"]`**
  (plus `"viz"` if a chart was asked). DB first, document second, optional chart.
- **analytics / rag / web** → emits a synthetic tool call with that route.
- **viz intent maps to `run_analytics`, NOT `run_viz`** — a chart query is also a data
  question, so fetch data first; the post-tool second-hop logic adds the chart. This
  avoids the "no data to visualize" dead end. The classifier *never* emits a direct viz
  call.

**Human-in-the-loop gate (hybrid)** — runs *between* Layer 1 and Layer 2:
- Fires only when **all three** are true: classifier confidence < 0.65 **AND** the regex
  `matched_route(query)` finds no keyword (`None`) **AND** there's no prior conversation
  to lean on (`_has_prior_context(state)` is False — i.e. this isn't a follow-up).
- When it fires → return a `_clarification_message(intent)` (a plain `AIMessage`, no tool
  call) that asks the user to rephrase, with a best-guess hint and 2-3 example prompts →
  graph goes straight to END. This is the **ask-don't-guess** human-in-the-loop.
- **Why gate before Layer 2?** The Layer-2 LLM is tool-bound and almost never abstains —
  it would confidently route even gibberish. Asking *before* the LLM is what makes the
  clarification actually trigger.
- **Why the "no prior context" condition?** A no-keyword *follow-up* like "show me more"
  should be resolved by Layer 2 using conversation history, not interrupted with a clarify
  prompt. So we only short-circuit on **cold** vague queries.

**Layer 2 — full supervisor LLM** (only when ambiguous or confidence < 0.65):
- 8B model with `.bind_tools(TOOLS)` and `SUPERVISOR_SYSTEM` (rich disambiguation rules).
- If it returns a valid tool call → route through it. If it answers directly → pass
  through. If neither → raise.
- **Exception fallback:** weak models sometimes emit malformed tool calls (e.g.
  `{"sql": …}` instead of `{"question": …}`), which Groq rejects with HTTP 400
  `tool_use_failed`. On *any* exception it falls back to the **deterministic regex
  matcher** (`router.matched_route`). If that returns a route → emit a schema-valid
  `{"question": …}` tool call (the "graph can never hard-fail on routing" guarantee). If
  even the regex returns `None` → the **last-resort human-in-the-loop**: ask the user to
  clarify instead of silently defaulting to RAG.

### 4.2 Post-tool decisions (the loop control)

After a tool runs, the last message is a `ToolMessage`. `supervisor_node` then:
1. If `pending_hops` non-empty → pop the next hop and emit its tool call. The list
   *shrinks every pass*, so **it can't loop forever**.
2. Else if `wants_viz(query)` and we have `sql_result` and no `chart_output` yet →
   auto-add a `run_viz` hop.
3. Else → build the final tagged answer and END.

> **Why this exists:** without the post-tool branch, the supervisor would re-route the
> same query forever until LangGraph's recursion limit throws. (This was one of the
> documented bugs that got fixed.)

### 4.3 Tool execution — `tools_node_wrapper`

The `@tool`-decorated functions (`run_analytics`, etc.) are **schema carriers only** —
their docstrings tell the LLM what each tool does; **their bodies never run.** The real
logic lives in `tools_node_wrapper`, which dispatches by tool name:

- **`run_analytics`** → `SQLEngine.run(q)`. Then:
  - Serializes the DataFrame → JSON-safe records (NaN/Inf→None, numpy `.item()` coercion).
  - Records `RequestMetrics` (tokens, judge score, latency, sql error).
  - Stores `sql_result` dict (answer, sql, records, judge_score, attempts, tokens, cost…).
- **`run_rag`** → `RAGPipeline.answer(q)` → stores `{answer, contexts}`.
- **`run_web`** → Tavily search (top 3, `include_answer=True`) → 8B synthesis over
  snippets. Catches errors → `"Web search failed: …"`.
- **`run_viz`** → rebuilds a DataFrame from `sql_result.result_records` → `render(df, q)`,
  then writes a **3-4 line `summary`** of the plotted rows via `_summarize()` (8B). A chart
  has no prose of its own, so the summary is what the UI shows under it. **Viz plots
  already-executed rows, never re-runs SQL.**

> **Note — the old "recommendations" feature was removed.** It used to make an extra 8B
> call for 2 "actionable recommendations", but on chart/viz requests it produced nonsense
> (e.g. matplotlib code). It's replaced by the `summary` above, generated **only for viz
> responses** (analytics/RAG/web already return a prose answer).

Every branch is wrapped in try/except → on failure emits `"Tool {name} failed: {e}"` as
the ToolMessage so the graph keeps moving.

### 4.4 Final answer + hybrid synthesis

`_final_answer_message`:
- If **both** `sql_result.answer` and `rag_result.answer` exist → `_synthesize_hybrid()`
  merges them into one comparison (8B call). **Caveat baked into the prompt:** DB holds
  raw exam marks, SLAS holds survey achievement %, so they may be apples-to-oranges — the
  prompt is told to say so. Falls back to plain concatenation if the LLM call fails.
- Else stamps the single result with its tag: `[ANALYTICS]`, `[RAG]`, or `[WEB]`.
- If a tool failed before storing a structured result → surfaces the last ToolMessage.

### 4.5 Answer extraction (`api/main.py`)

Scans messages for an `[ANALYTICS]`/`[RAG]`/`[WEB]` tag, strips it. **Fallback:** if no
tagged message (conversational replies aren't tagged), uses the last non-empty
`AIMessage`. Everything is `_safe()`-coerced before returning.

---

## 5. Routing — Two Mechanisms (don't confuse them)

There are **two** routers and they serve different purposes:

1. **`agents/router.py::classify`** — pure-regex deterministic router (always returns a
   route; defaults to **rag**). Used:
   - by the API as the *response label* (`query_type`),
   - by the evaluator's routing-accuracy suite.
2. **`agents/router.py::matched_route`** — same regex priority but with **no default**:
   returns the route on a keyword hit, else **`None`** (unroutable). This is the signal the
   supervisor uses for (a) the crash-proof Layer-2 fallback and (b) the human-in-the-loop
   "can we route this at all?" gate. The `None` is what makes "ask the human" possible —
   `classify` could never express "I don't know" because it always falls back to rag.
3. **`graph.py::classify_intent`** — the LLM Layer-1 classifier inside the supervisor.

**Human-in-the-loop (hybrid gate).** When the classifier is unsure (confidence < 0.65),
`matched_route` is `None`, **and** there's no prior conversation context, the supervisor
returns a clarification message asking the user to rephrase (with example prompts) instead
of guessing. No-keyword *follow-ups* (with context) still go to Layer 2, which routes them
using history. See §4.1.

### `router.classify` order (fast→slow)
1. `_RAG` regex (samacheer, syllabus, slas, grading policy…) → **rag**.
2. `_WEB` regex (government order, NEP, latest/news, blooms taxonomy, T###/E###/M### codes…) → **web**.
3. `_ANALYTICS` regex (count, average, top, pass/fail, school, subject, location names, chart words…):
   - if it *also* matches `_DOC_PHRASING` ("what does the report say", "according to",
     "explain", "summary"…) → ambiguous → **LLM fallback** `_llm_classify` (~0.5s).
   - else → **analytics** (fast).
4. Nothing matched → default **rag**.

`_llm_classify` is a 5-token 8B call returning one word; on any error → returns `"rag"`
(safe default). `wants_viz` is a separate regex (chart|plot|graph|bar|pie|line|heatmap…).

**Ordering rationale:** RAG checked before WEB so doc keywords don't leak to web search;
analytics keywords are common, so doc-phrasing disambiguation guards against
false-positives like "what does the report say about *school* performance."

---

## 6. SQL Analytics Engine — `sql/engine.py`

The crown jewel. Six stages: **dynamic prompt → validate → execute → answer → judge →
retry**, plus a result cache.

### 6.1 Dynamic prompt (`_build_prompt`)
Built **from the live DuckDB schema** every call — *nothing hardcoded*:
- `get_schema_dict()` for tables/columns,
- `get_sample_values()` injects real **subjects**, **schools** (sample of 10), and **years**.
- Includes a `LOCATION_MAP` (ooty→`LIKE '%OOTY%' OR '%UDHAGAMANDALAM%'`, gudalur, coonoor,
  kotagiri, pandalur), critical rules (marks<35=FAIL, UPPER() for all text matches, never
  `SELECT *`, percentage formula with `FILTER`), and 6 worked examples (few-shot).

This is a **USP**: adding a new school/subject/year requires zero prompt edits — the
schema introspection picks it up.

### 6.2 Validation (`_validate`) — deterministic gate
- Strips ```` ``` ```` fences.
- **Blocks DDL/DML** via `_UNSAFE` regex (DROP/DELETE/TRUNCATE/INSERT/UPDATE/ALTER/
  CREATE/REPLACE/ATTACH/PRAGMA/COPY) → SQL injection / destructive-op guard.
- Must **start with `SELECT`**. Empty → invalid.

### 6.3 The retry loop (`run`) — what actually happens per attempt

```
for attempt in 1..SQL_MAX_RETRIES(=2):
  generate SQL  (SQL_MODEL=120B on attempt 1; compact RETRY_PROMPT on retries)
  gate 1: validate           ── fail → feedback, sleep 0.5, retry
  gate 2: execute on DuckDB  ── fail → feedback="DuckDB error: …", retry
  gate 3: empty-result heuristic ── if empty & not yet retried → 1 free soft retry (no judge)
  generate NL answer (8B, ANSWER_PROMPT, numbers only from data)
  if not judged yet:
     judge ONCE (8B, JUDGE_PROMPT → JSON score 1–5)
     if score >= JUDGE_SCORE_THRESHOLD(=3): return (success)
     else: feedback = judge issues+suggestion, retry
  else:
     deterministic accept (already validated+executed) — no re-judge
```

Key design decisions (all driven by **Groq TPM limits** — token optimization plan):
- **Item 1 — result cache:** keyed on the normalized question (`question.lower().strip()`).
  Cache hit → 0 tokens, returns the cached `SQLResult` (answer/sql/records/judge). Backed by a
  process-local **LRU cache (256 entries)** held inside `SQLEngine`; only successful results are
  cached, least-recently-used entries evict past 256, and the cache is cleared on backend restart
  or on re-ingest (`/api/ingest` calls `SQLEngine.clear_cache()`).
- **Item 3 — compact retry prompt:** the full generation prompt is ~2,400 tokens. On
  retries it sends the much smaller `RETRY_PROMPT` (just enough schema + the failed SQL +
  the problem) to stay under TPM.
- **Item 4 — judge at most ONCE:** the LLM-as-Judge is expensive; it runs once on the
  first clean success. If it fails, one corrective retry, then **accept on deterministic
  gates** (valid + executes) without re-judging. Prevents judge-call blowup.
- **SQL on a separate model bucket:** `SQL_MODEL=gpt-oss-120b` has its own Groq quota, so
  heavy SQL generation doesn't starve the shared 8B bucket used by everything else.
- `time.sleep(0.5)` between retries — rate-limit courtesy.

### 6.4 LLM-as-Judge (`_judge`)
A second LLM scores the (question, SQL, result, answer) tuple 1–5 with
`sql_correct`/`answer_grounded`/`issues`/`suggestion`. Strips fences, `json.loads`. On
parse failure → defaults to `JudgeResult(score=3)` (neutral pass at threshold). Judge
tokens are added to the request total so cost accounting is honest.

### 6.5 Exhaustion fallback
If all retries fail: returns `answer="Could not generate a reliable answer after N
attempts."`, `judge.score=0`, and the last error — never throws to the API.

---

## 7. RAG Pipeline — `rag/pipeline.py`

Full hybrid RAG. Four stages.

### 7.1 Parent-Child Chunking (`ParentChildChunker`)
- **Parent chunks = 512 chars** (config: PARENT_CHUNK_SIZE) — large, hold context.
- **Child chunks = 128 chars** (CHILD_CHUNK_SIZE) — small, are what gets *embedded and
  retrieved*. Overlap = 20.
- Each child stores its `parent_id`. **Retrieve on children (precise), return parents
  (context-rich).** This is the small-to-big / parent-document retrieval pattern — a USP
  that beats naive fixed-chunk RAG: precise matching + enough surrounding context for the
  LLM to ground its answer.
- (Note: sizes are *characters* in code, described as "tokens" in comments — worth knowing.)

### 7.2 Index building (`build_index`)
- Loads all `.pdf` (via `pypdf`) and `.docx` (via `python-docx`) from `DATA_DIR`,
  concatenates, chunks.
- **BM25** built on lowercased, whitespace-split child texts.
- **Embeddings:** `BAAI/bge-small-en-v1.5` via `HuggingFaceEmbeddings` (runs **locally** —
  no API cost). `embed_documents(child_texts)` → float32 matrix.
- **FAISS:** `faiss.normalize_L2` then `IndexFlatIP` (inner product on normalized
  vectors = **cosine similarity**, exact search).
- **Persisted** to `VECTOR_STORE_DIR/rag_store.pkl` (pickles parents, children,
  child_embs). `build_index` **loads the cache if present** → instant startup. **Delete
  the pkl to force rebuild** (`make ingest-rebuild` or `/api/ingest rebuild_index=True`).
- Fallbacks: doc load errors are logged + skipped; if no docs, a placeholder string is
  used; if no children built, returns 0 with a warning.

### 7.3 Hybrid Retrieval + RRF (`retrieve`) — explain this clearly in interview

```
k_ret = TOP_K_RETRIEVAL = 10   (retrieve this many from each method)
k_fin = TOP_K_RERANK    = 5    (keep this many after rerank)

1. DENSE:  embed query → normalize_L2 → FAISS search → top-10 child ids (by cosine)
2. SPARSE: BM25 scores over all children → argsort → top-10 child ids (by keyword)
3. RRF (Reciprocal Rank Fusion):
      fused[id] += 1 / (61 + rank)   for each method's ranked list
      (rank is 0-based position; constant k=61)
   → sort by fused score → take top-10 candidates
4. RERANK: Groq LLM cross-encoder scores each candidate 1–5 (RERANK_PROMPT), sleep 0.3
      → sort by score → keep top-5
5. PARENTS: map each surviving child → its parent_id, dedupe → return parent texts
```

**Why RRF?** Dense and sparse scores live on totally different scales (cosine ∈ [0,1] vs
BM25 unbounded). RRF ignores the raw scores and fuses purely on **rank position**, so the
two signals combine fairly. The constant (here **61**, a common choice ≈60) dampens the
gap between top ranks and prevents any single high rank from dominating. A doc that ranks
well in *both* lists rises to the top.

**Why a cross-encoder rerank?** Bi-encoder (embedding) retrieval is fast but approximate —
it scores query and passage independently. The rerank feeds *both together* to an LLM for
a precise relevance judgment, fixing the ordering on the shortlist. Here it's an
LLM-as-cross-encoder (scores 1–5). On any error a candidate defaults to score 3.0.
`time.sleep(0.3)` per candidate for rate limits.

### 7.4 Grounded answer (`answer`)
- Joins the parent contexts (truncated to `PARENT_CHUNK_SIZE * TOP_K_RERANK`), fills
  `GROUNDED_PROMPT`. **Strict grounding:** "answer using ONLY the context… if not present
  say 'Not found in the available documents.' Do not make up statistics." → reduces
  hallucination.
- Returns `(answer_text, contexts)`. Contexts flow back so the evaluator can score
  retrieval/RAGAS.

---

## 8. Visualization Agent — `visualization/agent.py`

`render(df, query)` picks a chart and emits Plotly JSON.

- **LLM-first selection** (`_llm_plan`, VIZ_MODEL=70B): given columns+dtypes+sample rows,
  returns JSON `{chart_type, x, y, y2}`. Validates `chart_type ∈ {line,bar,pie,heatmap}`
  and that referenced columns actually exist (drops bad `y2`, else rejects plan).
- **Deterministic fallback** (`_pick_type`) when the LLM plan is missing/invalid: trend
  words→line, "across"/matrix→heatmap, ≤3 rows→pie, else bar.
- Sensible axis defaults if the planner omits them (first categorical for x, first numeric
  for y, second categorical for y2/heatmap).
- Builds the figure with Plotly Express, applies a dark transparent theme, returns both
  `chart_json` (frontend renders) and standalone `html`.
- **`_decode_typed_arrays` (important fix):** Plotly 6 serializes numeric arrays as base64
  typed-array blobs (`{"dtype":"f8","bdata":"…","shape":"…"}`). The recharts frontend
  expects plain JSON lists, so without decoding, bar/line `y` values arrive as an object →
  every point reads as 0 → **blank chart**. `render()` walks the chart JSON and decodes any
  such blob back into a list (and reshapes 2-D for heatmaps). Categorical `x` stays a list,
  which is why the symptom was "axis labels present, bars empty."
- **Exceptions:** empty df → `ChartOutput(chart_type="none", error="Empty result")`; build
  failure → `chart_type="error"` with the message. Never throws.

The **frontend** (recharts) then renders the decoded `x`/`y`, adds on-bar / on-point value
labels (`LabelList`), and shows the `summary` in a labeled box under the chart.

---

## 9. Evaluation Suite — `evaluation/evaluator.py`

Eight metrics across four families, with ground-truth suites hardcoded at the top.

**Retrieval** (over `RAG_QA_SUITE`, keyword-based relevance):
- **Recall@5 / Recall@10** — binary: 1 if any relevant keyword appears in top-k chunks.
- **MRR** (Mean Reciprocal Rank) — `1/rank` of first relevant chunk.
- **nDCG@10** — DCG `Σ rel/log2(i+2)` over IDCG; rewards ranking relevant docs higher.

**Generation:**
- **RAGAS** (`eval_ragas`): faithfulness, answer_relevancy, context_precision,
  context_recall via ragas 0.4.x. Notable hacks:
  - **Stubs `langchain_community.chat_models.vertexai.ChatVertexAI`** because ragas
    hard-imports a symbol that the sunset langchain-community no longer ships — we use
    Groq, so a dummy class lets the import succeed.
  - Wraps Groq + HF embeddings in `LangchainLLMWrapper`/`LangchainEmbeddingsWrapper`.
  - `ResponseRelevancy(strictness=1)` forces n=1 — **Groq rejects multi-completion
    (n>1)** requests.
  - `RunConfig(max_workers=1)` — serial, to respect rate limits.
  - `_mean()` coerces all-NaN columns to 0.0 so errored metrics don't leak into the report.
  - On any failure → `RAGASMetrics(computed=False, error=…)`; the rest of eval still runs.
- **SQL accuracy** (`eval_sql`): runs `SQL_TEST_SUITE`, scores 1 if it executed + returned
  non-empty data, also tracks judge score + latency.

**System:**
- **Routing accuracy** (`eval_routing`): `classify()` vs expected over `ROUTING_TEST_SUITE`.
- **LLM-as-Judge** — already integrated in the SQL engine; aggregated here.

**Overall pass thresholds:** routing ≥ 0.85, sql ≥ 0.80, recall@5 ≥ 0.70, nDCG ≥ 0.65,
ragas_faithfulness ≥ 0.75 (only if RAGAS computed). `overall_pass` = all thresholds met.
`make eval-fast` skips RAGAS (much faster). Inter-call `delay=1.5s` for rate limits.

---

## 10. Cross-Cutting Concerns

### Config (`core/config.py`)
Single pydantic-settings `Settings`, loaded via **`get_settings()` (lru_cached)** so it's
parsed once. Reads `backend/.env`. All knobs live here (chunk sizes, top-k, models,
thresholds, cost-per-token). `GROQ_API_KEY` required; `TAVILY_ENABLED` auto-flips True if
the key is present; setting `LANGCHAIN_API_KEY` auto-wires LangSmith tracing.

### Database (`core/database.py`)
**Only module that imports duckdb.** Thread-safe **singleton** connection (double-checked
lock). **Fallback:** if the DB file is locked by another process (`IOException`) it falls
back to a read-only connection. Helpers: `execute_query` (→DataFrame), `execute_ddl`,
`table_exists`, `get_schema_dict`, `get_sample_values`.

### Cache (`SQLEngine` LRU)
The SQL result cache lives **inside `SQLEngine`** (no separate module) as a process-local
**bounded LRU** — an `OrderedDict` of up to 256 entries guarded by a lock, keyed on the
normalized question (`question.lower().strip()`). Only successful results are stored; the
least-recently-used entry is evicted past 256. A hit returns the cached `SQLResult` with token
usage zeroed (no LLM call happened). No TTL: the cache survives browser refreshes and is cleared
on backend restart or via `SQLEngine.clear_cache()` (called by `/api/ingest`).

### Telemetry (`monitoring/telemetry.py`)
- **structlog** JSON logging to stdout.
- **`track_llm_call`** extracts token usage from either LangChain shape
  (`usage_metadata` or `response_metadata["token_usage"]`), coerces objects→dict, computes
  cost from Groq per-1M pricing in config.
- **`MetricsStore`** — thread-safe, in-memory ring buffer (last 1000 requests). Tracks
  totals (tokens, cost, requests) and a rolling summary (avg latency, routing accuracy,
  avg judge, type breakdown, recent 20). Backs `GET /api/metrics`.

### JSON serialization (`api/main.py::_safe`)
DuckDB/pandas/numpy results aren't JSON-safe (NaN, numpy scalars, DataFrames). `_safe`
recursively coerces everything: NaN/Inf→null, numpy→`.item()`, DataFrame→records,
dataclass→dict. **If you touch query output, preserve this** or the API response breaks.
`analytics_agent` does the same DataFrame→records conversion inline.

### Frontend (`frontend/src/App.js`, `index.css`)
- **React + recharts.** The backend emits Plotly JSON, but the UI re-renders the decoded
  `x`/`y` with **recharts** (Bar/Line/Pie) — with on-bar/on-point value labels (`LabelList`).
- **Per-response UI:** the answer text, a judge badge, latency·tokens, and — for chart
  responses — the chart plus a labeled **Summary** box. The query-type badge, per-message
  cost, and the "Show generated SQL" toggle were intentionally removed for a cleaner view.
- **Removed the metrics topbar** (DB rows / query count / total cost / 📊 Metrics panel) and
  its `/health`+`/metrics` polling. The endpoints still exist; the UI just no longer shows
  them.
- **Theme:** `Geist` font; an indigo-plum gradient backdrop with the sidebar/topbar/borders
  harmonized to the same palette (the blue accent is kept).
- Conversation continuity: a stable per-session `thread_id` (in `useRef`) is sent on every
  `/api/query` so the backend checkpointer threads context across turns.

---

## 11. The Five API Endpoints (`api/main.py`)

| Endpoint | Purpose |
|---|---|
| `POST /api/query` | main NL query → routed answer (+ optional sql/judge, chart, chart `summary`) or a clarify prompt |
| `POST /api/ingest` | reload Excel → DuckDB; `rebuild_index=True` deletes the pkl + rebuilds RAG |
| `POST /api/evaluate` | run the full eval suite (`include_ragas`, `include_sql` flags) |
| `GET /api/health` | liveness — counts `marks` rows; "ok" vs "degraded" |
| `GET /api/metrics` | runtime token/cost/latency/routing summary |

`QueryRequest` carries `thread_id` (optional) → **conversation memory**: the LangGraph
`MemorySaver` checkpointer keys state by `thread_id`, so follow-up turns remember context.
`show_sql`/`show_judge` flags gate whether SQL and judge details appear in the response.
The graph is **pre-warmed at startup** (lifespan) so the first real query isn't slow.

---

## 12. "What Activates" — Worked Query Examples

| Query | Layer-1 intent | Path | Specialists hit |
|---|---|---|---|
| "hi there" | conversational (≥0.65) | direct reply, no tools | 8B chat only |
| "how many students failed in Physics?" | analytics | synthetic run_analytics | SQLEngine (gen 120B → validate → execute → answer 8B → judge 8B) |
| "show top 5 schools by Math average" | analytics (not viz!) | run_analytics | SQLEngine |
| "bar chart of subject failures" | viz→maps to analytics | run_analytics → auto viz hop | SQLEngine then `render` (70B plan) + 8B `summary` |
| "what is the re-examination policy?" | rag | run_rag | RAG (dense+sparse+RRF+rerank+grounded) |
| "what does the SLAS report say about Class 8?" | rag | run_rag | RAG |
| "latest NEP circular" | web | run_web | Tavily + 8B synth |
| "compare our Math marks with the SLAS Nilgiris report" | hybrid | run_analytics → queued run_rag → `_synthesize_hybrid` | SQLEngine + RAG + 8B merge |
| ambiguous *but routable* phrasing | low confidence + keyword | Layer-2 supervisor LLM; on failure → `matched_route` | depends |
| "how's it looking there?" (cold, vague) | low confidence + no keyword + no context | **HITL clarify** → ask user to rephrase | none (no LLM route) |
| "show me more" (after an analytics turn) | low confidence + no keyword **but has context** | Layer-2 uses history → run_analytics | SQLEngine |

**Hybrid + chart:** "compare our Math marks with SLAS and show a bar chart" →
`pending_hops=["rag","viz"]` → analytics, then rag, then viz (chart renders the **DB
side** only).

---

## 13. Every Exception / Fallback (cheat sheet)

| Where | Failure | Fallback |
|---|---|---|
| `classify_intent` | LLM error / bad JSON | `("ambiguous", 0.0)` → Layer-2 |
| routing | cold query, low confidence + no keyword + no context | **HITL** clarify message (ask user to rephrase) |
| supervisor Layer-2 | malformed tool call (Groq 400 `tool_use_failed`) | `matched_route`; if it routes → tool call, else **HITL** clarify |
| `router._llm_classify` | LLM error | returns `"rag"` (safe default) |
| `get_conversational_reply` | LLM error | canned friendly fallback string |
| SQL `_validate` | DDL/DML or non-SELECT | rejected → retry with feedback |
| SQL execute | DuckDB error | feedback="DuckDB error" → retry |
| SQL empty result | 0 rows | one free soft retry (no judge) |
| SQL `_judge` | bad JSON | `JudgeResult(score=3)` (neutral) |
| SQL all retries fail | — | "Could not generate a reliable answer", score 0, no throw |
| `_synthesize_hybrid` | LLM error | plain concatenation of both answers |
| RAG doc load | bad PDF/DOCX | log + skip; placeholder if none |
| RAG rerank | LLM error per candidate | score 3.0 |
| viz `_llm_plan` | invalid/missing | rule-based `_pick_type` |
| viz `render` | empty df / build error | `chart_type="none"`/`"error"`, no throw |
| web tool | Tavily/LLM error | "Web search failed: …" |
| RAGAS | import/eval error | `computed=False`, rest of eval continues |
| Cache | SQL pipeline failure | result not cached (only successes stored); query re-runs next time |
| DuckDB connect | file locked | read-only connection |
| tool wrapper | any tool exception | "Tool {name} failed: {e}" ToolMessage |
| API graph.invoke | any | HTTP 500 with detail |

---

## 14. USPs to Lead With (interview headline answers)

1. **Self-correcting SQL** — generate→validate→execute→answer→judge→retry, with an
   LLM-as-Judge gate and deterministic safety (no DDL/DML, SELECT-only).
2. **Full hybrid RAG** — parent-child (small-to-big) chunking + dense (BGE/FAISS) +
   sparse (BM25) fused by **RRF** + **LLM cross-encoder rerank** + strict grounded
   generation. Not a toy "embed + top-k."
3. **Two-layer routing that can't crash, with a human-in-the-loop floor** — fast 8B
   classifier (cached) for the 95% clear cases, heavier supervisor LLM for ambiguity,
   **deterministic regex fallback** as the floor, and a **HITL clarification** for truly
   cold/unroutable queries (ask the user instead of guessing). The "ask vs guess" decision
   is gated on deterministic signals (confidence + keyword + conversation context), so it
   fires reliably — a tool-bound LLM would otherwise never abstain.
4. **Schema-agnostic, prompt-from-live-schema design** — Excel ingestion normalizes any
   wide format to long; the SQL prompt is built from live DuckDB introspection. New
   files/schools/subjects need zero code or prompt changes.
5. **Hybrid analytics+RAG queries** — compares live DB numbers against SLAS survey figures
   in one synthesized answer (with the apples-to-oranges caveat built in).
6. **Production-grade ops** — token/cost accounting per call, structlog JSON, in-memory
   metrics endpoint, optional LangSmith tracing, conversation memory via checkpointer.
7. **Rate-limit engineering** — every design choice (result cache, compact retry prompt,
   judge-once, separate model quota buckets, `sleep(0.3–0.5)`) exists to survive **Groq's
   TPM limits**. This is a great "what was hard" story.
8. **Graceful degradation everywhere** — DB locked, RAGAS broken, bad PDF,
   LLM 400 — every path has a fallback; the system never hard-fails on a query.

---

## 15. Likely Interview Q&A

- **"Why DuckDB not Postgres?"** Embedded OLAP, zero server, reads pandas directly, fast
  aggregations for analytical (not transactional) workloads.
- **"Why RRF over weighted score fusion?"** Dense and sparse scores aren't comparable in
  scale; RRF fuses on rank, parameter-free except the constant (61).
- **"Why parent-child chunks?"** Embed/retrieve small for precision, return large for
  context — best of both; avoids the chunk-size dilemma.
- **"What stops SQL injection / data loss?"** `_UNSAFE` regex blocks all DDL/DML; SELECT-
  only; everything goes through the single DB module.
- **"How do you control hallucination in RAG?"** Strict grounded prompt ("only from
  context", explicit "Not found" escape), cross-encoder rerank to keep only relevant
  context, faithfulness scored by RAGAS.
- **"How is quality measured?"** Routing accuracy, SQL accuracy + judge score, retrieval
  Recall@k/MRR/nDCG, RAGAS faithfulness/relevancy/precision/recall, all with pass
  thresholds.
- **"Why two models?"** 8B for cheap/fast generation, 120B for SQL (separate quota bucket
  + harder task), 70B for viz planning. The named `JUDGE_MODEL` (70B) is actually unused —
  judge runs on 8B to save TPM.
- **"Biggest engineering challenge?"** Groq TPM rate limits → drove caching,
  compact-retry, judge-once, model-bucket separation, and sleeps.
- **"What's the conversation memory?"** LangGraph `MemorySaver` checkpointer keyed by
  `thread_id`; `_query()` reads the *latest* HumanMessage so accumulated history doesn't
  pin routing to the first question.

---

*Generated as an interview-prep companion. Canonical code lives under `backend/app/`.*
