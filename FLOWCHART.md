# DMU Analytics Platform v2 — Architecture & Flow Diagrams

> Mermaid diagrams. They render on GitHub and in VS Code (Markdown Preview Mermaid
> extension). To export an image for a slide/interviewer, paste any block into
> <https://mermaid.live>.

---

## 1. System architecture (high level)

```mermaid
flowchart LR
  User([User])
  FE["React UI<br/>recharts charts + summary"]
  API["FastAPI<br/>api/main.py"]
  G{{"LangGraph Supervisor<br/>MemorySaver checkpointer (thread_id)"}}

  SQL["Analytics<br/>SQLEngine"]
  RAG["RAG Pipeline"]
  WEB["Web search"]
  VIZ["Visualization<br/>Plotly plan"]

  DB[("DuckDB<br/>marks / students")]
  VS[("rag_store.pkl<br/>parents / children / embeddings")]
  TAV[("Tavily API")]
  GROQ[("Groq LLMs<br/>8B gen / 120B SQL / 70B viz")]

  User --> FE
  FE -->|"POST /api/query (thread_id)"| API
  API --> G
  G --> SQL
  G --> RAG
  G --> WEB
  G --> VIZ
  SQL --> DB
  RAG --> VS
  WEB --> TAV
  SQL -. LLM .-> GROQ
  RAG -. LLM .-> GROQ
  WEB -. LLM .-> GROQ
  VIZ -. LLM .-> GROQ
  API -->|"answer + chart + summary (or clarify)"| FE
```

---

## 2. The brain — routing, HITL, and the tool loop

This is the one to walk an interviewer through. It shows the **two-layer router**, the
**human-in-the-loop** clarify gate, the crash-proof regex fallback, and the post-tool hop
loop (analytics → rag → viz).

```mermaid
flowchart TD
  Q["Query + thread_id"] --> L1["Layer 1: classify_intent<br/>8B, lru_cached → (intent, confidence)"]
  L1 --> C{"confidence ≥ 0.65 ?"}

  C -->|yes| FP{"intent"}
  FP -->|conversational| CONV["8B friendly reply → END"]
  FP -->|"analytics / rag / web / viz"| TOOL["synthetic tool call"]
  FP -->|hybrid| HYB["run_analytics + queue rag/viz hops"]

  C -->|no| GATE{"matched_route == None<br/>AND no prior context ?"}
  GATE -->|"yes — cold + vague"| CLAR["HITL: ask user to clarify<br/>+ example prompts → END"]
  GATE -->|no| L2["Layer 2: supervisor LLM<br/>bind_tools + full history"]

  L2 -->|tool call| TOOL
  L2 -->|direct answer| FIN
  L2 -->|"error / tool_use_failed"| FB{"matched_route ?"}
  FB -->|route| TOOL
  FB -->|None| CLAR

  HYB --> EXE
  TOOL --> EXE["tools_node_wrapper<br/>executes analytics / rag / web / viz"]
  EXE --> POST{"post-tool:<br/>more hops?"}
  POST -->|"pending hop / auto-viz"| TOOL
  POST -->|done| FIN["tagged final answer<br/>[ANALYTICS]/[RAG]/[WEB] → END"]
```

**HITL gate in words:** ask the human **only** when the classifier is unsure **and** the
regex finds no keyword **and** it isn't a follow-up (no prior context). Everything with a
keyword *or* conversation history goes to Layer 2.

---

## 3. SQL analytics — self-correcting loop

```mermaid
flowchart TD
  S["run(question)"] --> CK{"LRU cache hit?<br/>(normalized question)"}
  CK -->|yes| RET["return cached SQLResult<br/>0 tokens"]
  CK -->|no| GEN["generate SQL<br/>120B, prompt built from live schema"]
  GEN --> V{"validate<br/>SELECT-only, no DDL/DML"}
  V -->|fail| RT["feedback + sleep 0.5<br/>retry ≤ SQL_MAX_RETRIES"]
  V -->|ok| EX{"execute on DuckDB"}
  EX -->|error| RT
  EX -->|"empty (1st time)"| SOFT["one free soft retry<br/>(no judge)"]
  EX -->|rows| A["generate NL answer<br/>8B, numbers only from data"]
  SOFT --> GEN
  A --> J{"LLM-as-Judge once<br/>score 1–5"}
  J -->|"score 3 or more"| OK["cache success + return"]
  J -->|"score under 3"| RT
  RT --> GEN
  RT -->|retries exhausted| EXH["'Could not generate a reliable answer'<br/>score 0, never throws"]
```

---

## 4. RAG — hybrid retrieval pipeline

```mermaid
flowchart LR
  Q["query"] --> D["Dense<br/>BGE embed → FAISS cosine<br/>top-10 children"]
  Q --> SP["Sparse<br/>BM25 keyword<br/>top-10 children"]
  D --> RRF["RRF fuse on rank<br/>score += 1 / (61 + rank)"]
  SP --> RRF
  RRF --> RR["LLM cross-encoder rerank<br/>score 1–5 → keep top-5"]
  RR --> P["map children → parent chunks<br/>dedupe"]
  P --> GEN["grounded answer<br/>ONLY from context, else 'Not found'"]
```

---

## 5. Chart rendering path (backend → frontend)

```mermaid
flowchart LR
  REC["sql_result.result_records"] --> RENDER["render(df, query)<br/>VIZ_MODEL plan → Plotly figure"]
  RENDER --> DEC["_decode_typed_arrays<br/>base64 typed-array → plain lists"]
  DEC --> CJ["chart_json (x / y lists)"]
  RENDER --> SUM["_summarize<br/>3–4 line summary of the rows"]
  CJ --> FE["recharts Bar/Line/Pie<br/>+ on-bar value labels"]
  SUM --> FE2["Summary box under chart"]
```

> Why `_decode_typed_arrays` exists: Plotly 6 serializes numeric arrays as base64
> typed-array blobs; recharts needs plain lists, otherwise every `y` reads as 0 and the
> bars render blank.

---

*Canonical code lives under `backend/app/`. See `PROJECT_DOCUMENTATION.md` for the full
written deep-dive.*
