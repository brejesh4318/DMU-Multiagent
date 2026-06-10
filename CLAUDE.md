# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

**DMU Analytics Platform v2** — an AI-powered educational decision-intelligence system for the Nilgiris district (Tamil Nadu). A natural-language query hits a LangGraph multi-agent supervisor that routes to one of four specialists (SQL analytics over DuckDB, RAG over PDF/DOCX, web search via Tavily, or visualization via Plotly). All LLM calls go through **Groq** (`langchain-groq`). A React frontend talks to a FastAPI backend.

## Commands

All `make` targets run from the repo root. The backend is a Python package rooted at `backend/`; commands `cd backend` first.

```bash
make install          # pip install -r backend/requirements.txt
make ingest           # load all data/*.xlsx into DuckDB (backend/ingest.py)
make ingest-rebuild   # also rebuild RAG index for new PDFs (deletes rag_store.pkl)
make dev              # run backend: uvicorn app.api.main:app --reload --port 8000
make frontend         # cd frontend && npm install && npm start  (port 3000)
make eval             # full evaluation suite (backend/evaluate.py)
make eval-fast        # evaluation without RAGAS (much faster)
make check            # preflight: parse all app/*.py, check .env, check data/*.xlsx
make docker-up        # docker compose up -d (backend + redis + frontend)
make clean            # remove __pycache__, .pytest_cache, faiss_store
```

Run a single eval or ingest directly with flags: `cd backend && python evaluate.py --no-ragas`, `python ingest.py --rebuild-rag`.

**Tests:** `make test` runs `pytest tests/` but **there is no `tests/` directory** — no test suite exists yet. Use `make check` for a syntax/config preflight instead.

**Setup before anything works:** copy `backend/.env.example` → `backend/.env` and fill in `GROQ_API_KEY` (required), `TAVILY_API_KEY`, `LANGCHAIN_API_KEY`. Then drop `.xlsx` files into `data/` and run `make ingest` before `make dev`.

## Architecture

### Request flow
`POST /api/query` (`app/api/main.py`) → `app/agents/graph.py` LangGraph → specialist agent → JSON response.

The graph (`build_graph`) is a **supervisor loop**: every specialist returns to `supervisor`, which inspects tagged `AIMessage`s (`[ANALYTICS]`, `[RAG]`, `[WEB]`, `[VIZ]`) already in state to decide the next hop or `END`. Routing decision for a fresh query comes from `app/agents/router.py::classify` (regex-based: analytics keywords checked first, then web, else rag). Analytics queries that also match `wants_viz` get a second hop through `viz_agent`. State (`AgentState` TypedDict) carries all intermediate results across nodes; the compiled graph uses a `MemorySaver` checkpointer keyed by `thread_id`.

### The four specialists (all under `app/`)
- **`sql/engine.py` (`SQLEngine`)** — text-to-SQL with a self-correcting retry loop. Builds the prompt **dynamically from live DuckDB schema** (`get_schema_dict` + `get_sample_values` inject real column names, subjects, schools, years — nothing hardcoded). Validates generated SQL (blocks DDL/DML via `_UNSAFE` regex, must start with `SELECT`), executes, generates an NL answer, then runs an **LLM-as-Judge** (`JUDGE_MODEL`, stronger model) scoring 1–5. If score < `JUDGE_SCORE_THRESHOLD`, it feeds the failure back and retries up to `SQL_MAX_RETRIES`.
- **`rag/pipeline.py` (`RAGPipeline`)** — full hybrid RAG: parent-child chunking (512-token parents for context, 128-token children embedded for retrieval), dense (BGE embeddings + FAISS cosine) + sparse (BM25) retrieval fused via RRF, then a **Groq LLM cross-encoder rerank** (scores each candidate 1–5, with `time.sleep(0.3)` for rate limits), returns deduplicated parent chunks, grounded answer generation. The whole index is pickled to `VECTOR_STORE_DIR/rag_store.pkl`; `build_index` loads the cache if present, so **delete that pkl to force a rebuild** (the `/api/ingest` endpoint with `rebuild_index=True` and `make ingest-rebuild` do this).
- **`visualization/agent.py::render`** — picks chart type (line/heatmap/pie/bar) from query keywords + data shape, emits Plotly JSON.
- **web agent** (inline in `graph.py`) — Tavily search + Groq synthesis. Auto-enabled only when `TAVILY_API_KEY` is set.

### Data model
`app/ingestion/excel_loader.py` is **schema-agnostic by design**: it scans `DATA_DIR` for any `.xlsx`, reads every sheet, and normalizes wildly varying wide-format columns into a fixed **long format** (one row per student × subject). Column names are matched case-insensitively against candidate lists (`_normalize_col`); year is inferred from the filename digits, management type (`TW`/`GOVT`/`AIDED`) from the sheet name. Produces two DuckDB tables: **`marks`** (long, one row per subject) and **`students`** (deduplicated by rollno). Business rules live here and in the SQL prompt: `marks < 35` = subject FAIL, `fail_flag=1` = student failed overall, `pass_flag`/`fail_flag`/`subject_result` columns.

### Cross-cutting
- **`core/config.py`** — single `Settings` (pydantic-settings) loaded via `get_settings()` (lru_cached). All tuning knobs (chunk sizes, top-k, models, cost-per-token, thresholds) live here. Reads `backend/.env`. Setting `LANGCHAIN_API_KEY` auto-wires LangSmith tracing.
- **`core/database.py`** — the **only** module that imports `duckdb`. Thread-safe singleton connection; all SQL goes through `execute_query`/`execute_ddl`.
- **`monitoring/telemetry.py`** — structlog JSON logging, `track_llm_call` (token + cost accounting from Groq pricing in config), and an in-memory `MetricsStore` (`get_metrics_store`) behind `GET /api/metrics`.
- **`evaluation/evaluator.py`** — routing accuracy, SQL accuracy, retrieval (Recall@k, MRR, nDCG), and optional RAGAS. Backs `POST /api/evaluate` and `evaluate.py`.

### JSON serialization
DuckDB/pandas/numpy results are not JSON-safe (NaN, numpy scalars, DataFrames). `app/api/main.py::_safe` recursively coerces everything to JSON primitives (NaN/Inf → null), and `analytics_agent` does the same DataFrame→records conversion. **If you touch query output, preserve this — raw DataFrames or numpy types will break the API response.**

## Important: stale duplicate code

`backend/` contains **two copies** of most modules. The **canonical package is `backend/app/`** (`app.agents`, `app.core`, etc.) — this is what the Makefile, `uvicorn app.api.main:app`, `ingest.py`, and `evaluate.py` all import. The top-level `backend/agents/`, `backend/api/`, `backend/core/`, `backend/rag/`, etc. are **stale duplicates** and are NOT imported by anything. There is also a junk directory literally named `{backend,frontend}/...` at the repo root (a shell brace-expansion that ran as a literal path) and one-off patch scripts (`fix_serialization.py`, `patch_graph.py`). **Always edit files under `backend/app/`** unless you have a specific reason not to; ignore the top-level duplicates and the `{backend` directory.

## Conventions

- Singletons are lazy module-level globals behind `_get()`-style accessors (`get_graph`, `get_connection`, `get_settings`, `_sql`/`_rag` in `graph.py`). Reuse them rather than constructing fresh engines.
- Groq is rate-limited; the codebase sprinkles `time.sleep(0.3–0.5)` in retry/rerank loops. Keep this in mind when adding LLM-call loops.
- Two models by role: `LLM_MODEL` (fast, `llama-3.1-8b-instant`) for generation, `JUDGE_MODEL` (stronger, `llama-3.1-70b-versatile`) for evaluation.
