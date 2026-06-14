# DMU Analytics Platform v2

AI-Powered Educational Decision Intelligence for Nilgiris District

---

## Features

| Category | What's included |
|---|---|
| **Agents** | LangGraph Supervisor, Analytics (SQL), RAG, Visualization, Web Search |
| **Routing** | Two-layer (LLM classifier + regex), crash-proof fallback, **human-in-the-loop clarify** for cold/unroutable queries |
| **Retrieval** | BGE Embeddings, BM25, RRF Fusion, Cross-Encoder Reranking, Parent-Child Chunking |
| **Visualization** | Plotly chart planning → recharts render with on-bar value labels + auto 3-4 line summary |
| **Evaluation** | Recall@5, Recall@10, MRR, nDCG, RAGAS, SQL Accuracy, Routing Accuracy, LLM-as-Judge |
| **Monitoring** | LangSmith, Structured JSON Logs, Token Tracking, Cost Tracking |
| **Deployment** | FastAPI, React, In-memory LRU Result Cache, Docker Compose |
| **Data** | Auto-detect any .xlsx files + any sheet names, long-format normalization |

---

## Quick Start

### 1. Setup
```bash
cp backend/.env.example backend/.env
# Fill in: GROQ_API_KEY, TAVILY_API_KEY, LANGCHAIN_API_KEY
```

### 2. Add data
```
data/
  your_file.xlsx      ← any xlsx file, any sheet names
  SLAS_report.pdf     ← any PDF for RAG
```

### 3. Install & ingest
```bash
make install
make ingest
```

### 4. Run
```bash
# Terminal 1 — backend
make dev

# Terminal 2 — frontend
make frontend
```

Open: http://localhost:3000

### Or with Docker
```bash
docker compose up -d
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| POST | /api/query | Natural language query |
| POST | /api/ingest | Reload Excel + rebuild RAG |
| POST | /api/evaluate | Run full evaluation suite |
| GET | /api/health | Liveness probe |
| GET | /api/metrics | Token/cost/latency metrics |

### Query example
```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Which subject has the most failures?", "show_sql": true, "show_judge": true}'
```

### Evaluate example
```bash
curl -X POST http://localhost:8000/api/evaluate \
  -H "Content-Type: application/json" \
  -d '{"include_ragas": true, "include_sql": true}'
```

---

## Evaluation Metrics

Run: `make eval` or `make eval-fast` (skips RAGAS)

| Metric | Target | What it measures |
|---|---|---|
| Routing Accuracy | ≥ 85% | Correct agent selected |
| SQL Accuracy | ≥ 80% | SQL executed + returned data |
| LLM-as-Judge | ≥ 3.5/5 | Answer quality per query |
| Recall@5 | ≥ 0.70 | Relevant doc in top 5 |
| Recall@10 | ≥ 0.80 | Relevant doc in top 10 |
| MRR | ≥ 0.60 | Mean Reciprocal Rank |
| nDCG@10 | ≥ 0.65 | Normalized Discounted Cumulative Gain |
| RAGAS Faithfulness | ≥ 0.80 | No hallucination |
| RAGAS Answer Relevancy | ≥ 0.75 | On-topic answers |

---

## Adding New Data

Drop any `.xlsx` file into `data/` — the system auto-detects all sheets and file names. Then:
```bash
make ingest        # load new Excel data
make ingest-rebuild  # also rebuild RAG index for new PDFs
```

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | ✅ | Groq LLM key |
| `TAVILY_API_KEY` | ✅ | Web search key |
| `LANGCHAIN_API_KEY` | ✅ | LangSmith tracing |
| `LLM_MODEL` | optional | Default: llama-3.1-8b-instant |
| `JUDGE_MODEL` | optional | Default: llama-3.1-70b-versatile |
