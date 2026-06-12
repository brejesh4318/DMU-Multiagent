"""
FastAPI application — all endpoints.

POST /api/query      — main query endpoint
POST /api/ingest     — reload Excel + rebuild RAG index
POST /api/evaluate   — run full evaluation suite
GET  /api/health     — liveness probe
GET  /api/metrics    — runtime metrics (tokens, cost, latency, routing)
"""
from __future__ import annotations
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field

from app.agents.graph import get_graph
from app.agents.router import classify
from app.core.config import get_settings
from app.core.database import execute_query, table_exists
from app.evaluation.evaluator import Evaluator
from app.ingestion.excel_loader import ingest_all_excel
from app.monitoring.telemetry import configure_logging, get_logger, get_metrics_store
from app.rag.pipeline import RAGPipeline
from app.sql.engine import SQLEngine

logger = get_logger(__name__)



import math as _math

def _safe(obj):
    """Recursively convert any object to JSON-serializable Python primitives."""
    if obj is None:
        return None
    if isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, float):
        if _math.isnan(obj) or _math.isinf(obj):
            return None
        return obj
    if isinstance(obj, int):
        return obj
    type_name = type(obj).__name__
    module    = type(obj).__module__ or ""
    if module.startswith("numpy"):
        try:
            v = obj.item()
            return None if (isinstance(v, float) and (_math.isnan(v) or _math.isinf(v))) else v
        except Exception:
            return str(obj)
    if type_name == "DataFrame":
        try:
            safe = obj.astype(object).where(obj.notna(), other=None)
            records = safe.to_dict(orient="records")
            return [{str(k): _safe(v) for k, v in row.items()} for row in records]
        except Exception:
            return []
    if type_name == "Series":
        try:
            return [_safe(v) for v in obj.tolist()]
        except Exception:
            return []
    if isinstance(obj, dict):
        return {str(k): _safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe(v) for v in obj]
    try:
        import dataclasses
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return _safe(dataclasses.asdict(obj))
    except Exception:
        pass
    return str(obj)

@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = get_settings()
    configure_logging(cfg.LOG_LEVEL)
    logger.info("startup", version=cfg.APP_VERSION)
    try:
        get_graph()
        logger.info("graph_prewarmed")
    except Exception as e:
        logger.warning("prewarm_failed", error=str(e))
    yield
    logger.info("shutdown")


def create_app() -> FastAPI:
    cfg = get_settings()
    application = FastAPI(
        title=cfg.APP_NAME, version=cfg.APP_VERSION,
        description="Districting Monitoring Unit",
        lifespan=lifespan,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
        allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
    )
    application.include_router(router, prefix="/api")
    return application


from fastapi import APIRouter
router = APIRouter()


# ── Schemas ───────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=1000)
    show_sql: bool = False
    show_judge: bool = False
    thread_id: Optional[str] = None


class QueryResponse(BaseModel):
    query:           str
    query_type:      str
    answer:          str
    sql:             Optional[str]        = None
    judge_score:     Optional[int]        = None
    judge_issues:    list[str]            = []
    chart:           Optional[dict]       = None
    recommendations: list[dict]           = []
    latency_ms:      float                = 0.0
    tokens:          int                  = 0
    cost_usd:        float                = 0.0
    attempts:        int                  = 1
    error:           Optional[str]        = None


class IngestRequest(BaseModel):
    data_dir:      Optional[str] = None
    rebuild_index: bool          = False


class IngestResponse(BaseModel):
    status:       str
    marks_rows:   int = 0
    student_rows: int = 0
    files:        int = 0
    error:        Optional[str] = None


class EvalRequest(BaseModel):
    include_ragas: bool = True
    include_sql:   bool = True


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest) -> QueryResponse:
    t0         = time.perf_counter()
    thread_id  = req.thread_id or f"t_{uuid.uuid4().hex[:8]}"
    query_type = classify(req.query)

    try:
        state = get_graph().invoke(
            {
                "messages":        [HumanMessage(content=req.query)],
                "next_node":       "",
                "query_type":      query_type,
                "request_id":      thread_id,
                "sql_result":      {},
                "rag_result":      {},
                "web_result":      {},
                "chart_output":    {},
                "recommendations": [],
                "metrics":         {},
            },
            config={"configurable": {"thread_id": thread_id}},
        )
    except Exception as e:
        logger.error("graph_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

    latency = round((time.perf_counter() - t0) * 1000, 1)

    # Extract answer
    answer = ""
    for msg in state["messages"]:
        if not isinstance(msg, AIMessage):
            continue
        c = msg.content
        for tag in ("[ANALYTICS]", "[RAG]", "[WEB]"):
            if tag in c:
                answer = c.replace(tag, "").split("[SQL]")[0].strip()
                break

    # Fallback: no tagged message (e.g. conversational reply) → use the last
    # AIMessage that actually has text content.
    if not answer:
        for msg in reversed(state["messages"]):
            if isinstance(msg, AIMessage) and (msg.content or "").strip():
                answer = msg.content.strip()
                break

    sql_r = _safe(state.get("sql_result", {}))
    return QueryResponse(
        query=req.query,
        query_type=query_type,
        answer=answer,
        sql=sql_r.get("sql") if req.show_sql else None,
        judge_score=sql_r.get("judge_score") if req.show_judge else None,
        judge_issues=sql_r.get("judge_issues", []) if req.show_judge else [],
        chart=state.get("chart_output") or None,
        recommendations=state.get("recommendations", []),
        latency_ms=latency,
        tokens=sql_r.get("tokens", 0),
        cost_usd=sql_r.get("cost_usd", 0.0),
        attempts=sql_r.get("attempts", 1),
        error=sql_r.get("error"),
    )


@router.post("/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest) -> IngestResponse:
    try:
        counts = ingest_all_excel(req.data_dir)
    except Exception as e:
        return IngestResponse(status="error", error=str(e))

    if req.rebuild_index:
        try:
            from pathlib import Path
            cache = Path(get_settings().VECTOR_STORE_DIR) / "rag_store.pkl"
            cache.unlink(missing_ok=True)
            rag = RAGPipeline()
            rag.build_index(req.data_dir)
        except Exception as e:
            logger.warning("rag_rebuild_error", error=str(e))

    return IngestResponse(
        status="ok",
        marks_rows=counts["marks_rows"],
        student_rows=counts["student_rows"],
        files=counts["files"],
    )


@router.post("/evaluate")
async def evaluate(req: EvalRequest) -> dict:
    try:
        rag = RAGPipeline(); rag.build_index()
        sql = SQLEngine() if req.include_sql else None
        ev  = Evaluator(rag=rag, sql=sql)
        rep = ev.run_all(include_ragas=req.include_ragas)
        return {
            "overall_pass":  rep.overall_pass,
            "summary":       rep.summary,
            "routing": {
                "accuracy":  rep.routing.accuracy,
                "total":     rep.routing.total,
                "correct":   rep.routing.correct,
                "failures":  rep.routing.failures,
            },
            "sql": {
                "accuracy":        rep.sql.accuracy,
                "avg_judge_score": rep.sql.avg_judge_score,
                "avg_latency_ms":  rep.sql.avg_latency_ms,
                "per_query":       rep.sql.per_query,
            },
            "retrieval": {
                "recall_at_5":  rep.retrieval.recall_at_5,
                "recall_at_10": rep.retrieval.recall_at_10,
                "mrr":          rep.retrieval.mrr,
                "ndcg_at_10":   rep.retrieval.ndcg_at_10,
                "per_question": rep.retrieval.per_question,
            },
            "ragas": {
                "faithfulness":      rep.ragas.faithfulness,
                "answer_relevancy":  rep.ragas.answer_relevancy,
                "context_precision": rep.ragas.context_precision,
                "context_recall":    rep.ragas.context_recall,
                "computed":          rep.ragas.computed,
                "error":             rep.ragas.error,
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health() -> dict:
    cfg     = get_settings()
    db_ok   = False
    rows    = 0
    try:
        df   = execute_query("SELECT COUNT(*) AS n FROM marks")
        rows = int(df["n"].iloc[0])
        db_ok = True
    except Exception:
        pass
    return {
        "status":       "ok" if db_ok else "degraded",
        "version":      cfg.APP_VERSION,
        "db_connected": db_ok,
        "marks_rows":   rows,
    }


@router.get("/metrics")
async def metrics() -> dict:
    return get_metrics_store().summary()


app = create_app()
