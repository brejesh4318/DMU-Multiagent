"""
FastAPI application — all endpoints.

POST /api/query      — main query endpoint
POST /api/ingest     — reload Excel + rebuild RAG index
POST /api/evaluate   — run full evaluation suite
GET  /api/health     — liveness probe
GET  /api/metrics    — runtime metrics (tokens, cost, latency, routing)
"""
from __future__ import annotations
import math
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
from app.core.database import execute_query
from app.evaluation.evaluator import Evaluator
from app.ingestion.excel_loader import ingest_all_excel
from app.monitoring.telemetry import configure_logging, get_logger, get_metrics_store
from app.rag.pipeline import RAGPipeline
from app.sql.engine import SQLEngine

logger = get_logger(__name__)


# ── Safe JSON serializer ──────────────────────────────────────────────────────

def _safe(obj: Any) -> Any:
    """
    Recursively convert any object to a JSON-safe Python primitive.
    Handles: numpy types, pandas DataFrames, NaN/inf floats, dataclasses, etc.
    """
    # None passthrough
    if obj is None:
        return None

    # Already primitive
    if isinstance(obj, (str, bool)):
        return obj

    # Numeric — catch NaN and inf
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, int):
        return obj

    # numpy scalars (import lazily to avoid hard dependency at module level)
    type_name = type(obj).__name__
    module     = type(obj).__module__
    if module and module.startswith("numpy"):
        try:
            v = obj.item()  # converts numpy scalar → Python native
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                return None
            return v
        except Exception:
            return str(obj)

    # pandas DataFrame
    if type_name == "DataFrame":
        try:
            # astype(object) ensures NaN becomes Python None, not float nan
            safe = obj.astype(object).where(obj.notna(), other=None)
            records = safe.to_dict(orient="records")
            # Second pass to catch any remaining numpy scalars in values
            return [{str(k): _safe(v) for k, v in row.items()} for row in records]
        except Exception:
            return []

    # pandas Series
    if type_name == "Series":
        try:
            return obj.tolist()
        except Exception:
            return []

    # dict
    if isinstance(obj, dict):
        return {str(k): _safe(v) for k, v in obj.items()}

    # list / tuple
    if isinstance(obj, (list, tuple)):
        return [_safe(v) for v in obj]

    # dataclass
    try:
        import dataclasses
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return _safe(dataclasses.asdict(obj))
    except Exception:
        pass

    # Pydantic model
    try:
        if hasattr(obj, "model_dump"):
            return _safe(obj.model_dump())
        if hasattr(obj, "dict"):
            return _safe(obj.dict())
    except Exception:
        pass

    # Fallback
    return str(obj)


# ── Lifespan ──────────────────────────────────────────────────────────────────

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
        description="AI-Powered Educational Decision Intelligence — Nilgiris DMU",
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
    query:      str  = Field(..., min_length=3, max_length=1000)
    show_sql:   bool = False
    show_judge: bool = False
    thread_id:  Optional[str] = None


class QueryResponse(BaseModel):
    query:           str
    query_type:      str
    answer:          str
    sql:             Optional[str]  = None
    judge_score:     Optional[int]  = None
    judge_issues:    list[str]      = []
    chart:           Optional[dict] = None
    recommendations: list[dict]     = []
    latency_ms:      float          = 0.0
    tokens:          int            = 0
    cost_usd:        float          = 0.0
    attempts:        int            = 1
    error:           Optional[str]  = None


class IngestRequest(BaseModel):
    data_dir:      Optional[str] = None
    rebuild_index: bool          = False


class IngestResponse(BaseModel):
    status:       str
    marks_rows:   int          = 0
    student_rows: int          = 0
    files:        int          = 0
    error:        Optional[str] = None


class EvalRequest(BaseModel):
    include_ragas: bool = True
    include_sql:   bool = True


# ── /api/query ────────────────────────────────────────────────────────────────

@router.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest) -> QueryResponse:
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

    # ── Extract answer from agent messages ────────────────────────────────
    answer = ""
    for msg in state.get("messages", []):
        if not isinstance(msg, AIMessage):
            continue
        c = msg.content
        for tag in ("[ANALYTICS]", "[RAG]", "[WEB]"):
            if tag in c:
                answer = c.replace(tag, "").split("[SQL]")[0].strip()
                break

    # ── Extract sql_result safely ─────────────────────────────────────────
    sql_r      = _safe(state.get("sql_result", {}))
    chart_raw  = _safe(state.get("chart_output") or {})
    recs_raw   = _safe(state.get("recommendations") or [])

    # Normalise chart — must be a dict or None
    chart: Optional[dict] = None
    if chart_raw and isinstance(chart_raw, dict) and chart_raw.get("chart_json"):
        chart = chart_raw

    # Normalise recommendations — must be list[dict]
    recs: list[dict] = []
    for r in (recs_raw if isinstance(recs_raw, list) else []):
        if isinstance(r, dict):
            recs.append(r)
        elif isinstance(r, str):
            recs.append({"text": r})

    return QueryResponse(
        query=req.query,
        query_type=query_type,
        answer=answer or "No answer generated.",
        sql=str(sql_r.get("sql", "")) if req.show_sql and sql_r.get("sql") else None,
        judge_score=int(sql_r["judge_score"]) if req.show_judge and sql_r.get("judge_score") else None,
        judge_issues=list(sql_r.get("judge_issues") or []) if req.show_judge else [],
        chart=chart,
        recommendations=recs,
        latency_ms=latency,
        tokens=int(sql_r.get("tokens") or 0),
        cost_usd=float(sql_r.get("cost_usd") or 0.0),
        attempts=int(sql_r.get("attempts") or 1),
        error=str(sql_r["error"]) if sql_r.get("error") else None,
    )


# ── /api/ingest ───────────────────────────────────────────────────────────────

@router.post("/ingest", response_model=IngestResponse)
async def ingest_endpoint(req: IngestRequest) -> IngestResponse:
    try:
        counts = ingest_all_excel(req.data_dir)
    except Exception as e:
        return IngestResponse(status="error", error=str(e))

    if req.rebuild_index:
        try:
            from pathlib import Path
            cfg   = get_settings()
            cache = Path(cfg.VECTOR_STORE_DIR) / "rag_store.pkl"
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


# ── /api/evaluate ─────────────────────────────────────────────────────────────

@router.post("/evaluate")
async def evaluate_endpoint(req: EvalRequest) -> dict:
    try:
        rag = RAGPipeline()
        rag.build_index()
        sql = SQLEngine() if req.include_sql else None
        ev  = Evaluator(rag=rag, sql=sql)
        rep = ev.run_all(include_ragas=req.include_ragas)
        return _safe({
            "overall_pass": rep.overall_pass,
            "summary":      rep.summary,
            "routing": {
                "accuracy": rep.routing.accuracy,
                "total":    rep.routing.total,
                "correct":  rep.routing.correct,
                "failures": rep.routing.failures,
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
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── /api/health ───────────────────────────────────────────────────────────────

@router.get("/health")
async def health_endpoint() -> dict:
    cfg   = get_settings()
    db_ok = False
    rows  = 0
    try:
        df    = execute_query("SELECT COUNT(*) AS n FROM marks")
        rows  = int(df["n"].iloc[0])
        db_ok = True
    except Exception:
        pass
    return {
        "status":       "ok" if db_ok else "degraded",
        "version":      cfg.APP_VERSION,
        "db_connected": db_ok,
        "marks_rows":   rows,
    }


# ── /api/metrics ──────────────────────────────────────────────────────────────

@router.get("/metrics")
async def metrics_endpoint() -> dict:
    return _safe(get_metrics_store().summary())


app = create_app()