"""
Monitoring module:
  - Structured JSON logging (structlog)
  - Token tracking per request
  - Cost tracking (Groq pricing)
  - LangSmith integration via config
"""
from __future__ import annotations
import logging
import sys
import time
import threading
from dataclasses import dataclass, field
from typing import Optional
import structlog
from app.core.config import get_settings


# ── Structured logging ────────────────────────────────────────────────────────

def configure_logging(level: str = "INFO") -> None:
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    handler = logging.StreamHandler(sys.stdout)
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)


# ── Token + cost tracking ─────────────────────────────────────────────────────

@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    model: str = ""
    latency_ms: float = 0.0


@dataclass
class RequestMetrics:
    request_id: str = ""
    query: str = ""
    query_type: str = ""
    agent_path: list[str] = field(default_factory=list)
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    judge_score: Optional[int] = None
    routing_correct: Optional[bool] = None
    sql_executed: bool = False
    sql_error: Optional[str] = None
    latency_ms: float = 0.0
    timestamp: str = ""


class MetricsStore:
    """Thread-safe in-memory metrics store (last 1000 requests)."""

    def __init__(self, maxlen: int = 1000):
        self._lock = threading.Lock()
        self._records: list[RequestMetrics] = []
        self._maxlen = maxlen
        self._total_tokens = 0
        self._total_cost = 0.0
        self._total_requests = 0

    def record(self, m: RequestMetrics) -> None:
        with self._lock:
            self._records.append(m)
            if len(self._records) > self._maxlen:
                self._records.pop(0)
            self._total_tokens += m.token_usage.total_tokens
            self._total_cost += m.token_usage.cost_usd
            self._total_requests += 1

    def summary(self) -> dict:
        with self._lock:
            recent = self._records[-100:]
            avg_latency = (
                sum(r.latency_ms for r in recent) / len(recent) if recent else 0
            )
            routing = [r for r in recent if r.routing_correct is not None]
            routing_acc = (
                sum(1 for r in routing if r.routing_correct) / len(routing)
                if routing else None
            )
            judge_scores = [r.judge_score for r in recent if r.judge_score]
            avg_judge = sum(judge_scores) / len(judge_scores) if judge_scores else None
            type_counts: dict[str, int] = {}
            for r in recent:
                type_counts[r.query_type] = type_counts.get(r.query_type, 0) + 1
            return {
                "total_requests": self._total_requests,
                "total_tokens": self._total_tokens,
                "total_cost_usd": round(self._total_cost, 6),
                "avg_latency_ms": round(avg_latency, 1),
                "routing_accuracy": round(routing_acc, 3) if routing_acc else None,
                "avg_judge_score": round(avg_judge, 2) if avg_judge else None,
                "query_type_breakdown": type_counts,
                "recent_queries": [
                    {
                        "query": r.query[:80],
                        "type": r.query_type,
                        "latency_ms": r.latency_ms,
                        "tokens": r.token_usage.total_tokens,
                        "cost_usd": r.token_usage.cost_usd,
                        "judge_score": r.judge_score,
                    }
                    for r in list(reversed(self._records))[:20]
                ],
            }


# Singleton
_metrics_store = MetricsStore()


def get_metrics_store() -> MetricsStore:
    return _metrics_store


def compute_cost(input_tokens: int, output_tokens: int) -> float:
    cfg = get_settings()
    return (
        input_tokens * cfg.COST_PER_1M_INPUT / 1_000_000
        + output_tokens * cfg.COST_PER_1M_OUTPUT / 1_000_000
    )


def track_llm_call(response, model: str, t0: float) -> TokenUsage:
    """Extract token usage from a Groq ChatCompletion response."""
    usage = getattr(response, "usage_metadata", None) or getattr(response, "response_metadata", {}).get("token_usage", {})
    input_t  = getattr(usage, "input_tokens",  0) if hasattr(usage, "input_tokens")  else usage.get("prompt_tokens", 0)
    output_t = getattr(usage, "output_tokens", 0) if hasattr(usage, "output_tokens") else usage.get("completion_tokens", 0)
    total_t  = input_t + output_t
    cost     = compute_cost(input_t, output_t)
    latency  = (time.perf_counter() - t0) * 1000
    return TokenUsage(
        input_tokens=input_t,
        output_tokens=output_t,
        total_tokens=total_t,
        cost_usd=cost,
        model=model,
        latency_ms=round(latency, 1),
    )
