"""
Redis cache wrapper — degrades gracefully if Redis is unavailable.
"""
from __future__ import annotations
import hashlib, json
from typing import Any, Optional
import redis
from app.core.config import get_settings
from app.monitoring.telemetry import get_logger

logger = get_logger(__name__)
_client: Optional[redis.Redis] = None


def _get() -> Optional[redis.Redis]:
    global _client
    if _client is None:
        try:
            cfg = get_settings()
            _client = redis.from_url(cfg.REDIS_URL, decode_responses=True)
            _client.ping()
        except Exception as e:
            logger.warning("redis_unavailable", error=str(e))
            _client = None
    return _client


def cache_key(prefix: str, query: str) -> str:
    h = hashlib.sha256(query.lower().strip().encode()).hexdigest()[:16]
    return f"dmu:{prefix}:{h}"


def get_cached(key: str) -> Optional[Any]:
    c = _get()
    if not c:
        return None
    try:
        raw = c.get(key)
        return json.loads(raw) if raw else None
    except Exception:
        return None


def set_cached(key: str, value: Any, ttl: Optional[int] = None) -> None:
    c = _get()
    if not c:
        return
    try:
        cfg = get_settings()
        c.setex(key, ttl or cfg.CACHE_TTL, json.dumps(value, default=str))
    except Exception:
        pass


def invalidate(prefix: str) -> int:
    c = _get()
    if not c:
        return 0
    try:
        keys = c.keys(f"dmu:{prefix}:*")
        return c.delete(*keys) if keys else 0
    except Exception:
        return 0
