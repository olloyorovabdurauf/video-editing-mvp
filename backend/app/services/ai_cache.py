"""
Content-addressed cache for AI results (Redis).

Why: the two expensive, deterministic-enough operations in the pipeline —
segment picking and (optionally) keyword extraction — get re-run whenever a
user retries the same video or tweaks an unrelated knob. Caching on a hash of
the *inputs* makes repeats free and instant.

Design: fail-open. A cache miss OR a Redis hiccup must never break the
pipeline — we just recompute. So every Redis call is wrapped.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

import redis
from loguru import logger

from app.config import get_settings

_r = redis.from_url(get_settings().redis_url, decode_responses=True)


def key(namespace: str, *parts: Any) -> str:
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:24]
    return f"aicache:{namespace}:{digest}"


def get_json(k: str) -> Any | None:
    try:
        raw = _r.get(k)
        return json.loads(raw) if raw else None
    except Exception as e:                       # redis down / bad json → recompute
        logger.debug("ai_cache get miss ({}): {}", k, e)
        return None


def set_json(k: str, value: Any, *, ttl_s: int) -> None:
    try:
        _r.setex(k, ttl_s, json.dumps(value))
    except Exception as e:
        logger.debug("ai_cache set skipped ({}): {}", k, e)
