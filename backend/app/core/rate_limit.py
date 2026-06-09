"""
Lightweight per-key sliding-window rate limiter.

Redis ZSET trick: each request is a zset member with score=epoch_ms.
We ZREMRANGEBYSCORE to drop entries outside the window, then ZCARD to
count. Atomic via a single Lua script.

This is intentionally tiny. If you outgrow it, slowapi/limits/starlette-
limiter are drop-in upgrades — but most apps never need to upgrade.
"""
from __future__ import annotations

import time

import redis
from fastapi import HTTPException, Request, status

from app.config import get_settings

_r = redis.from_url(get_settings().redis_url, decode_responses=True)

# Single Lua = atomic. If we did this in three Python calls there'd be a
# race where two requests both read count=N-1, both insert, both pass.
_LIMITER_LUA = _r.register_script("""
local key = KEYS[1]
local now_ms = tonumber(ARGV[1])
local window_ms = tonumber(ARGV[2])
local max = tonumber(ARGV[3])
redis.call('ZREMRANGEBYSCORE', key, 0, now_ms - window_ms)
local count = redis.call('ZCARD', key)
if count >= max then
    return {0, count}
end
redis.call('ZADD', key, now_ms, now_ms .. ':' .. math.random())
redis.call('PEXPIRE', key, window_ms)
return {1, count + 1}
""")


def _bucket_key(scope: str, identity: str) -> str:
    return f"rl:{scope}:{identity}"


def _check(scope: str, identity: str, *, max_per_window: int, window_s: int) -> tuple[bool, int]:
    now_ms = int(time.time() * 1000)
    allowed, count = _LIMITER_LUA(
        keys=[_bucket_key(scope, identity)],
        args=[now_ms, window_s * 1000, max_per_window],
    )
    return bool(int(allowed)), int(count)


def rate_limit(scope: str, *, max_per_window: int, window_s: int):
    """
    FastAPI dependency factory.

        @router.post("/reels", dependencies=[Depends(rate_limit("reels", 10, 60))])

    Identity = authenticated user_id when present (header X-User-Id), else
    the client IP. Behind a proxy you'll want to trust X-Forwarded-For —
    that's a separate ProxyHeadersMiddleware concern.
    """
    async def dep(request: Request) -> None:
        identity = request.headers.get("X-User-Id") or (
            request.client.host if request.client else "anon"
        )
        allowed, count = _check(scope, identity, max_per_window=max_per_window, window_s=window_s)
        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"rate limit: {max_per_window} per {window_s}s (you: {count})",
                headers={"Retry-After": str(window_s)},
            )
    return dep
