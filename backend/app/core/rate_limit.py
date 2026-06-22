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
from datetime import datetime, timezone

import redis
from fastapi import Depends, HTTPException, Request, status

from app.config import get_settings
from app.core.auth import optional_user

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
    async def dep(request: Request, user: str | None = Depends(optional_user)) -> None:
        # Prefer the *verified* user id (forgery-proof: it came from a signed
        # JWT) so a logged-in user's limit follows them across IPs and a NAT'd
        # office doesn't share one bucket. Fall back to peer IP for anonymous
        # callers (we run behind --proxy-headers, so client.host is the real peer).
        identity = (f"u:{user}" if user and user != "anonymous"
                    else f"ip:{request.client.host if request.client else 'anon'}")
        allowed, count = _check(scope, identity, max_per_window=max_per_window, window_s=window_s)
        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"rate limit: {max_per_window} per {window_s}s (you: {count})",
                headers={"Retry-After": str(window_s)},
            )
    return dep


# ---------------------------------------------------------------------------
# Daily per-user job quota — abuse / cost guard distinct from the burst limiter.
# Each video job costs real money (Whisper + GPT-4o); a single user shouldn't be
# able to launch hundreds/day. Counter resets at UTC midnight via key TTL.
# ---------------------------------------------------------------------------

class QuotaExceeded(Exception):
    def __init__(self, used: int, limit: int):
        self.used, self.limit = used, limit
        super().__init__(f"daily job quota reached: {used}/{limit}")


def consume_daily_job_quota(user_id: str, *, limit: int) -> int:
    """
    Atomically increment today's job count for a user; raise QuotaExceeded if
    over `limit`. Returns the new count. Anonymous/dev callers are not capped
    here (billing/credits guard those paths).
    """
    if not user_id or user_id == "anonymous" or limit <= 0:
        return 0
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    key = f"quota:jobs:{user_id}:{today}"
    pipe = _r.pipeline()
    pipe.incr(key)
    pipe.expire(key, 60 * 60 * 26)   # ~1 day + slack; self-cleans
    used, _ = pipe.execute()
    used = int(used)
    if used > limit:
        raise QuotaExceeded(used, limit)
    return used
