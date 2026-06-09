"""
Health endpoints.

Two endpoints by convention — your load balancer needs both:

  /livez   — am I alive? (process responding). Used by orchestrator to
             decide whether to restart the container. Should ALWAYS return
             200 unless the process is wedged.

  /healthz — am I ready to serve traffic? Checks dependencies (Redis,
             ffmpeg binary). Returns 503 if any hard dep is unavailable so
             the LB stops routing to us until we recover.

We do NOT check OpenAI / Pexels here — those are upstream failures users
should see as job failures, not API 503s.
"""
from __future__ import annotations

import asyncio
import shutil
import time

import redis.asyncio as aioredis
from fastapi import APIRouter, Response, status

from app.config import get_settings

router = APIRouter(tags=["meta"])

_started_at = time.time()


@router.get("/livez")
async def livez() -> dict:
    return {"ok": True, "uptime_s": round(time.time() - _started_at, 1)}


@router.get("/healthz")
async def healthz(response: Response) -> dict:
    settings = get_settings()
    checks: dict[str, dict] = {}
    healthy = True

    # --- Redis ---
    try:
        r = aioredis.from_url(settings.redis_url, decode_responses=True)
        pong = await asyncio.wait_for(r.ping(), timeout=1.0)
        await r.aclose()
        checks["redis"] = {"ok": bool(pong), "detail": "ping"}
    except Exception as e:
        checks["redis"] = {"ok": False, "error": str(e)[:200]}
        healthy = False

    # --- ffmpeg binary ---
    ffmpeg_path = shutil.which(settings.ffmpeg_binary)
    checks["ffmpeg"] = {"ok": bool(ffmpeg_path), "path": ffmpeg_path}
    if not ffmpeg_path:
        healthy = False

    # --- ffprobe binary ---
    ffprobe_path = shutil.which(settings.ffprobe_binary)
    checks["ffprobe"] = {"ok": bool(ffprobe_path), "path": ffprobe_path}
    if not ffprobe_path:
        healthy = False

    if not healthy:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return {"ok": healthy, "checks": checks}
