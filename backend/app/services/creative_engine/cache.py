"""
Content-addressable cache for generated b-roll.

Why: a single generation costs $0.40-$0.50 and takes 30-120s. If the same
prompt+settings was generated yesterday (likely for evergreen content),
we should return the cached asset path instantly.

Cache key = sha256(provider + canonicalized request JSON). We store a
manifest in Redis (provider job_id, video path, cost, timestamp) and the
file on disk (or S3 in prod).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import redis

from app.config import get_settings
from app.services.creative_engine.providers.base import GenerationJob, GenerationRequest

_settings = get_settings()
_r = redis.from_url(_settings.redis_url, decode_responses=True)


def _key(provider: str, req: GenerationRequest) -> str:
    payload = {
        "provider": provider,
        "prompt": req.prompt,
        "aspect": req.aspect_ratio,
        "duration": round(req.duration_s, 2),
        "negative": req.negative_prompt or "",
        "motion": round(req.motion_strength, 2),
        "ref": req.reference_image_url or "",
        "seed": req.seed,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    h = hashlib.sha256(blob.encode()).hexdigest()[:24]
    return f"genvideo:cache:{h}"


def lookup(provider: str, req: GenerationRequest) -> Path | None:
    """Return a local Path if we already have this asset; else None."""
    raw = _r.get(_key(provider, req))
    if not raw:
        return None
    data = json.loads(raw)
    p = Path(data["path"])
    if p.exists():
        return p
    # Stale entry — purge it.
    _r.delete(_key(provider, req))
    return None


def store(provider: str, job: GenerationJob, local_path: Path) -> None:
    if not job.request:
        return
    _r.setex(
        _key(provider, job.request),
        60 * 60 * 24 * 30,                      # 30-day TTL
        json.dumps({
            "path": str(local_path),
            "cost_usd": job.cost_usd,
            "provider_job_id": job.provider_job_id,
            "thumbnail_url": job.thumbnail_url,
        }),
    )
