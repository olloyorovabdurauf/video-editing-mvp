"""
Per-job observability: cost + processing-time, written into the job's Redis
state so they surface in the status API (and later an admin/analytics view).

Deliberately tiny — no metrics backend to run. When you outgrow it, these same
numbers feed a Prometheus exporter or a `usage` table without changing callers.
"""
from __future__ import annotations

import time

# --- Unit costs (USD). Update when provider pricing changes. ---
WHISPER_USD_PER_MIN = 0.006
# gpt-4o-mini analysis is a few cents; flat per-job estimate is fine at MVP scale.
ANALYSIS_USD_PER_JOB = 0.01
RENDER_USD_PER_CLIP = 0.0          # our own ffmpeg/CPU; track when on metered GPU


def estimate_job_cost_usd(*, audio_minutes: float, n_clips: int, ai_broll_clips: int,
                          ai_broll_usd_each: float = 0.35) -> float:
    """Transparent, auditable cost estimate for a finished job."""
    return round(
        audio_minutes * WHISPER_USD_PER_MIN
        + ANALYSIS_USD_PER_JOB
        + n_clips * RENDER_USD_PER_CLIP
        + ai_broll_clips * ai_broll_usd_each,
        4,
    )


def now() -> float:
    return time.time()
