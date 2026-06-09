"""
POST /api/v1/reels        — enqueue a job, return job_id
GET  /api/v1/reels/{id}   — poll status / get artifacts
"""
from fastapi import APIRouter, Depends, HTTPException, status

from app.config import get_settings
from app.core.auth import require_user
from app.core.rate_limit import rate_limit
from app.schemas.reel import ReelCreateRequest, ReelJobResponse
from app.tasks.video_tasks import enqueue_reel_job, get_job

router = APIRouter(prefix="/reels", tags=["reels"])

_s = get_settings()
_reel_rate_limit = rate_limit(
    "reels",
    max_per_window=_s.rate_limit_reels_per_min,
    window_s=_s.rate_limit_reels_window_s,
)


@router.post(
    "",
    response_model=ReelJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Create a reel-generation job",
    dependencies=[Depends(_reel_rate_limit)],
)
async def create_reel(
    req: ReelCreateRequest,
    user_id: str = Depends(require_user),
) -> ReelJobResponse:
    # Override whatever the client put in the body — the token is the truth.
    req.user_id = user_id
    """
    Kicks off the pipeline asynchronously. The response is immediate; poll
    GET /reels/{job_id} (or subscribe to the SSE channel — v2) for progress.
    """
    job_id = enqueue_reel_job(req)
    job = get_job(job_id)
    if job is None:  # should never happen, but be loud if it does
        raise HTTPException(500, "job state failed to persist")
    return job


@router.get("/{job_id}", response_model=ReelJobResponse)
async def read_reel(job_id: str) -> ReelJobResponse:
    job = get_job(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    return job
