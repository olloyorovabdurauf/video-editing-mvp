"""
POST /api/v1/reels        — enqueue a job, return job_id
GET  /api/v1/reels/{id}   — poll status / get artifacts (owner only)
"""
from fastapi import APIRouter, Depends, HTTPException, status

from app.config import get_settings
from app.core.auth import require_user
from app.core.rate_limit import rate_limit
from app.schemas.reel import ReelCreateRequest, ReelJobResponse
from app.services import billing
from app.services.url_guard import UnsafeURLError, validate_source_url
from app.tasks.video_tasks import enqueue_reel_job, get_job, get_job_owner

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
    """
    Kicks off the pipeline asynchronously. The response is immediate; poll
    GET /reels/{job_id} for progress.
    """
    # The token is the truth — override whatever the client put in the body.
    req.user_id = user_id

    # SSRF guard: we fetch this URL server-side; refuse anything that
    # resolves to internal/private address space.
    try:
        validate_source_url(str(req.source_url))
    except UnsafeURLError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"source_url rejected: {e}")

    try:
        job_id = enqueue_reel_job(req)
    except billing.InsufficientCredits as e:
        raise HTTPException(status.HTTP_402_PAYMENT_REQUIRED, str(e))

    job = get_job(job_id)
    if job is None:  # should never happen, but be loud if it does
        raise HTTPException(500, "job state failed to persist")
    return job


@router.get("/{job_id}", response_model=ReelJobResponse)
async def read_reel(
    job_id: str,
    user_id: str = Depends(require_user),
) -> ReelJobResponse:
    job = get_job(job_id)
    # 404 for both "doesn't exist" and "not yours" — don't leak existence.
    if job is None or get_job_owner(job_id) != user_id:
        raise HTTPException(404, "job not found")
    return job
