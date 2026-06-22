"""
POST /api/v1/reels        — enqueue a job, return job_id
GET  /api/v1/reels/{id}   — poll status / get artifacts (owner only)
"""
from fastapi import APIRouter, Depends, HTTPException, status

from app.config import get_settings
from app.core.auth import require_user
from app.core.rate_limit import QuotaExceeded, consume_daily_job_quota, rate_limit
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

    # Daily per-user job quota (cost/abuse guard, separate from burst limit).
    try:
        consume_daily_job_quota(user_id, limit=_s.max_jobs_per_user_per_day)
    except QuotaExceeded as e:
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, str(e),
                            headers={"Retry-After": "3600"})

    # SSRF guard applies only to user-supplied URLs we fetch server-side.
    # Uploads come from our own private R2 bucket (no SSRF surface), and the
    # key must belong to this user (no reading another account's upload).
    if req.is_upload:
        if not req.upload_key.startswith(f"uploads/{user_id}/"):
            raise HTTPException(status.HTTP_403_FORBIDDEN, "upload_key does not belong to you")
    else:
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
