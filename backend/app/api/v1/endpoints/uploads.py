"""
Signed-upload endpoint — the entry for "user uploads a long video".

Flow (no bytes ever touch this server):
  1. Client POSTs filename + content_type + size here.
  2. We return a presigned POST (R2) the browser uploads directly to.
  3. Client then creates a reel job referencing the returned `key`
     (source_kind=upload) — wired in the DB/job PR.

Security: auth-required, daily-quota counted, and the presigned POST carries a
server-enforced content-length-range so a client can't exceed the size cap.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.config import get_settings
from app.core.auth import require_user
from app.core.rate_limit import rate_limit
from app.services.storage import get_storage

router = APIRouter(prefix="/uploads", tags=["uploads"])

_s = get_settings()
# Max source upload — generous for long videos, bounded for abuse/cost.
MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024   # 2 GB

_ALLOWED_CT = {"video/mp4", "video/quicktime", "video/webm", "video/x-matroska"}


class UploadIntent(BaseModel):
    filename: str = Field(..., max_length=255)
    content_type: str
    size_bytes: int = Field(..., gt=0, le=MAX_UPLOAD_BYTES)


@router.post("", dependencies=[Depends(rate_limit("uploads", max_per_window=20, window_s=60))])
async def create_upload(intent: UploadIntent, user_id: str = Depends(require_user)) -> dict:
    if intent.content_type not in _ALLOWED_CT:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                            f"unsupported type {intent.content_type!r}")
    # Namespaced by user so keys can't collide or be guessed across accounts.
    import uuid
    key = f"uploads/{user_id}/{uuid.uuid4().hex}/{intent.filename}"
    signed = get_storage().presigned_upload(
        key, content_type=intent.content_type,
        max_bytes=min(intent.size_bytes, MAX_UPLOAD_BYTES), expires_s=3600,
    )
    return {"upload": signed, "key": key,
            "next": "POST /api/v1/reels with source_kind=upload and this key"}
