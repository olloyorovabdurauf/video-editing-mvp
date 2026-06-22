"""
Storage adapter — local filesystem or S3/R2-compatible.

Why an interface?
-----------------
In dev you want files on disk so you can `ls storage/output/` and grep them.
In prod you need an object store: API and worker containers are different
machines, so a local file on the worker is invisible to the API serving
the URL. R2 has zero egress fees — best default unless you're already on AWS.

This module is a thin facade. There's deliberately no caching, retry logic,
or fancy abstraction. ~100 lines is enough.
"""
from __future__ import annotations

import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Protocol

from loguru import logger

from app.config import get_settings


class StorageBackend(Protocol):
    def put(self, local_path: Path, *, key: str) -> str:
        """Upload local_path to `key`; return a URL the client can fetch."""

    def presigned_upload(self, key: str, *, content_type: str, max_bytes: int,
                         expires_s: int = 3600) -> dict:
        """Return {url, fields, key} for a direct browser→bucket upload,
        constrained to <= max_bytes. Bucket stays private."""

    def presigned_download(self, key: str, *, expires_s: int = 3600) -> str:
        """Time-limited GET URL for a private object."""

    def download_to(self, key: str, dst: Path) -> Path:
        """Fetch object `key` into local `dst` (worker pulling an upload)."""

    def delete(self, key: str) -> None:
        ...


# ---------------------------------------------------------------------------
# Local backend — for dev / docker-compose. Files served by FastAPI's
# StaticFiles mount at /storage/.
# ---------------------------------------------------------------------------

class LocalStorage:
    def __init__(self, root: Path) -> None:
        self.root = root

    def put(self, local_path: Path, *, key: str) -> str:
        dst = self.root / key
        dst.parent.mkdir(parents=True, exist_ok=True)
        if local_path.resolve() != dst.resolve():
            shutil.copy2(local_path, dst)
        # Returns the path under /storage. Frontend resolves against API origin.
        return f"/storage/{key}"

    def presigned_upload(self, key: str, *, content_type: str, max_bytes: int,
                         expires_s: int = 3600) -> dict:
        # Dev convenience: PUT straight to the local dev upload endpoint.
        return {"url": f"/storage-upload/{key}", "fields": {}, "key": key,
                "method": "PUT", "max_bytes": max_bytes}

    def presigned_download(self, key: str, *, expires_s: int = 3600) -> str:
        return f"/storage/{key}"

    def download_to(self, key: str, dst: Path) -> Path:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.root / key, dst)
        return dst

    def delete(self, key: str) -> None:
        (self.root / key).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# S3 / R2 backend. R2 is S3-API-compatible; difference is the endpoint URL
# and the fact that there are no egress fees (massive deal for video).
# ---------------------------------------------------------------------------

class S3Storage:
    def __init__(self) -> None:
        # Lazy import — boto3 is heavy and only needed in this branch.
        import boto3
        s = get_settings()
        self.bucket = s.s3_bucket
        self.client = boto3.client(
            "s3",
            region_name=s.s3_region or None,
            endpoint_url=s.s3_endpoint_url or None,    # set for R2 / MinIO
            aws_access_key_id=s.aws_access_key_id or None,
            aws_secret_access_key=s.aws_secret_access_key or None,
        )
        # CDN/custom-domain in front of R2: serve files via your own domain.
        # If unset, falls back to the raw bucket URL (works but reveals R2).
        self.public_base = s.s3_public_base_url.rstrip("/")

    def put(self, local_path: Path, *, key: str) -> str:
        extra = {
            "ContentType": _guess_ct(key),
            # 30-day cache for finished MP4s. CDN-friendly.
            "CacheControl": "public, max-age=2592000",
        }
        self.client.upload_file(str(local_path), self.bucket, key, ExtraArgs=extra)
        logger.info("storage.put → s3://{}/{}", self.bucket, key)

        if self.public_base:
            return f"{self.public_base}/{key}"
        # Fallback: presigned URL. Lives 7 days. Use a CDN for permanent serving.
        return self.client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": key},
            ExpiresIn=7 * 24 * 3600,
        )

    def presigned_upload(self, key: str, *, content_type: str, max_bytes: int,
                         expires_s: int = 3600) -> dict:
        """
        Presigned POST → browser uploads directly to R2, bucket stays private.
        The content-length-range condition is server-enforced abuse prevention:
        R2 rejects the upload if it exceeds max_bytes (can't be bypassed client-side).
        """
        post = self.client.generate_presigned_post(
            Bucket=self.bucket, Key=key,
            Fields={"Content-Type": content_type},
            Conditions=[
                {"Content-Type": content_type},
                ["content-length-range", 1, max_bytes],
            ],
            ExpiresIn=expires_s,
        )
        return {"url": post["url"], "fields": post["fields"], "key": key,
                "method": "POST", "max_bytes": max_bytes}

    def presigned_download(self, key: str, *, expires_s: int = 3600) -> str:
        return self.client.generate_presigned_url(
            "get_object", Params={"Bucket": self.bucket, "Key": key}, ExpiresIn=expires_s,
        )

    def download_to(self, key: str, dst: Path) -> Path:
        dst.parent.mkdir(parents=True, exist_ok=True)
        self.client.download_file(self.bucket, key, str(dst))
        return dst

    def delete(self, key: str) -> None:
        self.client.delete_object(Bucket=self.bucket, Key=key)


def _guess_ct(key: str) -> str:
    if key.endswith(".mp4"):  return "video/mp4"
    if key.endswith(".webm"): return "video/webm"
    if key.endswith(".jpg"):  return "image/jpeg"
    if key.endswith(".png"):  return "image/png"
    if key.endswith(".srt"):  return "application/x-subrip"
    return "application/octet-stream"


# ---------------------------------------------------------------------------
# Module-level singleton, picked from settings on first call.
# ---------------------------------------------------------------------------

_backend: StorageBackend | None = None


def get_storage() -> StorageBackend:
    global _backend
    if _backend is not None:
        return _backend
    s = get_settings()
    if s.storage_backend.lower() == "s3":
        _backend = S3Storage()
        logger.info("storage backend: S3 ({})", s.s3_bucket)
    else:
        _backend = LocalStorage(s.storage_local_dir)
        logger.info("storage backend: local ({})", s.storage_local_dir)
    return _backend
