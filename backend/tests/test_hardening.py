"""
Production-hardening tests: daily quota, body-size limit, signed uploads,
auth enforcement.
"""
from __future__ import annotations

import pytest
from fastapi import Depends, FastAPI
from httpx import ASGITransport, AsyncClient

from app.api.v1.endpoints import uploads as uploads_ep
from app.core.middleware import MaxBodySizeMiddleware
from app.core.rate_limit import QuotaExceeded, consume_daily_job_quota


# ---------------------------------------------------------------------------
# Daily per-user job quota
# ---------------------------------------------------------------------------

def test_quota_counts_and_blocks(patched_redis):
    for i in range(3):
        assert consume_daily_job_quota("alice", limit=3) == i + 1
    with pytest.raises(QuotaExceeded):
        consume_daily_job_quota("alice", limit=3)


def test_quota_ignores_anonymous(patched_redis):
    # Anonymous/dev callers aren't capped here (billing guards those).
    for _ in range(100):
        assert consume_daily_job_quota("anonymous", limit=3) == 0


def test_quota_is_per_user(patched_redis):
    consume_daily_job_quota("u1", limit=2)
    consume_daily_job_quota("u1", limit=2)
    # u2 has its own bucket
    assert consume_daily_job_quota("u2", limit=2) == 1


# ---------------------------------------------------------------------------
# Body-size limit middleware
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_oversized_body_rejected():
    app = FastAPI()
    app.add_middleware(MaxBodySizeMiddleware, max_bytes=100)

    @app.post("/x")
    async def x():
        return {"ok": True}

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        small = await c.post("/x", content=b"a" * 50)
        big = await c.post("/x", content=b"a" * 500)
    assert small.status_code == 200
    assert big.status_code == 413


# ---------------------------------------------------------------------------
# Signed uploads (auth + size cap + type allow-list)
# ---------------------------------------------------------------------------

@pytest.fixture
def upload_client(patched_redis):
    app = FastAPI()
    app.include_router(uploads_ep.router, prefix="/api/v1")
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://t")


@pytest.mark.asyncio
async def test_signed_upload_returns_url(upload_client):
    async with upload_client as c:
        r = await c.post("/api/v1/uploads",
                         headers={"X-User-Id": "alice"},
                         json={"filename": "talk.mp4", "content_type": "video/mp4",
                               "size_bytes": 50_000_000})
    assert r.status_code == 200
    body = r.json()
    assert body["key"].startswith("uploads/alice/")     # namespaced by user
    assert "url" in body["upload"]


@pytest.mark.asyncio
async def test_signed_upload_rejects_bad_type(upload_client):
    async with upload_client as c:
        r = await c.post("/api/v1/uploads",
                         headers={"X-User-Id": "alice"},
                         json={"filename": "x.exe", "content_type": "application/x-msdownload",
                               "size_bytes": 1000})
    assert r.status_code == 415


@pytest.mark.asyncio
async def test_signed_upload_rejects_oversize(upload_client):
    async with upload_client as c:
        r = await c.post("/api/v1/uploads",
                         headers={"X-User-Id": "alice"},
                         json={"filename": "huge.mp4", "content_type": "video/mp4",
                               "size_bytes": 5 * 1024 * 1024 * 1024})  # 5GB > 2GB cap
    assert r.status_code == 422   # pydantic validation on size_bytes
