"""
Endpoint authorization tests — the audit's IDOR + fail-open findings.

  1. GET /reels/{id} must 404 for a non-owner (no existence leak).
  2. GET /billing/balance must be self-only (no /balance/{user_id} route).
  3. Stripe webhook must FAIL CLOSED in prod when the secret is unset.

We build a minimal app from the routers (not create_app) to avoid the
StaticFiles mount; auth runs in dev mode (X-User-Id header) per conftest env.
"""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.api.v1.endpoints import billing as billing_ep
from app.api.v1.endpoints import reels as reels_ep
from app.tasks import video_tasks


@pytest.fixture
def app() -> FastAPI:
    a = FastAPI()
    a.include_router(reels_ep.router, prefix="/api/v1")
    a.include_router(billing_ep.router, prefix="/api/v1")
    return a


@pytest.fixture
def client(app):
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://t")


# ---------------------------------------------------------------------------
# IDOR: job reads are owner-only
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_job_read_owner_sees_job(client, patched_redis):
    video_tasks._update("job-abc", status="queued", progress=0.0,
                        artifacts=[], user_id="alice")
    async with client as c:
        r = await c.get("/api/v1/reels/job-abc", headers={"X-User-Id": "alice"})
    assert r.status_code == 200
    assert r.json()["job_id"] == "job-abc"


@pytest.mark.asyncio
async def test_job_read_non_owner_gets_404(client, patched_redis):
    video_tasks._update("job-abc", status="queued", progress=0.0,
                        artifacts=[], user_id="alice")
    async with client as c:
        r = await c.get("/api/v1/reels/job-abc", headers={"X-User-Id": "mallory"})
    # 404, not 403 — never confirm the job exists to a non-owner.
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_job_read_unknown_job_404(client, patched_redis):
    async with client as c:
        r = await c.get("/api/v1/reels/nope", headers={"X-User-Id": "alice"})
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# Balance: self-only
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_balance_returns_callers_own(client, patched_redis):
    from app.services import billing as billing_svc
    billing_svc.credit("carol", 300, source="t", idempotency_key="sec-1")
    async with client as c:
        r = await c.get("/api/v1/billing/balance", headers={"X-User-Id": "carol"})
    assert r.status_code == 200
    assert r.json() == {"user_id": "carol", "balance": 300}


@pytest.mark.asyncio
async def test_balance_by_user_id_route_is_gone(client, patched_redis):
    """The old /balance/{user_id} route must not exist — it leaked wallets."""
    async with client as c:
        r = await c.get("/api/v1/billing/balance/alice", headers={"X-User-Id": "mallory"})
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# Stripe webhook: fail closed in prod
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_webhook_fails_closed_in_prod_without_secret(client, monkeypatch):
    from app import config as config_mod
    s = config_mod.get_settings()
    monkeypatch.setattr(s, "app_env", "production")
    monkeypatch.setattr(s, "stripe_webhook_secret", "")

    fake_event = b'{"type": "checkout.session.completed", "data": {"object": {"id": "cs_fake", "metadata": {"user_id": "mallory", "pack": "agency"}}}}'
    async with client as c:
        r = await c.post("/api/v1/billing/webhook/stripe", content=fake_event,
                         headers={"content-type": "application/json"})
    # Without a verifiable signature in prod: refuse. No free credits.
    assert r.status_code == 503

    # And mallory must NOT have been credited.
    from app.services import billing as billing_svc
    assert billing_svc.get_balance("mallory") == 0
