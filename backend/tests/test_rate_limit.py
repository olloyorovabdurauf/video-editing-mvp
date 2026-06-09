"""
Rate limiter tests.

This is the difference between "one bad actor with curl" and "your OpenAI
bill at $50k". Test the math, test the atomic Lua, test the 429 response.
"""
from __future__ import annotations

import pytest
from fastapi import FastAPI, Depends
from httpx import ASGITransport, AsyncClient

from app.core.rate_limit import rate_limit, _check


def test_first_request_allowed():
    allowed, count = _check("test-scope", "ip-1", max_per_window=3, window_s=60)
    assert allowed
    assert count == 1


def test_burst_then_block():
    """3rd request still allowed, 4th must be rejected."""
    for i in range(3):
        allowed, count = _check("test-burst", "ip-2", max_per_window=3, window_s=60)
        assert allowed, f"request {i} should be allowed"
    allowed, count = _check("test-burst", "ip-2", max_per_window=3, window_s=60)
    assert not allowed
    assert count == 3


def test_separate_identities_dont_collide():
    """Two different IPs / users get separate buckets."""
    for _ in range(3):
        ok, _ = _check("test-iso", "ip-a", max_per_window=3, window_s=60)
        assert ok
    # ip-a is now blocked
    blocked, _ = _check("test-iso", "ip-a", max_per_window=3, window_s=60)
    assert not blocked
    # ip-b should still get through
    ok, _ = _check("test-iso", "ip-b", max_per_window=3, window_s=60)
    assert ok


@pytest.mark.asyncio
async def test_endpoint_returns_429_after_burst():
    """End-to-end: a real ASGI app with the dep gets a 429 after the limit."""
    app = FastAPI()

    @app.get("/x", dependencies=[Depends(rate_limit("e2e", max_per_window=2, window_s=60))])
    async def ep():
        return {"ok": True}

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r1 = await c.get("/x")
        r2 = await c.get("/x")
        r3 = await c.get("/x")

    assert r1.status_code == 200
    assert r2.status_code == 200
    assert r3.status_code == 429
    assert "Retry-After" in r3.headers
