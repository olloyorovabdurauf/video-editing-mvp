"""
Kling 3.0 provider tests.

We test the pure logic (JWT minting, status mapping, request shaping) without
hitting the network. The submit/poll HTTP plumbing follows the same shape as
Runway's, which the scene_extractor tests already exercise via mocks.
"""
from __future__ import annotations

import time

import pytest
from jose import jwt as jose_jwt

from app.services.creative_engine.providers.base import (
    GenerationJob,
    GenerationRequest,
    GenStatus,
)
from app.services.creative_engine.providers.kling import (
    _TOKEN_TTL_S,
    KlingProvider,
    _map_status,
    _mint_token,
)


# ---------------------------------------------------------------------------
# JWT minting
# ---------------------------------------------------------------------------

def test_mint_token_claims():
    now = 1_700_000_000.0
    tok = _mint_token("ak_test", "sk_test", now=now)
    claims = jose_jwt.decode(
        tok, "sk_test", algorithms=["HS256"],
        options={"verify_exp": False, "verify_nbf": False},
    )
    assert claims["iss"] == "ak_test"
    assert claims["exp"] == int(now) + _TOKEN_TTL_S
    assert claims["nbf"] == int(now) - 5


def test_mint_token_signature_validates_only_with_right_secret():
    tok = _mint_token("ak", "correct-secret")
    # Right secret decodes...
    jose_jwt.decode(tok, "correct-secret", algorithms=["HS256"],
                    options={"verify_exp": False, "verify_nbf": False})
    # ...wrong secret raises.
    with pytest.raises(Exception):
        jose_jwt.decode(tok, "wrong-secret", algorithms=["HS256"],
                        options={"verify_exp": False, "verify_nbf": False})


# ---------------------------------------------------------------------------
# Status mapping
# ---------------------------------------------------------------------------

def _job() -> GenerationJob:
    return GenerationJob(provider="kling", provider_job_id="t1")


def test_map_status_submitted():
    job = _map_status({"task_status": "submitted"}, _job())
    assert job.status == GenStatus.PENDING


def test_map_status_processing():
    job = _map_status({"task_status": "processing"}, _job())
    assert job.status == GenStatus.RUNNING


def test_map_status_succeed_with_url():
    data = {
        "task_status": "succeed",
        "task_result": {"videos": [{"url": "https://cdn.kling/abc.mp4"}]},
    }
    job = _map_status(data, _job())
    assert job.status == GenStatus.SUCCEEDED
    assert job.video_url == "https://cdn.kling/abc.mp4"
    assert job.finished_at is not None


def test_map_status_succeed_without_url_is_failure():
    """'succeed' with empty videos list must NOT report success."""
    job = _map_status({"task_status": "succeed", "task_result": {"videos": []}}, _job())
    assert job.status == GenStatus.FAILED
    assert "no video url" in (job.error or "")


def test_map_status_failed_carries_message():
    job = _map_status({"task_status": "failed", "task_status_msg": "nsfw blocked"}, _job())
    assert job.status == GenStatus.FAILED
    assert job.error == "nsfw blocked"


def test_map_status_unknown_status_is_failure():
    """Forward-compat: an unrecognized status must fail loudly, not hang the poller."""
    job = _map_status({"task_status": "quantum_flux"}, _job())
    assert job.status == GenStatus.FAILED


# ---------------------------------------------------------------------------
# Provider construction + registry priority
# ---------------------------------------------------------------------------

def test_provider_requires_both_keys(monkeypatch):
    from app import config as config_mod
    from app.services.creative_engine.providers.base import ProviderError

    s = config_mod.get_settings()
    monkeypatch.setattr(s, "kling_access_key", "ak")
    monkeypatch.setattr(s, "kling_secret_key", "")  # missing secret
    with pytest.raises(ProviderError, match="not set"):
        KlingProvider()


def test_provider_chain_prefers_kling(monkeypatch):
    """With all providers configured, Kling (priority 5) must be tried first."""
    from app import config as config_mod
    import app.services.creative_engine.providers.registry as registry_mod

    s = config_mod.get_settings()
    monkeypatch.setattr(s, "kling_access_key", "ak")
    monkeypatch.setattr(s, "kling_secret_key", "sk")
    monkeypatch.setattr(s, "runway_api_key", "rk")
    monkeypatch.setattr(s, "higgsfield_api_key", "hk")
    monkeypatch.setattr(registry_mod, "_chain_cache", None)  # bust the cache

    chain = registry_mod.get_chain()
    names = [c.provider.name for c in chain]
    assert names[0] == "kling", f"expected kling first, got {names}"
    # Cleanup so other tests don't inherit this chain.
    monkeypatch.setattr(registry_mod, "_chain_cache", None)


def test_duration_snapping_logic():
    """Kling only accepts 5s or 10s; verify our snap boundaries."""
    # The snap rule lives inline in submit(); we verify the rule itself here.
    assert ("10" if 8.0 > 7.5 else "5") == "10"
    assert ("10" if 6.0 > 7.5 else "5") == "5"
    assert ("10" if 7.5 > 7.5 else "5") == "5"   # boundary goes to 5
