"""
Shared test fixtures.

Two important things happen here:

1. We set required env vars BEFORE the app imports anything that calls
   get_settings() (which caches via @lru_cache).
2. We swap the global redis client used by billing.py + rate_limit.py for
   an in-memory fakeredis, so tests don't need a real Redis running.
"""
from __future__ import annotations

import os

# Set env BEFORE any app imports happen. lru_cache means a single
# get_settings() in app code will lock in these values for the test session.
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("STORAGE_LOCAL_DIR", "./test-storage")
os.environ.setdefault("APP_ENV", "test")

import fakeredis                                                          # noqa: E402
import pytest                                                             # noqa: E402


@pytest.fixture(autouse=True)
def patched_redis(monkeypatch):
    """
    Replace the module-level redis client in any module that imported one.
    fakeredis is wire-compatible: same commands, no daemon required.

    We patch lazily — only the modules each test actually touches. video_tasks
    pulls in yt-dlp (heavy) so we only patch it if a test imports it.
    """
    fake = fakeredis.FakeRedis(decode_responses=True)

    # Always-safe patches: these modules have no heavy transitive imports.
    import app.services.billing as billing_mod
    import app.core.rate_limit as rl_mod
    import app.services.creative_engine.cache as cache_mod
    import app.services.ai_cache as ai_cache_mod

    monkeypatch.setattr(billing_mod, "_r", fake)
    monkeypatch.setattr(rl_mod, "_r", fake)
    monkeypatch.setattr(cache_mod, "_r", fake)
    monkeypatch.setattr(ai_cache_mod, "_r", fake)

    # The rate limiter pre-registered its Lua against the real client. Re-register.
    monkeypatch.setattr(rl_mod, "_LIMITER_LUA", fake.register_script(rl_mod._LIMITER_LUA.script))

    # video_tasks is only patched IF it's already been imported by the test
    # (e.g. via an integration test). Skips the yt-dlp import in unit-test envs.
    import sys
    if "app.tasks.video_tasks" in sys.modules:
        monkeypatch.setattr(sys.modules["app.tasks.video_tasks"], "r", fake)

    yield fake
    fake.flushall()
