"""
Sentry init — no-op when SENTRY_DSN is empty.

Why a shared module: both the API process (main.py) and the Celery worker
process (celery_app.py) need to init Sentry independently. Putting it here
keeps the integration list canonical.
"""
from __future__ import annotations

from loguru import logger

from app.config import get_settings


def init_sentry() -> None:
    s = get_settings()
    if not s.sentry_dsn:
        return  # opt-in
    try:
        import sentry_sdk
        from sentry_sdk.integrations.celery import CeleryIntegration
        from sentry_sdk.integrations.fastapi import FastApiIntegration
        from sentry_sdk.integrations.starlette import StarletteIntegration
    except ImportError:
        logger.warning("sentry-sdk not installed; skipping Sentry init")
        return

    sentry_sdk.init(
        dsn=s.sentry_dsn,
        environment=s.app_env,
        # 10% trace sampling — enough to debug perf issues, cheap enough
        # to leave on in prod. Raise during investigations, drop after.
        traces_sample_rate=0.1,
        # Send PII (user_id, IP) only if you've updated your privacy policy.
        send_default_pii=False,
        integrations=[
            FastApiIntegration(),
            StarletteIntegration(),
            CeleryIntegration(monitor_beat_tasks=True),
        ],
    )
    logger.info("sentry initialized ({})", s.app_env)
