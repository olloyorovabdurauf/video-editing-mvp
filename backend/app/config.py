"""Centralized settings. Read once at startup, injected everywhere."""
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # App
    app_env: str = "development"
    app_port: int = 8000
    log_level: str = "INFO"

    # Redis / Celery
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"

    # AI
    openai_api_key: str = ""
    openai_transcribe_model: str = "whisper-1"
    openai_reasoning_model: str = "gpt-4o"

    # Stock footage
    pexels_api_key: str = ""
    pixabay_api_key: str = ""

    # Generative video (any/all optional — registry skips providers w/o keys)
    runway_api_key: str = ""
    higgsfield_api_key: str = ""

    # Kling 3.0 (primary provider — priority 5 in the chain).
    # Kling auths with an AccessKey + SecretKey pair (JWT minted per request),
    # not a single bearer token.
    kling_access_key: str = ""
    kling_secret_key: str = ""
    kling_api_base: str = "https://api-singapore.klingai.com"
    kling_model_name: str = "kling-v3-0"   # verify against current Kling docs
    kling_mode: str = "std"                # std | pro (pro ≈ 2x cost, higher fidelity)

    # Hard upper bound on AI b-roll spend per job. Belt-and-suspenders next to
    # the per-job budget in the request payload.
    max_ai_broll_budget_usd: float = 8.0

    # Billing (Stripe)
    stripe_api_key: str = ""
    stripe_webhook_secret: str = ""
    # If empty, requests with user_id="anonymous" bypass credits.
    require_credits: bool = False

    # Production knobs
    cors_origins: str = "https://your.domain"     # comma-separated in prod
    rate_limit_reels_per_min: int = 10
    rate_limit_reels_window_s: int = 60

    # Auth — see app/core/auth.py
    auth_mode: str = "none"                       # none | clerk | custom
    clerk_jwks_url: str = ""                      # e.g. https://<your>.clerk.accounts.dev/.well-known/jwks.json
    custom_auth_jwks_url: str = ""

    # Observability
    sentry_dsn: str = ""                          # empty = no-op

    # Storage
    storage_backend: str = "local"
    storage_local_dir: Path = Field(default=Path("./storage"))
    s3_bucket: str = ""
    s3_region: str = ""
    s3_endpoint_url: str = ""           # set for R2 / MinIO
    s3_public_base_url: str = ""        # CDN / public domain in front of bucket
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""

    # FFmpeg
    ffmpeg_binary: str = "ffmpeg"
    ffprobe_binary: str = "ffprobe"
    ffmpeg_threads: int = 0
    ffmpeg_hwaccel: str = ""

    @property
    def is_prod(self) -> bool:
        return self.app_env.lower() == "production"


@lru_cache
def get_settings() -> Settings:
    return Settings()
