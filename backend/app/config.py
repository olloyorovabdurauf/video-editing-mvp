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
    # Cap source length to bound Whisper cost/time (chunked above 20MB audio).
    # 120 min ≈ $0.72 of Whisper at most. Raise once you have paying users.
    max_source_minutes: int = 120

    # YouTube blocks data downloads from datacenter IPs after repeated use.
    # Set this to a residential/rotating proxy (e.g. http://user:pass@host:port)
    # for reliable YouTube downloads at scale. Empty = direct (works for direct
    # media URLs and intermittently for YouTube from a fresh IP).
    ytdlp_proxy: str = ""

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
    # Free credits granted once, on first login (atomic with user creation).
    # ~one AI-b-roll job (≈450) or several basic 3-reel jobs (≈90 each). Doubles
    # as a cost-abuse cap when require_credits is on but Stripe top-up isn't live.
    signup_credits: int = 500

    # Production knobs
    cors_origins: str = "https://your.domain"     # comma-separated in prod
    rate_limit_reels_per_min: int = 10
    rate_limit_reels_window_s: int = 60
    # Abuse / cost guards
    max_jobs_per_user_per_day: int = 50           # distinct from burst limit
    max_request_body_bytes: int = 2 * 1024 * 1024  # JSON API only; video → R2 signed URLs

    # AI cost optimization (Area 5)
    # Cheap model does first-pass analysis; we escalate to the strong model
    # only when the cheap pass returns nothing usable.
    openai_analysis_model: str = "gpt-4o-mini"
    openai_escalation_model: str = "gpt-4o"
    segment_cache_ttl_s: int = 7 * 24 * 3600       # re-runs of same video reuse picks
    # Script generation is quality-sensitive (retention structure + native-language
    # copywriting) → use the strong model. One short JSON response, so cheap.
    openai_script_model: str = "gpt-4o"
    script_cache_ttl_s: int = 24 * 3600

    # Auth — see app/core/auth.py
    auth_mode: str = "none"                       # none | clerk | custom
    clerk_jwks_url: str = ""                      # e.g. https://<your>.clerk.accounts.dev/.well-known/jwks.json
    custom_auth_jwks_url: str = ""

    # Observability
    sentry_dsn: str = ""                          # empty = no-op

    # Database (durable records). Empty = Redis-only mode (DB layer disabled).
    # Set to a Neon Postgres URL to enable users/jobs/usage/ledger persistence.
    database_url: str = ""

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
    # x264 speed/size trade-off, applied to every re-encode pass (cut, reframe,
    # smart-crop, b-roll overlay, caption burn). superfast ≈ 1.5x faster than
    # veryfast at the same CRF/quality, for ~moderately larger files — the right
    # call for short social reels where render latency matters more than bytes.
    # Drop to "ultrafast" for max speed, or "veryfast"/"faster" for smaller files.
    ffmpeg_preset: str = "superfast"
    # How many clips to render concurrently within one job. The render is the
    # multi-clip bottleneck; on dedicated CPUs 3 parallel ffmpeg pipelines beat a
    # sequential loop without thrashing. Bound by core count in practice.
    render_concurrency: int = 3

    @property
    def is_prod(self) -> bool:
        return self.app_env.lower() == "production"


@lru_cache
def get_settings() -> Settings:
    return Settings()
