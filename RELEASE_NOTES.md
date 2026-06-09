# Reel Forge — v0.1.0-mvp

First public release. End-to-end pipeline: long-form video → ranked viral
segments → 9:16 reels with AI b-roll, animated captions, and music.

## What ships

**Pipeline.** Celery chain across four queues — `io / ai / ffmpeg / gpu` —
each scaling independently. Per-stage retry + idempotent intermediates so
expensive calls (Whisper, GPT-4o) are reusable when a downstream stage
fails.

**Creative engine.** Plug-in providers (Runway, Higgsfield) behind a
single interface with budget guard, content-addressable cache, and a
self-rescheduling poll task so generative video doesn't block other work.

**Visual quality stack.**
- Speaker-tracking smart crop (mediapipe → ffmpeg `crop` filter, frame-by-frame).
- Four animated caption styles (karaoke, popup, minimal, none) rendered via libass.
- Mood-detected music with sidechain ducking.
- Cross-dissolve composite for b-roll, not hard cuts.

**Billing.** Append-only Redis ledger with atomic hold/settle/refund Lua,
Stripe webhook idempotency, per-job budget cap. `test_credit_is_idempotent`
is your guard against double-charges.

**Production hardening (added this release).**
- Authentication dependency (`AUTH_MODE=none|clerk|custom`) — refuses to
  start with `auth_mode=none` in production.
- Pluggable storage adapter (`local | s3`) — render outputs land in R2/S3
  when configured. No more "the URL works on the worker but not the API"
  pitfalls.
- Sentry integration for both the API and Celery workers — no-op without
  `SENTRY_DSN`, zero-config when set.
- Real `/livez` + `/healthz` endpoints (process vs dependency liveness).
- Per-request `X-Request-ID` middleware bound to loguru context.
- Atomic Redis-Lua rate limit on `POST /reels` (default 10/min/user).
- CORS lockdown in prod; `/docs` hidden in prod.
- Multi-stage non-root Dockerfile with `tini` PID 1 and `HEALTHCHECK`.

**Deployment artifacts (added this release).**
- `deploy/fly/fly.api.toml` — API service, auto-stop machines, persistent
  volume mount, health checks wired.
- `deploy/fly/fly.worker.toml` — worker pool, rolling deploy strategy,
  larger volume.
- `deploy/fly/deploy.sh` — idempotent first-deploy script that creates
  apps + volumes + sets secrets in one go.
- GitHub Actions CI: backend tests with Redis service + Docker build matrix.

## Quality bar

- **38/38 tests passing** (`pytest -q`).
- **Coverage on load-bearing logic:** `rate_limit.py` 100% · `schemas/reel.py`
  100% · `captions.py` 98% · `billing.py` 90% · `scene_extractor.py` 92%.
- **Frontend production build:** 90.3 kB first-load, all routes prerender
  or stream cleanly.
- **Smoke import:** the FastAPI app boots, 10 routes register, no missing
  dependencies.

## Known limitations to fix in v0.2

- YouTube downloads via `yt-dlp` violate YouTube ToS. Replace with
  user-upload-only or YouTube Data API OAuth before scaling marketing.
  Tracked in `PUBLISHING.md` §0.
- No DB yet. The Redis ledger is fine through ~1k paying users; migrate
  to Postgres + Alembic when you need analytics or disputes.
- Job progress polls every 2.5s. SSE/WebSocket is a v0.2 polish.
- Smart crop uses face detection only. Multi-person scenes default to the
  largest face; multi-speaker tracking is post-MVP.

## Upgrade path from v0.1 dev → v0.1 prod

1. `cp backend/.env.example .env.production`, fill in:
   - `OPENAI_API_KEY` (required)
   - `STRIPE_API_KEY` + `STRIPE_WEBHOOK_SECRET`
   - `AUTH_MODE=clerk` + `CLERK_JWKS_URL`
   - `STORAGE_BACKEND=s3` + R2/S3 credentials
   - `REDIS_URL` pointing at Upstash
   - `SENTRY_DSN` (optional but strongly recommended)
   - `CORS_ORIGINS=https://your.domain` (your real domain)
   - `APP_ENV=production`
2. `bash deploy/fly/deploy.sh` from the repo root.
3. Set DNS: `api.your.domain` → Fly, root → Vercel.
4. Smoke test: `curl https://api.your.domain/api/v1/healthz` — expect
   `{"ok": true}` and all checks green.

See `PUBLISHING.md` for the full launch runbook including DMCA, music
licensing, and the five questions to answer before deploy.
