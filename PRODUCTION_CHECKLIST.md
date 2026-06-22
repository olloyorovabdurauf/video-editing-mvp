# Production Checklist — first paying users

Tick every box before charging. Each line is "what + why + how to verify".

## Environment variables (Fly secrets)

| Secret | Purpose | Required when |
|---|---|---|
| `APP_ENV=production` | enforces auth, hides /docs, locks CORS | before public |
| `OPENAI_API_KEY` | Whisper + GPT-4o | already set ✅ |
| `REDIS_URL` / `CELERY_*` | broker + hot state | already set ✅ |
| `AUTH_MODE=clerk` + `CLERK_JWKS_URL` | verify every JWT | before public — see CLERK_SETUP.md |
| `CORS_ORIGINS=https://yourdomain.com` | block other origins | before public |
| `DATABASE_URL` | durable users/jobs/usage/ledger (Neon) | before charging — see backend/app/db/README.md |
| `STORAGE_BACKEND=s3` + `S3_*` + `AWS_*` | R2 private bucket | before real traffic |
| `S3_PUBLIC_BASE_URL` | CDN domain in front of R2 | recommended |
| `STRIPE_API_KEY` + `STRIPE_WEBHOOK_SECRET` | payments (fail-closed in prod) | before charging |
| `SENTRY_DSN` | error tracking | before public |
| `MAX_SOURCE_MINUTES`, `MAX_JOBS_PER_USER_PER_DAY` | cost/abuse caps | tune per plan |
| `YTDLP_PROXY` | reliable YouTube at scale | when volume blocks the IP |

Verify: `flyctl ssh console -C "env | grep -E 'APP_ENV|AUTH_MODE|STORAGE'"` (secrets are masked).
`curl https://reelforge-mvp-x7k2.fly.dev/api/v1/healthz` → all checks ok.

## Monitoring
- [ ] **Uptime**: Better Stack/Cronitor hitting `/api/v1/livez` every 30s → alert on 2 failures.
- [ ] **Health**: same on `/api/v1/healthz` (returns 503 if Redis/ffmpeg down).
- [ ] **Cost**: OpenAI usage dashboard alert at your daily budget; per-job `cost_usd` is in the status API and (with DB) the `usage_counters` table.
- [ ] **Processing time**: `processing_time_s` per job; alert if p95 > 5 min.
- [ ] **Queue depth**: Flower (`:5555`) or `celery inspect active` — alert if backlog grows.

## Error tracking
- [ ] **Sentry** wired in API + workers (set `SENTRY_DSN`). Verify by forcing one error.
- [ ] **Request IDs**: every response has `X-Request-ID`; logs carry it — grep one ID across API + worker.
- [ ] **User-facing failures**: ingestion errors map to clear messages (private/age-restricted/rate-limited); job `message` field shows them.
- [ ] **Stripe**: alert on any webhook delivery failure (Stripe dashboard).

## Backup strategy
- [ ] **Postgres (Neon)**: enable PITR / daily branch backups (Neon does automatic backups — confirm retention ≥ 7 days). This holds the money ledger — non-negotiable.
- [ ] **R2**: enable object versioning on the bucket (recover deleted/overwritten reels); lifecycle rule to expire intermediates after 30 days.
- [ ] **Redis (Upstash)**: it's cache/queue only now — nothing to back up. Confirm no durable data lives there after the DB cutover.
- [ ] **Secrets**: store a copy of all Fly secrets in a password manager (you can't read them back from Fly).
- [ ] **Code**: GitHub is the backup; tag each release.

## Pre-launch smoke (run after flipping secrets)
1. `curl …/healthz` → ok.
2. Logged-out API call → `401`. Logged-in → works, other user's job → `404`.
3. One real job (URL + upload) end-to-end → reel served from R2 signed URL.
4. Test Stripe purchase → credits land in the ledger (`get_balance`).
5. Kill a worker mid-job → hold refunded, job marked `failed` with a clear message.

## The five questions (answer "yes" before launch)
1. If someone spent $1000 on the OpenAI key in an hour, would I know? (budget alert)
2. If Redis dies, what does the user see? (`/healthz` 503 → LB reroutes)
3. Double Stripe webhook → double credit? (No — idempotent, tested)
4. Worker dies mid-job → hold refunded? (Yes — `task_failure` handler)
5. DMCA takedown → response process? (email + 24h SLA — register agent, PUBLISHING.md §0)
