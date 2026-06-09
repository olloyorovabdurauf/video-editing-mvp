# Publishing — what you actually have to do to go live

A real launch checklist for **Reel Forge**. Every step has a "why" so you can
skip ones that don't apply, not skip ones because you're tired.

---

## 0. The honest gating question first

**YouTube downloading via `yt-dlp` violates YouTube's ToS.** That's the
single biggest legal risk in this codebase. You have three options before
launch:

1. **Don't accept YouTube URLs** — make users upload a file. Safer.
   Change `t_download` to accept a presigned-S3-upload URL only.
2. **Use the YouTube Data API** — official, requires OAuth from the user
   whose video it is. Legal but UX-heavy.
3. **Ship anyway and accept takedown risk.** Many AI-clipping competitors
   do this. Plan a migration path; expect a "cease and desist" within
   12 months of getting traction.

Pick before you take customer #1.

Same conversation applies to:
- **Music**: only ship tracks where `_licenses.json` has a real license.
  CC0 / royalty-free libraries (Pixabay Music, Free Music Archive,
  Epidemic Sound paid) are fine. Don't ship anything from "TikTok sounds".
- **Generated b-roll**: Runway, Higgsfield, Pika all currently grant the
  user commercial rights. Read each provider's ToS — it shifts.

---

## 1. Hosting — what to put where

Stage 0 (≤100 users) recommendation. Each row is "service + why".

| Component       | Service                                     | Why                                           |
|-----------------|---------------------------------------------|-----------------------------------------------|
| Frontend        | **Vercel** (free → $20/mo)                  | Native Next.js, free SSL, instant rollback    |
| API + workers   | **Fly.io** or **Railway** ($10–50/mo)       | Multi-region, persistent volumes, Docker-native |
| Redis           | **Upstash** (free → $10/mo at scale)        | Pay-per-command, no provisioning              |
| Object storage  | **Cloudflare R2** (free 10GB, $0.015/GB)   | S3 API, no egress fees — critical for serving MP4s |
| Postgres        | Skip for now. Add Neon when you need it.    | Redis ledger is fine ≤1k paying users        |
| Email           | **Resend** ($0–20/mo)                       | Transactional only; receipts + password reset |
| Error tracking  | **Sentry** (free 5k events/mo)             | Wire `SENTRY_DSN` into FastAPI + Celery       |
| Logs            | Fly.io / Railway built-in → Better Stack    | Request-ID makes them grep-able               |
| Monitoring      | **Better Stack Uptime** (free)              | Hits `/api/v1/livez` every 30s                |
| Billing         | **Stripe** (2.9% + 30¢)                     | Already wired                                 |

**Total monthly to start: ~$30–80**, scaling roughly linearly with users.

Why not AWS/GCP? You will spend more time on infra than on product.
Migrate later, when you actually have a load shape worth optimizing.

---

## 2. Pre-launch checklist (do these in order)

### Secrets & config
- [ ] **Rotate every default**. `OPENAI_API_KEY`, Pexels, Runway, Stripe live
      keys. None should ever be in `.env.example` or git history.
      `git log --all -p | grep -i 'sk-'` — if anything appears, rotate it.
- [ ] **Set `APP_ENV=production`** in Fly/Railway secrets.
- [ ] **`STRIPE_WEBHOOK_SECRET`** must be set. The webhook handler accepts
      *unverified* events when this is empty (intentional for dev, fatal
      in prod). The code logs `STRIPE_WEBHOOK_SECRET unset — accepting
      unverified event` — set up a log-based alert on that string.
- [ ] **`CORS_ORIGINS`** = `https://reelforge.com` (your real domain),
      not `*`.
- [ ] **`REQUIRE_CREDITS=true`** when you start charging.

### Auth
- [ ] Wire **Clerk** or **Auth.js** into the frontend. ~30 min.
      Replace `user_id="anonymous"` in `ReelCreateRequest` with the
      authenticated user's id. Frontend sends `X-User-Id` header; backend
      reads it via FastAPI dep (write a 10-line dep).
- [ ] Add a `verify_user_id` dep that compares the header against a
      JWT/session — never trust a raw header. ⚠️ Without this anyone can
      spend anyone else's credits.

### Stripe
- [ ] Create **live mode** products + prices that match `CREDIT_PACKS` in
      `app/api/v1/endpoints/billing.py`. Use price IDs in the frontend.
- [ ] Set up the **Stripe webhook endpoint** in your dashboard pointing at
      `https://api.reelforge.com/api/v1/billing/webhook/stripe`.
      Subscribe to `checkout.session.completed`.
- [ ] **Test the webhook** with Stripe CLI:
      `stripe trigger checkout.session.completed --add metadata.user_id=u1 --add metadata.pack=starter`
- [ ] Read the **idempotency test** in `tests/test_billing.py::test_credit_is_idempotent`.
      If you change `billing.py`, that test is your guard against double-charging.

### Storage migration
- [ ] In `config.py`, set `STORAGE_BACKEND=s3`, point `S3_*` at R2.
- [ ] Implement `services/storage.py` (currently the local mount is hardcoded).
      The interface is two methods: `put(local_path) -> public_url`,
      `presigned_upload(filename) -> {url, fields}`.
- [ ] Update `t_render` final step to upload to S3 and return the public URL
      instead of writing to `/storage/output/`.
- [ ] Set a 30-day lifecycle rule on the R2 bucket. Renders aren't valuable
      forever; users download or they don't.

### Database (when you're ready)
- [ ] Spin up **Neon** Postgres. Free tier is generous.
- [ ] Migrate the *ledger* to Postgres only — keep wallet balance in
      Redis for low-latency reads. SQL is for analytics, disputes, and
      "how much did user X spend in Jan".
- [ ] Schema: `users`, `jobs` (one row per reel job, JSON `payload`), `ledger_entries`
      (append-only, never UPDATE). Use Alembic for migrations.

### Operations
- [ ] **Health endpoints**: configure Fly's `services.tcp_checks` to hit
      `/api/v1/livez` (always-200 if process up) and a separate readiness
      check on `/api/v1/healthz` (503 if Redis/ffmpeg unavailable).
- [ ] **Set `MAX_AI_BROLL_BUDGET_USD=8.0`** (or your real cap). This is
      the belt-and-suspenders against an OpenAI cost runaway.
- [ ] **Set the rate limit**: `RATE_LIMIT_REELS_PER_MIN=5` for free tier,
      override per-user via Redis when they pay.
- [ ] **Wire Sentry**: `pip install sentry-sdk`, add 5 lines to `app/main.py`
      and `app/core/celery_app.py`.
- [ ] **Alert on `task_failure` from `creative_tasks`**: anything Runway
      adjacent failing > 10/hour means a provider outage and you want
      to know.

### Legal
- [ ] **Privacy policy + ToS** at the bottom of the home page. Cookies notice
      if you use anything beyond first-party session. Termly / iubenda
      generators do an acceptable job; have a lawyer review the
      auto-generated text before launching paid.
- [ ] **DMCA agent** registered with the US Copyright Office (free, online).
      Users *will* try to download copyrighted content; you need
      a takedown process the day you ship.
- [ ] **AI disclosure**: if you publish generated content to users' social
      accounts, comply with each platform's AI-content labeling rule
      (Meta and TikTok both require it for synthetic media now).

---

## 3. The actual go-live sequence

```bash
# 1. Final local sanity — make sure CI is green on the branch you ship.
cd backend && pytest -q                                  # must show "38 passed"
docker build -t reelforge-backend:rc1 .
docker build -t reelforge-frontend:rc1 ../frontend --target prod

# 2. Push images. Fly.io example:
flyctl auth login
flyctl launch --no-deploy --name reelforge-api
# answer questions, accept generated fly.toml — then edit it to add:
#   [[mounts]] source="storage" destination="/app/storage" size_gb=10
#   [services] internal_port=8000

flyctl secrets set \
  OPENAI_API_KEY=sk-prod-... \
  STRIPE_API_KEY=sk_live_... \
  STRIPE_WEBHOOK_SECRET=whsec_... \
  REDIS_URL=rediss://default:...@... \
  CORS_ORIGINS=https://reelforge.com \
  APP_ENV=production

flyctl deploy
flyctl scale count 1 --process api
flyctl scale count 2 --process worker-ffmpeg
flyctl scale count 1 --process worker-ai
flyctl scale count 1 --process worker-io

# 3. Frontend — Vercel
cd ../frontend
vercel --prod
# in Vercel dashboard set env: NEXT_PUBLIC_BACKEND_URL=https://api.reelforge.com
# remove NEXT_PUBLIC_DEMO entirely

# 4. DNS — Cloudflare or your registrar
#    reelforge.com         → Vercel
#    api.reelforge.com     → Fly.io
#    Wait for cert provisioning (~2 min)

# 5. Smoke test live
curl https://api.reelforge.com/api/v1/livez               # → {"ok": true}
curl https://api.reelforge.com/api/v1/healthz             # → all checks ok
# Open https://reelforge.com, paste a real YouTube URL, run a real job.
# Watch flyctl logs in another terminal.
```

---

## 4. First-week monitoring — what to actually watch

| Metric                                  | Where                          | Alert when                            |
|-----------------------------------------|--------------------------------|---------------------------------------|
| OpenAI spend per day                    | platform.openai.com → Usage    | > $50 day 1, > $200 day 7             |
| Runway spend                            | dev.runwayml.com → Usage       | > 2x your forecast                    |
| `task_failure` rate                     | Sentry / Flower                | > 5% of jobs                          |
| p95 job latency                         | log analysis on `request_id`   | > 4 min                               |
| Stripe webhook failures                 | Stripe dashboard               | any                                   |
| `/healthz` 503s                         | Better Stack                   | any                                   |
| Free→paid conversion                    | Stripe + your wallet ledger    | < 3% week 1                           |

Pick **one signal** to look at every morning the first 30 days. Mine is
"unique users who completed at least 1 successful reel yesterday". If
that's growing week-over-week, things are fine. If it's flat, your
problem isn't an engineering one.

---

## 5. Rollback plan

When (not if) a deploy breaks production:

```bash
# Fly
flyctl releases                                          # list releases
flyctl deploy --image-label v123                         # roll to previous

# Vercel — instant rollback in dashboard, or:
vercel rollback <deployment-url>
```

The credit ledger is **append-only and idempotent** by design — you can
roll back code without worrying about charge consistency. The `storage/`
volume is the one piece that's not rollback-safe; failed renders just
re-run, no harm done.

---

## 6. Pricing recommendations (because someone always asks)

- Starter pack: **$5 = 500 credits** (~5 reels with AI b-roll, or ~20 stock-only)
- Pro pack:    **$10 = 1,200 credits** (~20% bonus — most popular)
- Agency pack: **$40 = 6,000 credits** (~33% bonus)

Margin at OpenAI + Runway costs: ~70%. You can absorb a 2x cost
increase from providers without changing prices.

Don't do unlimited subscriptions. Generative video costs scale with use,
unlimited is how AI startups go bankrupt.

---

## 7. What I deliberately omit from this runbook

- Kubernetes manifests, Terraform, complex CI matrices — none of these
  make customers happier or you richer until you have ~$10k MRR.
- Multi-region. One region is fine until you have customers complaining
  about latency from another continent.
- A separate staging environment. Use Fly's preview deploys + Vercel's
  preview URLs, ship to "production" behind a feature flag for risky changes.

---

## 8. The one-page test before pressing the deploy button

Ask yourself, in order, the night before:

1. If a stranger spent $1000 on my OpenAI key in the next hour, would I
   know? (Cost alert wired.) ✅ Required.
2. If Redis goes down, what does the user see? (503 from `/healthz` →
   LB routes elsewhere, OR an honest error page.) ✅ Required.
3. If Stripe webhook arrives twice for the same purchase, does the user
   get credited twice? (No — `test_credit_is_idempotent`.) ✅ Done.
4. If a Celery worker dies mid-job, does the user's hold get refunded?
   (Yes — `task_failure` signal in `video_tasks.py`.) ✅ Done.
5. If I get a DMCA takedown, what's my response time? (Have the email
   address + form ready. 24h response is industry norm.) ⚠️ Set up before launch.

If you can answer "yes" or "I have a runbook" to all five, you're ready.
