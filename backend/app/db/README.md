# Database layer (Postgres) — migration plan

**Status:** schema defined (`models.py`), not yet wired. Redis remains the hot
job-state store. This is the durable-records layer for users / billing / usage.

## Why split Redis + Postgres
- **Redis** = hot, ephemeral execution state (job progress, artifacts). Losing it
  costs a re-run, nothing more. Keep it.
- **Postgres** = things you must never lose or must query historically: users,
  subscriptions, the money ledger, usage counters. An earlier audit flagged
  "ledger in Redis" as the #1 durability risk — this fixes it.

## Provision (Neon — free tier is plenty for first 1000 users)
1. Create a project at neon.tech → copy the `postgresql://...` connection string.
2. `flyctl secrets set DATABASE_URL="postgresql+psycopg://..." -a reelforge-mvp-x7k2`

## Wire (next PR — ~half a day)
1. `app/db/session.py`: async engine + `async_sessionmaker` from `DATABASE_URL`.
2. `alembic init alembic` → point `target_metadata = Base.metadata` → autogenerate
   the first migration → `alembic upgrade head` (run on deploy via release_command
   in fly.toml).
3. On Clerk login (first request from a new `sub`): upsert a `User`.
4. On job completion (`t_render`): write a `ProcessingJob` row (id == Redis job_id)
   + bump the `UsageCounter` for the period. Live progress still comes from Redis.
5. Move the credit ledger writes into a transaction alongside `Subscription`.

## Migration ordering (safe, zero-downtime)
1. Ship tables (additive) — app ignores them.
2. Start dual-writing durable records (Postgres) while still reading from Redis.
3. Backfill is unnecessary (no historical durable data worth keeping from the MVP).
4. Flip reads for user/subscription/usage to Postgres. Job progress stays Redis.

Lean on purpose. Add tables when a feature needs them, not before.
