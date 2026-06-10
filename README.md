# Reel Forge — AI Video Automation SaaS

> Long-form video → viral reels with AI-generated b-roll, captions, and music.

## What runs the show

```
┌─────────────────────────────────────────────────────────────────────┐
│  Next.js 14 (App Router, Tailwind)                                  │
│  • Submit form (URL + aspect + AI b-roll budget)                    │
│  • Job page with live polling, progress, MP4 previews               │
└──────────────────────┬──────────────────────────────────────────────┘
                       │ /api/v1/* (proxied)
                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│  FastAPI                                                            │
│  • POST /api/v1/reels   → 202 Accepted, returns job_id              │
│  • GET  /api/v1/reels/* → status, progress, artifacts               │
└──────────────────────┬──────────────────────────────────────────────┘
                       │ enqueue
                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Celery chain                                                       │
│  download → transcribe → analyze → [creative chord] → render        │
│                                       │                             │
│                                       ▼                             │
│              ┌──────────────────────────────────────────┐           │
│              │ creative_engine (per segment, parallel)  │           │
│              │  scene_extractor → prompts.compile()     │           │
│              │  → provider chain: Runway → Higgsfield   │           │
│              │  → poll (self-rescheduling) → download   │           │
│              │  → content cache (redis)                 │           │
│              └──────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────────────────┘
```

## Quickstart

```bash
cp backend/.env.example backend/.env
# fill at minimum: OPENAI_API_KEY
# optional but recommended: RUNWAY_API_KEY, HIGGSFIELD_API_KEY, PEXELS_API_KEY

docker compose up --build
```

- App:    http://localhost:3000
- API:    http://localhost:8000/docs
- Flower: http://localhost:5555

## End-to-end MVP flow

1. Open http://localhost:3000
2. Paste a YouTube URL
3. Pick aspect (9:16 for Reels), count, max duration
4. Toggle "Generate AI b-roll" + set budget
5. Submit → routed to `/jobs/{id}`
6. Watch the 7 stages tick green; reels appear as they finish rendering

## What happens under the hood

| Stage              | Queue   | What it does                                                |
|--------------------|---------|-------------------------------------------------------------|
| `download`         | io      | yt-dlp pulls source, normalizes to mp4                      |
| `transcribe`       | ai      | Whisper word-level timestamps                               |
| `analyze`          | ai      | GPT-4o picks top-N viral segments (hook_score, reason)      |
| `bridge_creative`  | io      | Plans b-roll + waits for parallel generation                |
| `creative.plan`    | ai      | Per-segment: scene_extractor finds insertion windows        |
| `creative.submit`  | ai      | Try providers in fallback order, return job handle          |
| `creative.poll`    | ai      | Self-rescheduling state machine (10s, cap 10 min wall time) |
| `creative.download`| io      | Stream provider asset → local intermediate                  |
| `render`           | ffmpeg  | Cut → reframe → composite (cross-dissolve) → captions       |

## Provider fallback chain

`Kling 3.0 (priority 5)` → `Higgsfield (priority 10)` → `Runway (priority 20)` → `Pexels stock` → no b-roll

Kling 3.0 is the primary: native text-to-video (no reference image needed),
cheapest per clip (~$0.35 std / ~$0.70 pro), 5s/10s at 9:16. Each provider
self-disables if its API key isn't set. Cache hits short-circuit the whole
submit/poll/download path.

## Repository layout

```
backend/app/
  api/v1/endpoints/         HTTP surface (reels.py, health.py)
  core/                     celery_app, logging
  schemas/                  Pydantic request/response/job models
  services/
    transcription.py        Whisper + SRT generation
    segment_picker.py       GPT-4o viral pick
    broll.py                Pexels stock fallback
    creative_engine/        AI b-roll subsystem
      engine.py             public orchestrator
      prompts.py            structured VisualPrompt + compiler
      scene_extractor.py    where-to-overlay analyzer
      compositor.py         cross-dissolve sequencer
      cache.py              content-addressable Redis cache
      providers/
        base.py             VideoGenProvider ABC + GenerationJob
        kling.py            Kling 3.0 — primary (JWT auth, native t2v)
        runway.py
        higgsfield.py
        registry.py         fallback chain + budget guard
  tasks/
    video_tasks.py          download/transcribe/analyze/render + bridge
    creative_tasks.py       plan/submit/poll/download/collect
  utils/
    ffmpeg.py               THE only place ffmpeg is invoked

frontend/
  app/page.tsx              submit form
  app/jobs/[id]/page.tsx    live status + reel grid
  components/JobProgress.tsx
  components/ReelCard.tsx
  lib/api.ts                typed fetch wrapper
```

## Cost notes

- Whisper: ~$0.006/min of audio
- GPT-4o segment picker: ~$0.02/job
- Kling 3.0 text-to-video: ~$0.35/clip std, ~$0.70 pro × 5-10s (primary)
- Runway Gen-3 image-to-video: ~$0.50/clip × 5s (fallback)
- Higgsfield: ~$0.40/clip × up to 8s (fallback)

A typical 3-reel job with 2 AI b-rolls per reel: **~$3-4 in inference**.
Cache reduces repeat-prompt cost to zero. Set `ai_broll_budget_usd` per request
to cap downside.

## Production hardening checklist (not in MVP)

- Postgres for job/billing history (Redis is hot state only)
- S3 / Cloudflare R2 for `storage/` instead of bind mount
- Auth (Clerk / Auth.js) + per-user quotas
- SSE/WebSocket progress instead of 2.5s polling
- Webhook receiver as the primary `creative.poll` path (poll task becomes fallback)
- Replace chord-of-chains with explicit callback to harden against worker restarts
- Speaker-tracking smart crop (mediapipe) instead of center crop
