# Reel Forge — Architecture

AI video **repurposing**: a long YouTube video → several complete, captioned,
face-tracked 45–60s vertical clips ("reels"). Read this before changing code so
you don't have to re-scan the repo.

## System overview

```
Browser (Next.js / Vercel)
   │  POST /api/v1/reels  (Clerk JWT)         ← returns 202 + job_id immediately
   ▼
FastAPI (Fly.io)  ──enqueue──►  Redis (broker + hot job state)
   │  GET /api/v1/reels/:id (poll)                 ▲
   ▼                                               │ writes job state each stage
Celery worker (same Fly machine)  ────────────────┘
   download → transcribe → analyze → render(parallel clips) → storage
        │            │           │            │
     yt-dlp      OpenAI       OpenAI       ffmpeg (libx264/NVENC)
                 Whisper      GPT-4o        + MediaPipe smart-crop
                                            + libass captions
Postgres (Fly MPG): users, jobs, clips, append-only credit ledger, usage
```

The async job layer already exists — POST returns a `job_id` at once, a Celery
chain processes in the background, the frontend polls, and clips **stream** into
job state as each finishes. It is NOT synchronous.

## Pipeline stages (the order things happen)

| Stage | Code | What it does |
|-------|------|--------------|
| Download | `tasks/video_tasks.py::t_download` → `services/ingestion.py` | yt-dlp (platform) or direct HTTP (media URL). Sweeps stale disk first. |
| Transcribe | `t_transcribe` → `services/transcription.py` (+ `services/google_stt.py`) | OpenAI Whisper, chunked >20MB, parallel. **Forces language only if Whisper supports it** (`_WHISPER_LANGS`). Languages in `GOOGLE_STT_LANGUAGES` (Uzbek) route to **Google STT** (native `uz-UZ`) when creds are set; any failure falls back to Whisper+translate. `Transcript.is_source_language` decides whether downstream translates. |
| Analyze | `t_analyze` → `services/segment_picker.py` | GPT picks COMPLETE 45–60s clips (hook→context→value→payoff), 4-factor scoring, sentence-boundary snapping, distribution fallback to guarantee N. |
| Render | `t_render` → `_render_all`/`_render_segment` | Per clip, **in parallel** (semaphore): cut → smart-crop → b-roll → captions → music → upload → metadata. Streams each finished clip to job state. |
| Caption | `services/captions.py` (+ `services/translation.py`) | Animated word-level ASS (libass). If `translate_to` set, translate lines to target language (line-level). |
| Metadata | `services/clip_metadata.py` | Per-clip title/caption/hashtags, locked to source language (translated when needed). |
| Storage | `services/storage.py` | Local volume now (`/app/storage`, 7-day retention sweep); R2 adapter ready. |

## Services (responsibility map)

- **video-processing**: `tasks/video_tasks.py`, `services/ingestion.py`, `utils/ffmpeg.py`
- **ai-analysis**: `services/segment_picker.py`, `services/transcription.py`
- **caption-service**: `services/captions.py`, `services/translation.py`, `services/clip_metadata.py`
- **render-service**: `services/smart_crop.py`, `utils/ffmpeg.py` (`VideoEncoder`/`CPUEncoder`/`GPUEncoder`), `services/music.py`, `services/creative_engine/*` (AI b-roll)
- **storage-service**: `services/storage.py`
- **billing**: `services/billing.py` + `db/repositories.py` (append-only credit ledger)
- **auth**: `core/auth.py` (Clerk JWKS), `db/repositories.upsert_user`
- **api**: `api/v1/endpoints/{reels,scripts,uploads,billing,health}.py`

## Data flow for one job

`source_url` → ctx dict threaded through the Celery chain. Hot state lives in
Redis at `job:{id}` (status/progress/artifacts/message/total_clips); durable
history in Postgres (`processing_jobs`). The frontend only ever sees the Redis
state via `GET /reels/:id`.

## Important files

- `app/tasks/video_tasks.py` — the whole pipeline orchestration (download→render).
- `app/services/segment_picker.py` — clip selection (the "viral moment" brain).
- `app/services/smart_crop.py` — face-tracking vertical crop.
- `app/services/transcription.py` — Whisper + language gating.
- `app/services/translation.py` — caption/title language correction.
- `app/utils/ffmpeg.py` — typed command builder + encoder strategy.
- `app/config.py` — all settings/env. `app/core/auth.py` — auth modes.
- `deploy/fly/fly.mvp.toml` — the live single-machine deploy (api+worker+volume).

## Environment / secrets (Fly)

`OPENAI_API_KEY`, `DATABASE_URL` (Fly MPG, pgbouncer), `REDIS_URL` + `CELERY_*`
(Upstash), `AUTH_MODE=clerk` + `CLERK_JWKS_URL`, `APP_ENV=production`,
`CORS_ORIGINS`, `REQUIRE_CREDITS=true`. Optional: `FFMPEG_HWACCEL` (=`none`
forces CPU), `FFMPEG_PRESET` (superfast), `RENDER_CONCURRENCY` (3), `YTDLP_PROXY`,
`GOOGLE_STT_CREDENTIALS` (service-account JSON → native Uzbek ASR),
`GOOGLE_STT_LANGUAGES` (default `uz`), `GOOGLE_STT_MODEL` (default `latest_long`).

## Known decisions / gotchas

- **`flyctl deploy` resets the VM size** (ignores `[[vm]]`) → re-run
  `flyctl scale vm performance-4x` after every deploy.
- **Whisper can't do Uzbek** (transcribes as Kazakh). Fixed with native Google
  STT (`services/google_stt.py`, `uz-UZ`), routed for `GOOGLE_STT_LANGUAGES` when
  `GOOGLE_STT_CREDENTIALS` is set; the Whisper+translate layer is now the
  fallback. Enable: set the `GOOGLE_STT_CREDENTIALS` Fly secret (service-account
  JSON) — no rebuild, the client ships in the image.
- **No GPU on the current Fly host** → `CPUEncoder` (libx264). `GPUEncoder`
  (NVENC) auto-engages on a GPU machine, zero code change.
- **Storage is a 40GB volume, not object storage** → 7-day retention sweep +
  cleanup-on-failure. R2 offload is the permanent fix.
- **Token efficiency**: the transcript is compressed to ~4s buckets before the
  LLM sees it (`segment_picker._compress`); segment picks + scripts are cached
  (`services/ai_cache.py`). Don't send raw full transcripts to the LLM.
