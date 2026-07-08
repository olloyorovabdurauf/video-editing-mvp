"""
The reel-generation pipeline as a Celery chain.

Stages are intentionally small so each is independently retryable and the
result of an expensive call (Whisper, GPT-4o) is reusable if a later stage
fails. Job state is mirrored into Redis under `job:{id}` so the API can
serve status without hitting Celery's result backend on every poll.
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from pathlib import Path

import redis
from celery import chain, shared_task
from loguru import logger

from app.config import get_settings
from app.core.celery_app import celery_app  # noqa: F401  (registers tasks)
from app.schemas.reel import (
    AspectRatio,
    ReelArtifact,
    ReelCreateRequest,
    ReelJobResponse,
    ReelJobStatus,
    Segment,
)
from app.services import broll as broll_svc
from app.services import ingestion, segment_picker, transcription
from app.utils import ffmpeg as ff

settings = get_settings()
r = redis.from_url(settings.redis_url, decode_responses=True)


# ---------------------------------------------------------------------------
# Job state helpers
# ---------------------------------------------------------------------------

def _key(job_id: str) -> str:
    return f"job:{job_id}"


_NON_TERMINAL = {"queued", "downloading", "transcribing", "analyzing",
                 "generating_broll", "rendering"}
# If a non-terminal job hasn't advanced in this long, treat it as dead (the
# worker was killed by a deploy/restart, or a stage hung). A streaming render
# updates state every clip (~1-3 min), so this won't false-positive a slow job.
_STALE_AFTER_S = 12 * 60


def _update(job_id: str, **patch) -> None:
    raw = r.get(_key(job_id))
    state = json.loads(raw) if raw else {"job_id": job_id, "artifacts": []}
    state.update(patch)
    state["updated_at"] = time.time()        # heartbeat for the stale-job guard
    r.setex(_key(job_id), 60 * 60 * 24, json.dumps(state))


def get_job(job_id: str) -> ReelJobResponse | None:
    raw = r.get(_key(job_id))
    if not raw:
        return None
    state = json.loads(raw)
    # Fail orphaned/hung jobs instead of letting the UI spin forever.
    if state.get("status") in _NON_TERMINAL:
        last = state.get("updated_at") or state.get("created_at") or 0
        if last and (time.time() - last) > _STALE_AFTER_S:
            state["status"] = ReelJobStatus.FAILED.value
            state["message"] = "Processing stopped unexpectedly (the server may have restarted). Please try again."
            r.setex(_key(job_id), 60 * 60 * 24, json.dumps(state))
    return ReelJobResponse(**state)


def get_job_owner(job_id: str) -> str | None:
    """Owner user_id for authorization checks. None if job unknown."""
    raw = r.get(_key(job_id))
    if not raw:
        return None
    return json.loads(raw).get("user_id")


# ---------------------------------------------------------------------------
# Failure handler — refunds the credit hold if any stage of the chain dies
# ---------------------------------------------------------------------------

from celery.signals import task_failure                                  # noqa: E402


@task_failure.connect
def _on_task_failure(sender=None, task_id=None, exception=None, **kw):
    """Any task in the chain failing → refund the credit hold + mark job failed."""
    # We can only refund if the failing task can give us the job_id from ctx.
    # The chain's tasks all carry ctx as their first arg-or-result; we grab
    # it from the AsyncResult if possible. Best-effort — production should
    # link this via task chord callback for stricter semantics.
    try:
        args = kw.get("args") or []
        # Chained tasks (transcribe/analyze/render) carry a ctx dict as arg0;
        # t_download carries the job_id string as arg0. Handle both.
        job_id = None
        if args:
            if isinstance(args[0], dict):
                job_id = args[0].get("job_id")
            elif isinstance(args[0], str):
                job_id = args[0]
        if not job_id:
            return
        state = json.loads(r.get(_key(job_id)) or "{}")
        from app.services import billing as _billing
        if state.get("credit_hold_id") and state.get("user_id") not in (None, "anonymous"):
            _billing.refund(state["user_id"], state["credit_hold_id"])
        msg = (_AI_QUOTA_MSG if _is_quota_error(exception)
               else f"{sender.name if sender else 'unknown'}: {exception}")
        _update(job_id, status=ReelJobStatus.FAILED.value, message=msg)
        _purge_job_files(job_id)        # don't leak the download of a failed job
    except Exception as e:
        logger.warning("failure-handler couldn't refund/clean: {}", e)


def _run(coro):
    """Bridge async helpers into sync Celery tasks (one loop per task call)."""
    return asyncio.run(coro)


# Friendly message + detector for an exhausted OpenAI account (HTTP 429
# insufficient_quota). Retrying won't help — it needs a billing top-up — so we
# fail fast and show this instead of the raw provider JSON.
_AI_QUOTA_MSG = "AI service is temporarily unavailable (provider quota reached). Please try again later."


def _is_quota_error(exc: object) -> bool:
    s = str(exc).lower()
    return "insufficient_quota" in s or "exceeded your current quota" in s


# ---------------------------------------------------------------------------
# Disk hygiene — the storage volume is finite (no R2 yet). raw/ + intermediate/
# are transient (never needed once a job ends); output/ holds served reels.
# ---------------------------------------------------------------------------

def _purge_job_files(job_id: str) -> None:
    """Delete every on-disk trace of one job (used when a job fails)."""
    import shutil
    base = settings.storage_local_dir
    for sub in ("raw", "intermediate", "output"):
        shutil.rmtree(base / sub / job_id, ignore_errors=True)


def _sweep_storage(*, transient_age_h: int = 2, output_age_days: int = 7) -> None:
    """
    Bound disk usage so the volume can't fill (the "No space left on device" bug).
    raw/ + intermediate/ older than a couple hours are leftovers from finished or
    dead jobs → always safe to delete. output/ is kept `output_age_days` so users
    can still fetch recent reels (permanent storage arrives with R2). Best-effort.
    """
    import shutil
    base = settings.storage_local_dir
    now = time.time()
    rules = (("raw", transient_age_h * 3600), ("intermediate", transient_age_h * 3600),
             ("output", output_age_days * 86400))
    for sub, max_age in rules:
        d = base / sub
        if not d.exists():
            continue
        for child in d.iterdir():
            try:
                if now - child.stat().st_mtime > max_age:
                    shutil.rmtree(child, ignore_errors=True)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Source download helpers
#
# Two distinct paths, because they have different failure modes:
#   - Direct media file URL (…/clip.mp4) → plain HTTP stream. yt-dlp's generic
#     extractor sends a non-browser User-Agent that many CDNs (w3.org, GCS,
#     etc.) answer with 403, so we must NOT route direct files through it.
#   - Platform URL (YouTube/Vimeo/…) → yt-dlp, but hardened with a socket
#     timeout, retry cap, and a browser UA so a stalled host can't hang a
#     worker forever.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

@shared_task(name="pipeline.download", queue="io", bind=True, max_retries=3)
def t_download(self, job_id: str, payload: dict) -> dict:
    req = ReelCreateRequest(**payload)
    _update(job_id, status=ReelJobStatus.DOWNLOADING.value, progress=0.05)

    # Reclaim disk before pulling a fresh (possibly 500MB+) download, so the
    # volume can't fill from leftovers of old/failed jobs.
    _sweep_storage()

    work = settings.storage_local_dir / "raw" / job_id
    work.mkdir(parents=True, exist_ok=True)

    audio_only = False
    try:
        if req.is_upload:
            src = ingestion.fetch_upload(req.upload_key, work)
        elif ingestion.looks_like_direct_media(str(req.source_url)):
            src = ingestion.download_source(str(req.source_url), work)
        else:
            # Platform URL (YouTube/…): audio-first. The full video is never
            # pulled — render fetches ONLY the AI-selected sections later.
            # Through the metered ~300KB/s residential proxy this is the
            # difference between ~90s and ~25min for a long video.
            try:
                src = ingestion.download_audio(str(req.source_url), work)
                audio_only = True
            except ingestion.IngestionError:
                raise
            except Exception:
                logger.exception("audio-first download failed; falling back to full")
                src = ingestion.download_source(str(req.source_url), work)
    except ingestion.IngestionError as e:
        logger.warning("ingestion failed for job {}: {}", job_id, e)
        # Retry transient blocks (403/429/timeout); fail permanent ones (private,
        # age-restricted) immediately with the user-facing message.
        if e.retryable and self.request.retries < self.max_retries:
            raise self.retry(exc=e, countdown=15)
        _update(job_id, status=ReelJobStatus.FAILED.value, message=e.user_message)
        raise

    return {"job_id": job_id, "payload": payload, "source_path": str(src),
            "audio_only": audio_only, "source_url": str(req.source_url or "")}


@shared_task(name="pipeline.transcribe", queue="ai", bind=True, max_retries=2)
def t_transcribe(self, ctx: dict) -> dict:
    _update(ctx["job_id"], status=ReelJobStatus.TRANSCRIBING.value, progress=0.20)
    # User-locked source language (forces Whisper) or auto-detect.
    lang = (ctx.get("payload") or {}).get("language") or None
    try:
        transcript = _run(transcription.transcribe(Path(ctx["source_path"]), language=lang))
    except transcription.VideoTooLong as e:
        # Permanent — retrying won't help. Fail the job with a user-facing message.
        logger.warning("transcription rejected: {}", e)
        _update(ctx["job_id"], status=ReelJobStatus.FAILED.value, message=str(e))
        raise
    except Exception as e:
        if _is_quota_error(e):           # OpenAI billing — don't waste retries
            logger.error("OpenAI quota exhausted — top up the OpenAI account: {}", e)
            _update(ctx["job_id"], status=ReelJobStatus.FAILED.value, message=_AI_QUOTA_MSG)
            raise
        logger.exception("transcription failed")
        raise self.retry(exc=e, countdown=10)

    # Persist transcript so re-runs of downstream stages don't re-pay OpenAI.
    tpath = Path(ctx["source_path"]).with_suffix(".transcript.json")
    tpath.write_text(json.dumps({
        "language": transcript.language,
        "text": transcript.text,
        "words": [w.__dict__ for w in transcript.words],
    }), encoding="utf-8")

    audio_minutes = round((transcript.words[-1].end / 60.0) if transcript.words else 0.0, 2)
    # Lock the language for all downstream text (captions/titles) to the source.
    # Only translate when the transcript ISN'T already in the requested language —
    # i.e. Whisper was forced toward Uzbek and emitted Kazakh. When Google STT
    # produced native Uzbek (is_source_language=True), no translation is needed.
    translate_to = lang if (lang and not transcript.is_source_language) else None
    source_language = transcript.language
    if not lang:
        # Auto-detect mode: verify the ASR's label against the TEXT. Whisper
        # mislabels Uzbek as Kazakh; without this check the wrong language
        # flowed straight into captions/titles whenever the user left the
        # selector on Auto. The guard's verdict wins on disagreement.
        from app.services import language_guard
        detected = _run(language_guard.detect_text_language(transcript.text))
        if detected and detected != transcript.language:
            logger.warning("language guard: ASR labeled '{}' but text is '{}' — "
                           "translating captions/metadata to '{}'",
                           transcript.language, detected, detected)
            source_language = detected
            translate_to = detected
    return {**ctx, "transcript_path": str(tpath), "audio_minutes": audio_minutes,
            "source_language": source_language, "translate_to": translate_to}


@shared_task(name="pipeline.analyze", queue="ai", bind=True, max_retries=2)
def t_analyze(self, ctx: dict) -> dict:
    _update(ctx["job_id"], status=ReelJobStatus.ANALYZING.value, progress=0.40)
    req = ReelCreateRequest(**ctx["payload"])

    raw = json.loads(Path(ctx["transcript_path"]).read_text(encoding="utf-8"))
    transcript = transcription.Transcript(
        language=raw["language"],
        text=raw["text"],
        words=[transcription.Word(**w) for w in raw["words"]],
    )

    segments = _run(segment_picker.pick_segments(
        transcript,
        n=req.target_count,
        min_duration_s=req.min_duration_s,
        max_duration_s=req.max_duration_s,
        prompt=req.prompt,
    ))
    return {**ctx, "segments": [s.model_dump() for s in segments]}


@shared_task(name="pipeline.render", queue="ffmpeg", bind=True, max_retries=1)
def t_render(self, ctx: dict) -> dict:
    """
    Render every segment into a finished MP4. Per-segment pipeline:

        cut → smart_crop (or center reframe) → composite b-roll
            → burn animated captions → mix music → final mp4

    Each step is a single ffmpeg pass and produces a new intermediate file;
    we never re-encode for free. The encoder settings (preset/CRF) stay
    consistent across passes so quality doesn't degrade.
    """
    req = ReelCreateRequest(**ctx["payload"])
    out_dir = settings.storage_local_dir / "output" / ctx["job_id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    broll_dir = settings.storage_local_dir / "intermediate" / ctx["job_id"]
    broll_dir.mkdir(parents=True, exist_ok=True)

    segments = [Segment(**s) for s in ctx["segments"]]
    raw_transcript = json.loads(Path(ctx["transcript_path"]).read_text(encoding="utf-8"))
    is_vertical = req.aspect == AspectRatio.VERTICAL
    target_dims = (1080, 1920) if is_vertical else (1920, 1080)

    _update(ctx["job_id"], status=ReelJobStatus.RENDERING.value, progress=0.60,
            total_clips=len(segments), completed_clips=0,
            message=(f"fetching + rendering {len(segments)} clips"
                     if ctx.get("audio_only") else f"rendering {len(segments)} clips"))

    # Render ALL clips CONCURRENTLY (was a sequential loop — the multi-clip
    # bottleneck). Each clip streams into job state the instant it finishes, so
    # the UI shows the first reel in tens of seconds instead of at the very end.
    artifacts = _run(_render_all(
        ctx, req, segments, raw_transcript, out_dir, broll_dir, target_dims, is_vertical,
    ))

    # Settle credits — refund the over-estimate, charge any overage.
    job_state = json.loads(r.get(_key(ctx["job_id"])) or "{}")
    if job_state.get("credit_hold_id") and job_state.get("user_id") not in (None, "anonymous"):
        from app.services import billing as _billing
        # Actual = base + smart_crop + per-broll generations actually paid for.
        n_ai = sum(len(v) for v in (ctx.get("ai_broll_by_segment") or {}).values())
        actual = (
            _billing.PRICING_CREDITS["reel_base"] * len(artifacts)
            + (_billing.PRICING_CREDITS["smart_crop"] * len(artifacts) if req.smart_crop else 0)
            + _billing.PRICING_CREDITS["ai_broll_gen"] * n_ai
        )
        _billing.settle(
            job_state["user_id"], job_state["credit_hold_id"],
            actual_amount=actual,
        )

    # Reclaim disk: the raw download (a long video can be 500MB+) and the
    # intermediate b-roll clips are no longer needed once finals are rendered.
    # Final reels live under output/ and are kept (served to the user).
    import shutil
    raw_dir = settings.storage_local_dir / "raw" / ctx["job_id"]
    inter_dir = settings.storage_local_dir / "intermediate" / ctx["job_id"]
    for d in (raw_dir, inter_dir):
        shutil.rmtree(d, ignore_errors=True)

    # Observability: cost + processing time, written into job state.
    from app.core import metrics
    n_ai = sum(len(v) for v in (ctx.get("ai_broll_by_segment") or {}).values())
    cost = metrics.estimate_job_cost_usd(
        audio_minutes=float(ctx.get("audio_minutes", 0.0)),
        n_clips=len(artifacts), ai_broll_clips=n_ai,
    )
    prior = json.loads(r.get(_key(ctx["job_id"])) or "{}")
    proc_s = round(metrics.now() - prior.get("created_at", metrics.now()), 1)

    _update(
        ctx["job_id"],
        status=ReelJobStatus.SUCCEEDED.value,
        progress=1.0,
        artifacts=[a.model_dump(mode="json") for a in artifacts],
        cost_usd=cost,
        processing_time_s=proc_s,
        audio_minutes=ctx.get("audio_minutes", 0.0),
    )
    # Durable history (no-op until DATABASE_URL is set). Never let a DB hiccup
    # fail a job that already succeeded — the reels are rendered and served.
    try:
        from app.db import repositories
        repositories.record_completed_job(
            job_id=ctx["job_id"], user_id=prior.get("user_id", "anonymous"),
            status="completed", cost_usd=cost, processing_time_s=proc_s,
            clips=len(artifacts), audio_minutes=float(ctx.get("audio_minutes", 0.0)),
        )
    except Exception as e:
        logger.warning("durable job record skipped: {}", e)

    logger.info("job {} done: {} reels, ${} cost, {}s", ctx["job_id"], len(artifacts), cost, proc_s)
    return {"job_id": ctx["job_id"], "ok": True}


# ---------------------------------------------------------------------------
# Concurrent, streaming render — render all clips in parallel and push each one
# into job state the moment it finishes (so the UI shows results incrementally).
# ---------------------------------------------------------------------------

def _render_concurrency() -> int:
    return max(1, settings.render_concurrency)


def _commit_clip(job_id: str, artifact: ReelArtifact, total: int) -> None:
    """Append a finished clip to job state immediately. Runs without awaiting, so
    the Redis read-modify-write is atomic within the event loop (no lock needed)."""
    st = json.loads(r.get(_key(job_id)) or "{}")
    arts = st.get("artifacts", [])
    arts.append(artifact.model_dump(mode="json"))
    st["artifacts"] = arts
    st["completed_clips"] = len(arts)
    st["progress"] = round(min(0.95, 0.60 + 0.35 * (len(arts) / max(1, total))), 3)
    st.setdefault("status", ReelJobStatus.RENDERING.value)
    st["message"] = f"{len(arts)}/{total} clips ready"
    r.setex(_key(job_id), 60 * 60 * 24, json.dumps(st))


async def _render_all(ctx, req, segments, raw_transcript, out_dir, broll_dir,
                      target_dims, is_vertical) -> list[ReelArtifact]:
    source = Path(ctx["source_path"])
    total = len(segments)
    results: list[ReelArtifact | None] = [None] * total
    sem = asyncio.Semaphore(_render_concurrency())
    # MediaPipe/TF-Lite (smart crop face detection) is NOT safe to run several
    # times concurrently in one process — it corrupts the ffmpeg filter graph
    # ("Error reinitializing filters"). Serialize just that step; everything else
    # (cut, captions, music, encode, upload) still runs in parallel.
    crop_lock = asyncio.Lock()

    async def worker(i: int, seg: Segment) -> None:
        async with sem:
            try:
                art = await _render_segment(
                    i, seg, ctx=ctx, req=req, source=source, raw_transcript=raw_transcript,
                    out_dir=out_dir, broll_dir=broll_dir, target_dims=target_dims,
                    is_vertical=is_vertical, crop_lock=crop_lock)
            except Exception as e:                       # one clip failing must not sink the rest
                logger.warning("clip {} render failed, skipping: {}", i, e)
                return
            results[i] = art
            _commit_clip(ctx["job_id"], art, total)      # stream it to the UI now

    await asyncio.gather(*[worker(i, s) for i, s in enumerate(segments)])
    return [a for a in results if a is not None]


async def _render_segment(i, seg, *, ctx, req, source, raw_transcript, out_dir,
                          broll_dir, target_dims, is_vertical, crop_lock) -> ReelArtifact:
    """One clip's pipeline: cut → reframe → b-roll → captions → music → upload →
    metadata. Mirrors the old loop body but fully awaitable so clips run in
    parallel under the semaphore."""
    from app.services import captions as captions_mod
    from app.services import clip_metadata
    from app.services import music as music_svc
    from app.services import translation
    from app.services.captions import write_ass
    from app.services.smart_crop import smart_crop_to_vertical
    from app.services.storage import get_storage

    translate_to = ctx.get("translate_to")     # set when the source language needs translating
    translated_text: str | None = None         # reused for metadata if captions translate

    # 1. Cut (re-encode for frame-accurate boundaries).
    #    Audio-only jobs never downloaded the video — fetch JUST this clip's
    #    window instead (already frame-accurate via force_keyframes_at_cuts),
    #    each clip through its own proxy session so downloads run in parallel
    #    on separate residential lines. Falls back to nothing: a section
    #    failure only skips this clip, the others keep rendering.
    if ctx.get("audio_only"):
        raw_dir = settings.storage_local_dir / "raw" / ctx["job_id"]
        try:
            cut_path = await asyncio.to_thread(
                ingestion.download_section, ctx["source_url"], raw_dir, i,
                seg.start, seg.end, session=i)
        except Exception as e:
            # One transient section failure shouldn't cost the user a clip —
            # retry once on a DIFFERENT residential line (offset session).
            logger.warning("section {} download failed ({}); retrying on another line", i, e)
            cut_path = await asyncio.to_thread(
                ingestion.download_section, ctx["source_url"], raw_dir, i,
                seg.start, seg.end, session=i + 5)
    else:
        cut_path = out_dir / f"seg_{i}_cut.mp4"
        await ff.cut(source, cut_path, start=seg.start, end=seg.end, reencode=True)
    current = cut_path

    # 2. Reframe — smart crop for vertical, identity for horizontal.
    if is_vertical:
        framed = out_dir / f"seg_{i}_framed.mp4"
        if req.smart_crop:
            try:
                async with crop_lock:        # serialize face detection (not concurrency-safe)
                    await smart_crop_to_vertical(current, framed, target_w=1080, target_h=1920)
            except Exception as e:           # never lose a clip to a crop hiccup — center-crop it
                logger.warning("smart_crop failed for seg {}, center-cropping: {}", i, e)
                await ff.reframe_to_vertical(current, framed)
        else:
            await ff.reframe_to_vertical(current, framed)
        current = framed

    # 3. B-roll: AI first (premium), stock as fallback.
    ai_clips_for_seg = (ctx.get("ai_broll_by_segment") or {}).get(str(i)) \
        or (ctx.get("ai_broll_by_segment") or {}).get(i, [])
    broll_meta = []
    if req.use_ai_broll and ai_clips_for_seg:
        from app.services.creative_engine.compositor import CompositeClip, composite
        composite_clips = [
            CompositeClip(
                path=Path(c["path"]),
                start=max(0.0, float(c["window_start_abs"]) - seg.start),
                duration=max(0.5, float(c["window_end_abs"]) - float(c["window_start_abs"])),
                dissolve=0.5,
            )
            for c in ai_clips_for_seg
        ]
        try:
            current = await composite(current, composite_clips, out_dir, name_prefix=f"seg_{i}_aibroll")
        except Exception as e:
            logger.warning("AI b-roll composite failed for seg {}: {}", i, e)
            ai_clips_for_seg = []
    if req.add_broll and not ai_clips_for_seg:
        try:
            broll_meta = await broll_svc.find_broll_for_segment(
                seg, i, orientation="portrait" if is_vertical else "landscape",
                download_to=broll_dir)
        except Exception as e:
            logger.warning("stock b-roll lookup failed for seg {}: {}", i, e)
            broll_meta = []
        for j, clip in enumerate(broll_meta):
            local = broll_dir / f"broll_seg{i}_{j}.mp4"
            if not local.exists():
                continue
            overlaid = out_dir / f"seg_{i}_broll{j}.mp4"
            await ff.overlay_with_dissolve(current, local, overlaid,
                                           start=clip.start_offset, duration=clip.duration, dissolve=0.4)
            current = overlaid

    # 4. Animated captions.
    if req.caption_style != "none":
        words_in_segment = [
            transcription.Word(text=w["text"],
                               start=max(0.0, w["start"] - seg.start),
                               end=max(0.0, w["end"] - seg.start))
            for w in raw_transcript["words"]
            if w["start"] >= seg.start and w["end"] <= seg.end
        ]
        if words_in_segment:
            ass_path = out_dir / f"seg_{i}.ass"
            if translate_to:
                # Whisper produced the wrong language → translate caption lines to
                # the target (line-level; per-word karaoke can't survive translation).
                phrases = captions_mod.group_into_phrases(words_in_segment, max_words=4)
                originals = [" ".join(w.text for w in ph.words) for ph in phrases]
                translated = await translation.translate_lines(originals, translate_to)
                translated_text = " ".join(t for t in translated if t.strip())
                ass_path.write_text(
                    captions_mod.render_ass_lines(phrases, translated,
                                                  style=req.caption_style, resolution=target_dims),
                    encoding="utf-8")
            else:
                write_ass(words_in_segment, ass_path, style=req.caption_style, resolution=target_dims)
            captioned = out_dir / f"seg_{i}_cap.mp4"
            await ff.burn_ass(current, ass_path, captioned)
            current = captioned

    # 5. Music bed (mood-selected, auto-ducked under speech).
    if req.add_music:
        try:
            track = await music_svc.pick_track(seg, override_mood=req.mood)
        except Exception as e:
            logger.warning("music pick failed for seg {}: {}", i, e)
            track = None
        if track:
            with_music = out_dir / f"seg_{i}_mix.mp4"
            try:
                await ff.mix_music(current, track.path, with_music, music_volume=0.18, duck=True)
                current = with_music
            except Exception as e:
                logger.warning("music mix failed for seg {}: {}", i, e)

    # 6. Promote to final name + faststart + poster, then upload to durable storage.
    final = out_dir / f"reel_{i}.mp4"
    if current != final:
        current.rename(final)

    # Progressive playback: relocate the moov atom to the front so the browser can
    # start rendering on the FIRST bytes instead of waiting for the whole file.
    # burn_ass/mix_music (the last transforms) don't set faststart themselves, so
    # without this the served clip has moov-at-end and feels slow to load.
    try:
        fast = out_dir / f"reel_{i}_fast.mp4"
        await ff.finalize_faststart(final, fast)
        fast.replace(final)
    except Exception as e:
        logger.warning("faststart remux failed for seg {} (serving as-is): {}", i, e)

    output_url = get_storage().put(final, key=f"output/{ctx['job_id']}/{final.name}")

    # Poster/thumbnail so the UI shows an instant image and only fetches the video
    # on demand (lazy loading). Best-effort — a poster hiccup never drops the clip.
    thumbnail_url: str | None = None
    try:
        poster = out_dir / f"reel_{i}.jpg"
        await ff.poster_frame(final, poster, at=min(1.5, max(0.3, seg.duration * 0.1)))
        thumbnail_url = get_storage().put(poster, key=f"output/{ctx['job_id']}/{poster.name}")
    except Exception as e:
        logger.warning("poster generation failed for seg {}: {}", i, e)

    art = ReelArtifact(segment=seg, output_url=output_url,
                       thumbnail_url=thumbnail_url, broll=broll_meta)

    # Per-clip metadata so each clip streams to the UI fully formed (title +
    # caption + hashtags). Best-effort — a metadata hiccup never drops the clip.
    try:
        # Feed metadata the TRANSLATED transcript when we have it, so titles/
        # captions come out in the target language instead of echoing Kazakh.
        meta_transcript = seg.transcript
        if translate_to:
            meta_transcript = translated_text or await translation.translate_text(
                seg.transcript, translate_to)
        metas = await clip_metadata.generate_for_clips(
            [clip_metadata.ClipInput(transcript=meta_transcript, reason=seg.reason)],
            language=ctx.get("source_language"))
        if metas:
            art.title, art.caption, art.hashtags = metas[0].title, metas[0].caption, metas[0].hashtags
    except Exception as e:
        logger.warning("clip {} metadata skipped: {}", i, e)
    return art


# ---------------------------------------------------------------------------
# Public entry point — called from the API
# ---------------------------------------------------------------------------

def enqueue_reel_job(req: ReelCreateRequest) -> str:
    """
    Reserve credits, persist job state, dispatch the Celery pipeline. The
    credit hold lives until t_render completes (success → settle to estimate,
    failure → refund). We never charge a user for a job we never started.
    """
    from app.services import billing

    job_id = uuid.uuid4().hex

    # 1. Up-front credit reservation. Reject early — before yt-dlp starts —
    #    if the user can't afford the upper-bound estimate.
    estimated = billing.estimate_job_credits(
        target_count=req.target_count,
        use_ai_broll=req.use_ai_broll,
        use_smart_crop=req.smart_crop,
    )
    # Only reserve credits when billing is actually enabled. With
    # REQUIRE_CREDITS=false (dev / free beta) nobody is charged, even
    # authenticated users. InsufficientCredits propagates to the API layer,
    # which maps it to 402 — don't wrap it (erasing the type forces string parsing).
    hold_id = ""
    if settings.require_credits and req.user_id != "anonymous":
        hold_id = billing.hold(req.user_id, estimated, job_id=job_id)

    from app.core import metrics
    _update(
        job_id,
        status=ReelJobStatus.QUEUED.value, progress=0.0, artifacts=[],
        user_id=req.user_id, credit_hold_id=hold_id, credits_held=estimated,
        created_at=metrics.now(),     # for processing-time metric
    )

    # 2. Build pipeline. Bridge into creative engine only if AI b-roll on.
    payload = req.model_dump(mode="json")
    use_ai_broll = bool(payload.get("use_ai_broll", False))

    if use_ai_broll:
        # Lazy import to avoid pulling the creative subgraph for builds that
        # might run with AI b-roll fully disabled (e.g. CI without API keys).
        from app.tasks.creative_tasks import t_plan_broll, _t_fanout_generations

        # We can't build the dynamic chord until analyze() returns, so we
        # bounce through a small bridge task that gets ctx and constructs
        # the chord topology with the actual segment count.
        chain(
            t_download.s(job_id, payload),
            t_transcribe.s(),
            t_analyze.s(),
            _t_bridge_to_creative.s(job_id),    # builds + waits on chord
            t_render.s(),
        ).apply_async()
    else:
        chain(
            t_download.s(job_id, payload),
            t_transcribe.s(),
            t_analyze.s(),
            t_render.s(),
        ).apply_async()
    return job_id


@shared_task(name="pipeline.bridge_creative", queue="io")
def _t_bridge_to_creative(ctx: dict, job_id: str) -> dict:
    """
    Runs between analyze and render. Builds the parallel planning group,
    waits for it, then runs the dynamic submit→poll→download chord. This
    task itself is on the io queue and only blocks on .get() — no CPU.

    Why a bridge task instead of a chord callback?
    Celery chords inside dynamic chains are flaky across worker restarts;
    a single bridge task with explicit .get() is more debuggable for MVP.
    Move to a chord-callback once latency hits the budget for that pattern.
    """
    from celery import group
    from app.tasks.creative_tasks import (
        t_plan_broll, t_submit_one, t_poll_one, t_download_one, t_collect_broll,
    )

    _update(job_id, message="planning AI b-roll", progress=0.45)

    num_segments = len(ctx.get("segments", []))
    if num_segments == 0:
        ctx["ai_broll_by_segment"] = {}
        return ctx

    # Phase 1: plan in parallel, one task per segment.
    plan_group = group(t_plan_broll.s(ctx, i) for i in range(num_segments))
    plan_results = plan_group.apply_async().get(disable_sync_subtasks=False)

    all_plans: list[dict] = []
    for r in plan_results:
        all_plans.extend(r.get("plans", []))
    logger.info("planned {} AI b-roll insertions across {} segments", len(all_plans), num_segments)

    if not all_plans:
        ctx["ai_broll_by_segment"] = {}
        ctx["ai_broll_total_cost_usd"] = 0.0
        return ctx

    _update(job_id, message=f"generating {len(all_plans)} AI b-roll clip(s)", progress=0.50)

    # Phase 2: submit → poll → download, all in parallel.
    budget = float(ctx["payload"].get("ai_broll_budget_usd", 4.0))
    per_plan_budget = budget / max(1, len(all_plans))
    pipelines = [
        (t_submit_one.s(plan, job_id, per_plan_budget)
         | t_poll_one.s(job_id)
         | t_download_one.s(job_id))
        for plan in all_plans
    ]
    download_results = group(pipelines).apply_async().get(disable_sync_subtasks=False)

    return t_collect_broll(download_results, ctx)
