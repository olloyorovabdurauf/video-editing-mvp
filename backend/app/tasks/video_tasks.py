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


def _update(job_id: str, **patch) -> None:
    raw = r.get(_key(job_id))
    state = json.loads(raw) if raw else {"job_id": job_id, "artifacts": []}
    state.update(patch)
    r.setex(_key(job_id), 60 * 60 * 24, json.dumps(state))


def get_job(job_id: str) -> ReelJobResponse | None:
    raw = r.get(_key(job_id))
    if not raw:
        return None
    return ReelJobResponse(**json.loads(raw))


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
        _update(job_id,
                status=ReelJobStatus.FAILED.value,
                message=f"{sender.name if sender else 'unknown'}: {exception}")
    except Exception as e:
        logger.warning("failure-handler couldn't refund: {}", e)


def _run(coro):
    """Bridge async helpers into sync Celery tasks (one loop per task call)."""
    return asyncio.run(coro)


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

    work = settings.storage_local_dir / "raw" / job_id
    work.mkdir(parents=True, exist_ok=True)

    try:
        if req.is_upload:
            src = ingestion.fetch_upload(req.upload_key, work)
        else:
            src = ingestion.download_source(str(req.source_url), work)
    except ingestion.IngestionError as e:
        logger.warning("ingestion failed for job {}: {}", job_id, e)
        # Retry transient blocks (403/429/timeout); fail permanent ones (private,
        # age-restricted) immediately with the user-facing message.
        if e.retryable and self.request.retries < self.max_retries:
            raise self.retry(exc=e, countdown=15)
        _update(job_id, status=ReelJobStatus.FAILED.value, message=e.user_message)
        raise

    return {"job_id": job_id, "payload": payload, "source_path": str(src)}


@shared_task(name="pipeline.transcribe", queue="ai", bind=True, max_retries=2)
def t_transcribe(self, ctx: dict) -> dict:
    _update(ctx["job_id"], status=ReelJobStatus.TRANSCRIBING.value, progress=0.20)
    try:
        transcript = _run(transcription.transcribe(Path(ctx["source_path"])))
    except transcription.VideoTooLong as e:
        # Permanent — retrying won't help. Fail the job with a user-facing message.
        logger.warning("transcription rejected: {}", e)
        _update(ctx["job_id"], status=ReelJobStatus.FAILED.value, message=str(e))
        raise
    except Exception as e:
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
    return {**ctx, "transcript_path": str(tpath), "audio_minutes": audio_minutes}


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
    _update(ctx["job_id"], status=ReelJobStatus.RENDERING.value, progress=0.60)
    req = ReelCreateRequest(**ctx["payload"])
    source = Path(ctx["source_path"])
    out_dir = settings.storage_local_dir / "output" / ctx["job_id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    broll_dir = settings.storage_local_dir / "intermediate" / ctx["job_id"]
    broll_dir.mkdir(parents=True, exist_ok=True)

    # Lazy imports — keep the task module light at import time.
    from app.services import music as music_svc
    from app.services.captions import write_ass
    from app.services.smart_crop import smart_crop_to_vertical

    artifacts: list[ReelArtifact] = []
    segments = [Segment(**s) for s in ctx["segments"]]
    raw_transcript = json.loads(Path(ctx["transcript_path"]).read_text(encoding="utf-8"))
    is_vertical = req.aspect == AspectRatio.VERTICAL
    target_dims = (1080, 1920) if is_vertical else (1920, 1080)

    for i, seg in enumerate(segments):
        progress = 0.60 + (0.35 * (i / max(1, len(segments))))
        _update(ctx["job_id"], progress=progress, message=f"rendering clip {i + 1}/{len(segments)}")

        # 1. Cut (re-encode for frame-accurate boundaries).
        cut_path = out_dir / f"seg_{i}_cut.mp4"
        _run(ff.cut(source, cut_path, start=seg.start, end=seg.end, reencode=True))
        current = cut_path

        # 2. Reframe — smart crop for vertical, identity for horizontal.
        if is_vertical:
            framed = out_dir / f"seg_{i}_framed.mp4"
            if req.smart_crop:
                _run(smart_crop_to_vertical(current, framed,
                                            target_w=1080, target_h=1920))
            else:
                _run(ff.reframe_to_vertical(current, framed))
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
                current = _run(composite(current, composite_clips, out_dir,
                                         name_prefix=f"seg_{i}_aibroll"))
            except Exception as e:
                logger.warning("AI b-roll composite failed for seg {}: {}", i, e)
                ai_clips_for_seg = []  # fall through to stock

        if req.add_broll and not ai_clips_for_seg:
            try:
                broll_meta = _run(broll_svc.find_broll_for_segment(
                    seg, i,
                    orientation="portrait" if is_vertical else "landscape",
                    download_to=broll_dir,
                ))
            except Exception as e:
                logger.warning("stock b-roll lookup failed for seg {}: {}", i, e)
                broll_meta = []
            for j, clip in enumerate(broll_meta):
                local = broll_dir / f"broll_seg{i}_{j}.mp4"
                if not local.exists():
                    continue
                overlaid = out_dir / f"seg_{i}_broll{j}.mp4"
                _run(ff.overlay_with_dissolve(
                    current, local, overlaid,
                    start=clip.start_offset,
                    duration=clip.duration,
                    dissolve=0.4,
                ))
                current = overlaid

        # 4. Animated captions (the single biggest "this looks pro" lever).
        if req.caption_style != "none":
            words_in_segment = [
                transcription.Word(
                    text=w["text"],
                    start=max(0.0, w["start"] - seg.start),
                    end=max(0.0, w["end"] - seg.start),
                )
                for w in raw_transcript["words"]
                if w["start"] >= seg.start and w["end"] <= seg.end
            ]
            if words_in_segment:
                ass_path = out_dir / f"seg_{i}.ass"
                write_ass(
                    words_in_segment, ass_path,
                    style=req.caption_style,
                    resolution=target_dims,
                )
                captioned = out_dir / f"seg_{i}_cap.mp4"
                _run(ff.burn_ass(current, ass_path, captioned))
                current = captioned

        # 5. Music bed (mood-selected, auto-ducked under speech).
        if req.add_music:
            try:
                track = _run(music_svc.pick_track(seg, override_mood=req.mood))
            except Exception as e:
                logger.warning("music pick failed for seg {}: {}", i, e)
                track = None
            if track:
                with_music = out_dir / f"seg_{i}_mix.mp4"
                try:
                    _run(ff.mix_music(current, track.path, with_music,
                                      music_volume=0.18, duck=True))
                    current = with_music
                except Exception as e:
                    logger.warning("music mix failed for seg {}: {}", i, e)

        # 6. Promote to final name + upload to durable storage.
        # In dev, LocalStorage returns "/storage/...". In prod (S3/R2),
        # it returns the public CDN URL. Either way, the frontend can fetch it.
        final = out_dir / f"reel_{i}.mp4"
        if current != final:
            current.rename(final)

        from app.services.storage import get_storage
        output_url = get_storage().put(
            final, key=f"output/{ctx['job_id']}/{final.name}",
        )

        artifacts.append(ReelArtifact(
            segment=seg,
            output_url=output_url,
            broll=broll_meta,
        ))

    # Generate ready-to-post title/caption/hashtags per clip. Best-effort: the
    # clips are already rendered, so a metadata hiccup must never fail the job.
    try:
        from app.services import clip_metadata
        metas = _run(clip_metadata.generate_for_clips(
            [clip_metadata.ClipInput(transcript=a.segment.transcript, reason=a.segment.reason)
             for a in artifacts]
        ))
        for a, m in zip(artifacts, metas):
            a.title, a.caption, a.hashtags = m.title, m.caption, m.hashtags
    except Exception as e:
        logger.warning("clip metadata step skipped: {}", e)

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
