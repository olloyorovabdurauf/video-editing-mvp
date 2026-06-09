"""
Generative b-roll tasks.

The polling pattern is the key piece here. Video generation takes 30s-3min.
Three bad alternatives we explicitly REJECT:

  1. asyncio.sleep in a long-running task → pins a worker for minutes,
     starves the queue, lost on worker restart.
  2. Celery beat periodic scan → fine for cron, terrible for per-job latency.
  3. Webhooks from provider → ideal but providers don't all support it; we'd
     end up implementing this fallback anyway.

What we do: each poll task self-reschedules via `self.retry(countdown=...)`.
The exception we raise (`_StillGenerating`) is *expected* and doesn't count
as a failure for alerting purposes. `max_retries` provides the wall-clock cap.

Pipeline shape per reel job:

    t_plan_broll  --(chord)-->  [t_submit_one, t_submit_one, ...]
                                     |                |
                                t_poll_one        t_poll_one
                                     |                |
                                t_download_one    t_download_one
                                     \\              /
                                      \\            /
                                  t_collect_broll_plans
                                          |
                                  (back into render in video_tasks.py)
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from celery import chord, group, shared_task
from loguru import logger

from app.config import get_settings
from app.core.celery_app import celery_app  # noqa: F401
from app.schemas.reel import Segment
from app.services.creative_engine import (
    engine as creative,
    cache,
)
from app.services.creative_engine.providers.base import (
    GenerationJob,
    GenStatus,
    ProviderError,
)
from app.services.creative_engine.providers.registry import get_provider

settings = get_settings()


class _StillGenerating(Exception):
    """Sentinel — not a real error, just a 'check back later' signal for self.retry."""


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# 1. Planning task — runs once per segment in parallel
# ---------------------------------------------------------------------------

@shared_task(name="creative.plan", queue="ai", bind=True, max_retries=2)
def t_plan_broll(self, ctx: dict, segment_index: int) -> dict:
    """
    Take one segment, ask scene_extractor + prompts to produce a list of plans.
    Returns the plans + the remaining b-roll budget so downstream submit tasks
    can decline if the kitty's empty.
    """
    req = ctx["payload"]
    segments = [Segment(**s) for s in ctx["segments"]]
    segment = segments[segment_index]

    reel_context = req.get("prompt") or f"viral short, mood={req.get('mood', 'neutral')}"
    try:
        plans = _run(creative.plan_broll_for_segment(
            segment, segment_index,
            reel_context=reel_context,
            aspect_ratio=req["aspect"],
        ))
    except Exception as e:
        logger.exception("planning failed for segment {}", segment_index)
        raise self.retry(exc=e, countdown=5)

    return {
        "segment_index": segment_index,
        "plans": [p.to_dict() for p in plans],
    }


# ---------------------------------------------------------------------------
# 2. Submit task — one per plan; tries providers in fallback order
# ---------------------------------------------------------------------------

@shared_task(name="creative.submit", queue="ai", bind=True, max_retries=3)
def t_submit_one(self, plan_dict: dict, job_id: str, budget_remaining_usd: float) -> dict:
    """Submit a single b-roll generation. Returns the job state (PENDING or SUCCEEDED-from-cache)."""
    plan = creative.BRollPlan.from_dict(plan_dict)

    result = _run(creative.submit_generation(plan, budget_remaining_usd=budget_remaining_usd))
    if result is None:
        logger.warning("submit_generation declined plan {}", plan_dict)
        return {"plan": plan_dict, "gen_job": None, "outcome": "declined"}

    job, new_budget = result
    return {
        "plan": plan_dict,
        "gen_job": job.to_dict(),
        "budget_after": new_budget,
        "submitted_at": time.time(),
    }


# ---------------------------------------------------------------------------
# 3. Poll task — self-rescheduling state machine
# ---------------------------------------------------------------------------

# Cap = max_retries * countdown. 60 * 10s = 10 minutes of wall-clock budget.
POLL_MAX_RETRIES = 60
POLL_COUNTDOWN_S = 10


@shared_task(
    name="creative.poll",
    queue="ai",
    bind=True,
    max_retries=POLL_MAX_RETRIES,
    default_retry_delay=POLL_COUNTDOWN_S,
)
def t_poll_one(self, submit_result: dict, job_id: str) -> dict:
    """
    Poll a generation job until terminal. Self-reschedules every 10s by raising
    _StillGenerating into self.retry — the worker is FREE between checks.
    """
    if submit_result.get("gen_job") is None:
        # Submission was declined upstream; nothing to poll.
        return submit_result

    job = GenerationJob.from_dict(submit_result["gen_job"])

    # Already finalized (e.g. cache hit short-circuit)?
    if job.status == GenStatus.SUCCEEDED:
        return submit_result
    if job.status in (GenStatus.FAILED, GenStatus.CANCELED):
        return submit_result

    try:
        provider = get_provider(job.provider)
        job = _run(provider.poll(job))
    except ProviderError as e:
        logger.warning("poll error {}: {}", job.provider, e)
        job.status = GenStatus.FAILED
        job.error = str(e)
    except Exception as e:
        logger.warning("poll crashed {}: {}", job.provider, e)
        # Network blip — retry the poll itself.
        if self.request.retries < self.max_retries:
            raise self.retry(exc=e, countdown=POLL_COUNTDOWN_S)
        job.status = GenStatus.FAILED
        job.error = f"poll crashed after max retries: {e}"

    submit_result["gen_job"] = job.to_dict()

    if job.status in (GenStatus.PENDING, GenStatus.RUNNING):
        # Not done. Hand worker back, check again later.
        raise self.retry(exc=_StillGenerating(), countdown=POLL_COUNTDOWN_S)

    return submit_result


# ---------------------------------------------------------------------------
# 4. Download task — bring the asset local
# ---------------------------------------------------------------------------

@shared_task(name="creative.download", queue="io", bind=True, max_retries=3)
def t_download_one(self, poll_result: dict, job_id: str) -> dict:
    if poll_result.get("gen_job") is None:
        return poll_result

    job = GenerationJob.from_dict(poll_result["gen_job"])
    if job.status != GenStatus.SUCCEEDED:
        return poll_result

    out_dir = settings.storage_local_dir / "intermediate" / job_id / "ai_broll"
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f"{job.provider}_{job.provider_job_id.replace('/', '_')}.mp4"

    try:
        path = _run(creative.finalize_generation(job, dst))
    except Exception as e:
        logger.exception("download failed for {}", job.provider_job_id)
        raise self.retry(exc=e, countdown=5)

    poll_result["local_path"] = str(path)
    return poll_result


# ---------------------------------------------------------------------------
# 5. Collector — chord callback, reshapes per-plan results into the
#    composite-clip list the renderer needs
# ---------------------------------------------------------------------------

@shared_task(name="creative.collect", queue="io")
def t_collect_broll(download_results: list[dict], ctx: dict) -> dict:
    """
    Collapse parallel download results into a flat list of CompositeClip-ready
    dicts indexed by segment_index. Failures and declined plans are silently
    skipped — the reel still renders, just without that overlay.
    """
    by_segment: dict[int, list[dict]] = {}
    total_cost = 0.0

    for r in download_results:
        plan = creative.BRollPlan.from_dict(r["plan"])
        local = r.get("local_path")
        gen = r.get("gen_job") or {}
        if not local or gen.get("status") != "succeeded":
            continue
        by_segment.setdefault(plan.segment_index, []).append({
            "path": local,
            # The renderer needs CLIP-LOCAL start (not source-absolute).
            "window_start_abs": plan.window_start_abs,
            "window_end_abs":   plan.window_end_abs,
            "rationale": plan.rationale,
            "prompt": plan.visual_prompt.render(),
            "provider": gen.get("provider"),
        })
        total_cost += float(gen.get("cost_usd", 0.0))

    ctx["ai_broll_by_segment"] = by_segment
    ctx["ai_broll_total_cost_usd"] = total_cost
    return ctx


# ---------------------------------------------------------------------------
# 6. Public entry: build the chord that does all of the above
# ---------------------------------------------------------------------------

def build_creative_chord(ctx: dict, job_id: str):
    """
    Construct (but don't execute) a Celery primitive that, when chained after
    `t_analyze`, will plan + generate + download all b-roll in parallel and
    return an enriched ctx ready for `t_render`.

    Called from video_tasks.enqueue_reel_job — kept here to keep the chord
    topology close to the tasks that compose it.
    """
    from celery import chain as celery_chain

    num_segments = len(ctx["segments"])
    req = ctx["payload"]
    budget = float(req.get("ai_broll_budget_usd", 4.0))

    # Plan step per segment → list of plans. We need to know all plans before
    # we know the per-plan budget split, so we plan-all first, then submit-all.
    # In practice the planning calls are short (LLM only), so a group is fine.
    planning = group(t_plan_broll.s(ctx, i) for i in range(num_segments))

    # Once planning is done, we kick off a second chord: submit→poll→download
    # for every plan. The shape is built dynamically by the fan-out task.
    return celery_chain(
        planning,
        _t_fanout_generations.s(ctx, job_id, budget),
    )


@shared_task(name="creative.fanout", queue="io")
def _t_fanout_generations(plan_results: list[dict], ctx: dict, job_id: str, budget: float) -> dict:
    """
    Receives the list of per-segment plans. Builds the submit→poll→download
    chord and waits for it via .get() — safe here because this task itself
    runs on the io queue and only waits, never holds CPU.

    NOTE: chord-of-chains is a Celery anti-pattern across worker restarts.
    For production hardening, switch this to apply_async + a callback rather
    than .get(). For MVP latency, .get() is fine.
    """
    from celery.result import GroupResult

    # Flatten and budget-split.
    all_plans: list[dict] = []
    for r in plan_results:
        all_plans.extend(r.get("plans", []))

    if not all_plans:
        ctx["ai_broll_by_segment"] = {}
        ctx["ai_broll_total_cost_usd"] = 0.0
        return ctx

    per_plan_budget = budget / max(1, len(all_plans))

    pipelines = [
        (t_submit_one.s(plan, job_id, per_plan_budget)
         | t_poll_one.s(job_id)
         | t_download_one.s(job_id))
        for plan in all_plans
    ]
    g = group(pipelines).apply_async()
    download_results = g.get(disable_sync_subtasks=False)
    return t_collect_broll(download_results, ctx)
