"""
Creative engine orchestrator.

Public surface (called from the Celery tasks):

    plan_broll_for_segment(segment, reel_context, aspect, budget_remaining_usd)
        → list[BRollPlan]   # one per insertion window, with provider+prompt

    submit_generation(plan, budget_remaining_usd)
        → GenerationJob     # try providers in fallback order, return the
                              accepted job (or raise if all fail / budget)

    finalize_generation(job, local_dst)
        → Path               # download (or cache hit) → local file

The Celery tasks call these from sync code via asyncio.run; the engine
itself is fully async so a single submit_generation can do prompt-compile
+ provider-call concurrently if needed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from loguru import logger

from app.schemas.reel import Segment
from app.services.creative_engine import cache
from app.services.creative_engine.prompts import VisualPrompt, compile_visual_prompt
from app.services.creative_engine.providers.base import (
    GenerationJob,
    GenerationRequest,
    GenStatus,
    ProviderError,
)
from app.services.creative_engine.providers.registry import (
    get_provider,
    iter_candidates_for,
)
from app.services.creative_engine.scene_extractor import InsertionWindow, extract_scenes


# ---------------------------------------------------------------------------
# Planning: segment → list of {window, prompt} pairs
# ---------------------------------------------------------------------------

@dataclass
class BRollPlan:
    """One concrete b-roll insertion plan, ready to submit to a provider."""

    segment_index: int                  # which reel segment this belongs to
    window_start_abs: float             # absolute timestamp in source video
    window_end_abs: float
    rationale: str                      # why scene_extractor chose this spot
    visual_prompt: VisualPrompt
    aspect_ratio: str

    def to_request(self) -> GenerationRequest:
        return GenerationRequest(
            prompt=self.visual_prompt.render(),
            negative_prompt=self.visual_prompt.negative,
            aspect_ratio=self.aspect_ratio,
            duration_s=max(2.5, min(self.window_end_abs - self.window_start_abs, 5.0)),
            motion_strength=0.4,    # b-roll wants subtle motion, not chaos
        )

    def to_dict(self) -> dict:
        return {
            "segment_index": self.segment_index,
            "window_start_abs": self.window_start_abs,
            "window_end_abs": self.window_end_abs,
            "rationale": self.rationale,
            "visual_prompt": asdict(self.visual_prompt),
            "aspect_ratio": self.aspect_ratio,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BRollPlan":
        return cls(
            segment_index=d["segment_index"],
            window_start_abs=d["window_start_abs"],
            window_end_abs=d["window_end_abs"],
            rationale=d["rationale"],
            visual_prompt=VisualPrompt(**d["visual_prompt"]),
            aspect_ratio=d["aspect_ratio"],
        )


async def plan_broll_for_segment(
    segment: Segment,
    segment_index: int,
    *,
    reel_context: str,
    aspect_ratio: str,
    max_windows: int = 3,
) -> list[BRollPlan]:
    """LLM-driven planning. Returns 0..max_windows plans, ranked by priority."""
    windows: list[InsertionWindow] = await extract_scenes(segment, max_windows=max_windows)
    if not windows:
        return []

    plans: list[BRollPlan] = []
    for w in windows:
        try:
            vp = await compile_visual_prompt(
                w.transcript or segment.transcript,
                duration=w.end - w.start,
                reel_context=reel_context,
            )
        except Exception as e:
            logger.warning("prompt compile failed for window {}: {}", w, e)
            continue
        plans.append(BRollPlan(
            segment_index=segment_index,
            window_start_abs=w.start,
            window_end_abs=w.end,
            rationale=w.rationale,
            visual_prompt=vp,
            aspect_ratio=aspect_ratio,
        ))
    return plans


# ---------------------------------------------------------------------------
# Submission: plan → submitted GenerationJob (with fallback chain)
# ---------------------------------------------------------------------------

async def submit_generation(
    plan: BRollPlan,
    *,
    budget_remaining_usd: float,
) -> tuple[GenerationJob, float] | None:
    """
    Try providers in priority order until one accepts the job.
    Returns (job, new_budget_remaining) or None if all providers declined.

    Cache-hit shortcut: if we've already generated this exact prompt with
    *any* provider, we synthesize a SUCCEEDED job pointing at the cached path
    so the polling task can skip straight to compositing.
    """
    req = plan.to_request()

    # Cheap path: someone already paid for this exact generation.
    for c in iter_candidates_for(req, budget_remaining_usd):
        cached = cache.lookup(c.name, req)
        if cached:
            logger.info("cache hit ({}): {}", c.name, cached.name)
            return GenerationJob(
                provider=c.name,
                provider_job_id=f"cached:{cached.name}",
                status=GenStatus.SUCCEEDED,
                video_url=f"file://{cached.resolve()}",
                cost_usd=0.0,
                request=req,
            ), budget_remaining_usd

    # Real path: try providers in order.
    last_err: Exception | None = None
    for provider in iter_candidates_for(req, budget_remaining_usd):
        try:
            job = await provider.submit(req)
            logger.info("submitted to {} job_id={}", provider.name, job.provider_job_id)
            return job, budget_remaining_usd - provider.cost_usd_per_gen
        except ProviderError as e:
            if not e.retryable:
                logger.warning("provider {} hard-failed: {}; trying next", provider.name, e)
                last_err = e
                continue
            raise
        except Exception as e:
            logger.warning("provider {} crashed: {}; trying next", provider.name, e)
            last_err = e
            continue

    logger.error("all providers declined or failed; last_err={}", last_err)
    return None


# ---------------------------------------------------------------------------
# Finalization: succeeded job → local file on disk
# ---------------------------------------------------------------------------

async def finalize_generation(job: GenerationJob, local_dst: Path) -> Path:
    """Download the asset (or copy from cache). Stores in cache for next time."""
    if not job.video_url:
        raise ProviderError(job.provider, "finalize called before video_url is set")

    # Cache pre-hit: video_url points to a local file (synthesized in submit).
    if job.video_url.startswith("file://"):
        return Path(job.video_url.removeprefix("file://"))

    provider = get_provider(job.provider)
    path = await provider.download(job, local_dst)
    cache.store(job.provider, job, path)
    return path
