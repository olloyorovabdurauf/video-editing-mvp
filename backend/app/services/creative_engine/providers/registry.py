"""
Provider registry + fallback chain + budget guard.

Why a registry instead of importing providers directly?
- The orchestrator only knows about *capabilities* (aspect, duration, budget).
- Providers are activated by config (env vars). If RUNWAY_API_KEY isn't set,
  Runway silently drops out of the chain — no code change needed.
- The fallback chain is data, not control flow: easy to reorder, A/B test,
  or override per request.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from loguru import logger

from app.config import get_settings
from app.services.creative_engine.providers.base import (
    GenerationRequest,
    ProviderError,
    VideoGenProvider,
)


@dataclass
class ProviderCandidate:
    provider: VideoGenProvider
    priority: int               # lower = tried first


def _build_chain() -> list[ProviderCandidate]:
    """Construct providers from settings. Each is optional — missing keys skip it."""
    chain: list[ProviderCandidate] = []
    settings = get_settings()

    if settings.runway_api_key:
        try:
            from app.services.creative_engine.providers.runway import RunwayProvider
            chain.append(ProviderCandidate(RunwayProvider(), priority=20))
        except ProviderError as e:
            logger.warning("runway disabled: {}", e)

    if settings.higgsfield_api_key:
        try:
            from app.services.creative_engine.providers.higgsfield import HiggsfieldProvider
            chain.append(ProviderCandidate(HiggsfieldProvider(), priority=10))  # cheaper → preferred
        except ProviderError as e:
            logger.warning("higgsfield disabled: {}", e)

    chain.sort(key=lambda c: c.priority)
    if not chain:
        logger.warning("no video-gen providers configured; falling back to stock-only b-roll")
    return chain


_chain_cache: list[ProviderCandidate] | None = None


def get_chain() -> list[ProviderCandidate]:
    global _chain_cache
    if _chain_cache is None:
        _chain_cache = _build_chain()
    return _chain_cache


def get_provider(name: str) -> VideoGenProvider:
    """Look up a single provider by name. Used by the polling task."""
    for c in get_chain():
        if c.provider.name == name:
            return c.provider
    raise ProviderError(name, f"provider {name!r} not registered or not configured")


def iter_candidates_for(req: GenerationRequest, budget_remaining_usd: float) -> Iterator[VideoGenProvider]:
    """
    Yield providers that can serve this request and fit the remaining budget.

    Used by `submit_with_fallback`: try each in order until one succeeds.
    """
    for c in get_chain():
        p = c.provider
        if req.aspect_ratio not in p.supported_aspects:
            logger.debug("skip {}: aspect {} unsupported", p.name, req.aspect_ratio)
            continue
        if req.duration_s > p.max_duration_s:
            logger.debug("skip {}: duration {} > max {}", p.name, req.duration_s, p.max_duration_s)
            continue
        if p.cost_usd_per_gen > budget_remaining_usd:
            logger.debug("skip {}: cost {} > budget {}", p.name, p.cost_usd_per_gen, budget_remaining_usd)
            continue
        yield p
