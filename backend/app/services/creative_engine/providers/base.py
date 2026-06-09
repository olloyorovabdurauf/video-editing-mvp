"""
Provider abstraction for video-generation APIs.

Why an ABC instead of duck typing?
- Runway, Pika, Higgsfield, Luma all have *similar* but *not identical* APIs.
- We want the orchestrator (engine.py) and the Celery tasks to be 100%
  provider-agnostic. Adding a new provider should be one file + one registry
  entry, never a change in the pipeline.
- A normalized `GenerationJob` is what crosses task boundaries and what we
  serialize into Redis for the polling state machine.
"""
from __future__ import annotations

import abc
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class GenStatus(str, Enum):
    PENDING = "pending"      # submitted, not yet processing
    RUNNING = "running"      # actively generating
    SUCCEEDED = "succeeded"  # video_url is set
    FAILED = "failed"        # error is set
    CANCELED = "canceled"


@dataclass
class GenerationRequest:
    """Provider-agnostic request. Providers translate to their own schema."""

    prompt: str
    aspect_ratio: str = "9:16"          # "9:16" | "16:9" | "1:1"
    duration_s: float = 5.0
    negative_prompt: str | None = None
    seed: int | None = None
    motion_strength: float = 0.5        # 0..1, normalized
    reference_image_url: str | None = None
    # Free-form for provider-specific knobs the caller wants to pass through.
    # Use sparingly — anything you set here breaks portability.
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationJob:
    """The normalized job handle that flows through Celery."""

    provider: str
    provider_job_id: str
    status: GenStatus = GenStatus.PENDING
    video_url: str | None = None
    thumbnail_url: str | None = None
    error: str | None = None
    cost_usd: float = 0.0
    progress: float = 0.0                # 0..1 if provider exposes it
    request: GenerationRequest | None = None
    # Wall-clock metrics — useful for SLO dashboards.
    submitted_at: float = 0.0
    finished_at: float | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "GenerationJob":
        d = dict(d)
        d["status"] = GenStatus(d.get("status", "pending"))
        if d.get("request") and isinstance(d["request"], dict):
            d["request"] = GenerationRequest(**d["request"])
        return cls(**d)


class ProviderError(RuntimeError):
    """Raised on hard provider failure. Retryable errors should be inside it."""

    def __init__(self, provider: str, message: str, *, retryable: bool = False):
        super().__init__(f"[{provider}] {message}")
        self.provider = provider
        self.retryable = retryable


class VideoGenProvider(abc.ABC):
    """Each concrete provider implements exactly these four methods."""

    name: str
    # Approximate cost per generation; used by the registry's budget guard.
    cost_usd_per_gen: float = 0.0
    # Supported aspect ratios — used for routing.
    supported_aspects: tuple[str, ...] = ("9:16", "16:9", "1:1")
    # Max duration in seconds.
    max_duration_s: float = 10.0

    @abc.abstractmethod
    async def submit(self, req: GenerationRequest) -> GenerationJob: ...

    @abc.abstractmethod
    async def poll(self, job: GenerationJob) -> GenerationJob: ...

    @abc.abstractmethod
    async def cancel(self, job: GenerationJob) -> None: ...

    async def download(self, job: GenerationJob, dst_path) -> "Path":  # type: ignore[name-defined]
        """Default: streaming HTTP GET. Override if provider needs auth on the asset URL."""
        from pathlib import Path
        import httpx

        if not job.video_url:
            raise ProviderError(self.name, "download called before video_url is set")
        dst = Path(dst_path)
        async with httpx.AsyncClient(timeout=120, follow_redirects=True) as c:
            async with c.stream("GET", job.video_url) as resp:
                resp.raise_for_status()
                with dst.open("wb") as f:
                    async for chunk in resp.aiter_bytes(chunk_size=1 << 16):
                        f.write(chunk)
        return dst
