"""
Runway Gen-3 provider.

NOTE: Runway's public API surface has shifted a few times. This module follows
the v1 task pattern (image_to_video → tasks/{id}). When you onboard, verify
against current docs at https://docs.dev.runwayml.com/ and adjust the two
constants below — the rest of the pipeline doesn't care.
"""
from __future__ import annotations

import time

import httpx
from loguru import logger
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.config import get_settings
from app.services.creative_engine.providers.base import (
    GenerationJob,
    GenerationRequest,
    GenStatus,
    ProviderError,
    VideoGenProvider,
)

API_BASE = "https://api.dev.runwayml.com/v1"
API_VERSION = "2024-11-06"


class RunwayProvider(VideoGenProvider):
    name = "runway"
    cost_usd_per_gen = 0.50
    supported_aspects = ("9:16", "16:9", "1:1")
    max_duration_s = 10.0

    def __init__(self) -> None:
        self._key = get_settings().runway_api_key
        if not self._key:
            raise ProviderError(self.name, "RUNWAY_API_KEY not set")

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._key}",
            "X-Runway-Version": API_VERSION,
            "Content-Type": "application/json",
        }

    @retry(
        retry=retry_if_exception_type(httpx.HTTPError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
    )
    async def submit(self, req: GenerationRequest) -> GenerationJob:
        if req.duration_s > self.max_duration_s:
            raise ProviderError(self.name, f"duration {req.duration_s}s > max {self.max_duration_s}s")

        # Runway requires a reference image for Gen-3 image_to_video. If the
        # caller didn't supply one, we'd need to generate it via a T2I provider
        # first — for now, require it explicitly.
        if not req.reference_image_url:
            raise ProviderError(
                self.name,
                "Runway Gen-3 image_to_video requires reference_image_url; "
                "generate a still first or use a different provider.",
            )

        body = {
            "promptImage": req.reference_image_url,
            "promptText": req.prompt,
            "model": "gen3a_turbo",
            "duration": int(round(req.duration_s)),
            "ratio": {"9:16": "768:1280", "16:9": "1280:768", "1:1": "960:960"}[req.aspect_ratio],
            **({"seed": req.seed} if req.seed is not None else {}),
        }
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.post(f"{API_BASE}/image_to_video", headers=self._headers(), json=body)
            if r.status_code >= 500:
                raise httpx.HTTPError(f"runway 5xx: {r.text}")
            if r.status_code >= 400:
                raise ProviderError(self.name, f"submit failed {r.status_code}: {r.text}")
            data = r.json()

        return GenerationJob(
            provider=self.name,
            provider_job_id=data["id"],
            status=GenStatus.PENDING,
            request=req,
            submitted_at=time.time(),
            cost_usd=self.cost_usd_per_gen,
        )

    async def poll(self, job: GenerationJob) -> GenerationJob:
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.get(f"{API_BASE}/tasks/{job.provider_job_id}", headers=self._headers())
            if r.status_code == 404:
                job.status = GenStatus.FAILED
                job.error = "task not found"
                return job
            r.raise_for_status()
            data = r.json()

        # Runway status: "PENDING" | "RUNNING" | "SUCCEEDED" | "FAILED" | "CANCELED" | "THROTTLED"
        upstream = data.get("status", "PENDING").upper()
        if upstream in ("PENDING", "THROTTLED"):
            job.status = GenStatus.PENDING
        elif upstream == "RUNNING":
            job.status = GenStatus.RUNNING
            job.progress = float(data.get("progress", 0.0))
        elif upstream == "SUCCEEDED":
            job.status = GenStatus.SUCCEEDED
            outputs = data.get("output", [])
            job.video_url = outputs[0] if outputs else None
            job.finished_at = time.time()
            if not job.video_url:
                job.status = GenStatus.FAILED
                job.error = "succeeded but no output url"
        elif upstream == "CANCELED":
            job.status = GenStatus.CANCELED
            job.finished_at = time.time()
        else:  # FAILED + unknown
            job.status = GenStatus.FAILED
            job.error = data.get("failure", "unknown failure")
            job.finished_at = time.time()
        return job

    async def cancel(self, job: GenerationJob) -> None:
        async with httpx.AsyncClient(timeout=15) as c:
            r = await c.delete(f"{API_BASE}/tasks/{job.provider_job_id}", headers=self._headers())
            if r.status_code >= 400 and r.status_code != 404:
                logger.warning("runway cancel returned {}: {}", r.status_code, r.text)
