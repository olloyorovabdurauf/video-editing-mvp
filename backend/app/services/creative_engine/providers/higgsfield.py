"""
Higgsfield provider.

Higgsfield's official public REST API is evolving — when you sign up, the
contract you receive may differ. The two operations we need are stable
across every video-gen provider on the planet:

    submit(prompt, opts) -> job_id
    poll(job_id)         -> {status, progress, asset_url}

This module is shaped around that contract. Adjust `_endpoints` to match the
exact URLs you're issued and the rest of the pipeline is unchanged.
"""
from __future__ import annotations

import time

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.config import get_settings
from app.services.creative_engine.providers.base import (
    GenerationJob,
    GenerationRequest,
    GenStatus,
    ProviderError,
    VideoGenProvider,
)


class HiggsfieldProvider(VideoGenProvider):
    name = "higgsfield"
    cost_usd_per_gen = 0.40
    supported_aspects = ("9:16", "16:9", "1:1")
    max_duration_s = 8.0

    _endpoints = {
        "submit": "https://api.higgsfield.ai/v1/videos",
        "poll":   "https://api.higgsfield.ai/v1/videos/{job_id}",
        "cancel": "https://api.higgsfield.ai/v1/videos/{job_id}",
    }

    def __init__(self) -> None:
        self._key = get_settings().higgsfield_api_key
        if not self._key:
            raise ProviderError(self.name, "HIGGSFIELD_API_KEY not set")

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._key}",
            "Content-Type": "application/json",
        }

    @retry(
        retry=retry_if_exception_type(httpx.HTTPError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
    )
    async def submit(self, req: GenerationRequest) -> GenerationJob:
        body = {
            "prompt": req.prompt,
            "negative_prompt": req.negative_prompt or "",
            "aspect_ratio": req.aspect_ratio,
            "duration_seconds": min(req.duration_s, self.max_duration_s),
            "motion_intensity": req.motion_strength,
            **({"seed": req.seed} if req.seed is not None else {}),
            **({"reference_image_url": req.reference_image_url} if req.reference_image_url else {}),
            **req.extras,
        }
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.post(self._endpoints["submit"], headers=self._headers(), json=body)
            if r.status_code >= 500:
                raise httpx.HTTPError(f"higgsfield 5xx: {r.text}")
            if r.status_code >= 400:
                raise ProviderError(self.name, f"submit failed {r.status_code}: {r.text}")
            data = r.json()

        return GenerationJob(
            provider=self.name,
            provider_job_id=data.get("id") or data.get("job_id"),
            status=GenStatus.PENDING,
            request=req,
            submitted_at=time.time(),
            cost_usd=self.cost_usd_per_gen,
        )

    async def poll(self, job: GenerationJob) -> GenerationJob:
        url = self._endpoints["poll"].format(job_id=job.provider_job_id)
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.get(url, headers=self._headers())
            if r.status_code == 404:
                job.status = GenStatus.FAILED
                job.error = "task not found"
                return job
            r.raise_for_status()
            data = r.json()

        # Normalize to our GenStatus enum.
        upstream = (data.get("status") or "pending").lower()
        if upstream in ("pending", "queued"):
            job.status = GenStatus.PENDING
        elif upstream in ("running", "processing", "in_progress"):
            job.status = GenStatus.RUNNING
            job.progress = float(data.get("progress", 0.0))
        elif upstream in ("succeeded", "completed", "done"):
            job.status = GenStatus.SUCCEEDED
            job.video_url = data.get("video_url") or data.get("output_url")
            job.thumbnail_url = data.get("thumbnail_url")
            job.finished_at = time.time()
            if not job.video_url:
                job.status = GenStatus.FAILED
                job.error = "succeeded but no video_url"
        elif upstream in ("canceled", "cancelled"):
            job.status = GenStatus.CANCELED
            job.finished_at = time.time()
        else:
            job.status = GenStatus.FAILED
            job.error = data.get("error", f"unknown status {upstream}")
            job.finished_at = time.time()
        return job

    async def cancel(self, job: GenerationJob) -> None:
        url = self._endpoints["cancel"].format(job_id=job.provider_job_id)
        async with httpx.AsyncClient(timeout=15) as c:
            await c.delete(url, headers=self._headers())
