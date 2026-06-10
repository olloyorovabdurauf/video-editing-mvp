"""
Kling 3.0 provider (Kuaishou / KlingAI).

Why Kling leads our chain:
- Native text-to-video — no reference image required (Runway Gen-3 needs one,
  which forces an extra T2I hop). Our b-roll prompts are text-first, so this
  removes a whole failure mode.
- Strong cinematic motion quality at 9:16 — the aspect we render most.
- Cost-competitive in std mode.

Auth model
----------
Kling does NOT use a static bearer token. You hold an AccessKey + SecretKey
pair and mint a short-lived JWT (HS256) per request:

    header  {"alg": "HS256", "typ": "JWT"}
    payload {"iss": <access_key>, "exp": now+30min, "nbf": now-5s}

NOTE: model identifiers and the API host occasionally shift between Kling
releases. `KLING_MODEL_NAME` and `KLING_API_BASE` are config so an ops change
never needs a code deploy. Verify against https://app.klingai.com/global/dev
when onboarding.
"""
from __future__ import annotations

import time

import httpx
from jose import jwt
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

# Token lifetime. Kling rejects exp > 30 min out; we mint per call, so short.
_TOKEN_TTL_S = 1800
_TOKEN_NBF_SKEW_S = 5


def _mint_token(access_key: str, secret_key: str, *, now: float | None = None) -> str:
    """Build the short-lived HS256 JWT Kling expects. Pure — unit-testable."""
    now = time.time() if now is None else now
    payload = {
        "iss": access_key,
        "exp": int(now) + _TOKEN_TTL_S,
        "nbf": int(now) - _TOKEN_NBF_SKEW_S,
    }
    return jwt.encode(payload, secret_key, algorithm="HS256", headers={"typ": "JWT"})


def _map_status(data: dict, job: GenerationJob) -> GenerationJob:
    """
    Translate Kling task payload → normalized GenerationJob. Pure — unit-testable.

    Kling task_status: "submitted" | "processing" | "succeed" | "failed"
    Result shape: data["task_result"]["videos"][0]["url"]
    """
    upstream = (data.get("task_status") or "submitted").lower()
    if upstream == "submitted":
        job.status = GenStatus.PENDING
    elif upstream == "processing":
        job.status = GenStatus.RUNNING
    elif upstream == "succeed":
        videos = (data.get("task_result") or {}).get("videos") or []
        job.video_url = videos[0].get("url") if videos else None
        job.finished_at = time.time()
        if job.video_url:
            job.status = GenStatus.SUCCEEDED
        else:
            job.status = GenStatus.FAILED
            job.error = "succeed but no video url in task_result"
    else:  # "failed" + anything unrecognized
        job.status = GenStatus.FAILED
        job.error = data.get("task_status_msg") or f"kling status {upstream!r}"
        job.finished_at = time.time()
    return job


class KlingProvider(VideoGenProvider):
    name = "kling"
    # std mode 5s ≈ $0.35; pro mode roughly doubles it. Registry budget guard
    # uses this, so keep it aligned with the mode you configure.
    cost_usd_per_gen = 0.35
    supported_aspects = ("9:16", "16:9", "1:1")
    max_duration_s = 10.0

    def __init__(self) -> None:
        s = get_settings()
        if not (s.kling_access_key and s.kling_secret_key):
            raise ProviderError(self.name, "KLING_ACCESS_KEY / KLING_SECRET_KEY not set")
        self._access_key = s.kling_access_key
        self._secret_key = s.kling_secret_key
        self._base = s.kling_api_base.rstrip("/")
        self._model = s.kling_model_name
        self._mode = s.kling_mode

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {_mint_token(self._access_key, self._secret_key)}",
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

        # Kling accepts only 5 or 10 second durations — snap to nearest.
        duration = "10" if req.duration_s > 7.5 else "5"

        body: dict = {
            "model_name": self._model,
            "prompt": req.prompt[:2500],            # Kling caps prompt length
            "mode": self._mode,                      # "std" | "pro"
            "duration": duration,
            "aspect_ratio": req.aspect_ratio,        # Kling takes "9:16" verbatim
            "cfg_scale": 0.5,
        }
        if req.negative_prompt:
            body["negative_prompt"] = req.negative_prompt[:2500]
        # If a reference image is supplied we switch to image2video — same
        # task shape, different endpoint.
        endpoint = "/v1/videos/text2video"
        if req.reference_image_url:
            body["image"] = req.reference_image_url
            endpoint = "/v1/videos/image2video"

        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.post(f"{self._base}{endpoint}", headers=self._headers(), json=body)
            if r.status_code >= 500:
                raise httpx.HTTPError(f"kling 5xx: {r.text}")
            if r.status_code >= 400:
                raise ProviderError(self.name, f"submit failed {r.status_code}: {r.text}")
            payload = r.json()

        # Kling envelope: {"code": 0, "message": "ok", "data": {"task_id": ...}}
        if payload.get("code") != 0:
            raise ProviderError(self.name, f"submit rejected: {payload.get('message')}")
        task_id = payload["data"]["task_id"]
        logger.info("kling submit ok: task {} ({}s {} {})", task_id, duration, self._mode, req.aspect_ratio)

        return GenerationJob(
            provider=self.name,
            provider_job_id=task_id,
            status=GenStatus.PENDING,
            request=req,
            submitted_at=time.time(),
            cost_usd=self.cost_usd_per_gen if self._mode == "std" else self.cost_usd_per_gen * 2,
        )

    async def poll(self, job: GenerationJob) -> GenerationJob:
        # Poll endpoint mirrors the submit endpoint family; text2video covers both
        # because Kling task ids are globally unique per account.
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.get(
                f"{self._base}/v1/videos/text2video/{job.provider_job_id}",
                headers=self._headers(),
            )
            if r.status_code == 404:
                job.status = GenStatus.FAILED
                job.error = "task not found"
                return job
            r.raise_for_status()
            payload = r.json()

        if payload.get("code") != 0:
            job.status = GenStatus.FAILED
            job.error = f"poll rejected: {payload.get('message')}"
            return job
        return _map_status(payload.get("data") or {}, job)

    async def cancel(self, job: GenerationJob) -> None:
        # Kling has no public cancel endpoint as of this writing — we just stop
        # polling. The budget was committed at submit time, which is honest:
        # the generation does run to completion on their side.
        logger.info("kling cancel: no-op (provider lacks cancel API); job {}", job.provider_job_id)
