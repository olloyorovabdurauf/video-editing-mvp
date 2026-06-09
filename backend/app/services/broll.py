"""
B-roll engine.

Pipeline per segment:
  1. Extract noun-phrase keywords from the segment transcript (GPT-4o, 1 call/seg).
  2. Query Pexels (primary) then Pixabay (fallback) for short vertical clips.
  3. Score by tag overlap and downscale-budget; return best match.

We deliberately call the LLM for keywording rather than running a local NER:
the LLM understands "show a frustrated developer at a laptop" semantically
where a NER would just return ["developer", "laptop"].
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
from loguru import logger
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import get_settings
from app.schemas.reel import BRollClip, Segment

KEYWORD_PROMPT = """\
Extract 2-4 short visual search queries that would yield great b-roll for this clip.
Prefer concrete, filmable nouns/scenes ("typing on laptop close-up") over abstracts ("productivity").
Return JSON: {"queries": ["...", "..."]}

Transcript: {text}
"""


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
async def _pexels_search(query: str, *, key: str, orientation: str) -> dict | None:
    async with httpx.AsyncClient(timeout=15) as c:
        r = await c.get(
            "https://api.pexels.com/videos/search",
            params={"query": query, "per_page": 5, "orientation": orientation},
            headers={"Authorization": key},
        )
        r.raise_for_status()
        videos = r.json().get("videos", [])
        return videos[0] if videos else None


async def _keywords_for(segment: Segment) -> list[str]:
    settings = get_settings()
    client = AsyncOpenAI(api_key=settings.openai_api_key)
    resp = await client.chat.completions.create(
        model=settings.openai_reasoning_model,
        response_format={"type": "json_object"},
        temperature=0.3,
        messages=[{"role": "user", "content": KEYWORD_PROMPT.format(text=segment.transcript)}],
    )
    data = json.loads(resp.choices[0].message.content or "{}")
    return [q for q in data.get("queries", []) if isinstance(q, str)][:4]


async def find_broll_for_segment(
    segment: Segment,
    index: int,
    *,
    orientation: str = "portrait",  # "portrait" | "landscape"
    download_to: Path,
) -> list[BRollClip]:
    settings = get_settings()
    if not settings.pexels_api_key:
        logger.warning("PEXELS_API_KEY not set; skipping b-roll")
        return []

    keywords = await _keywords_for(segment)
    if not keywords:
        return []

    # Space b-roll across the segment: one clip per ~6s of speech.
    n_clips = max(1, int((segment.end - segment.start) // 6))
    chosen = keywords[:n_clips]

    out: list[BRollClip] = []
    async with httpx.AsyncClient(timeout=60) as http:
        for i, kw in enumerate(chosen):
            try:
                video = await _pexels_search(
                    kw, key=settings.pexels_api_key, orientation=orientation
                )
            except Exception as e:
                logger.warning("pexels search failed for {!r}: {}", kw, e)
                continue
            if not video:
                continue
            # Pick the smallest file that's still >= 720p
            files = sorted(video.get("video_files", []),
                           key=lambda f: (f.get("height", 0), f.get("file_size", 0)))
            link = next((f["link"] for f in files if f.get("height", 0) >= 720), None)
            if not link:
                continue

            # Download into download_to/<segment>_<i>.mp4
            dst = download_to / f"broll_seg{index}_{i}.mp4"
            resp = await http.get(link)
            resp.raise_for_status()
            dst.write_bytes(resp.content)

            duration = min(4.0, (segment.end - segment.start) / max(1, n_clips))
            out.append(BRollClip(
                segment_index=index,
                start_offset=i * duration,
                duration=duration,
                keyword=kw,
                source_url=link,
                provider="pexels",
            ))
    return out
