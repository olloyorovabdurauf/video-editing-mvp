"""
Per-clip social metadata: a viral title, a ready-to-post caption, and hashtags
for each short clip we cut from the source video.

One batched LLM call for all clips (cheap model — this is light copywriting, not
reasoning). Best-effort: any failure falls back to a deterministic title/caption
derived from the clip's own transcript + hook reason, so a clip is never left
without usable text. Output is in the SOURCE video's language (the transcript
carries it), so a Russian podcast gets Russian captions.

`_call_llm` is the only network seam, so tests can stub it without OpenAI.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from loguru import logger
from openai import AsyncOpenAI

from app.config import get_settings


@dataclass
class ClipMeta:
    title: str
    caption: str
    hashtags: list[str] = field(default_factory=list)


@dataclass
class ClipInput:
    transcript: str
    reason: str


SYSTEM_PROMPT = """\
You are a viral short-form social copywriter. For each clip you receive its
transcript and why it hooks. For EACH clip return:
  - "title": a punchy, scroll-stopping title (max ~60 chars, no quotes)
  - "caption": a ready-to-post caption — a strong first line, 1-2 short lines of
    value, then a clear CTA
  - "hashtags": 4-6 relevant hashtags (no # symbol needed)

Write everything in the SAME LANGUAGE as that clip's transcript (do not
translate). Return STRICT JSON only:
{"clips": [{"title": "...", "caption": "...", "hashtags": ["..."]}, ...]}
The clips array MUST be in the same order and length as the input.
"""


def _make_client() -> AsyncOpenAI:
    return AsyncOpenAI(api_key=get_settings().openai_api_key)


async def _call_llm(messages: list[dict], *, model: str, max_tokens: int) -> str:
    client = _make_client()
    resp = await client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        temperature=0.7,
        max_tokens=max_tokens,
        messages=messages,
    )
    return (resp.choices[0].message.content or "").strip()


def _fallback(clip: ClipInput) -> ClipMeta:
    """Deterministic metadata from the clip itself — never leave a clip blank."""
    words = clip.transcript.split()
    title = " ".join(words[:9]).strip().rstrip(",.;:") or "Untitled clip"
    caption = (clip.reason or clip.transcript[:140]).strip()
    return ClipMeta(title=title[:80], caption=caption[:300], hashtags=[])


def _user_payload(clips: list[ClipInput]) -> str:
    lines = []
    for i, c in enumerate(clips):
        lines.append(f"Clip {i + 1}:\n  why_it_hooks: {c.reason}\n  transcript: {c.transcript[:600]}")
    return "\n\n".join(lines)


async def generate_for_clips(clips: list[ClipInput]) -> list[ClipMeta]:
    """Title + caption + hashtags per clip. Always returns one ClipMeta per input."""
    if not clips:
        return []
    settings = get_settings()
    fallbacks = [_fallback(c) for c in clips]

    try:
        content = await _call_llm(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _user_payload(clips)},
            ],
            model=settings.openai_analysis_model,   # cheap model — light copywriting
            max_tokens=1200,
        )
        raw = json.loads(content).get("clips", [])
    except Exception as e:                            # network / parse error
        logger.warning("clip metadata generation failed, using fallbacks: {}", e)
        return fallbacks

    out: list[ClipMeta] = []
    for i, fb in enumerate(fallbacks):
        item = raw[i] if i < len(raw) and isinstance(raw[i], dict) else {}
        title = str(item.get("title", "")).strip() or fb.title
        caption = str(item.get("caption", "")).strip() or fb.caption
        hashtags = [str(h).lstrip("#") for h in (item.get("hashtags") or []) if str(h).strip()][:8]
        out.append(ClipMeta(title=title[:120], caption=caption[:500], hashtags=hashtags))
    return out
