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

{lang_directive}

Return STRICT JSON only:
{{"clips": [{{"title": "...", "caption": "...", "hashtags": ["..."]}}, ...]}}
The clips array MUST be in the same order and length as the input.
"""

# ISO-639-1 → human name for the language lock.
_LANG_NAMES = {
    "uz": "Uzbek (Latin alphabet)", "en": "English", "ru": "Russian",
    "kk": "Kazakh", "ar": "Arabic", "tr": "Turkish", "es": "Spanish",
    "fr": "French", "de": "German", "pt": "Portuguese", "hi": "Hindi",
}


def _lang_directive(language: str | None) -> str:
    if not language:
        return ("Write everything in the SAME LANGUAGE as the clip's transcript. "
                "Do NOT translate or switch languages.")
    name = _LANG_NAMES.get(language, language)
    extra = ""
    if language == "uz":
        extra = (" Use natural Uzbek in the LATIN alphabet — NOT Cyrillic, NOT Arabic "
                 "script, and NOT Kazakh/Turkish. Correct: "
                 "\"Bugungi kunda sun'iy intellekt juda tez rivojlanmoqda\".")
    return (f"Write EVERYTHING (title, caption, hashtags) ONLY in {name}. "
            f"NEVER translate. NEVER switch to another language.{extra}")


def _has_range(text: str, lo: int, hi: int) -> bool:
    return any(lo <= ord(c) <= hi for c in text)


def _wrong_language(text: str, language: str | None) -> bool:
    """Cheap script check that catches the common failures (uz→Cyrillic/Arabic)."""
    if not text or not language:
        return False
    cyrillic = _has_range(text, 0x0400, 0x04FF)
    arabic = _has_range(text, 0x0600, 0x06FF)
    if language == "uz":
        return cyrillic or arabic          # Uzbek must be Latin
    if language in ("ru", "kk"):
        return not cyrillic                 # must be Cyrillic
    if language == "ar":
        return not arabic
    if language == "en":
        return cyrillic or arabic
    return False


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


def _assemble(raw: list, fallbacks: list[ClipMeta]) -> list[ClipMeta]:
    out: list[ClipMeta] = []
    for i, fb in enumerate(fallbacks):
        item = raw[i] if i < len(raw) and isinstance(raw[i], dict) else {}
        title = str(item.get("title", "")).strip() or fb.title
        caption = str(item.get("caption", "")).strip() or fb.caption
        hashtags = [str(h).lstrip("#") for h in (item.get("hashtags") or []) if str(h).strip()][:8]
        out.append(ClipMeta(title=title[:120], caption=caption[:500], hashtags=hashtags))
    return out


async def generate_for_clips(clips: list[ClipInput], *, language: str | None = None) -> list[ClipMeta]:
    """
    Title + caption + hashtags per clip, LOCKED to `language` (the source video's
    language). Always returns one ClipMeta per input. If the model answers in the
    wrong script (e.g. Cyrillic for Uzbek), regenerate once with a sharper nudge.
    """
    if not clips:
        return []
    settings = get_settings()
    fallbacks = [_fallback(c) for c in clips]
    system = SYSTEM_PROMPT.format(lang_directive=_lang_directive(language))
    user = _user_payload(clips)

    metas = fallbacks
    for attempt in (1, 2):
        try:
            content = await _call_llm(
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                model=settings.openai_analysis_model,   # cheap model — light copywriting
                max_tokens=1200,
            )
            raw = json.loads(content).get("clips", [])
        except Exception as e:                           # network / parse error → fallback
            logger.warning("clip metadata generation failed, using fallbacks: {}", e)
            return fallbacks

        metas = _assemble(raw, fallbacks)
        wrong = language and any(_wrong_language(f"{m.title} {m.caption}", language) for m in metas)
        if not wrong:
            return metas
        logger.warning("clip metadata in wrong language (expected {}) → regenerating", language)
        system = system + (f"\n\nYOUR PREVIOUS OUTPUT WAS IN THE WRONG LANGUAGE. Output ONLY in "
                           f"{_LANG_NAMES.get(language, language)}. Do it again, correctly.")
    return metas                                          # accept best effort after one retry
