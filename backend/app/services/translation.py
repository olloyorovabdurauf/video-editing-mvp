"""
Translation / transliteration to a target language.

Why this exists: OpenAI Whisper can't transcribe some languages (Uzbek comes
back as Kazakh). When the user locks a source language Whisper can't produce, we
translate whatever Whisper returned into the target language — for both the
burned subtitles and the AI titles/captions — so the output always matches the
language the creator asked for.

Uses the strong model (gpt-4o) because faithful translation (esp. Kazakh→Uzbek
Latin) is beyond the cheap model. `_call_llm` is the only network seam (stubbed
in tests).
"""
from __future__ import annotations

import json

from loguru import logger
from openai import AsyncOpenAI

from app.config import get_settings

_LANG_NAMES = {
    "uz": "Uzbek (Latin alphabet)", "en": "English", "ru": "Russian",
    "kk": "Kazakh", "ar": "Arabic", "tr": "Turkish", "es": "Spanish",
    "fr": "French", "de": "German", "pt": "Portuguese", "hi": "Hindi",
}


def language_name(code: str) -> str:
    return _LANG_NAMES.get(code, code)


def _make_client() -> AsyncOpenAI:
    return AsyncOpenAI(api_key=get_settings().openai_api_key)


async def _call_llm(messages: list[dict], *, model: str, max_tokens: int) -> str:
    client = _make_client()
    resp = await client.chat.completions.create(
        model=model, response_format={"type": "json_object"},
        temperature=0.2, max_tokens=max_tokens, messages=messages)
    return (resp.choices[0].message.content or "").strip()


async def translate_lines(lines: list[str], target_lang: str) -> list[str]:
    """
    Translate each line into target_lang, preserving order AND count (lines map
    1:1 so caption timings stay aligned). Best-effort: returns the originals on
    any failure so a clip is never blocked. Empty lines pass through unchanged.
    """
    if not lines or not target_lang:
        return lines
    name = language_name(target_lang)
    idx = [i for i, ln in enumerate(lines) if ln.strip()]
    if not idx:
        return lines
    payload = {"lines": [lines[i] for i in idx]}
    extra = ""
    if target_lang == "uz":
        extra = (" Use natural Uzbek in the LATIN alphabet only (never Cyrillic/Arabic, "
                 "never Kazakh/Turkish).")
    system = (
        f"You are a professional subtitle translator. Translate EACH line into "
        f"{name}, keeping it natural and spoken. The input may be in a different or "
        f"mis-detected language — translate the MEANING, do not copy the source "
        f"language.{extra} Return STRICT JSON {{\"lines\": [...]}} with EXACTLY the "
        f"same number of lines, in the same order."
    )
    out: list = []
    for attempt in (1, 2):                         # one retry if output validation fails
        try:
            content = await _call_llm(
                [{"role": "system", "content": system},
                 {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
                model=get_settings().openai_escalation_model, max_tokens=2000)
            out = json.loads(content).get("lines", [])
        except Exception as e:
            logger.warning("translate_lines failed ({}) — keeping originals", e)
            return lines
        # Output validation: the translated lines must actually BE the target
        # language's script (e.g. Latin Uzbek). Catching Cyrillic/Arabic here is
        # what stops Kazakh captions from being burned into an Uzbek reel.
        from app.services.clip_metadata import _wrong_language
        joined = " ".join(str(x) for x in out)
        if len(out) == len(idx) and not _wrong_language(joined, target_lang):
            break
        if attempt == 1:
            logger.warning("translate_lines output failed validation for '{}' — regenerating",
                           target_lang)
            system += " PREVIOUS ATTEMPT WAS REJECTED for wrong language/script — comply exactly."
    if len(out) != len(idx):                       # count drift → unsafe to map, keep originals
        logger.warning("translate_lines count mismatch ({} vs {}) — keeping originals",
                       len(out), len(idx))
        return lines
    result = list(lines)
    for j, i in enumerate(idx):
        if str(out[j]).strip():
            result[i] = str(out[j]).strip()
    return result


async def translate_text(text: str, target_lang: str) -> str:
    """Translate a single block of text (e.g. a clip transcript) into target_lang."""
    if not text or not target_lang:
        return text
    out = await translate_lines([text], target_lang)
    return out[0] if out else text
