"""
Language guard — catches ASR language mislabeling from the transcript TEXT.

Why this exists: Whisper has no Uzbek model and transcribes Uzbek audio AS
KAZAKH (Cyrillic). When the user explicitly picks Uzbek we already correct via
the translation layer — but in auto-detect mode the wrong label used to flow
through to captions/titles untouched. This guard runs a cheap text-based
language identification on a transcript sample right after transcription; when
it disagrees with the ASR label, downstream switches to the guard's verdict
(captions translated, metadata locked to the real language).

Uzbek vs Kazakh vs Kyrgyz from TEXT is easy for an LLM even when the audio
fooled the ASR — vocabulary and grammar differ clearly.
"""
from __future__ import annotations

import json

from loguru import logger

from app.config import get_settings

_SAMPLE_CHARS = 1200          # plenty for LID, tiny token cost

_PROMPT = (
    "Identify the language ACTUALLY spoken in this transcript sample. "
    "IMPORTANT: speech recognizers often mis-transcribe Uzbek as Kazakh — if the "
    "vocabulary/grammar is Uzbek (even written in Cyrillic), answer uz. "
    'Reply with JSON only: {"lang": "<ISO 639-1 code>"}.'
)


async def detect_text_language(text: str) -> str | None:
    """Best-effort ISO 639-1 code from transcript text; None on any failure."""
    sample = (text or "").strip()[:_SAMPLE_CHARS]
    if len(sample) < 40:                       # too little signal to judge
        return None
    try:
        from openai import AsyncOpenAI
        settings = get_settings()
        client = AsyncOpenAI(api_key=settings.openai_api_key)
        resp = await client.chat.completions.create(
            model=settings.openai_analysis_model,
            messages=[{"role": "system", "content": _PROMPT},
                      {"role": "user", "content": sample}],
            response_format={"type": "json_object"},
            max_tokens=20, temperature=0,
        )
        lang = (json.loads(resp.choices[0].message.content).get("lang") or "").strip().lower()
        return lang[:2] if len(lang) >= 2 else None
    except Exception as e:                     # guard must never fail a job
        logger.warning("language guard failed (keeping ASR label): {}", e)
        return None
