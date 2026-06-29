"""
Transcription via OpenAI Whisper.

We chunk audio above 25MB (the API limit) and stitch results back together
with corrected timestamps. For multi-hour content, switch to faster-whisper
on a GPU worker; the interface stays the same.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

from loguru import logger
from openai import AsyncOpenAI

from app.config import get_settings
from app.utils import ffmpeg as ff


@dataclass
class Word:
    text: str
    start: float
    end: float


@dataclass
class Transcript:
    language: str
    text: str
    words: list[Word]
    # True when `text` is genuinely in `language`. False when we had to settle for
    # a model that produces the WRONG language (Whisper forced toward Uzbek emits
    # Kazakh) → downstream must translate. Native providers (Google STT) set True.
    is_source_language: bool = True

    def to_srt(self) -> str:
        """Group words into ~1.2s caption rows. Good for animated burn-ins."""
        lines: list[str] = []
        idx = 1
        buf: list[Word] = []
        for w in self.words:
            buf.append(w)
            if buf[-1].end - buf[0].start >= 1.2 or len(buf) >= 6:
                lines.append(_srt_block(idx, buf))
                idx += 1
                buf = []
        if buf:
            lines.append(_srt_block(idx, buf))
        return "\n".join(lines)


def _srt_block(idx: int, words: list[Word]) -> str:
    start, end = _srt_ts(words[0].start), _srt_ts(words[-1].end)
    return f"{idx}\n{start} --> {end}\n{' '.join(w.text for w in words)}\n"


def _srt_ts(t: float) -> str:
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02}:{int(m):02}:{s:06.3f}".replace(".", ",")


# Whisper API hard limit is 25MB. We chunk well under it. At 16kHz mono
# 16-bit PCM the byte-rate is fixed (32000 B/s), so duration ⇔ size is exact.
_SAMPLE_RATE = 16_000
_BYTES_PER_SEC = _SAMPLE_RATE * 2            # mono, 16-bit
_WAV_HEADER = 44
_CHUNK_SECONDS = 600                          # 10 min ≈ 19.2MB per chunk (< 25MB)
_DIRECT_LIMIT_BYTES = 20 * 1024 * 1024        # send whole file if under this
_CHUNK_CONCURRENCY = 4                         # parallel Whisper calls (rate-limit safe)
_CHUNK_CONCURRENCY = 4                         # parallel Whisper calls (rate-limit safe)


class VideoTooLong(ValueError):
    """Source exceeds the configured maximum duration — permanent, don't retry."""


def _wav_duration_s(path: Path) -> float:
    """Exact for 16k mono 16-bit PCM: (bytes - header) / byte-rate."""
    return max(0.0, (path.stat().st_size - _WAV_HEADER) / _BYTES_PER_SEC)


# ISO-639-1 codes OpenAI's transcription API accepts for the `language` param.
# Uzbek ('uz') is deliberately ABSENT — OpenAI rejects it ("unsupported_language",
# HTTP 400). Forcing only these; everything else auto-detects (+ optional bias).
_WHISPER_LANGS = frozenset({
    "af", "ar", "hy", "az", "be", "bs", "bg", "ca", "zh", "hr", "cs", "da", "nl",
    "en", "et", "fi", "fr", "gl", "de", "el", "he", "hi", "hu", "is", "id", "it",
    "ja", "kn", "kk", "ko", "lv", "lt", "mk", "ms", "mr", "mi", "ne", "no", "fa",
    "pl", "pt", "ro", "ru", "sr", "sk", "sl", "es", "sw", "sv", "tl", "ta", "th",
    "tr", "uk", "ur", "vi", "cy",
})

# For languages Whisper can't be FORCED into, bias it with a prompt in that
# language/script so the transcript leans the right way (best-effort).
_LANG_HINTS = {
    "uz": "Bu audio o'zbek tilida. Iltimos, o'zbek tilida, lotin alifbosida yozing.",
}


def _whisper_extra(language: str | None) -> dict:
    """Extra Whisper args: force a supported language, else bias an unsupported
    one with a prompt, else nothing (auto-detect). Never sends an unsupported
    `language` (which 400s)."""
    if language in _WHISPER_LANGS:
        return {"language": language}
    if language and language in _LANG_HINTS:
        return {"prompt": _LANG_HINTS[language]}
    return {}


async def _transcribe_file(client: AsyncOpenAI, path: Path, *, offset: float,
                           model: str, language: str | None = None) -> Transcript:
    """Transcribe one (≤limit) audio file; shift word timestamps by `offset`.

    `language` (ISO-639-1) forces Whisper when OpenAI supports it; for codes it
    rejects (e.g. Uzbek 'uz') we auto-detect and bias with a prompt instead. The
    returned Transcript is still LABELLED with the requested language so the
    captions/titles downstream lock to it.
    """
    extra = _whisper_extra(language)
    with path.open("rb") as f:
        resp = await client.audio.transcriptions.create(
            model=model, file=f,
            response_format="verbose_json",
            timestamp_granularities=["word"],
            **extra,
        )
    words = [
        Word(text=w.word, start=w.start + offset, end=w.end + offset)
        for w in (resp.words or [])
    ]
    # When forced, trust the requested language over Whisper's echo.
    lang = language or resp.language or "en"
    # If we asked for a language Whisper can't produce (e.g. uz), the text is in
    # the WRONG language even though we label it `lang` → flag for translation.
    native = language is None or language in _WHISPER_LANGS
    return Transcript(language=lang, text=resp.text or "", words=words,
                      is_source_language=native)


async def transcribe(video_path: Path, *, language: str | None = None) -> Transcript:
    settings = get_settings()

    # Languages Whisper can't do (Uzbek → transcribed as Kazakh) route to Google
    # STT's native model when credentials are configured. On any failure we fall
    # through to Whisper + the translation layer so a job never hard-fails here.
    if language and language in settings.google_stt_language_set:
        from app.services import google_stt
        try:
            return await google_stt.transcribe(video_path, language=language)
        except Exception as e:                       # noqa: BLE001 — degrade, don't crash
            logger.warning("Google STT failed for '{}' ({}); falling back to Whisper",
                           language, e)

    # Generous per-request timeout: a 10-min chunk upload + transcription.
    client = AsyncOpenAI(api_key=settings.openai_api_key, timeout=300.0)

    wav = video_path.with_suffix(".whisper.wav")
    await ff.extract_audio(video_path, wav)
    total_s = _wav_duration_s(wav)

    # Guard against runaway cost/time from someone pasting a 10-hour stream.
    max_s = settings.max_source_minutes * 60
    if total_s > max_s:
        wav.unlink(missing_ok=True)
        raise VideoTooLong(
            f"video is {total_s/60:.0f} min; max supported is "
            f"{settings.max_source_minutes} min"
        )

    model = settings.openai_transcribe_model
    size = wav.stat().st_size

    # --- Small enough: one request. ---
    if size <= _DIRECT_LIMIT_BYTES:
        logger.info("transcribing {} ({:.1f}MB) in one request (lang={})",
                    wav.name, size / 1e6, language or "auto")
        try:
            return await _transcribe_file(client, wav, offset=0.0, model=model, language=language)
        finally:
            wav.unlink(missing_ok=True)

    # --- Too big: chunk, transcribe in PARALLEL, stitch with corrected offsets. ---
    logger.info("audio {:.0f}MB > limit → chunking into {}s pieces",
                size / 1e6, _CHUNK_SECONDS)
    chunks = await ff.segment_audio(
        wav, wav.parent, segment_seconds=_CHUNK_SECONDS, prefix=wav.stem + "_seg"
    )

    # Offsets are knowable up front from each chunk's size (PCM byte-rate is
    # fixed), so we don't need to process chunks in order — transcribe them
    # concurrently with a bounded pool to keep a 2hr video at ~1-2 min while
    # respecting Whisper rate limits.
    offsets: list[float] = []
    acc = 0.0
    for chunk in chunks:
        offsets.append(acc)
        acc += _wav_duration_s(chunk)

    sem = asyncio.Semaphore(_CHUNK_CONCURRENCY)

    async def _do(chunk: Path, offset: float) -> Transcript:
        async with sem:
            return await _transcribe_file(client, chunk, offset=offset, model=model, language=language)

    try:
        # gather preserves order → results stay chunk-ordered.
        results = await asyncio.gather(*(_do(c, o) for c, o in zip(chunks, offsets)))
    finally:
        for chunk in chunks:
            chunk.unlink(missing_ok=True)
        wav.unlink(missing_ok=True)

    all_words = [w for r in results for w in r.words]
    texts = [r.text for r in results if r.text]
    detected = language or next((r.language for r in results if r.language), "en")
    native = language is None or language in _WHISPER_LANGS
    return Transcript(language=detected, text=" ".join(texts), words=all_words,
                      is_source_language=native)
