"""
Native transcription for languages OpenAI Whisper can't do, via Google Cloud
Speech-to-Text.

Why this exists: Whisper has no Uzbek model — it transcribes Uzbek audio AS
Kazakh, which is the root cause of the wrong-language captions. Google STT has a
native ``uz-UZ`` model (Latin script), so for the languages listed in
``settings.google_stt_languages`` we route here instead of Whisper-then-translate.

Design: we reuse the exact chunk-and-stitch pattern the Whisper path already uses,
so this needs **no Cloud Storage bucket**. Google's synchronous ``recognize`` caps
at 60s of audio per call, so we split the WAV into ≤55s pieces, transcribe them in
parallel, and shift each chunk's word timestamps by its known PCM byte-offset.

Auth: a service-account JSON, supplied via ``settings.google_stt_credentials`` as
either the raw JSON string (e.g. a Fly secret) or a path to the ``.json`` file.
"""
from __future__ import annotations

import asyncio
import json
from functools import lru_cache
from pathlib import Path

from loguru import logger

from app.config import get_settings
from app.utils import ffmpeg as ff
# Reuse the Whisper path's data types + exact PCM duration math (one-way import:
# transcription imports this module lazily, so there's no import cycle).
from app.services.transcription import (
    Transcript,
    VideoTooLong,
    Word,
    _wav_duration_s,
)

# Google sync recognize caps at 60s / 10MB inline. 55s of 16k mono 16-bit PCM is
# ~1.76MB — comfortably under both, with headroom for segment rounding.
_CHUNK_SECONDS = 55
_CONCURRENCY = 8                      # parallel recognize calls (well under quota)

# ISO-639-1 → BCP-47 for the languages we actually route here. The heuristic
# fallback (uppercase the code) is wrong for some (ky→ky-KG), so prefer the map.
_BCP47 = {
    "uz": "uz-UZ", "kk": "kk-KZ", "ky": "ky-KG", "az": "az-AZ",
    "tg": "tg-TJ", "tk": "tk-TM", "mn": "mn-MN",
}


def _bcp47(code: str) -> str:
    return _BCP47.get(code, f"{code}-{code.upper()}")


@lru_cache(maxsize=1)
def _client():
    """Authenticated Google SpeechClient (cached). Raises if creds are bad."""
    from google.cloud import speech                       # type: ignore
    from google.oauth2 import service_account             # type: ignore

    raw = get_settings().google_stt_credentials.strip()
    if not raw:
        raise RuntimeError("google_stt_credentials is empty")
    if raw.startswith("{"):
        info = json.loads(raw)
        creds = service_account.Credentials.from_service_account_info(info)
    else:
        creds = service_account.Credentials.from_service_account_file(raw)
    return speech.SpeechClient(credentials=creds)


def _recognize_sync(content: bytes, language_code: str, model: str
                    ) -> tuple[list[tuple[str, float, float]], str]:
    """Blocking single-chunk recognize → ([(word, start, end)], full_text).

    Tries the configured model (e.g. ``latest_long``); if Google rejects it for
    this language we retry once with the default model so a model mismatch never
    breaks transcription.
    """
    from google.cloud import speech                       # type: ignore

    audio = speech.RecognitionAudio(content=content)

    def _run(use_model: str | None):
        cfg = dict(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16_000,
            language_code=language_code,
            enable_word_time_offsets=True,
            enable_automatic_punctuation=True,
        )
        if use_model:
            cfg["model"] = use_model
        resp = _client().recognize(config=speech.RecognitionConfig(**cfg), audio=audio)
        words: list[tuple[str, float, float]] = []
        texts: list[str] = []
        for result in resp.results:
            alt = result.alternatives[0]
            texts.append(alt.transcript)
            for w in alt.words:
                words.append((w.word,
                              w.start_time.total_seconds(),
                              w.end_time.total_seconds()))
        return words, " ".join(t.strip() for t in texts if t.strip())

    try:
        return _run(model or None)
    except Exception as e:                                # noqa: BLE001
        if model:
            logger.warning("Google STT model '{}' rejected ({}); retrying default", model, e)
            return _run(None)
        raise


async def transcribe(video_path: Path, *, language: str) -> Transcript:
    """Native Google STT transcription, chunked + stitched like the Whisper path.

    Returns a Transcript labelled with `language` and ``is_source_language=True``
    (the text really is in that language), so downstream skips translation.
    """
    settings = get_settings()
    bcp47 = _bcp47(language)
    model = settings.google_stt_model

    wav = video_path.with_suffix(".gstt.wav")
    await ff.extract_audio(video_path, wav)
    total_s = _wav_duration_s(wav)
    max_s = settings.max_source_minutes * 60
    if total_s > max_s:
        wav.unlink(missing_ok=True)
        raise VideoTooLong(
            f"video is {total_s/60:.0f} min; max supported is "
            f"{settings.max_source_minutes} min")

    chunks = await ff.segment_audio(
        wav, wav.parent, segment_seconds=_CHUNK_SECONDS, prefix=wav.stem + "_g")
    logger.info("Google STT [{}]: {:.0f}s → {} chunk(s)", bcp47, total_s, len(chunks))

    # Offsets are knowable up front from each chunk's size (PCM byte-rate fixed),
    # so chunks transcribe concurrently and still stitch back in order.
    offsets: list[float] = []
    acc = 0.0
    for c in chunks:
        offsets.append(acc)
        acc += _wav_duration_s(c)

    sem = asyncio.Semaphore(_CONCURRENCY)

    async def _do(chunk: Path, offset: float) -> tuple[list[Word], str]:
        async with sem:
            raw, text = await asyncio.to_thread(
                _recognize_sync, chunk.read_bytes(), bcp47, model)
            return [Word(text=t, start=s + offset, end=e + offset) for t, s, e in raw], text

    try:
        results = await asyncio.gather(*(_do(c, o) for c, o in zip(chunks, offsets)))
    finally:
        for c in chunks:
            c.unlink(missing_ok=True)
        wav.unlink(missing_ok=True)

    all_words = [w for words, _ in results for w in words]
    full_text = " ".join(t for _, t in results if t).strip()
    if not all_words and not full_text:
        # Genuine empty result (silence) is suspicious for a real video → let the
        # caller fall back to Whisper rather than ship an empty transcript.
        raise RuntimeError("Google STT returned no speech")
    return Transcript(language=language, text=full_text, words=all_words,
                      is_source_language=True)
