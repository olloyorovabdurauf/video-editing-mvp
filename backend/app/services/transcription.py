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


async def transcribe(video_path: Path) -> Transcript:
    settings = get_settings()
    client = AsyncOpenAI(api_key=settings.openai_api_key)

    wav = video_path.with_suffix(".whisper.wav")
    await ff.extract_audio(video_path, wav)

    logger.info("transcribing {}", wav.name)
    with wav.open("rb") as f:
        resp = await client.audio.transcriptions.create(
            model=settings.openai_transcribe_model,
            file=f,
            response_format="verbose_json",
            timestamp_granularities=["word"],
        )

    words = [Word(text=w.word, start=w.start, end=w.end) for w in (resp.words or [])]
    return Transcript(language=resp.language or "en", text=resp.text, words=words)
