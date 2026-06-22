"""
Long-video transcription tests.

A clipping product's bread and butter is long podcasts/streams, which blow past
Whisper's 25MB limit. We verify: the duration math, the too-long guard, and the
chunk-stitching with corrected offsets (Whisper + ffmpeg mocked).
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services import transcription as tr
from app.services.transcription import Transcript, Word


_HEADER = 44
_BPS = 16_000 * 2  # 16k mono 16-bit


def _make_wav(path: Path, seconds: float) -> Path:
    path.write_bytes(b"\x00" * (_HEADER + int(seconds * _BPS)))
    return path


# ---------------------------------------------------------------------------
# Duration math
# ---------------------------------------------------------------------------

def test_wav_duration_exact(tmp_path):
    w = _make_wav(tmp_path / "a.wav", 30.0)
    assert abs(tr._wav_duration_s(w) - 30.0) < 1e-6


# ---------------------------------------------------------------------------
# Too-long guard
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_video_too_long_raises(tmp_path, monkeypatch):
    from app import config as config_mod
    s = config_mod.get_settings()
    monkeypatch.setattr(s, "max_source_minutes", 5)   # 5-min cap

    src = tmp_path / "src.mp4"; src.write_bytes(b"x")

    async def fake_extract(video, dst, **kw):
        return _make_wav(Path(dst), 6 * 60)            # 6 min > 5-min cap
    monkeypatch.setattr(tr.ff, "extract_audio", fake_extract)

    with pytest.raises(tr.VideoTooLong, match="max supported"):
        await tr.transcribe(src)


# ---------------------------------------------------------------------------
# Single-request path (small file)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_small_file_single_request(tmp_path, monkeypatch):
    src = tmp_path / "src.mp4"; src.write_bytes(b"x")

    async def fake_extract(video, dst, **kw):
        return _make_wav(Path(dst), 60)                # 1 min → ~1.9MB, single
    monkeypatch.setattr(tr.ff, "extract_audio", fake_extract)

    one = Transcript("en", "hello world", [Word("hello", 0.1, 0.5), Word("world", 0.6, 1.0)])
    with patch.object(tr, "_transcribe_file", AsyncMock(return_value=one)) as mock_tf, \
         patch.object(tr, "AsyncOpenAI"):
        out = await tr.transcribe(src)

    assert mock_tf.call_count == 1                     # not chunked
    assert out.text == "hello world"
    # WAV cleaned up
    assert not (tmp_path / "src.whisper.wav").exists()


# ---------------------------------------------------------------------------
# Chunked path with offset stitching
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_large_file_is_chunked_and_offsets_stitched(tmp_path, monkeypatch):
    src = tmp_path / "src.mp4"; src.write_bytes(b"x")

    # 25 min of audio → ~48MB → over the 20MB direct limit → chunk.
    async def fake_extract(video, dst, **kw):
        return _make_wav(Path(dst), 25 * 60)
    monkeypatch.setattr(tr.ff, "extract_audio", fake_extract)

    # Mock ffmpeg segmentation: produce 3 chunk files of 10/10/5 min.
    async def fake_segment(wav, out_dir, *, segment_seconds, prefix):
        durations = [600, 600, 300]
        paths = []
        for i, d in enumerate(durations):
            p = _make_wav(Path(out_dir) / f"{prefix}_{i:04d}.wav", d)
            paths.append(p)
        return paths
    monkeypatch.setattr(tr.ff, "segment_audio", fake_segment)

    # Each chunk transcription returns one word at local t=1.0; _transcribe_file
    # itself applies the offset, so mock it to honor the offset arg.
    async def fake_tf(client, path, *, offset, model):
        return Transcript("en", f"chunk@{offset:.0f}", [Word("w", 1.0 + offset, 2.0 + offset)])
    with patch.object(tr, "_transcribe_file", side_effect=fake_tf) as mock_tf, \
         patch.object(tr, "AsyncOpenAI"):
        out = await tr.transcribe(src)

    # Three chunks → three calls, with cumulative offsets 0, 600, 1200.
    offsets = [c.kwargs["offset"] for c in mock_tf.call_args_list]
    assert offsets == [0.0, 600.0, 1200.0]
    # Words carry the offset → second/third chunk words are past 600/1200s.
    starts = sorted(w.start for w in out.words)
    assert starts == [1.0, 601.0, 1201.0]
    assert "chunk@0" in out.text and "chunk@1200" in out.text


# ---------------------------------------------------------------------------
# SRT generation still works on stitched transcript
# ---------------------------------------------------------------------------

def test_to_srt_from_offset_words():
    t = Transcript("en", "", [Word("late", 605.0, 605.4), Word("word", 605.5, 606.0)])
    srt = t.to_srt()
    assert "00:10:05" in srt   # 605s = 10m05s, proving offsets flow into captions
