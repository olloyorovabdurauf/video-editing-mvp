"""
Segment-picker fallback tests.

A clipping product must never return zero clips for a video that has speech.
We verify the deterministic pieces (clamp, synthesis) and the fallback wiring
with the LLM mocked.
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services import segment_picker as sp
from app.services.transcription import Transcript, Word


def _transcript(words: list[tuple[str, float, float]]) -> Transcript:
    return Transcript(language="en", text=" ".join(w[0] for w in words),
                      words=[Word(text=t, start=s, end=e) for t, s, e in words])


# ---------------------------------------------------------------------------
# _clamp
# ---------------------------------------------------------------------------

def test_clamp_drops_too_short_and_too_long():
    raw = [
        {"start": 0, "end": 2, "hook_score": 0.9, "reason": "r", "transcript": "t"},   # 2s < min
        {"start": 0, "end": 30, "hook_score": 0.8, "reason": "r", "transcript": "t"},  # ok (max 25+5)
        {"start": 0, "end": 60, "hook_score": 0.8, "reason": "r", "transcript": "t"},  # 60s > 25+5
    ]
    out = sp._clamp(raw, max_duration_s=25, min_dur=5.0)
    assert len(out) == 1
    assert out[0].end == 30


def test_clamp_relaxed_min_keeps_short_clip():
    raw = [{"start": 0, "end": 4.5, "hook_score": 0.4, "reason": "r", "transcript": "t"}]
    assert len(sp._clamp(raw, 25, min_dur=4.0)) == 1   # kept at min_dur=4
    assert len(sp._clamp(raw, 25, min_dur=5.0)) == 0   # dropped at min_dur=5


# ---------------------------------------------------------------------------
# _synthesize_from_transcript
# ---------------------------------------------------------------------------

def test_synthesize_covers_speech_window():
    t = _transcript([("hello", 1.0, 1.5), ("there", 1.6, 2.0), ("friend", 2.1, 2.6)])
    seg = sp._synthesize_from_transcript(t, max_duration_s=20)
    assert seg is not None
    assert seg.start == 1.0
    assert seg.end == 2.6
    assert seg.hook_score == 0.3
    assert "hello" in seg.transcript


def test_synthesize_caps_at_max_duration():
    words = [(f"w{i}", float(i), float(i) + 0.4) for i in range(60)]  # 60s of speech
    seg = sp._synthesize_from_transcript(_transcript(words), max_duration_s=20)
    assert seg is not None
    assert seg.end - seg.start <= 20.5


def test_synthesize_none_for_empty_transcript():
    assert sp._synthesize_from_transcript(_transcript([]), 20) is None


# ---------------------------------------------------------------------------
# Fallback wiring (LLM mocked)
# ---------------------------------------------------------------------------

def _mock_resp(payload: dict) -> MagicMock:
    m = MagicMock()
    m.choices = [MagicMock()]
    m.choices[0].message.content = json.dumps(payload)
    return m


@pytest.mark.asyncio
async def test_strict_pass_returns_when_found():
    t = _transcript([("contrarian", 10.0, 10.5), ("take", 10.6, 25.0)])
    strict = _mock_resp({"segments": [
        {"start": 10, "end": 24, "hook_score": 0.9, "reason": "r", "transcript": "t"}
    ]})
    with patch.object(sp, "AsyncOpenAI") as oai:
        oai.return_value.chat.completions.create = AsyncMock(return_value=strict)
        out = await sp.pick_segments(t, n=1, max_duration_s=30, prompt=None)
    assert len(out) == 1 and out[0].hook_score == 0.9


@pytest.mark.asyncio
async def test_lenient_fallback_used_when_strict_empty():
    t = _transcript([("mundane", 1.0, 1.5), ("chatter", 1.6, 18.0)])
    strict = _mock_resp({"segments": []})                       # strict finds nothing
    lenient = _mock_resp({"segments": [
        {"start": 1, "end": 17, "hook_score": 0.4, "reason": "best available", "transcript": "t"}
    ]})
    with patch.object(sp, "AsyncOpenAI") as oai:
        oai.return_value.chat.completions.create = AsyncMock(side_effect=[strict, lenient])
        out = await sp.pick_segments(t, n=1, max_duration_s=30, prompt=None)
    assert len(out) == 1
    assert out[0].hook_score == 0.4


@pytest.mark.asyncio
async def test_synthesis_when_both_llm_passes_empty():
    """Even if both LLM passes return nothing, a speech video yields one clip."""
    t = _transcript([("here", 2.0, 2.4), ("we", 2.5, 2.7), ("are", 2.8, 3.2)])
    empty = _mock_resp({"segments": []})
    with patch.object(sp, "AsyncOpenAI") as oai:
        oai.return_value.chat.completions.create = AsyncMock(side_effect=[empty, empty])
        out = await sp.pick_segments(t, n=1, max_duration_s=20, prompt=None)
    assert len(out) == 1
    assert out[0].reason.startswith("Best available")
