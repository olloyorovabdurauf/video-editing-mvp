"""
Complete-clip picker tests.

The product extracts COMPLETE 45-60s clips, not highlight fragments. We verify
the mechanical guarantees (sentence-boundary snapping, duration enforcement,
completeness scoring/filtering) and the fallback wiring with the LLM mocked.
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


def _talk(sentences: int = 9, words_per: int = 10, gap: float = 1.0) -> Transcript:
    """A talk with sentence punctuation every `words_per` words (≈ words_per sec)."""
    words: list[tuple[str, float, float]] = []
    t = 0.0
    for si in range(sentences):
        for wi in range(words_per):
            txt = f"w{si}x{wi}" + ("." if wi == words_per - 1 else "")
            words.append((txt, round(t, 2), round(t + 0.4, 2)))
            t += gap
    return _transcript(words)


MIN, MAX = 45, 60


# ---------------------------------------------------------------------------
# Sentence boundaries + window finalization (the editing core)
# ---------------------------------------------------------------------------

def test_is_sentence_end():
    assert sp._is_sentence_end("done.")
    assert sp._is_sentence_end("really?!".rstrip("!")) or sp._is_sentence_end("really?")
    assert sp._is_sentence_end("wait…")
    assert not sp._is_sentence_end("3.5")        # decimal, not a sentence end
    assert not sp._is_sentence_end("middle")


def test_finalize_window_snaps_and_enforces_duration():
    t = _talk()                                  # ~90s, sentence ends at 9.4, 19.4, ...
    win = sp._finalize_window(t.words, start=12.0, end=20.0, min_s=MIN, max_s=MAX)
    assert win is not None
    start, end = win
    dur = end - start
    assert MIN - 2 <= dur <= MAX                 # forced into range despite asking for 8s
    # both ends land on sentence boundaries (start of a sentence / end of a sentence)
    starts, ends = sp._boundaries(t.words)
    assert any(abs(start - s) < 0.01 for s in starts)
    assert any(abs(end - e) < 0.01 for e in ends)


def test_finalize_window_caps_at_max():
    t = _talk()
    _, end = sp._finalize_window(t.words, 0.0, 200.0, MIN, MAX)  # asks for way too long
    assert end <= 0.0 + MAX + 0.01


def test_finalize_window_none_without_boundaries():
    # No sentence punctuation → can't form a clean window → dropped.
    t = _transcript([("a", 0, 0.4), ("b", 1, 1.4), ("c", 2, 2.4)])
    assert sp._finalize_window(t.words, 0, 50, MIN, MAX) is None


# ---------------------------------------------------------------------------
# _clamp — scoring + duration + drop unformable
# ---------------------------------------------------------------------------

def test_clamp_builds_complete_scored_clip():
    t = _talk()
    raw = [{"start": 0, "end": 50, "hook_score": 0.9, "value_score": 0.7,
            "completeness_score": 0.8, "payoff_score": 0.6, "reason": "r", "summary": "s"}]
    out = sp._clamp(raw, t, MIN, MAX)
    assert len(out) == 1
    seg = out[0]
    assert MIN - 2 <= seg.duration <= MAX
    assert seg.completeness_score == 0.8 and seg.summary == "s"
    expected = round(0.20 * 0.9 + 0.25 * 0.7 + 0.35 * 0.8 + 0.20 * 0.6, 3)
    assert seg.score == expected
    assert "w0x0" in seg.transcript                # text filled from our words


def test_clamp_drops_unformable_and_malformed():
    t = _transcript([("a", 0, 0.4), ("b", 1, 1.4)])   # no boundaries
    raw = [
        {"start": 0},                                  # missing end → dropped
        {"start": 0, "end": 50, "completeness_score": 0.9, "reason": "x"},  # no boundaries → dropped
    ]
    assert sp._clamp(raw, t, MIN, MAX) == []


# ---------------------------------------------------------------------------
# Synthesis fallback
# ---------------------------------------------------------------------------

def test_synthesize_produces_complete_length_window():
    t = _talk()
    seg = sp._synthesize_from_transcript(t, MIN, MAX)
    assert seg is not None
    assert MIN - 2 <= seg.duration <= MAX
    assert seg.completeness_score == 0.6
    assert seg.reason.startswith("Best available")


def test_synthesize_none_for_empty():
    assert sp._synthesize_from_transcript(_transcript([]), MIN, MAX) is None


# ---------------------------------------------------------------------------
# Fallback wiring (LLM mocked)
# ---------------------------------------------------------------------------

def _resp(payload: dict) -> MagicMock:
    m = MagicMock()
    m.choices = [MagicMock()]
    m.choices[0].message.content = json.dumps(payload)
    return m


def _seg(start, end, comp=0.8):
    return {"start": start, "end": end, "hook_score": 0.8, "value_score": 0.8,
            "completeness_score": comp, "payoff_score": 0.8, "reason": "complete", "summary": "idea"}


@pytest.mark.asyncio
async def test_strict_returns_complete_clip():
    t = _talk()
    strict = _resp({"segments": [_seg(0, 50, comp=0.85)]})
    with patch.object(sp, "AsyncOpenAI") as oai:
        oai.return_value.chat.completions.create = AsyncMock(return_value=strict)
        out = await sp.pick_segments(t, n=1, min_duration_s=MIN, max_duration_s=MAX, prompt=None)
    assert len(out) == 1
    assert MIN - 2 <= out[0].duration <= MAX
    assert out[0].completeness_score == 0.85


@pytest.mark.asyncio
async def test_incomplete_clip_filtered_then_escalates():
    """A low-completeness pick is dropped by the strict bar → escalate model used."""
    t = _talk()
    weak = _resp({"segments": [_seg(0, 50, comp=0.4)]})    # below 0.6 strict bar
    strong = _resp({"segments": [_seg(0, 50, comp=0.75)]})
    from app.config import get_settings
    s = get_settings()
    with patch.object(sp, "AsyncOpenAI") as oai:
        create = AsyncMock(side_effect=[weak, strong])
        oai.return_value.chat.completions.create = create
        out = await sp.pick_segments(t, n=1, min_duration_s=MIN, max_duration_s=MAX, prompt=None)
    assert len(out) == 1 and out[0].completeness_score == 0.75
    models = [c.kwargs["model"] for c in create.call_args_list]
    assert models[0] == s.openai_analysis_model and models[1] == s.openai_escalation_model


@pytest.mark.asyncio
async def test_synthesis_when_all_passes_empty():
    t = _talk()
    empty = _resp({"segments": []})
    with patch.object(sp, "AsyncOpenAI") as oai:
        oai.return_value.chat.completions.create = AsyncMock(side_effect=[empty, empty, empty])
        out = await sp.pick_segments(t, n=1, min_duration_s=MIN, max_duration_s=MAX, prompt=None)
    assert len(out) == 1
    assert out[0].reason.startswith("Best available")
    assert MIN - 2 <= out[0].duration <= MAX


@pytest.mark.asyncio
async def test_cache_hit_short_circuits(patched_redis):
    t = _talk()
    found = _resp({"segments": [_seg(0, 50)]})
    with patch.object(sp, "AsyncOpenAI") as oai:
        create = AsyncMock(return_value=found)
        oai.return_value.chat.completions.create = create
        first = await sp.pick_segments(t, n=1, min_duration_s=MIN, max_duration_s=MAX, prompt=None)
        after = create.call_count
        second = await sp.pick_segments(t, n=1, min_duration_s=MIN, max_duration_s=MAX, prompt=None)
    assert len(first) == 1 and len(second) == 1
    assert create.call_count == after                  # no new LLM call on 2nd run


@pytest.mark.asyncio
async def test_truncated_json_does_not_crash():
    t = _talk()
    truncated = MagicMock()
    truncated.choices = [MagicMock()]
    truncated.choices[0].message.content = '{"segments": [{"start": 1, "end": 50, "reason": "an unterminated'
    with patch.object(sp, "AsyncOpenAI") as oai:
        oai.return_value.chat.completions.create = AsyncMock(side_effect=[truncated, truncated, truncated])
        out = await sp.pick_segments(t, n=1, min_duration_s=MIN, max_duration_s=MAX, prompt=None)
    assert len(out) == 1                               # never zero, never crashes
    assert out[0].reason.startswith("Best available")
