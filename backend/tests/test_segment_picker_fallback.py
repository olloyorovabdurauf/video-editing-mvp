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


def test_finalize_window_word_fallback_when_no_punctuation():
    # Whisper words often carry NO sentence punctuation. We must still produce a
    # 45-60s window aligned to word boundaries — never None, never the whole clip.
    words = [(f"w{i}", float(i), float(i) + 0.4) for i in range(200)]   # 200s, no '.'
    win = sp._finalize_window(_transcript(words).words, 30, 90, MIN, MAX)
    assert win is not None
    s, e = win
    assert MIN - 2 <= e - s <= MAX


def test_finalize_window_never_returns_whole_video():
    # The 895s bug: even asking for a huge window, the result is capped at max.
    words = [(f"w{i}", float(i), float(i) + 0.4) for i in range(900)]   # 900s, no '.'
    s, e = sp._finalize_window(_transcript(words).words, 0, 900, MIN, MAX)
    assert e - s <= MAX + 0.1


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
    # Overall folds the full rubric; dims the LLM omitted default to 0.5 (via _f).
    expected = sp._overall({
        "hook_score": 0.9, "value_score": 0.7, "completeness_score": 0.8, "payoff_score": 0.6,
        "curiosity_score": 0.5, "emotion_score": 0.5, "retention_score": 0.5, "virality_score": 0.5})
    assert seg.score == expected
    assert "w0x0" in seg.transcript                # text filled from our words


def test_weights_sum_to_one():
    assert abs(sum(sp._WEIGHTS.values()) - 1.0) < 1e-9


def test_clamp_folds_extended_rubric_into_overall():
    t = _talk()
    raw = [{"start": 0, "end": 50,
            "hook_score": 0.8, "curiosity_score": 0.9, "emotion_score": 0.7,
            "value_score": 0.8, "retention_score": 0.9, "completeness_score": 0.85,
            "payoff_score": 0.8, "standalone_score": 0.9, "virality_score": 0.95,
            "reason": "r", "summary": "s"}]
    out = sp._clamp(raw, t, MIN, MAX)
    assert len(out) == 1
    seg = out[0]
    assert seg.virality_score == 0.95 and seg.standalone_score == 0.9 and seg.curiosity_score == 0.9
    assert seg.score == sp._overall({
        "hook_score": 0.8, "curiosity_score": 0.9, "emotion_score": 0.7, "value_score": 0.8,
        "retention_score": 0.9, "completeness_score": 0.85, "payoff_score": 0.8, "virality_score": 0.95})
    assert 0 < seg.score <= 1


def test_clamp_drops_malformed_picks():
    t = _talk()
    raw = [
        {"start": 0},                 # missing end → dropped
        {"hook_score": 0.5},          # missing start + end → dropped
        "not a dict",                 # dropped
    ]
    assert sp._clamp(raw, t, MIN, MAX) == []


def test_synthesize_never_exceeds_max_without_punctuation():
    words = [(f"w{i}", float(i), float(i) + 0.4) for i in range(300)]   # 300s, no '.'
    seg = sp._synthesize_from_transcript(_transcript(words), MIN, MAX)
    assert seg is not None and seg.duration <= MAX + 0.1   # not the whole 300s


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
async def test_low_standalone_is_gated_then_escalates():
    """Issue 4: a clip with a complete arc but that DOESN'T stand alone is dropped
    by the standalone gate → the picker escalates to find a real standalone clip."""
    t = _talk()
    weak = _resp({"segments": [{**_seg(0, 50, comp=0.85), "standalone_score": 0.3}]})
    strong = _resp({"segments": [{**_seg(0, 50, comp=0.85), "standalone_score": 0.85}]})
    with patch.object(sp, "AsyncOpenAI") as oai:
        create = AsyncMock(side_effect=[weak, strong])
        oai.return_value.chat.completions.create = create
        out = await sp.pick_segments(t, n=1, min_duration_s=MIN, max_duration_s=MAX, prompt=None)
    assert len(out) == 1 and out[0].standalone_score == 0.85
    assert create.call_count == 3                  # escalated (+1 reviewer pass)


@pytest.mark.asyncio
async def test_story_map_in_response_is_ignored():
    """The model emits a story_map before segments; we read only segments."""
    t = _talk()
    resp = _resp({"story_map": [{"start": 0, "end": 90, "theme": "x", "type": "lesson"}],
                  "segments": [_seg(0, 50, comp=0.8)]})
    with patch.object(sp, "AsyncOpenAI") as oai:
        oai.return_value.chat.completions.create = AsyncMock(return_value=resp)
        out = await sp.pick_segments(t, n=1, min_duration_s=MIN, max_duration_s=MAX, prompt=None)
    assert len(out) == 1


@pytest.mark.asyncio
async def test_synthesis_when_all_passes_empty():
    t = _talk()
    empty = _resp({"segments": []})
    with patch.object(sp, "AsyncOpenAI") as oai:
        oai.return_value.chat.completions.create = AsyncMock(side_effect=[empty, empty, empty])
        out = await sp.pick_segments(t, n=1, min_duration_s=MIN, max_duration_s=MAX, prompt=None)
    assert len(out) >= 1                          # never zero
    assert MIN - 2 <= out[0].duration <= MAX      # a sized clip, never the whole video


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


def test_dedupe_overlap_keeps_distinct_drops_overlap():
    def S(start, end, score):
        from app.schemas.reel import Segment
        return Segment(start=start, end=end, score=score, completeness_score=0.8, transcript="t")
    # Two heavily-overlapping clips (keep the higher score) + one distinct.
    out = sp._dedupe_overlap([S(0, 55, 0.7), S(5, 58, 0.9), S(80, 135, 0.6)])
    assert len(out) == 2
    kept_starts = {s.start for s in out}
    assert 5 in kept_starts and 80 in kept_starts and 0 not in kept_starts


@pytest.mark.asyncio
async def test_accumulates_distinct_clips_toward_n():
    """The 'asked for 5, got 1' fix: keep gathering across passes until we reach n."""
    t = _talk(sentences=16)                       # ~160s → room for several clips
    cheap = _resp({"segments": [_seg(0, 50)]})    # 1 clip
    esc = _resp({"segments": [_seg(60, 110)]})    # +1 from the middle
    fill = _resp({"segments": [_seg(115, 160)]})  # +1 from the end
    with patch.object(sp, "AsyncOpenAI") as oai:
        create = AsyncMock(side_effect=[cheap, esc, fill])
        oai.return_value.chat.completions.create = create
        out = await sp.pick_segments(t, n=5, min_duration_s=MIN, max_duration_s=MAX, prompt=None)
    assert create.call_count == 4                 # escalated + filled (+1 reviewer pass)
    assert len(out) == 3                          # 3 distinct clips gathered
    starts = [s.start for s in out]
    assert starts == sorted(starts)               # returned chronologically
    for a, b in zip(out, out[1:]):                # non-overlapping
        assert a.end <= b.start + 1
    assert all(MIN - 2 <= s.duration <= MAX for s in out)


@pytest.mark.asyncio
async def test_distributes_to_n_when_ai_underdelivers_on_long_video():
    # The real-world bug: long video, AI keeps returning 1 → distribution fills
    # to n with complete windows spread across the timeline.
    words = [(f"w{i}", round(i * 0.5, 2), round(i * 0.5 + 0.42, 2)) for i in range(720)]  # ~360s
    t = _transcript(words)
    one = _resp({"segments": [_seg(0, 55)]})   # AI returns the same single clip every pass
    with patch.object(sp, "AsyncOpenAI") as oai:
        oai.return_value.chat.completions.create = AsyncMock(return_value=one)
        out = await sp.pick_segments(t, n=3, min_duration_s=MIN, max_duration_s=MAX, prompt=None)
    assert len(out) == 3                        # guaranteed N on a 360s source
    assert all(MIN - 2 <= s.duration <= MAX for s in out)
    starts = [s.start for s in out]
    assert starts == sorted(starts)
    for a, b in zip(out, out[1:]):
        assert a.end <= b.start + 1             # non-overlapping


@pytest.mark.asyncio
async def test_stops_escalating_once_n_met():
    t = _talk(sentences=16)
    cheap = _resp({"segments": [_seg(0, 50), _seg(60, 110), _seg(115, 160)]})  # 3 at once
    with patch.object(sp, "AsyncOpenAI") as oai:
        create = AsyncMock(side_effect=[cheap])
        oai.return_value.chat.completions.create = create
        out = await sp.pick_segments(t, n=3, min_duration_s=MIN, max_duration_s=MAX, prompt=None)
    assert create.call_count == 2                 # cheap pass + reviewer only — no escalation
    assert len(out) == 3


@pytest.mark.asyncio
async def test_truncated_json_does_not_crash():
    t = _talk()
    truncated = MagicMock()
    truncated.choices = [MagicMock()]
    truncated.choices[0].message.content = '{"segments": [{"start": 1, "end": 50, "reason": "an unterminated'
    with patch.object(sp, "AsyncOpenAI") as oai:
        oai.return_value.chat.completions.create = AsyncMock(side_effect=[truncated, truncated, truncated])
        out = await sp.pick_segments(t, n=1, min_duration_s=MIN, max_duration_s=MAX, prompt=None)
    assert len(out) >= 1                               # never zero, never crashes
    assert MIN - 2 <= out[0].duration <= MAX


def test_finalize_window_grace_completes_the_sentence():
    """A sentence ending a few seconds past max_s is kept — never truncate the payoff."""
    from app.services.segment_picker import _finalize_window
    from app.services.transcription import Word
    # Sentence ends at 64.0s; max_s=60. Grace (min(6, 10%)=6) must include it.
    words = ([Word(text="w", start=float(i), end=float(i) + 0.9) for i in range(63)]
             + [Word(text="end.", start=63.0, end=64.0)])
    win = _finalize_window(words, 0.0, 58.0, 45, 60)
    assert win is not None
    assert win[1] >= 64.0                        # completed the thought
    assert win[1] <= 66.1                        # but still bounded by max_s + grace


def test_reviewer_rejects_and_nudges(monkeypatch):
    """Agent 4: rejected clips never render; nudged boundaries are re-finalized."""
    import asyncio, json as _json
    from types import SimpleNamespace
    from app.services import segment_picker as sp
    from app.services.transcription import Transcript, Word
    from app.schemas.reel import Segment

    words = [Word(text=("end." if i % 20 == 4 else "w"), start=float(i), end=i + 0.9)
             for i in range(200)]   # sentences end at 4.9, 24.9, ... → starts at 5, 25, ...
    tr = Transcript(language="en", text="x", words=words)
    segs = [Segment(start=0.0, end=50.0, transcript="", score=0.9),
            Segment(start=100.0, end=150.0, transcript="", score=0.5)]

    verdicts = {"reviews": [
        {"i": 0, "verdict": "approve", "start_delta": 5, "end_delta": 0, "reason": "tighten"},
        {"i": 1, "verdict": "reject", "start_delta": 0, "end_delta": 0, "reason": "no conclusion"},
    ]}
    async def fake_create(**kw):
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=_json.dumps(verdicts)))])
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create)))

    out = asyncio.run(sp._review_clips(client, tr, segs, 45, 60))
    # Rejected clip is replaced from an uncovered story region ("choose another
    # story"), so the user still gets the requested count — but not that window.
    assert len(out) == 2
    assert out[0].start >= 4.0                # nudge applied (then boundary-aligned)
    assert out == sorted(out, key=lambda x: x.start)


# ---------------------------------------------------------------------------
# Semantic boundaries: meaning beats duration (soft 45-60s, hard 27-90s band)
# ---------------------------------------------------------------------------

def _mk_words(sentence_ends_at, total=120):
    from app.services.transcription import Word
    ends = set(sentence_ends_at)
    return [Word(text=("stop." if i in ends else "w"), start=float(i), end=i + 0.95)
            for i in range(total)]


def test_short_but_complete_idea_is_not_padded():
    """Idea genuinely ends at ~42s (min=45): keep 42s, don't stretch into filler."""
    from app.services.segment_picker import _finalize_window
    words = _mk_words({41, 80})
    win = _finalize_window(words, 0.0, 42.0, 45, 60)
    assert win is not None
    assert win[1] <= 43.0                    # ended where the idea ends
    assert win[1] - win[0] >= 27.0           # still above the hard fragment floor


def test_long_idea_completes_past_max():
    """Conclusion lands at ~67s (max=60): the thought finishes, within hard cap."""
    from app.services.segment_picker import _finalize_window
    words = _mk_words({66}, total=120)
    win = _finalize_window(words, 0.0, 58.0, 45, 60)
    assert win is not None
    assert win[1] >= 66.0                    # took the sentence end past max_s
    assert win[1] - win[0] <= 90.0           # bounded by the platform hard cap


def test_whole_video_guard_survives_semantic_bounds():
    """No punctuation at all: end is still hard-capped, never the whole video."""
    from app.services.segment_picker import _finalize_window
    from app.services.transcription import Word
    words = [Word(text="w", start=float(i), end=i + 0.95) for i in range(300)]
    win = _finalize_window(words, 0.0, 280.0, 45, 60)
    assert win is not None
    assert win[1] - win[0] <= 90.0 + 0.1     # 1.5x max, capped — not 300s
