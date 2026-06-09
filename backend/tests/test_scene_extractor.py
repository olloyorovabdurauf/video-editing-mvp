"""
Scene-extractor constraint enforcement.

The LLM call inside `extract_scenes` is mocked. What we're asserting is the
*post-LLM* invariants the function guarantees no matter what the model returns:
  - never overlay the first 2.5s of the clip
  - never overlay the last 1.5s
  - windows must be 2.5-5.0s
  - no overlapping windows (after dedupe)
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.schemas.reel import Segment
from app.services.creative_engine.scene_extractor import extract_scenes


def _segment(start=100.0, end=140.0) -> Segment:
    return Segment(
        start=start, end=end, hook_score=0.9,
        reason="t",
        transcript="A long enough transcript that has multiple beats to it.",
    )


def _mock_llm_response(windows: list[dict]):
    """Build a MagicMock that looks like an OpenAI chat completion."""
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = json.dumps({"windows": windows})
    return resp


@pytest.mark.asyncio
async def test_segment_too_short_returns_empty():
    """A <8s segment is never b-rolled — full speaker time."""
    seg = _segment(start=0, end=6)
    out = await extract_scenes(seg)
    assert out == []


@pytest.mark.asyncio
async def test_constraints_enforced_when_llm_violates():
    """LLM returns windows in forbidden zones; our code must clamp/drop them."""
    seg = _segment(start=100.0, end=140.0)  # 40s long; usable zone [102.5, 138.5]

    bad_windows = [
        # Violates head-zone (starts before 102.5)
        {"start": 100.5, "end": 105.0, "rationale": "r", "transcript": "", "priority": 0.9},
        # Violates tail-zone (ends after 138.5)
        {"start": 137.0, "end": 139.5, "rationale": "r", "transcript": "", "priority": 0.8},
        # Valid mid-clip window
        {"start": 115.0, "end": 119.0, "rationale": "r", "transcript": "", "priority": 0.7},
    ]
    resp = _mock_llm_response(bad_windows)

    with patch("app.services.creative_engine.scene_extractor.AsyncOpenAI") as mock_oai:
        mock_oai.return_value.chat.completions.create = AsyncMock(return_value=resp)
        out = await extract_scenes(seg)

    # The first one gets clamped to start=102.5; that may make it valid OR drop below 2.5s.
    # We just assert every returned window is within the safe band.
    for w in out:
        assert w.start >= 102.5 - 0.01, f"head-zone violation: {w}"
        assert w.end <= 138.5 + 0.01, f"tail-zone violation: {w}"
        assert 2.5 <= (w.end - w.start) <= 5.0 + 0.01


@pytest.mark.asyncio
async def test_overlapping_windows_are_deduplicated():
    """When LLM proposes two overlapping windows, we keep the higher-priority one."""
    seg = _segment(start=100.0, end=140.0)
    windows = [
        {"start": 110.0, "end": 114.0, "rationale": "lower", "transcript": "", "priority": 0.4},
        {"start": 111.0, "end": 115.0, "rationale": "higher", "transcript": "", "priority": 0.9},
    ]
    resp = _mock_llm_response(windows)

    with patch("app.services.creative_engine.scene_extractor.AsyncOpenAI") as mock_oai:
        mock_oai.return_value.chat.completions.create = AsyncMock(return_value=resp)
        out = await extract_scenes(seg)

    assert len(out) == 1
    assert out[0].priority == 0.9  # higher-priority survivor


@pytest.mark.asyncio
async def test_respects_max_windows():
    """max_windows=2 should trim a 5-candidate list to top-2 by priority."""
    seg = _segment(start=100.0, end=180.0)  # 80s, plenty of room
    windows = [
        {"start": 105.0 + i * 12, "end": 109.0 + i * 12,
         "rationale": "r", "transcript": "", "priority": 0.1 * (i + 1)}
        for i in range(5)
    ]
    resp = _mock_llm_response(windows)

    with patch("app.services.creative_engine.scene_extractor.AsyncOpenAI") as mock_oai:
        mock_oai.return_value.chat.completions.create = AsyncMock(return_value=resp)
        out = await extract_scenes(seg, max_windows=2)

    assert len(out) <= 2
    # Top picks must be the highest-priority ones
    priorities = [w.priority for w in out]
    assert all(p >= 0.3 for p in priorities)
