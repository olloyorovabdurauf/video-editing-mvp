"""
Semantic Ending Detection — stop at the first strong natural ending.
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from app.schemas.reel import Segment
from app.services import ending_detector as ed
from app.services.transcription import Transcript, Word


def _tr(n_words=200, sent_every=10):
    words = [Word(text=("stop." if i % sent_every == sent_every - 1 else "w"),
                  start=float(i), end=i + 0.9) for i in range(n_words)]
    return Transcript(language="en", text="x", words=words)


def _client(payload):
    async def create(**kw):
        _client.last_body = kw["messages"][1]["content"]
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=json.dumps(payload)))])
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def _run(client, tr, segs, min_s=45):
    return asyncio.run(ed.refine_endings(client, tr, segs, min_s=min_s))


def test_early_stop_at_payoff_sentence():
    # Clip 0..60s (6 sentences ending 9.9,19.9,...59.9); payoff = sentence 3 → trim to ~39.9
    tr = _tr()
    seg = Segment(start=0.0, end=60.0, transcript="", score=0.9)
    out = _run(_client({"endings": [{"i": 0, "end_sentence": 3, "confidence": 0.9,
                                     "reason": "payoff"}]}), tr, [seg])
    assert abs(out[0].end - 39.9) < 0.2
    assert out[0].start == 0.0                    # start untouched


def test_low_confidence_keeps_boundary():
    tr = _tr()
    seg = Segment(start=0.0, end=60.0, transcript="", score=0.9)
    out = _run(_client({"endings": [{"i": 0, "end_sentence": 2, "confidence": 0.4,
                                     "reason": "meh"}]}), tr, [seg])
    assert out[0].end == 60.0


def test_never_trims_below_fragment_floor():
    tr = _tr()
    seg = Segment(start=0.0, end=60.0, transcript="", score=0.9)
    # Sentence 0 ends at 9.9s — trimming there would be a fragment (< 0.6*45).
    out = _run(_client({"endings": [{"i": 0, "end_sentence": 0, "confidence": 0.95,
                                     "reason": "quote"}]}), tr, [seg])
    assert out[0].end == 60.0


def test_never_extends_past_current_end():
    tr = _tr()
    seg = Segment(start=0.0, end=60.0, transcript="", score=0.9)
    out = _run(_client({"endings": [{"i": 0, "end_sentence": 99, "confidence": 0.9,
                                     "reason": "bad index"}]}), tr, [seg])
    assert out[0].end == 60.0


def test_lookahead_content_is_shown_to_model():
    tr = _tr()
    seg = Segment(start=0.0, end=60.0, transcript="", score=0.9)
    client = _client({"endings": []})
    _run(client, tr, [seg])
    assert "AFTER-END" in _client.last_body      # repetition/topic-change evidence visible
