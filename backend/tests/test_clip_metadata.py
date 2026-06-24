"""Tests for per-clip social metadata. The LLM seam is stubbed (offline)."""
from __future__ import annotations

import asyncio
import json

from app.services import clip_metadata as cm


CLIPS = [
    cm.ClipInput(transcript="Most people think productivity is doing more. It isn't.",
                 reason="Contrarian hook on productivity."),
    cm.ClipInput(transcript="We cut our standup to six minutes and shipped faster.",
                 reason="Concrete story with numbers."),
]


def test_generates_metadata_per_clip(monkeypatch):
    payload = {"clips": [
        {"title": "Productivity is a lie", "caption": "Hook\n\nValue\n\nFollow for more",
         "hashtags": ["productivity", "#focus"]},
        {"title": "6-minute standups", "caption": "Cut the meeting.", "hashtags": ["startup"]},
    ]}

    async def fake(messages, *, model, max_tokens):
        return json.dumps(payload)
    monkeypatch.setattr(cm, "_call_llm", fake)

    out = asyncio.run(cm.generate_for_clips(CLIPS))
    assert len(out) == 2
    assert out[0].title == "Productivity is a lie"
    assert out[0].hashtags == ["productivity", "focus"]     # '#' stripped
    assert out[1].caption == "Cut the meeting."


def test_falls_back_when_llm_errors(monkeypatch):
    async def boom(messages, *, model, max_tokens):
        raise RuntimeError("openai down")
    monkeypatch.setattr(cm, "_call_llm", boom)

    out = asyncio.run(cm.generate_for_clips(CLIPS))
    assert len(out) == 2
    # Deterministic fallback derived from the clip itself.
    assert out[0].title.startswith("Most people think")
    assert out[0].caption == "Contrarian hook on productivity."


def test_short_llm_array_filled_with_fallbacks(monkeypatch):
    async def partial(messages, *, model, max_tokens):
        return json.dumps({"clips": [{"title": "Only one", "caption": "c", "hashtags": []}]})
    monkeypatch.setattr(cm, "_call_llm", partial)

    out = asyncio.run(cm.generate_for_clips(CLIPS))
    assert len(out) == 2
    assert out[0].title == "Only one"
    assert out[1].title.startswith("We cut our standup")   # fallback for the missing item


def test_empty_input():
    assert asyncio.run(cm.generate_for_clips([])) == []
