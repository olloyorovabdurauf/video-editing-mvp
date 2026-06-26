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


# ---------------------------------------------------------------------------
# Language locking (Issue 1: captions must match the source language)
# ---------------------------------------------------------------------------

def test_wrong_language_script_detection():
    assert cm._wrong_language("Бүгінгі таңда", "uz")       # Cyrillic for Uzbek → wrong
    assert not cm._wrong_language("Bugungi kunda AI", "uz")  # Latin Uzbek → ok
    assert cm._wrong_language("Hello world", "ru")          # Latin for Russian → wrong
    assert not cm._wrong_language("Привет мир", "ru")       # Cyrillic → ok
    assert cm._wrong_language("salom", "ar")                # no Arabic script → wrong


def test_uzbek_directive_demands_latin():
    d = cm._lang_directive("uz")
    assert "Uzbek" in d and "LATIN" in d and "Cyrillic" in d


def test_regenerates_when_language_wrong(monkeypatch):
    # 1st response is (wrongly) Cyrillic for an Uzbek lock → must regenerate.
    cyr = {"clips": [{"title": "Бүгінгі таңда", "caption": "Қазақша мәтін", "hashtags": ["ai"]},
                     {"title": "Тағы бір", "caption": "Текст", "hashtags": ["x"]}]}
    latin = {"clips": [{"title": "Bugungi kunda AI", "caption": "O'zbekcha matn", "hashtags": ["ai"]},
                       {"title": "Yana bir", "caption": "Matn", "hashtags": ["x"]}]}
    seq = [json.dumps(cyr), json.dumps(latin)]
    calls = {"n": 0}

    async def fake(messages, *, model, max_tokens):
        i = calls["n"]; calls["n"] += 1
        return seq[min(i, len(seq) - 1)]
    monkeypatch.setattr(cm, "_call_llm", fake)

    out = asyncio.run(cm.generate_for_clips(CLIPS, language="uz"))
    assert calls["n"] == 2                                  # regenerated once
    assert out[0].title == "Bugungi kunda AI"              # ended up in Latin Uzbek
    assert not cm._wrong_language(out[0].title, "uz")
