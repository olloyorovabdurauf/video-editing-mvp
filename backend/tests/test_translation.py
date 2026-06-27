"""Translation layer (Issue 2): captions/titles match the locked language even
when Whisper produced the wrong one (Uzbek → Kazakh). LLM seam stubbed."""
from __future__ import annotations

import asyncio
import json

from app.services import translation as tr
from app.services import captions as cap
from app.services.transcription import Word


def test_translate_lines_preserves_count(monkeypatch):
    async def fake(messages, *, model, max_tokens):
        return json.dumps({"lines": ["Bir", "Ikki"]})
    monkeypatch.setattr(tr, "_call_llm", fake)
    assert asyncio.run(tr.translate_lines(["One", "Two"], "uz")) == ["Bir", "Ikki"]


def test_translate_lines_count_mismatch_keeps_originals(monkeypatch):
    async def fake(messages, *, model, max_tokens):
        return json.dumps({"lines": ["only one"]})       # wrong count → unsafe to map
    monkeypatch.setattr(tr, "_call_llm", fake)
    assert asyncio.run(tr.translate_lines(["a", "b"], "uz")) == ["a", "b"]


def test_translate_lines_failure_keeps_originals(monkeypatch):
    async def boom(messages, *, model, max_tokens):
        raise RuntimeError("openai down")
    monkeypatch.setattr(tr, "_call_llm", boom)
    assert asyncio.run(tr.translate_lines(["a", "b"], "uz")) == ["a", "b"]


def test_translate_lines_noop_without_language():
    assert asyncio.run(tr.translate_lines(["x"], "")) == ["x"]


def test_render_ass_lines_uses_translated_text():
    ph = cap.Phrase([Word("hello", 0.0, 0.5), Word("world", 0.6, 1.0)])
    out = cap.render_ass_lines([ph], ["Salom dunyo"], style="karaoke", resolution=(1080, 1920))
    assert "Salom dunyo" in out
    assert "Dialogue:" in out
    assert "hello" not in out                            # original (wrong-lang) text replaced
