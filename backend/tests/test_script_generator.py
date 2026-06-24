"""
Tests for the Reels script generator. The only network seam (`_call_llm`) is
stubbed, so these are fast and offline. We assert the structural guarantees that
make the feature a *fix* for thin reels: 4 beats, 40s+ duration, hook payoff,
native-language prompting, and the exact output format.
"""
from __future__ import annotations

import asyncio
import json

import pytest

from app.schemas.script import ScriptGenerateRequest, ScriptLanguage, ScriptStyle
from app.services import script_generator as sg


def _valid_payload(duration_end: int = 60) -> dict:
    return {
        "title": "AI 3 ta montaj ishini yo'q qiladi",
        "hashtags": ["aivideo", "#montaj", "startup"],
        "sections": [
            {"name": "hook", "start_s": 0, "end_s": 5,
             "voiceover": "Video editing 3 soat vaqt olayaptimi?", "visual": "Founder kameraga qaraydi"},
            {"name": "problem", "start_s": 5, "end_s": 15,
             "voiceover": "Creatorlarning eng katta muammosi — montaj.", "visual": "Timeline ko'rsatiladi"},
            {"name": "value", "start_s": 15, "end_s": 45,
             "voiceover": "Mana AI avtomatlashtiradigan 3 ta ish: kesish, subtitr, reframe.",
             "visual": "Uchta misol ketma-ket"},
            {"name": "payoff", "start_s": 45, "end_s": duration_end,
             "voiceover": "Demak AI montajni daqiqalarga qisqartiradi. Profilni kuzating.",
             "visual": "Logo + CTA"},
        ],
        "caption": {
            "hook": "Video editing 3 soat vaqt olayaptimi?",
            "body": "Creatorlarning eng katta muammosi — montaj. AI buni necha daqiqaga qisqartiradi.",
            "cta": "AI video editing kelajagini kuzatib boring.",
        },
    }


def _short_payload() -> dict:
    """All four beats present but the whole thing is compressed to 22s (< 40 floor)."""
    p = _valid_payload()
    p["sections"] = [
        {"name": "hook", "start_s": 0, "end_s": 3, "voiceover": "Hook", "visual": "v"},
        {"name": "problem", "start_s": 3, "end_s": 8, "voiceover": "Problem", "visual": "v"},
        {"name": "value", "start_s": 8, "end_s": 16, "voiceover": "Value", "visual": "v"},
        {"name": "payoff", "start_s": 16, "end_s": 22, "voiceover": "Payoff", "visual": "v"},
    ]
    return p


def _stub_returning(*payloads: dict):
    """Async stub for _call_llm that yields the given payloads across calls."""
    calls = {"n": 0}
    seq = list(payloads)

    async def fake(messages, *, model, max_tokens):
        i = min(calls["n"], len(seq) - 1)
        calls["n"] += 1
        return json.dumps(seq[i])

    fake.calls = calls
    return fake


def test_generates_valid_uzbek_script(monkeypatch):
    monkeypatch.setattr(sg, "_call_llm", _stub_returning(_valid_payload()))
    req = ScriptGenerateRequest(topic="AI video editing future", language=ScriptLanguage.UZ,
                                style=ScriptStyle.FOUNDER, duration_seconds=60)
    res = asyncio.run(sg.generate(req))

    assert res.language is ScriptLanguage.UZ
    assert res.hook == "Video editing 3 soat vaqt olayaptimi?"
    assert [s.name for s in res.sections] == ["hook", "problem", "value", "payoff"]
    assert res.duration_seconds == 60
    # Full read-through includes every beat's voiceover.
    assert "subtitr" in res.script
    # CTA present and caption renders hashtags without duping '#'.
    assert res.caption.cta.endswith("kuzatib boring.")
    rendered = res.caption.render(res.hashtags)
    assert "#montaj" in rendered and "##" not in rendered


def test_output_format_block(monkeypatch):
    monkeypatch.setattr(sg, "_call_llm", _stub_returning(_valid_payload()))
    res = asyncio.run(sg.generate(ScriptGenerateRequest(topic="x", language=ScriptLanguage.UZ)))
    out = res.formatted
    for header in ("TITLE:", "HOOK:", "SCRIPT:", "SCENE BREAKDOWN:", "CAPTION:", "LANGUAGE:"):
        assert header in out
    assert "[0-5s] HOOK" in out
    assert out.rstrip().endswith("uz")


def test_retries_then_succeeds_on_short_draft(monkeypatch):
    stub = _stub_returning(_short_payload(), _valid_payload())  # 1st too short, 2nd valid
    monkeypatch.setattr(sg, "_call_llm", stub)
    res = asyncio.run(sg.generate(ScriptGenerateRequest(topic="x", language=ScriptLanguage.UZ)))
    assert res.duration_seconds == 60
    assert stub.calls["n"] == 2                       # it actually retried


def test_raises_after_two_bad_drafts(monkeypatch):
    monkeypatch.setattr(sg, "_call_llm", _stub_returning(_short_payload(), _short_payload()))
    with pytest.raises(sg.ScriptGenerationError):
        asyncio.run(sg.generate(ScriptGenerateRequest(topic="x", language=ScriptLanguage.UZ)))


def test_sections_are_reordered_to_canonical(monkeypatch):
    payload = _valid_payload()
    payload["sections"].reverse()                     # payoff, value, problem, hook
    monkeypatch.setattr(sg, "_call_llm", _stub_returning(payload))
    res = asyncio.run(sg.generate(ScriptGenerateRequest(topic="x", language=ScriptLanguage.UZ)))
    assert [s.name for s in res.sections] == ["hook", "problem", "value", "payoff"]


def test_prompt_is_native_language_and_styled():
    uz = sg._build_messages(ScriptGenerateRequest(topic="t", language=ScriptLanguage.UZ,
                                                  style=ScriptStyle.FOUNDER))
    system = uz[0]["content"]
    assert "O'ZBEK" in system and "TARJIMA EMAS" in system     # native, not translation
    assert "Video editing 3 soat" in system                   # Uzbek few-shot anchor
    assert "build-in-public" in system                        # founder style injected

    en = sg._build_messages(ScriptGenerateRequest(topic="t", language=ScriptLanguage.EN,
                                                  style=ScriptStyle.EDUCATIONAL))
    en_system = en[0]["content"]
    assert "conversational English" in en_system
    assert "Problem → Insight → Framework → Example" in en_system
    assert "Video editing 3 soat" not in en_system            # UZ anchor only for UZ


def test_invalid_json_then_valid(monkeypatch):
    async def fake(messages, *, model, max_tokens):
        fake.n += 1
        return "not json{" if fake.n == 1 else json.dumps(_valid_payload())
    fake.n = 0
    monkeypatch.setattr(sg, "_call_llm", fake)
    res = asyncio.run(sg.generate(ScriptGenerateRequest(topic="x", language=ScriptLanguage.RU)))
    assert res.language is ScriptLanguage.RU and len(res.sections) == 4
