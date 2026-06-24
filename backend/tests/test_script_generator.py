"""
Tests for the Reels script generator. The only network seam (`_call_llm`) is
stubbed, so these are fast and offline. We assert the structural guarantees that
make the feature a *fix* for thin reels: 4 beats, full duration, hook payoff,
native-language prompting, content-type adaptation (no forced founder voice),
and the exact output format.
"""
from __future__ import annotations

import asyncio
import json

import pytest

from app.schemas.script import (
    ContentType,
    Industry,
    ScriptGenerateRequest,
    ScriptLanguage,
)
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
            "body": "Creatorlarning eng katta muammosi — montaj.",
            "cta": "AI video editing kelajagini kuzatib boring.",
        },
    }


def _short_payload() -> dict:
    """All four beats present but compressed to 22s (< the 60s target's 48s floor)."""
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


def _req(**kw) -> ScriptGenerateRequest:
    base = dict(topic="x", language=ScriptLanguage.UZ)
    base.update(kw)
    return ScriptGenerateRequest(**base)


def test_generates_valid_script(monkeypatch):
    monkeypatch.setattr(sg, "_call_llm", _stub_returning(_valid_payload()))
    res = asyncio.run(sg.generate(_req(content_type=ContentType.EDUCATIONAL,
                                       industry=Industry.TECHNOLOGY, duration_seconds=60)))
    assert res.language is ScriptLanguage.UZ
    assert res.content_type is ContentType.EDUCATIONAL
    assert res.industry is Industry.TECHNOLOGY
    assert [s.name for s in res.sections] == ["hook", "problem", "value", "payoff"]
    assert "subtitr" in res.script
    rendered = res.caption.render(res.hashtags)
    assert "#montaj" in rendered and "##" not in rendered


def test_default_is_not_founder():
    # Default content type must be educational, and the prompt must NOT inject
    # founder/build-in-public voice unless explicitly chosen.
    req = ScriptGenerateRequest(topic="t")
    assert req.content_type is ContentType.EDUCATIONAL
    system = sg._build_messages(req)[0]["content"]
    assert "build-in-public" not in system.lower()
    assert "educational" in system.lower()


def test_founder_voice_only_when_selected():
    system = sg._build_messages(_req(content_type=ContentType.FOUNDER_STORY))[0]["content"]
    assert "build-in-public" in system.lower()
    sales = sg._build_messages(_req(content_type=ContentType.SALES))[0]["content"]
    assert "conversion" in sales.lower() and "build-in-public" not in sales.lower()


def test_output_format_block(monkeypatch):
    monkeypatch.setattr(sg, "_call_llm", _stub_returning(_valid_payload()))
    res = asyncio.run(sg.generate(_req()))
    out = res.formatted
    for header in ("TITLE:", "HOOK:", "SCRIPT:", "SCENE BREAKDOWN:", "CAPTION:", "LANGUAGE:"):
        assert header in out
    assert "[0-5s] HOOK" in out
    assert out.rstrip().endswith("uz")


def test_retries_then_succeeds_on_short_draft(monkeypatch):
    stub = _stub_returning(_short_payload(), _valid_payload())
    monkeypatch.setattr(sg, "_call_llm", stub)
    res = asyncio.run(sg.generate(_req(duration_seconds=60)))
    assert res.duration_seconds == 60
    assert stub.calls["n"] == 2


def test_raises_after_two_bad_drafts(monkeypatch):
    monkeypatch.setattr(sg, "_call_llm", _stub_returning(_short_payload(), _short_payload()))
    with pytest.raises(sg.ScriptGenerationError):
        asyncio.run(sg.generate(_req(duration_seconds=60)))


def test_sections_reordered_to_canonical(monkeypatch):
    payload = _valid_payload()
    payload["sections"].reverse()
    monkeypatch.setattr(sg, "_call_llm", _stub_returning(payload))
    res = asyncio.run(sg.generate(_req()))
    assert [s.name for s in res.sections] == ["hook", "problem", "value", "payoff"]


def test_prompt_native_language_and_industry():
    uz = sg._build_messages(_req(language=ScriptLanguage.UZ, industry=Industry.FINANCE))[0]["content"]
    assert "O'ZBEK" in uz and "TARJIMA EMAS" in uz       # native, not translation
    assert "Caption FORMATI" in uz                        # UZ caption-format anchor
    assert "finance" in uz.lower()                        # industry flows in

    en = sg._build_messages(ScriptGenerateRequest(
        topic="t", language=ScriptLanguage.EN, content_type=ContentType.TUTORIAL))[0]["content"]
    assert "conversational English" in en
    assert "Caption FORMATI" not in en                    # UZ anchor only for UZ
    assert "tutorial" in en.lower()


def test_timeline_scales_with_duration():
    assert sg._timeline(60) == (5, 15, 45)
    assert sg._min_total(60) == 48
    # 30s reel: shorter beats, lower floor — still four beats, not a 15s stub.
    h, p, v = sg._timeline(30)
    assert h == 4 and p < v < 30
    assert sg._min_total(30) == 24


def test_30s_script_passes(monkeypatch):
    p = _valid_payload()
    p["sections"] = [
        {"name": "hook", "start_s": 0, "end_s": 4, "voiceover": "Hook", "visual": "v"},
        {"name": "problem", "start_s": 4, "end_s": 9, "voiceover": "Problem", "visual": "v"},
        {"name": "value", "start_s": 9, "end_s": 23, "voiceover": "Value deeply explained", "visual": "v"},
        {"name": "payoff", "start_s": 23, "end_s": 30, "voiceover": "Payoff + CTA", "visual": "v"},
    ]
    monkeypatch.setattr(sg, "_call_llm", _stub_returning(p))
    res = asyncio.run(sg.generate(_req(duration_seconds=30)))
    assert res.duration_seconds == 30 and len(res.sections) == 4


def test_invalid_json_then_valid(monkeypatch):
    async def fake(messages, *, model, max_tokens):
        fake.n += 1
        return "not json{" if fake.n == 1 else json.dumps(_valid_payload())
    fake.n = 0
    monkeypatch.setattr(sg, "_call_llm", fake)
    res = asyncio.run(sg.generate(_req(language=ScriptLanguage.RU)))
    assert res.language is ScriptLanguage.RU and len(res.sections) == 4
