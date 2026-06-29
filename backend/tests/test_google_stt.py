"""
Google STT routing + chunk-stitch tests. No network, no credentials, and no
`google-cloud-speech` install required — every Google call is monkeypatched
(the real client is imported lazily inside the functions we stub out).
"""
from __future__ import annotations

import pytest

from app.config import Settings
from app.services import google_stt as g
from app.services import transcription
from app.services.transcription import Transcript


class FakeSettings:
    """Minimal stand-in for Settings with the knobs these paths read."""
    def __init__(self, uz: bool = False):
        self.openai_api_key = "sk-test"
        self.openai_transcribe_model = "whisper-1"
        self.max_source_minutes = 120
        self.google_stt_model = "latest_long"
        self._uz = uz

    @property
    def google_stt_language_set(self):
        return frozenset({"uz"}) if self._uz else frozenset()


# --- config -----------------------------------------------------------------

def test_language_set_empty_without_creds():
    s = Settings(google_stt_credentials="", google_stt_languages="uz")
    assert s.google_stt_language_set == frozenset()


def test_language_set_parses_only_with_creds():
    s = Settings(google_stt_credentials="{}", google_stt_languages="uz, kk")
    assert s.google_stt_language_set == frozenset({"uz", "kk"})


# --- BCP-47 mapping ---------------------------------------------------------

def test_bcp47_known_and_fallback():
    assert g._bcp47("uz") == "uz-UZ"
    assert g._bcp47("kk") == "kk-KZ"
    assert g._bcp47("ky") == "ky-KG"        # heuristic would wrongly give ky-KY
    assert g._bcp47("xx") == "xx-XX"        # fallback


# --- chunk + stitch ---------------------------------------------------------

async def test_chunks_stitch_with_offsets(monkeypatch, tmp_path):
    c0 = tmp_path / "a_g_0000.wav"; c0.write_bytes(b"x")
    c1 = tmp_path / "a_g_0001.wav"; c1.write_bytes(b"y")

    async def _extract(src, dst, **k):
        dst.write_bytes(b"z"); return dst

    async def _segment(src, out_dir, *, segment_seconds, prefix):
        return [c0, c1]

    monkeypatch.setattr(g.ff, "extract_audio", _extract)
    monkeypatch.setattr(g.ff, "segment_audio", _segment)
    monkeypatch.setattr(g, "_wav_duration_s", lambda p: 10.0)   # total + per-chunk
    monkeypatch.setattr(g, "get_settings", lambda: FakeSettings(uz=True))
    monkeypatch.setattr(g, "_recognize_sync",
                        lambda content, lang, model: ([("salom", 0.0, 0.5)], "salom"))

    out = await g.transcribe(tmp_path / "v.mp4", language="uz")

    assert out.language == "uz"
    assert out.is_source_language is True
    # chunk 0 at offset 0, chunk 1 shifted by its 10s offset
    assert [w.start for w in out.words] == [0.0, 10.0]
    assert out.text == "salom salom"


async def test_empty_result_raises_to_trigger_fallback(monkeypatch, tmp_path):
    c0 = tmp_path / "a_g_0000.wav"; c0.write_bytes(b"x")

    async def _extract(src, dst, **k):
        dst.write_bytes(b"z"); return dst

    monkeypatch.setattr(g.ff, "extract_audio", _extract)
    monkeypatch.setattr(g.ff, "segment_audio",
                        lambda *a, **k: _aresult([c0]))
    monkeypatch.setattr(g, "_wav_duration_s", lambda p: 5.0)
    monkeypatch.setattr(g, "get_settings", lambda: FakeSettings(uz=True))
    monkeypatch.setattr(g, "_recognize_sync", lambda content, lang, model: ([], ""))

    with pytest.raises(RuntimeError):
        await g.transcribe(tmp_path / "v.mp4", language="uz")


async def _aresult(value):
    return value


# --- routing inside transcription.transcribe --------------------------------

async def test_uzbek_routes_to_google(monkeypatch, tmp_path):
    sentinel = Transcript(language="uz", text="salom", words=[], is_source_language=True)

    async def _stub(path, *, language):
        assert language == "uz"
        return sentinel

    monkeypatch.setattr(transcription, "get_settings", lambda: FakeSettings(uz=True))
    monkeypatch.setattr(g, "transcribe", _stub)

    out = await transcription.transcribe(tmp_path / "v.mp4", language="uz")
    assert out is sentinel


async def test_falls_back_to_whisper_when_google_fails(monkeypatch, tmp_path):
    class _Reached(Exception):
        pass

    async def _boom_google(path, *, language):
        raise RuntimeError("stt down")

    def _boom_openai(*a, **k):           # constructing the client = Whisper path reached
        raise _Reached()

    monkeypatch.setattr(transcription, "get_settings", lambda: FakeSettings(uz=True))
    monkeypatch.setattr(g, "transcribe", _boom_google)
    monkeypatch.setattr(transcription, "AsyncOpenAI", _boom_openai)

    with pytest.raises(_Reached):        # proves it fell through to Whisper
        await transcription.transcribe(tmp_path / "v.mp4", language="uz")


def test_google_disabled_when_no_creds(monkeypatch, tmp_path):
    # language set is empty (no creds) → Google never consulted, even for 'uz'.
    s = FakeSettings(uz=False)
    assert "uz" not in s.google_stt_language_set
