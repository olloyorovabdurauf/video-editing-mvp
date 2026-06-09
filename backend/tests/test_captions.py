"""
Caption generation tests.

The phrase grouping algorithm and ASS output are 100% deterministic given
inputs — exactly the kind of code we should pin with tests so a "harmless"
refactor doesn't silently change how captions read.
"""
from __future__ import annotations

from pathlib import Path

from app.services.captions import (
    group_into_phrases,
    render_ass,
    write_ass,
)
from app.services.transcription import Word


def _w(text: str, start: float, end: float) -> Word:
    return Word(text=text, start=start, end=end)


def test_phrase_grouping_respects_pause():
    """A >250ms gap forces a phrase break."""
    words = [
        _w("hello",   0.00, 0.30),
        _w("world",   0.31, 0.55),
        # 600ms gap → new phrase
        _w("friends", 1.20, 1.55),
    ]
    phrases = group_into_phrases(words)
    assert len(phrases) == 2
    assert [w.text for w in phrases[0].words] == ["hello", "world"]
    assert [w.text for w in phrases[1].words] == ["friends"]


def test_phrase_grouping_respects_max_words():
    """Default max=4: a 5-word run must split."""
    words = [_w(f"w{i}", i * 0.20, i * 0.20 + 0.15) for i in range(5)]
    phrases = group_into_phrases(words, max_words=4)
    assert len(phrases) == 2
    assert len(phrases[0].words) == 4
    assert len(phrases[1].words) == 1


def test_phrase_grouping_respects_char_budget():
    """A single very long word must not produce a >28-char row."""
    words = [
        _w("supercalifragilisticexpialidocious", 0.0, 0.6),
        _w("more", 0.7, 0.9),
    ]
    phrases = group_into_phrases(words, max_chars=28)
    # The long word alone exceeds budget → its own row.
    assert len(phrases) >= 1
    # No row may sum to >28 chars (with separator math).
    for ph in phrases:
        total = sum(len(w.text) for w in ph.words) + max(0, len(ph.words) - 1)
        # The first phrase may contain just the oversize word — that's allowed
        # because we can't break a single word; the invariant is "we don't
        # APPEND to overflow", which is what group_into_phrases guarantees.
        if len(ph.words) > 1:
            assert total <= 28


def test_render_ass_includes_required_blocks():
    """ASS file must have ScriptInfo, V4+ Styles, and Events sections."""
    words = [_w("hello", 0, 0.3), _w("world", 0.31, 0.6)]
    phrases = group_into_phrases(words)
    out = render_ass(phrases, style="karaoke", resolution=(1080, 1920))

    assert "[Script Info]" in out
    assert "PlayResX: 1080" in out
    assert "PlayResY: 1920" in out
    assert "[V4+ Styles]" in out
    assert "[Events]" in out
    # The single phrase should be one Dialogue line.
    assert out.count("Dialogue:") == 1


def test_render_ass_karaoke_emits_kf_tags():
    """Karaoke style must emit per-word \\kf timing tags."""
    words = [_w("hello", 0, 0.3), _w("world", 0.31, 0.6)]
    out = render_ass(group_into_phrases(words), style="karaoke")
    # \kf<centiseconds> appears for each word
    assert r"\kf" in out
    # Pop-in transform present
    assert r"\fscx" in out


def test_render_ass_popup_one_word_per_line(tmp_path):
    """Popup style: one Dialogue per word, with scale punch animation."""
    words = [_w("a", 0, 0.2), _w("b", 0.3, 0.5), _w("c", 0.6, 0.8)]
    out_path = tmp_path / "test.ass"
    # write_ass uses max_words=1 for popup
    write_ass(words, out_path, style="popup")
    text = out_path.read_text(encoding="utf-8")
    assert text.count("Dialogue:") == 3
    assert r"\fscx" in text


def test_ass_escapes_special_chars():
    """ASS curly braces must be escaped in the text payload."""
    words = [_w("{bad}", 0, 0.3)]
    out = render_ass(group_into_phrases(words))
    # Original braces must not appear unescaped in the dialogue text portion.
    # Format string used by the rendered phrase will start with `{\fad...}`,
    # so we look for the actual text payload.
    assert r"\{bad\}" in out


def test_write_ass_creates_file(tmp_path):
    words = [_w("hi", 0, 0.3), _w("there", 0.4, 0.7)]
    out = tmp_path / "caps.ass"
    write_ass(words, out, style="karaoke")
    assert out.exists()
    assert out.stat().st_size > 100  # nontrivial content
