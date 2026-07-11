"""
Launch-blocker tests: teaser preservation, energy re-ranking, payoff protection.
"""
from __future__ import annotations

from app.schemas.reel import Segment
from app.services import audio_energy
from app.services.segment_picker import _finalize_window
from app.services.transcription import Word


def _words(n=200, sent_every=10):
    return [Word(text=("stop." if i % sent_every == sent_every - 1 else "w"),
                 start=float(i), end=i + 0.9) for i in range(n)]


# --- Teaser preservation -----------------------------------------------------

def test_teaser_pulled_back_to_story_block_start():
    """Pick starts 8s after the story opens (mid-teaser) → snapped BACK to the
    block opening so hook + open-loop stay in the clip."""
    words = _words()
    win = _finalize_window(words, 48.0, 100.0, 45, 75, block_starts=[40.0])
    assert win is not None
    assert win[0] <= 40.1                     # recovered the teaser


def test_distant_block_start_is_not_dragged_in():
    """A block that opened 40s earlier is a DIFFERENT story — no drag-back."""
    words = _words()
    win = _finalize_window(words, 80.0, 130.0, 45, 75, block_starts=[40.0])
    assert win is not None
    assert win[0] >= 75.0                     # stayed near the requested start


def test_payoff_end_still_honored_with_blocks():
    """Block snapping must not break meaning-first endings."""
    words = _words()
    win = _finalize_window(words, 48.0, 108.0, 45, 75, block_starts=[40.0])
    assert win is not None
    assert win[1] >= 105.0                    # ends at the idea's end, not early


# --- Viral Intelligence: vocal-energy re-ranking ------------------------------

def _seg(start, end, score):
    return Segment(start=float(start), end=float(end), transcript="", score=score)


def test_energetic_delivery_outranks_monotone():
    # 200s: flat 0.3 everywhere except an emphatic 0.9 stretch at 100-150s.
    curve = [0.3] * 200
    for i in range(100, 150):
        curve[i] = 0.9
    flat = _seg(0, 50, 0.70)
    emphatic = _seg(100, 150, 0.70)           # same semantic score
    out = audio_energy.rerank([flat, emphatic], curve)
    assert out[1].score > out[0].score        # delivery breaks the tie


def test_landing_peak_gets_bonus():
    curve = [0.4] * 100
    for i in range(40, 50):                   # clip 0..50 peaks in final third
        curve[i] = 1.0
    seg = _seg(0, 50, 0.70)
    out = audio_energy.rerank([seg], curve)
    assert out[0].score > 0.70                # emotional landing rewarded


def test_energy_is_a_nudge_not_a_takeover():
    curve = [1.0 if i < 50 else 0.01 for i in range(100)]
    strong_story = _seg(50, 100, 0.90)        # weak delivery, strong story
    weak_story = _seg(0, 50, 0.60)            # loud delivery, weak story
    out = audio_energy.rerank([strong_story, weak_story], curve)
    assert out[0].score > out[1].score        # semantics still dominate
