"""
Tests for the crop-expression generator — the reliability-critical part of smart
crop (the bit that used to overflow ffmpeg's parser). Pure math, no ffmpeg/cv2.
"""
from __future__ import annotations

import math

from app.services import smart_crop as sc


def test_static_speaker_returns_constant():
    times = [i * 0.5 for i in range(40)]
    centers = [960.0] * 40                       # dead-centre, never moves
    expr = sc._crop_origin_expr(times, centers, src_dim=1920, crop_dim=608)
    assert "if(" not in expr                      # a single constant, not piecewise
    assert expr.lstrip("-").isdigit()


def test_moving_speaker_caps_keyframes():
    # 200 samples (the old code emitted ~200 nested if()s → ffmpeg blew up).
    times = [i * 0.5 for i in range(200)]
    centers = [960 + 400 * math.sin(i / 8) for i in range(200)]
    expr = sc._crop_origin_expr(times, centers, src_dim=1920, crop_dim=608)
    assert expr.count("if(") <= sc._MAX_KEYFRAMES  # bounded, ffmpeg-safe


def test_origin_clamped_inside_frame():
    # Face pinned to the far right → crop origin clamps to max_origin, face stays in.
    expr = sc._crop_origin_expr([0, 1], [5000, 5000], src_dim=1920, crop_dim=608)
    val = int(expr)
    assert 0 <= val <= 1920 - 608


def test_no_pan_room_returns_zero():
    # crop fills the whole dimension → nothing to pan → constant 0.
    expr = sc._crop_origin_expr([0, 1, 2], [100, 500, 900], src_dim=1080, crop_dim=1080)
    assert expr == "0"


def test_ema_smooths_and_keeps_length():
    raw = [0.0, 100.0, 0.0, 100.0, 0.0]
    out = sc._ema(raw)
    assert len(out) == len(raw)
    # smoothed swings are gentler than the raw 0↔100 jumps
    assert max(abs(out[i + 1] - out[i]) for i in range(len(out) - 1)) < 100


# ---------------------------------------------------------------------------
# Podcast framing: multi-face + subtitle-safe vertical anchor
# ---------------------------------------------------------------------------

from app.services.smart_crop import Face, FrameFaces, _target_series   # noqa: E402


def test_frames_both_speakers_when_they_fit():
    # Two faces 200px apart, crop 608 wide → fit → centre between them (keep both).
    frames = [FrameFaces(t=0.0, faces=[Face(400, 500, 150, 150), Face(600, 500, 150, 150)])]
    _, centers = _target_series(frames, crop_dim=608, axis="x")
    assert centers[0] == 500.0


def test_follows_dominant_speaker_when_far_apart():
    # Faces 900px apart can't both fit → follow the larger (active/nearest) face.
    frames = [FrameFaces(t=0.0, faces=[Face(200, 500, 100, 100), Face(1100, 500, 300, 300)])]
    _, centers = _target_series(frames, crop_dim=608, axis="x")
    assert centers[0] == 1100.0


def test_holds_target_through_frames_with_no_face():
    frames = [
        FrameFaces(t=0.0, faces=[Face(500, 500, 150, 150)]),
        FrameFaces(t=0.5, faces=[]),                       # blink / turn → hold, don't snap
        FrameFaces(t=1.0, faces=[Face(520, 500, 150, 150)]),
    ]
    times, centers = _target_series(frames, crop_dim=608, axis="x")
    assert len(centers) == 3 and centers[1] == 500.0       # held the last target


def test_vertical_anchor_keeps_headroom_and_caption_space():
    # Tall source: anchoring higher (0.40) shifts the window down vs centred (0.50),
    # leaving headroom above the head and caption room below.
    centered = int(sc._crop_origin_expr([0.0, 1.0], [960, 960], src_dim=1920, crop_dim=1080, anchor=0.5))
    biased = int(sc._crop_origin_expr([0.0, 1.0], [960, 960], src_dim=1920, crop_dim=1080, anchor=0.40))
    assert biased > centered
    assert 0 <= biased <= 1920 - 1080
