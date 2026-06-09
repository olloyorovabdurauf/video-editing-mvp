"""
Eval-harness math.

Pure functions, deterministic. If these drift, our quality reports lie.
"""
from __future__ import annotations

from eval.run_benchmark import _iou, _score_picks
from app.schemas.reel import Segment


def _seg(start, end, hook=0.8):
    return Segment(start=start, end=end, hook_score=hook, reason="t", transcript="t")


def test_iou_identical_intervals():
    assert _iou((10, 20), (10, 20)) == 1.0


def test_iou_disjoint_intervals():
    assert _iou((0, 5), (10, 20)) == 0.0


def test_iou_partial_overlap():
    # |10..15| ∩ |12..17| = 3, ∪ = 7 → 3/7
    assert abs(_iou((10, 15), (12, 17)) - (3 / 7)) < 1e-9


def test_score_picks_perfect_match():
    predicted = [_seg(100, 120, hook=0.9), _seg(200, 230, hook=0.85)]
    golden = [(100, 120), (200, 230)]
    s = _score_picks(predicted, golden)
    assert s["matched"] == 2
    assert s["mean_iou"] == 1.0
    assert s["hook_at_matched"] > 0.8


def test_score_picks_partial_match():
    predicted = [
        _seg(100, 120, hook=0.9),   # matches golden 1
        _seg(500, 520, hook=0.4),   # no match
    ]
    golden = [(100, 120), (200, 230)]
    s = _score_picks(predicted, golden)
    assert s["matched"] == 1
    assert s["hook_at_unmatched"] == 0.4
    assert s["hook_at_matched"] == 0.9


def test_score_picks_no_predictions():
    s = _score_picks([], [(0, 10)])
    assert s["matched"] == 0
    assert s["mean_iou"] == 0.0
