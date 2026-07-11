"""
Audio energy analysis — the vocal-delivery signal for the Viral Intelligence
ranking (multimodal signal #2, after pause-based boundaries).

Transcript text can't tell an emphatic, energetic delivery from a monotone
reading of the same words. The per-second loudness curve can — cheaply, on CPU,
straight from the audio we already downloaded. Candidates whose window carries
above-average vocal energy, and especially those whose energy PEAKS in the
final third (the speaker "landing" the message), get a ranking boost; flat,
low-energy windows get a penalty. Deterministic — no extra LLM tokens.
"""
from __future__ import annotations

import struct
import subprocess
from pathlib import Path

from loguru import logger

_RATE = 8000                       # 8kHz mono is plenty for loudness
_BYTES_PER_S = _RATE * 2           # s16le

# Ranking adjustment bounds — a nudge, not a takeover: the semantic scores
# still dominate; energy breaks ties toward the better DELIVERY.
_MAX_BOOST = 0.06
_LANDING_BONUS = 0.04


def energy_curve(path: Path) -> list[float]:
    """Per-second RMS loudness, normalized 0..1 against the file's own peak."""
    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", str(path),
         "-ac", "1", "-ar", str(_RATE), "-f", "s16le", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    rms: list[float] = []
    assert proc.stdout is not None
    while True:
        chunk = proc.stdout.read(_BYTES_PER_S)
        if len(chunk) < 2:
            break
        n = len(chunk) // 2
        samples = struct.unpack(f"<{n}h", chunk[: n * 2])
        rms.append((sum(x * x for x in samples) / n) ** 0.5)
    proc.wait()
    peak = max(rms) or 1.0
    return [v / peak for v in rms]


def window_energy(curve: list[float], start: float, end: float) -> tuple[float, bool]:
    """(mean energy of the window, does energy peak in the final third?)"""
    a, b = int(max(0, start)), int(min(len(curve), end))
    win = curve[a:b]
    if not win:
        return 0.0, False
    mean = sum(win) / len(win)
    peak_at = win.index(max(win))
    return mean, peak_at >= (2 * len(win)) // 3


def rerank(segs: list, curve: list[float]) -> list:
    """Adjust each candidate's overall score by its vocal-delivery energy."""
    if not curve or not segs:
        return segs
    overall_mean = sum(curve) / len(curve)
    out = []
    for s in segs:
        mean, lands = window_energy(curve, s.start, s.end)
        # Relative to the SPEAKER's own baseline, not an absolute level.
        rel = (mean - overall_mean) / (overall_mean or 1.0)
        adj = max(-_MAX_BOOST, min(_MAX_BOOST, rel * _MAX_BOOST * 2))
        if lands:
            adj += _LANDING_BONUS
        if adj:
            s = s.model_copy(update={"score": round(max(0.0, min(1.0, s.score + adj)), 3)})
        out.append(s)
    boosted = sum(1 for a, b in zip(segs, out) if b.score > a.score)
    logger.info("energy rerank: {} of {} candidates boosted by vocal delivery", boosted, len(segs))
    return out
