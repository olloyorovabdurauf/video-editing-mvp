"""
Compositor: apply a sequence of generated b-roll clips onto a base reel
with cross-dissolves. Calls into utils.ffmpeg.overlay_with_dissolve in
order, each pass producing a new intermediate that becomes the next pass's
base.

We re-encode at each pass (necessary — overlay can't stream-copy), so this
is the second-most-expensive step in the pipeline after generation itself.
Three overlays on a 60s reel takes ~15-25s on a CPU worker.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from loguru import logger

from app.utils import ffmpeg as ff


@dataclass
class CompositeClip:
    """One b-roll insertion. start is RELATIVE to the reel (clip-local), not source."""
    path: Path
    start: float
    duration: float
    dissolve: float = 0.5


async def composite(
    base: Path,
    clips: Sequence[CompositeClip],
    out_dir: Path,
    *,
    name_prefix: str = "comp",
) -> Path:
    if not clips:
        return base

    # Sort so we apply earliest-first; the intermediate pipeline is
    # commutative in result but linear in execution.
    clips_sorted = sorted(clips, key=lambda c: c.start)
    current = base
    for i, clip in enumerate(clips_sorted):
        out = out_dir / f"{name_prefix}_{i}.mp4"
        logger.debug("compositing clip {} at t={}s dur={}s", clip.path.name, clip.start, clip.duration)
        current = await ff.overlay_with_dissolve(
            current, clip.path, out,
            start=clip.start,
            duration=clip.duration,
            dissolve=clip.dissolve,
        )
    return current
