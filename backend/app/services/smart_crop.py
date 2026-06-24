"""
Speaker-tracking smart crop.

Naive vertical reframe (`crop=W:H:(iw-W)/2:(ih-H)/2`) is the #1 reason
auto-cropped reels look amateurish: the talker walks/turns and exits frame.

What we do
----------
1. Sample the source at 2 fps with OpenCV.
2. Detect the largest face per sample using MediaPipe FaceDetection.
3. Build a smoothed (x, y) trajectory of where the *content* lives.
4. Emit a SINGLE ffmpeg filter pass whose `crop` expression evaluates the
   trajectory over time via chained `if(between(t, ...))` clauses.

That last point is the trick: one ffmpeg pass, no per-frame Python work,
no temp files, and the encoder doesn't run twice. Smooth pan + scan in
~realtime-ish on a CPU worker.

Graceful degradation
--------------------
If mediapipe / opencv aren't installed, we fall back to center-crop — the
pipeline still produces output, just less polished. The hard import is
inside the function so a missing dep doesn't break the rest of the app.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from app.config import get_settings
from app.utils import ffmpeg as ff
from app.utils.ffmpeg import FFmpegCommand, run


@dataclass
class FaceTrack:
    """Smoothed (t, cx, cy) keypoints in SOURCE-pixel coordinates."""
    t: list[float]
    cx: list[float]
    cy: list[float]


def _sample_faces(src: Path, *, sample_fps: float) -> FaceTrack | None:
    """OpenCV + MediaPipe at sample_fps. Returns None if libs missing or no face."""
    try:
        import cv2                                          # type: ignore
        import mediapipe as mp                              # type: ignore
    except ImportError:
        logger.warning("mediapipe/opencv not installed; smart crop will fall back to center")
        return None

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        return None
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames == 0:
        cap.release()
        return None

    step = max(1, int(round(src_fps / sample_fps)))
    detector = mp.solutions.face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5,
    )

    ts: list[float] = []
    xs: list[float] = []
    ys: list[float] = []

    for frame_idx in range(0, total_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        results = detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        t = frame_idx / src_fps
        if not results.detections:
            # No face — fall back to centroid; will be ignored if no other samples land near it
            continue
        # Largest face = highest area
        best = max(results.detections, key=lambda d: d.location_data.relative_bounding_box.width
                                                    * d.location_data.relative_bounding_box.height)
        box = best.location_data.relative_bounding_box
        cx = (box.xmin + box.width / 2) * w
        cy = (box.ymin + box.height / 2) * h
        ts.append(t)
        xs.append(cx)
        ys.append(cy)

    cap.release()
    detector.close()
    if len(ts) < 2:
        return None
    return FaceTrack(t=ts, cx=xs, cy=ys)


def _smooth(values: list[float], window: int = 5) -> list[float]:
    """Symmetric moving average — removes jitter without introducing lag."""
    if window <= 1 or len(values) <= 1:
        return values[:]
    out: list[float] = []
    half = window // 2
    for i in range(len(values)):
        lo, hi = max(0, i - half), min(len(values), i + half + 1)
        out.append(sum(values[lo:hi]) / (hi - lo))
    return out


def _build_crop_expr(samples: FaceTrack, *, src_dim: int, crop_dim: int, axis: str) -> str:
    """
    Build an ffmpeg expression evaluating the smoothed track over time.

    The output is the TOP-LEFT corner of the crop window, clamped so the
    crop stays inside the source frame. Format:

        if(lt(t, t1), v1, if(lt(t, t2), v2 + lerp..., ...))

    For ~30 keyframes this expression is ~600 chars; ffmpeg parses it once,
    evaluates per frame, no measurable overhead.
    """
    centers_raw = samples.cx if axis == "x" else samples.cy
    centers = _smooth(centers_raw, window=5)
    times = samples.t

    half = crop_dim / 2
    max_origin = max(0, src_dim - crop_dim)

    def clamp(c: float) -> float:
        return max(0.0, min(c - half, max_origin))

    # Build piecewise-linear expression: for each pair of keyframes,
    # linearly interpolate between clamped(center_i) and clamped(center_{i+1}).
    expr = f"{clamp(centers[-1]):.1f}"  # default after last keyframe
    for i in range(len(times) - 2, -1, -1):
        t0, t1 = times[i], times[i + 1]
        v0, v1 = clamp(centers[i]), clamp(centers[i + 1])
        slope = (v1 - v0) / max(0.0001, (t1 - t0))
        lerp = f"({v0:.1f}+({slope:.3f})*(t-{t0:.3f}))"
        expr = f"if(lt(t\\,{t1:.3f})\\,{lerp}\\,{expr})"
    return expr


async def smart_crop_to_vertical(
    src: Path,
    dst: Path,
    *,
    target_w: int = 1080,
    target_h: int = 1920,
    crf: int = 20,
    sample_fps: float = 2.0,
) -> Path:
    """
    Speaker-aware crop to 9:16. Falls back to center-crop if no faces found.
    """
    # Probe source dimensions.
    info = await ff.probe(src)
    src_w, src_h = info.width, info.height
    if src_w == 0 or src_h == 0:
        raise ValueError("smart_crop: source has zero dimension")

    # Determine the crop window inside the source: tallest 9:16 that fits.
    target_aspect = target_w / target_h
    if src_w / src_h > target_aspect:
        # Source wider than target → crop horizontally, full height.
        crop_h = src_h
        crop_w = int(round(src_h * target_aspect))
    else:
        # Source taller than target → crop vertically, full width.
        crop_w = src_w
        crop_h = int(round(src_w / target_aspect))

    track = _sample_faces(src, sample_fps=sample_fps)

    if track is None:
        logger.info("smart_crop: no face track, using center crop for {}", src.name)
        x_expr = f"{max(0, (src_w - crop_w) // 2)}"
        y_expr = f"{max(0, (src_h - crop_h) // 2)}"
    else:
        logger.info("smart_crop: tracking {} keyframes over {}", len(track.t), src.name)
        x_expr = _build_crop_expr(track, src_dim=src_w, crop_dim=crop_w, axis="x")
        y_expr = _build_crop_expr(track, src_dim=src_h, crop_dim=crop_h, axis="y")

    filt = (
        f"crop={crop_w}:{crop_h}:{x_expr}:{y_expr},"
        f"scale={target_w}:{target_h}:flags=lanczos,setsar=1"
    )
    cmd = (
        FFmpegCommand()
        .add_input(src)
        .with_filter_complex(filt)
        .add_output_args(
            "-c:v", "libx264", "-preset", get_settings().ffmpeg_preset, "-crf", str(crf),
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart",
        )
        .with_output(dst)
    )
    return await run(cmd)
