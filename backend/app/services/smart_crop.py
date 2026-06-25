"""
Speaker-tracking smart crop (16:9 → 9:16).

Naive center-crop is the #1 reason auto-reels look amateurish: the talker turns
or sits off-centre and gets half-cropped. We track the active speaker's face and
pan the crop window to keep them framed, like a human editor would.

How it works
------------
1. Sample the clip at ~1.5 fps, reading FRAMES SEQUENTIALLY (grab/skip) — never
   seeking per sample, which is slow and flaky.
2. Detect the largest face per sample (MediaPipe).
3. Smooth the trajectory (EMA) so the camera glides, never jitters.
4. Down-sample to at most ~20 keyframes and emit ONE ffmpeg `crop` pass whose x
   position interpolates between them. If the speaker barely moves, emit a single
   CONSTANT crop instead — cleaner and bullet-proof.

Why the keyframe cap matters
----------------------------
The old version emitted one `if()` per sample (90-120 for a 60s clip), which
overflowed ffmpeg's expression parser → "Error reinitializing filters" and the
clip fell back to center-crop. Capping at ~20 keyframes keeps the expression
small and reliable while still tracking the speaker smoothly.

Graceful degradation: if mediapipe/opencv are missing or no face is found, the
caller center-crops. Imports are local so a missing dep never breaks the app.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from app.config import get_settings
from app.utils import ffmpeg as ff
from app.utils.ffmpeg import FFmpegCommand, run

# Tracking/quality knobs.
_SAMPLE_FPS = 1.5        # face samples per second — enough to track a talking head
_MAX_KEYFRAMES = 20      # hard cap on crop-expression keyframes (ffmpeg-safe)
_EMA_ALPHA = 0.4         # trajectory smoothing (lower = smoother/slower camera)


@dataclass
class FaceTrack:
    """(t, cx, cy) face-centre keypoints in SOURCE-pixel coordinates."""
    t: list[float]
    cx: list[float]
    cy: list[float]


def _sample_faces(src: Path, *, sample_fps: float) -> FaceTrack | None:
    """MediaPipe face centres sampled sequentially. None if libs missing / no face."""
    try:
        import cv2                                          # type: ignore
        import mediapipe as mp                              # type: ignore
    except ImportError:
        logger.warning("mediapipe/opencv not installed; smart crop falls back to center")
        return None

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        return None
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(src_fps / sample_fps)))
    detector = mp.solutions.face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5)

    ts: list[float] = []
    xs: list[float] = []
    ys: list[float] = []
    idx = 0
    try:
        while True:
            ok = cap.grab()                                 # cheap: no decode
            if not ok:
                break
            if idx % step == 0:
                ok, frame = cap.retrieve()                  # decode only sampled frames
                if ok and frame is not None:
                    h, w = frame.shape[:2]
                    res = detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    if res.detections:
                        best = max(res.detections,
                                   key=lambda d: d.location_data.relative_bounding_box.width
                                   * d.location_data.relative_bounding_box.height)
                        box = best.location_data.relative_bounding_box
                        ts.append(idx / src_fps)
                        xs.append((box.xmin + box.width / 2) * w)
                        ys.append((box.ymin + box.height / 2) * h)
            idx += 1
    finally:
        cap.release()
        detector.close()

    if len(ts) < 2:
        return None
    return FaceTrack(t=ts, cx=xs, cy=ys)


def _ema(values: list[float], alpha: float = _EMA_ALPHA) -> list[float]:
    """Exponential moving average — a gliding camera, no jitter, minimal lag."""
    if not values:
        return values
    out = [values[0]]
    for v in values[1:]:
        out.append(alpha * v + (1 - alpha) * out[-1])
    # second (reverse) pass cancels the EMA's lag so motion stays centred on the face
    rev = [out[-1]]
    for v in reversed(out[:-1]):
        rev.append(alpha * v + (1 - alpha) * rev[-1])
    rev.reverse()
    return rev


def _downsample(times: list[float], values: list[float], k: int) -> tuple[list[float], list[float]]:
    """Evenly pick at most k (t, value) keyframes."""
    n = len(times)
    if n <= k:
        return times, values
    idx = [round(i * (n - 1) / (k - 1)) for i in range(k)]
    return [times[i] for i in idx], [values[i] for i in idx]


def _crop_origin_expr(times: list[float], centers: list[float], *,
                      src_dim: int, crop_dim: int) -> str:
    """
    ffmpeg expression for the crop window's top-left origin along one axis.

    Centres the window on the (smoothed) face and clamps it inside the frame so
    the face is never half-cut. Emits a CONSTANT when the speaker is ~static
    (most podcasts), else a short piecewise-linear interpolation (<= _MAX_KEYFRAMES).
    """
    half = crop_dim / 2
    max_origin = max(0, src_dim - crop_dim)

    def clamp(c: float) -> int:
        return int(round(max(0.0, min(c - half, float(max_origin)))))

    times, centers = _downsample(times, _ema(centers), _MAX_KEYFRAMES)
    vals = [clamp(c) for c in centers]

    # Static shot (or no room to pan) → single constant origin: clean + reliable.
    if max_origin == 0 or (max(vals) - min(vals)) <= max(2, int(0.02 * crop_dim)):
        return str(round(sum(vals) / len(vals)))

    expr = f"{vals[-1]}"
    for i in range(len(times) - 2, -1, -1):
        t0, t1 = times[i], times[i + 1]
        v0, v1 = vals[i], vals[i + 1]
        slope = (v1 - v0) / max(0.0001, (t1 - t0))
        lerp = f"({v0}+({slope:.2f})*(t-{t0:.2f}))"
        expr = f"if(lt(t\\,{t1:.2f})\\,{lerp}\\,{expr})"
    return expr


async def smart_crop_to_vertical(
    src: Path,
    dst: Path,
    *,
    target_w: int = 1080,
    target_h: int = 1920,
    crf: int = 20,
    sample_fps: float = _SAMPLE_FPS,
) -> Path:
    """Speaker-aware crop to 9:16. Falls back to center-crop when no face is found."""
    info = await ff.probe(src)
    src_w, src_h = info.width, info.height
    if src_w == 0 or src_h == 0:
        raise ValueError("smart_crop: source has zero dimension")

    # The tallest 9:16 window that fits the source.
    target_aspect = target_w / target_h
    if src_w / src_h > target_aspect:        # wider than 9:16 → pan horizontally, full height
        crop_h = src_h
        crop_w = int(round(src_h * target_aspect))
    else:                                    # taller → pan vertically, full width
        crop_w = src_w
        crop_h = int(round(src_w / target_aspect))
    crop_w = min(crop_w, src_w)
    crop_h = min(crop_h, src_h)

    track = _sample_faces(src, sample_fps=sample_fps)
    if track is None:
        logger.info("smart_crop: no face track → center crop for {}", src.name)
        x_expr = str(max(0, (src_w - crop_w) // 2))
        y_expr = str(max(0, (src_h - crop_h) // 2))
    else:
        logger.info("smart_crop: tracking face over {} ({} samples)", src.name, len(track.t))
        x_expr = _crop_origin_expr(track.t, track.cx, src_dim=src_w, crop_dim=crop_w)
        y_expr = _crop_origin_expr(track.t, track.cy, src_dim=src_h, crop_dim=crop_h)

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
