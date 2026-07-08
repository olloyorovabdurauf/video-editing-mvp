"""
Professional speaker-aware crop (16:9 → 9:16) for podcasts/interviews.

Goals (vs. naive center-crop, the #1 amateur tell):
- Track the ACTIVE speaker and keep their face fully inside the vertical frame.
- Two-speaker podcasts: frame BOTH when they fit, else follow the dominant
  speaker — never crop a face in half by accident.
- Safe margins: the crop is far wider than a face, so the face always keeps side
  padding; for taller-than-16:9 sources we bias the window UP so there's headroom
  above the head and a clean caption zone below the chin.
- Smooth, eased camera motion (EMA + bounded keyframes) — like a human operator,
  not a jump-cut.

Reliability: at most ~20 crop keyframes (ffmpeg expression stays small), a single
constant crop for static shots, and a graceful center-crop fallback if MediaPipe/
OpenCV are missing or no face is found.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

from app.config import get_settings
from app.utils import ffmpeg as ff
from app.utils.ffmpeg import FFmpegCommand, run

_SAMPLE_FPS = 1.5        # face samples per second
_MAX_KEYFRAMES = 20      # crop-expression keyframe cap (ffmpeg-safe)
_EMA_ALPHA = 0.4         # camera smoothing (lower = smoother/slower)
_BOTH_FIT = 0.72         # two faces fit one frame if their span ≤ this × crop width
                         # (0.72 → each outer face keeps ≥14% side margin; wider
                         # spans follow the dominant speaker instead of edge-cropping)
_Y_ANCHOR = 0.40         # subject sits at 40% down → headroom above, captions below


@dataclass
class Face:
    cx: float
    cy: float
    w: float
    h: float

    @property
    def area(self) -> float:
        return self.w * self.h


@dataclass
class FrameFaces:
    t: float
    faces: list[Face] = field(default_factory=list)


def _sample_frames(src: Path, *, sample_fps: float) -> list[FrameFaces] | None:
    """All faces per sampled frame (sequential read, no per-sample seeking)."""
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

    frames: list[FrameFaces] = []
    idx = 0
    try:
        while True:
            if not cap.grab():
                break
            if idx % step == 0:
                ok, frame = cap.retrieve()
                if ok and frame is not None:
                    h, w = frame.shape[:2]
                    res = detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    faces = []
                    for d in (res.detections or []):
                        b = d.location_data.relative_bounding_box
                        faces.append(Face(cx=(b.xmin + b.width / 2) * w,
                                          cy=(b.ymin + b.height / 2) * h,
                                          w=b.width * w, h=b.height * h))
                    frames.append(FrameFaces(t=idx / src_fps, faces=faces))
            idx += 1
    finally:
        cap.release()
        detector.close()

    return frames if any(f.faces for f in frames) else None


def _target_series(frames: list[FrameFaces], *, crop_dim: float, axis: str
                   ) -> tuple[list[float], list[float]]:
    """
    Per-frame target centre for the crop along one axis.

    - 2+ faces that FIT the crop → midpoint (keep both speakers in frame).
    - otherwise → the dominant (largest = nearest/active) face, holding the last
      target through frames with no detection so the camera doesn't snap.
    """
    times: list[float] = []
    centers: list[float] = []
    last: float | None = None
    for fr in frames:
        if not fr.faces:
            if last is not None:
                times.append(fr.t); centers.append(last)
            continue
        coords = sorted((f.cx if axis == "x" else f.cy) for f in fr.faces)
        if len(coords) >= 2 and (coords[-1] - coords[0]) <= crop_dim * _BOTH_FIT:
            tgt = (coords[0] + coords[-1]) / 2.0          # frame both
        else:
            dom = max(fr.faces, key=lambda f: f.area)     # active/nearest speaker
            tgt = dom.cx if axis == "x" else dom.cy
        last = tgt
        times.append(fr.t); centers.append(tgt)
    return times, centers


def _ema(values: list[float], alpha: float = _EMA_ALPHA) -> list[float]:
    """Two-pass EMA — a gliding camera with no jitter and minimal lag."""
    if not values:
        return values
    fwd = [values[0]]
    for v in values[1:]:
        fwd.append(alpha * v + (1 - alpha) * fwd[-1])
    rev = [fwd[-1]]
    for v in reversed(fwd[:-1]):
        rev.append(alpha * v + (1 - alpha) * rev[-1])
    rev.reverse()
    return rev


def _downsample(times: list[float], values: list[float], k: int) -> tuple[list[float], list[float]]:
    n = len(times)
    if n <= k:
        return times, values
    idx = [round(i * (n - 1) / (k - 1)) for i in range(k)]
    return [times[i] for i in idx], [values[i] for i in idx]


def _crop_origin_expr(times: list[float], centers: list[float], *,
                      src_dim: int, crop_dim: int, anchor: float = 0.5) -> str:
    """
    ffmpeg expression for the crop window's top-left origin on one axis.

    `anchor` is where the subject sits inside the crop (0.5 = centred; <0.5 puts
    the subject higher → headroom above + caption room below). Origin is clamped
    so the window — and therefore the face, with its side padding — stays inside
    the frame. Constant for static shots; ≤ _MAX_KEYFRAMES piecewise otherwise.
    """
    if not times:
        return str(max(0, int((src_dim - crop_dim) * anchor)))
    offset = crop_dim * anchor
    max_origin = max(0, src_dim - crop_dim)

    def clamp(c: float) -> int:
        return int(round(max(0.0, min(c - offset, float(max_origin)))))

    times, centers = _downsample(times, _ema(centers), _MAX_KEYFRAMES)
    vals = [clamp(c) for c in centers]

    if max_origin == 0 or (max(vals) - min(vals)) <= max(2, int(0.02 * crop_dim)):
        return str(round(sum(vals) / len(vals)))          # static → constant, bullet-proof

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
    """Speaker-aware crop to 9:16. Falls back to center-crop if no face is found."""
    info = await ff.probe(src)
    src_w, src_h = info.width, info.height
    if src_w == 0 or src_h == 0:
        raise ValueError("smart_crop: source has zero dimension")

    target_aspect = target_w / target_h
    if src_w / src_h > target_aspect:        # wider than 9:16 → pan horizontally, full height
        crop_h = src_h
        crop_w = int(round(src_h * target_aspect))
    else:                                    # taller → pan vertically, full width
        crop_w = src_w
        crop_h = int(round(src_w / target_aspect))
    crop_w, crop_h = min(crop_w, src_w), min(crop_h, src_h)

    frames = _sample_frames(src, sample_fps=sample_fps)
    if not frames:
        logger.info("smart_crop: no faces → center crop for {}", src.name)
        x_expr = str(max(0, (src_w - crop_w) // 2))
        y_expr = str(max(0, (src_h - crop_h) // 2))
    else:
        n_faces = max((len(f.faces) for f in frames), default=0)
        logger.info("smart_crop: tracking {} (≤{} faces) over {}", n_faces, n_faces, src.name)
        tx, cx = _target_series(frames, crop_dim=crop_w, axis="x")
        x_expr = _crop_origin_expr(tx, cx, src_dim=src_w, crop_dim=crop_w, anchor=0.5)
        if crop_h < src_h:                   # vertical room exists → bias up for headroom + captions
            ty, cy = _target_series(frames, crop_dim=crop_h, axis="y")
            y_expr = _crop_origin_expr(ty, cy, src_dim=src_h, crop_dim=crop_h, anchor=_Y_ANCHOR)
        else:
            y_expr = "0"                     # full-height crop (16:9 source) — nothing to pan

    filt = (
        f"crop={crop_w}:{crop_h}:{x_expr}:{y_expr},"
        f"scale={target_w}:{target_h}:flags=lanczos,setsar=1"
    )
    cmd = (
        FFmpegCommand()
        .add_input(src)
        .with_filter_complex(filt)
        .add_output_args(
            *ff.video_encoder_args(crf),
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart",
        )
        .with_output(dst)
    )
    return await run(cmd)
