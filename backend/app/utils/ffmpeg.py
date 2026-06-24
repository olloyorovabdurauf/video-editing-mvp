"""
FFmpeg utility layer.

Design goals
------------
1. **Typed, composable command builder** — no f-string spaghetti, no shell injection.
2. **Streaming progress** — parse ffmpeg stderr in real time so workers can report % done.
3. **Idempotent ops** — same inputs + same params → same output filename (content-hashed).
4. **Hardware acceleration aware** — opt-in via settings, falls back gracefully.
5. **Pure async** — never blocks the Celery worker's event loop / process beyond the child.

Everything that touches ffmpeg in this codebase MUST go through this module.
Direct subprocess calls to ffmpeg elsewhere are a code-review-blocking offense.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from loguru import logger

from app.config import get_settings

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class FFmpegError(RuntimeError):
    """ffmpeg/ffprobe exited non-zero."""

    def __init__(self, returncode: int, cmd: Sequence[str], stderr: str):
        self.returncode = returncode
        self.cmd = list(cmd)
        self.stderr = stderr
        super().__init__(
            f"ffmpeg exited {returncode}: {' '.join(shlex.quote(c) for c in cmd)}\n"
            f"---- stderr (last 2KB) ----\n{stderr[-2048:]}"
        )


# ---------------------------------------------------------------------------
# Probe (ffprobe wrapper)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MediaInfo:
    """Subset of ffprobe output we actually use downstream."""

    path: Path
    duration: float          # seconds
    width: int
    height: int
    fps: float
    has_audio: bool
    video_codec: str
    audio_codec: str | None
    bitrate: int | None      # bits/sec, may be None for some containers

    @property
    def aspect(self) -> float:
        return self.width / self.height if self.height else 0.0

    @property
    def is_portrait(self) -> bool:
        return self.height > self.width


async def probe(path: Path | str) -> MediaInfo:
    """Run ffprobe and parse a minimal, opinionated MediaInfo."""
    settings = get_settings()
    path = Path(path)
    cmd = [
        settings.ffprobe_binary,
        "-v", "error",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    out, err = await proc.communicate()
    if proc.returncode != 0:
        raise FFmpegError(proc.returncode or -1, cmd, err.decode("utf-8", "replace"))

    data = json.loads(out.decode("utf-8"))
    fmt = data.get("format", {})
    streams = data.get("streams", [])
    v = next((s for s in streams if s["codec_type"] == "video"), None)
    a = next((s for s in streams if s["codec_type"] == "audio"), None)
    if v is None:
        raise FFmpegError(0, cmd, "no video stream found")

    # fps may be "30000/1001" — eval safely
    num, _, den = v.get("r_frame_rate", "0/1").partition("/")
    fps = (float(num) / float(den)) if den and float(den) else 0.0

    return MediaInfo(
        path=path,
        duration=float(fmt.get("duration", 0.0)),
        width=int(v.get("width", 0)),
        height=int(v.get("height", 0)),
        fps=fps,
        has_audio=a is not None,
        video_codec=v.get("codec_name", "unknown"),
        audio_codec=(a.get("codec_name") if a else None),
        bitrate=(int(fmt["bit_rate"]) if fmt.get("bit_rate") else None),
    )


# ---------------------------------------------------------------------------
# Command builder
# ---------------------------------------------------------------------------


@dataclass
class FFmpegCommand:
    """
    Fluent ffmpeg builder. All paths/arguments are passed argv-style — never
    shell-interpolated — so user-supplied filenames or filter strings cannot
    inject options.
    """

    inputs: list[tuple[Path, list[str]]] = field(default_factory=list)
    global_args: list[str] = field(default_factory=list)
    output_args: list[str] = field(default_factory=list)
    filter_complex: str | None = None
    output: Path | None = None
    overwrite: bool = True

    # ---- builder methods ----

    def add_input(self, path: Path | str, *opts: str) -> "FFmpegCommand":
        self.inputs.append((Path(path), list(opts)))
        return self

    def with_filter_complex(self, expr: str) -> "FFmpegCommand":
        self.filter_complex = expr
        return self

    def with_output(self, path: Path | str, *opts: str) -> "FFmpegCommand":
        self.output = Path(path)
        self.output_args.extend(opts)
        return self

    def add_output_args(self, *opts: str) -> "FFmpegCommand":
        self.output_args.extend(opts)
        return self

    def add_global_args(self, *opts: str) -> "FFmpegCommand":
        self.global_args.extend(opts)
        return self

    # ---- realize ----

    def build(self) -> list[str]:
        settings = get_settings()
        if self.output is None:
            raise ValueError("FFmpegCommand: output not set")
        cmd: list[str] = [settings.ffmpeg_binary, "-hide_banner", "-nostdin"]
        if self.overwrite:
            cmd.append("-y")
        if settings.ffmpeg_threads:
            cmd += ["-threads", str(settings.ffmpeg_threads)]
        if settings.ffmpeg_hwaccel:
            cmd += ["-hwaccel", settings.ffmpeg_hwaccel]
        cmd += self.global_args
        for path, opts in self.inputs:
            cmd += opts + ["-i", str(path)]
        if self.filter_complex:
            cmd += ["-filter_complex", self.filter_complex]
        # progress on stdout so we can parse without fighting stderr
        cmd += ["-progress", "pipe:1", "-nostats"]
        cmd += self.output_args
        cmd.append(str(self.output))
        return cmd

    def cache_key(self, extra: str = "") -> str:
        """Content-addressable hash of the command — good for output naming."""
        h = hashlib.sha256()
        for p, opts in self.inputs:
            try:
                stat = p.stat()
                h.update(f"{p}:{stat.st_size}:{int(stat.st_mtime)}".encode())
            except FileNotFoundError:
                h.update(str(p).encode())
            h.update("|".join(opts).encode())
        h.update((self.filter_complex or "").encode())
        h.update("|".join(self.output_args).encode())
        h.update(extra.encode())
        return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Runner with streaming progress
# ---------------------------------------------------------------------------

_PROGRESS_LINE = re.compile(r"^(?P<key>[^=]+)=(?P<value>.*)$")


@dataclass
class Progress:
    out_time_us: int = 0   # microseconds processed
    frame: int = 0
    fps: float = 0.0
    speed: float = 0.0     # 1.0 = realtime
    done: bool = False

    @property
    def out_time_s(self) -> float:
        return self.out_time_us / 1_000_000


async def run(
    cmd: FFmpegCommand | Sequence[str],
    *,
    timeout: float | None = None,
    on_progress: Callable[[Progress], None] | None = None,
    total_duration: float | None = None,
) -> Path:
    """
    Run an ffmpeg command, streaming progress. Returns the output Path.

    Parameters
    ----------
    cmd
        Either a built FFmpegCommand or a raw argv list (escape hatch).
    timeout
        Hard wall-clock kill, in seconds.
    on_progress
        Called with a `Progress` snapshot whenever ffmpeg flushes a progress block.
        If `total_duration` is provided, the caller can derive percentage.
    total_duration
        Stream duration in seconds. If None and `cmd` is FFmpegCommand with one
        input, we'll probe it. Pass explicitly to skip the probe.
    """
    if isinstance(cmd, FFmpegCommand):
        argv = cmd.build()
        output = cmd.output
        # auto-probe if caller wants progress but didn't supply duration
        if on_progress and total_duration is None and len(cmd.inputs) == 1:
            try:
                total_duration = (await probe(cmd.inputs[0][0])).duration
            except FFmpegError:
                total_duration = None
    else:
        argv = list(cmd)
        output = None

    logger.debug("ffmpeg: {}", " ".join(shlex.quote(a) for a in argv))

    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    progress = Progress()
    stderr_tail: list[str] = []

    async def pump_stdout() -> None:
        assert proc.stdout is not None
        async for raw in proc.stdout:
            line = raw.decode("utf-8", "replace").strip()
            m = _PROGRESS_LINE.match(line)
            if not m:
                continue
            key, value = m["key"], m["value"]
            if key == "out_time_us" and value.isdigit():
                progress.out_time_us = int(value)
            elif key == "frame" and value.isdigit():
                progress.frame = int(value)
            elif key == "fps":
                try: progress.fps = float(value)
                except ValueError: pass
            elif key == "speed" and value.endswith("x"):
                try: progress.speed = float(value[:-1])
                except ValueError: pass
            elif key == "progress":
                progress.done = value == "end"
                if on_progress:
                    on_progress(progress)

    async def pump_stderr() -> None:
        assert proc.stderr is not None
        async for raw in proc.stderr:
            line = raw.decode("utf-8", "replace").rstrip()
            stderr_tail.append(line)
            if len(stderr_tail) > 400:  # bounded ring
                stderr_tail.pop(0)

    try:
        await asyncio.wait_for(
            asyncio.gather(pump_stdout(), pump_stderr(), proc.wait()),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise FFmpegError(-1, argv, "\n".join(stderr_tail) + "\n[timeout]")

    if proc.returncode != 0:
        raise FFmpegError(proc.returncode or -1, argv, "\n".join(stderr_tail))

    if output and not output.exists():
        raise FFmpegError(0, argv, "ffmpeg returned 0 but output file missing")
    return output  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# High-level operations (composed from the primitives above)
# ---------------------------------------------------------------------------


async def extract_audio(src: Path, dst: Path, *, sample_rate: int = 16_000) -> Path:
    """Mono 16k PCM WAV — what Whisper wants. Cheap and lossless to re-derive."""
    cmd = (
        FFmpegCommand()
        .add_input(src)
        .add_output_args("-vn", "-ac", "1", "-ar", str(sample_rate), "-f", "wav")
        .with_output(dst)
    )
    return await run(cmd)


async def segment_audio(
    src: Path, out_dir: Path, *, segment_seconds: int, prefix: str = "chunk"
) -> list[Path]:
    """
    Split a WAV into fixed-duration chunks (for Whisper's 25MB upload limit).
    PCM stream-copy → sample-accurate, near-instant. Returns chunks in order.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(out_dir / f"{prefix}_%04d.wav")
    settings = get_settings()
    argv = [
        settings.ffmpeg_binary, "-hide_banner", "-nostdin", "-y",
        "-i", str(src),
        "-f", "segment",
        "-segment_time", str(segment_seconds),
        "-c", "copy",
        "-reset_timestamps", "1",
        pattern,
    ]
    await run(argv)  # raw argv: run skips the single-output existence check
    return sorted(out_dir.glob(f"{prefix}_*.wav"))


async def cut(
    src: Path,
    dst: Path,
    *,
    start: float,
    end: float,
    reencode: bool = False,
) -> Path:
    """
    Hard-cut a segment. Stream-copy by default (fast, but cuts at keyframes).
    Set `reencode=True` for frame-accurate cuts at the cost of CPU.
    """
    duration = max(0.0, end - start)
    builder = (
        FFmpegCommand()
        # -ss BEFORE -i = fast seek (keyframe). -ss AFTER -i = accurate but slow.
        .add_input(src, "-ss", f"{start:.3f}")
        .add_output_args("-t", f"{duration:.3f}")
        .with_output(dst)
    )
    if reencode:
        builder.add_output_args("-c:v", "libx264", "-preset", get_settings().ffmpeg_preset,
                                "-crf", "20", "-c:a", "aac", "-b:a", "128k")
    else:
        builder.add_output_args("-c", "copy", "-avoid_negative_ts", "make_zero")
    return await run(builder)


async def reframe_to_vertical(
    src: Path,
    dst: Path,
    *,
    target_w: int = 1080,
    target_h: int = 1920,
    crf: int = 20,
    on_progress: Callable[[Progress], None] | None = None,
) -> Path:
    """
    Convert any aspect ratio to 9:16 by center-cropping + scaling.
    For talking-head content this is a reasonable default; a smarter
    speaker-tracking crop is a v2 upgrade (use mediapipe / yolo).
    """
    # cover-fit: scale so short edge matches target, then center-crop overflow
    filt = (
        f"scale={target_w}:{target_h}:force_original_aspect_ratio=increase,"
        f"crop={target_w}:{target_h},"
        f"setsar=1"
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
    return await run(cmd, on_progress=on_progress)


async def concat_segments(srcs: Sequence[Path], dst: Path) -> Path:
    """
    Lossless concat via the demuxer protocol. All inputs must share codecs/params
    (true after running them through the same reframe step).
    """
    list_path = dst.with_suffix(".concat.txt")
    list_path.write_text(
        "\n".join(f"file '{p.resolve().as_posix()}'" for p in srcs),
        encoding="utf-8",
    )
    cmd = (
        FFmpegCommand()
        .add_input(list_path, "-f", "concat", "-safe", "0")
        .add_output_args("-c", "copy", "-movflags", "+faststart")
        .with_output(dst)
    )
    try:
        return await run(cmd)
    finally:
        list_path.unlink(missing_ok=True)


async def overlay_broll(
    base: Path,
    broll: Path,
    dst: Path,
    *,
    start: float,
    duration: float,
    opacity: float = 1.0,
) -> Path:
    """
    Overlay b-roll footage on top of the base video for [start, start+duration].
    Keeps the base audio (b-roll audio is muted — almost always what you want).
    """
    # Scale b-roll to base size, set its PTS, enable only in the window.
    filt = (
        "[1:v]scale=iw:ih,setpts=PTS-STARTPTS,"
        f"format=yuva420p,colorchannelmixer=aa={opacity}[bv];"
        f"[0:v][bv]overlay=enable='between(t,{start:.3f},{start + duration:.3f})':"
        "shortest=0[v]"
    )
    cmd = (
        FFmpegCommand()
        .add_input(base)
        .add_input(broll, "-itsoffset", f"{start:.3f}")
        .with_filter_complex(filt)
        .add_output_args(
            "-map", "[v]", "-map", "0:a?",
            "-c:v", "libx264", "-preset", get_settings().ffmpeg_preset, "-crf", "20",
            "-c:a", "copy",
        )
        .with_output(dst)
    )
    return await run(cmd)


async def overlay_with_dissolve(
    base: Path,
    broll: Path,
    dst: Path,
    *,
    start: float,
    duration: float,
    dissolve: float = 0.5,
) -> Path:
    """
    Overlay b-roll with a true cross-dissolve in/out.

    Unlike `overlay_broll` (hard cut), this ramps the b-roll's alpha so it
    fades in over `dissolve` seconds at the start and fades out over the
    last `dissolve` seconds. The base video plays unaltered underneath; b-roll
    audio is dropped.

    The trick: we have to scale-pad the b-roll to the base's exact size
    (generated clips arrive at 1280x720 or 768x1280, never your timeline's
    1080x1920), apply yuva420p so alpha is a thing, then `fade=alpha=1`
    twice with start times relative to the b-roll's PTS, NOT the base.
    `setpts=PTS-STARTPTS` re-zeroes the b-roll's clock before the fade math.
    """
    dissolve = max(0.05, min(dissolve, duration / 2 - 0.05))
    end = start + duration
    fade_out_start = duration - dissolve   # in b-roll-local time

    # Probe base for output dimensions — the b-roll must match exactly.
    info = await probe(base)
    bw, bh = info.width, info.height

    filt = (
        f"[1:v]scale={bw}:{bh}:force_original_aspect_ratio=increase,"
        f"crop={bw}:{bh},setsar=1,setpts=PTS-STARTPTS,"
        f"format=yuva420p,"
        f"fade=t=in:st=0:d={dissolve}:alpha=1,"
        f"fade=t=out:st={fade_out_start:.3f}:d={dissolve}:alpha=1,"
        f"trim=duration={duration:.3f}[bv];"
        f"[0:v][bv]overlay=enable='between(t,{start:.3f},{end:.3f})':"
        f"x=0:y=0:shortest=0[v]"
    )
    cmd = (
        FFmpegCommand()
        .add_input(base)
        # itsoffset shifts the b-roll's start time to align with `start` on the base
        .add_input(broll, "-itsoffset", f"{start:.3f}")
        .with_filter_complex(filt)
        .add_output_args(
            "-map", "[v]", "-map", "0:a?",
            "-c:v", "libx264", "-preset", get_settings().ffmpeg_preset, "-crf", "20",
            "-c:a", "copy",
        )
        .with_output(dst)
    )
    return await run(cmd)


async def burn_ass(src: Path, ass: Path, dst: Path) -> Path:
    r"""
    Burn an ASS subtitle file into the video. Use this — not SRT — because
    word-level animations (\k karaoke, \t transitions, \fscx scale) are
    ASS-only. The .ass file is produced by services.captions.write_ass.
    """
    # ffmpeg's subtitles filter parses the path as a filter argument, so
    # backslashes and colons (Windows drive letters) need escaping.
    ass_escaped = str(ass).replace("\\", "/").replace(":", r"\:")
    filt = f"subtitles='{ass_escaped}'"
    cmd = (
        FFmpegCommand()
        .add_input(src)
        .with_filter_complex(filt)
        .add_output_args(
            "-c:v", "libx264", "-preset", get_settings().ffmpeg_preset, "-crf", "20",
            "-c:a", "copy",
        )
        .with_output(dst)
    )
    return await run(cmd)


# Backwards-compatible alias — call sites in tasks/ still import burn_captions.
async def burn_captions(src: Path, ass_or_srt: Path, dst: Path, *, style: str | None = None) -> Path:
    """Deprecated alias. New code should call burn_ass directly."""
    return await burn_ass(src, ass_or_srt, dst)


async def mix_music(
    src: Path,
    music: Path,
    dst: Path,
    *,
    music_volume: float = 0.18,
    duck: bool = True,
) -> Path:
    """
    Mix a music bed under the source audio. When `duck=True`, the music ducks
    automatically when speech is present (sidechain compression).
    """
    if duck:
        # Sidechain the music against the speech track.
        filt = (
            f"[1:a]volume={music_volume}[music];"
            "[0:a]asplit=2[speech1][speech2];"
            "[music][speech1]sidechaincompress=threshold=0.05:ratio=8:attack=5:release=300[ducked];"
            "[speech2][ducked]amix=inputs=2:duration=first:dropout_transition=0[a]"
        )
    else:
        filt = (
            f"[1:a]volume={music_volume}[music];"
            "[0:a][music]amix=inputs=2:duration=first:dropout_transition=0[a]"
        )
    cmd = (
        FFmpegCommand()
        .add_input(src)
        .add_input(music, "-stream_loop", "-1")  # loop music if shorter than video
        .with_filter_complex(filt)
        .add_output_args(
            "-map", "0:v", "-map", "[a]",
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest",
        )
        .with_output(dst)
    )
    return await run(cmd)
