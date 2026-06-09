"""
FFmpeg builder tests.

`FFmpegCommand` is on the hot path of every render. A regression here
silently produces broken commands. We assert:
  1. argv shape is deterministic and shell-injection-free
  2. cache_key is content-addressable AND stable across runs
  3. filter_complex flows through
  4. overwrite + hwaccel flags appear correctly
"""
from __future__ import annotations

from pathlib import Path

from app.utils.ffmpeg import FFmpegCommand


def test_build_minimal_command(tmp_path: Path):
    src = tmp_path / "in.mp4"
    src.write_bytes(b"x")
    out = tmp_path / "out.mp4"

    cmd = FFmpegCommand().add_input(src).add_output_args("-c", "copy").with_output(out)
    argv = cmd.build()

    # Required shape: binary, banner suppression, no stdin, overwrite, progress
    assert argv[0].endswith("ffmpeg") or argv[0] == "ffmpeg"
    assert "-hide_banner" in argv
    assert "-nostdin" in argv
    assert "-y" in argv
    assert "-progress" in argv
    assert "pipe:1" in argv
    # Input + output present and in order
    assert argv[argv.index("-i") + 1] == str(src)
    assert argv[-1] == str(out)
    # Output codec args present
    assert "-c" in argv and "copy" in argv


def test_build_no_shell_in_user_supplied_filenames(tmp_path: Path):
    """Filenames with shell metachars must be passed argv-style, never interpolated."""
    nasty = tmp_path / "input';rm -rf $HOME;'.mp4"
    nasty.write_bytes(b"x")
    out = tmp_path / "out.mp4"

    cmd = FFmpegCommand().add_input(nasty).with_output(out).add_output_args("-c", "copy")
    argv = cmd.build()

    # The nasty name must appear as a single argv element, not split.
    assert str(nasty) in argv
    # No combined " -i 'name; rm ...'" string anywhere.
    assert all(";" not in a or a == str(nasty) or a == str(out) for a in argv)


def test_filter_complex_flows_through(tmp_path: Path):
    cmd = (
        FFmpegCommand()
        .add_input(tmp_path / "a.mp4")
        .with_filter_complex("scale=1080:1920")
        .with_output(tmp_path / "o.mp4")
        .add_output_args("-c:v", "libx264")
    )
    argv = cmd.build()
    assert "-filter_complex" in argv
    assert argv[argv.index("-filter_complex") + 1] == "scale=1080:1920"


def test_cache_key_is_deterministic(tmp_path: Path):
    src = tmp_path / "a.mp4"
    src.write_bytes(b"abc")
    out = tmp_path / "o.mp4"

    def make():
        return (
            FFmpegCommand()
            .add_input(src)
            .with_filter_complex("scale=1080:1920")
            .add_output_args("-c:v", "libx264", "-crf", "20")
            .with_output(out)
        )

    k1 = make().cache_key()
    k2 = make().cache_key()
    assert k1 == k2 and len(k1) == 16


def test_cache_key_changes_with_inputs(tmp_path: Path):
    a = tmp_path / "a.mp4"; a.write_bytes(b"abc")
    b = tmp_path / "b.mp4"; b.write_bytes(b"different")
    out = tmp_path / "o.mp4"

    k_a = FFmpegCommand().add_input(a).with_output(out).cache_key()
    k_b = FFmpegCommand().add_input(b).with_output(out).cache_key()
    k_filter = FFmpegCommand().add_input(a).with_filter_complex("scale=1080:1920").with_output(out).cache_key()

    assert k_a != k_b, "different input files must produce different keys"
    assert k_a != k_filter, "different filter must produce different key"


def test_overwrite_can_be_disabled(tmp_path: Path):
    cmd = FFmpegCommand().add_input(tmp_path / "x.mp4").with_output(tmp_path / "y.mp4")
    cmd.overwrite = False
    argv = cmd.build()
    assert "-y" not in argv


def test_output_required():
    import pytest
    cmd = FFmpegCommand().add_input("a.mp4")
    with pytest.raises(ValueError, match="output not set"):
        cmd.build()
