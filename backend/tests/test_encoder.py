"""GPU/CPU encoder strategy (Issue 2). NVENC probe is stubbed — no GPU needed."""
from __future__ import annotations

from app.utils import ffmpeg as ff


def test_cpu_encoder_output_args():
    args = ff.CPUEncoder().output_args(20)
    assert args[:2] == ["-c:v", "libx264"]
    assert "-crf" in args and "20" in args


def test_gpu_encoder_output_args():
    args = ff.GPUEncoder().output_args(20)
    assert args[:2] == ["-c:v", "h264_nvenc"]
    assert "-cq" in args              # NVENC constant-quality control


def test_select_encoder_picks_cpu_when_no_gpu(monkeypatch):
    monkeypatch.setattr(ff, "_nvenc_available", lambda: False)
    assert isinstance(ff.select_encoder(), ff.CPUEncoder)
    assert ff.video_encoder_args(20)[:2] == ["-c:v", "libx264"]


def test_select_encoder_picks_gpu_when_available(monkeypatch):
    monkeypatch.setattr(ff, "_nvenc_available", lambda: True)
    assert isinstance(ff.select_encoder(), ff.GPUEncoder)
    assert ff.video_encoder_args(20)[:2] == ["-c:v", "h264_nvenc"]


def test_hwaccel_none_forces_cpu(monkeypatch):
    from app.config import get_settings
    ff._nvenc_available.cache_clear()
    monkeypatch.setattr(get_settings(), "ffmpeg_hwaccel", "none")
    assert ff._nvenc_available() is False
    ff._nvenc_available.cache_clear()
