"""Disk-hygiene helpers: prevent the storage volume filling ("No space left")."""
from __future__ import annotations

import os
import time

from app.tasks import video_tasks as vt


def _mkdir(p):
    p.mkdir(parents=True, exist_ok=True)
    return p


def test_sweep_removes_stale_transient_and_old_output(tmp_path, monkeypatch):
    monkeypatch.setattr(vt.settings, "storage_local_dir", tmp_path)

    old_raw = _mkdir(tmp_path / "raw" / "oldjob"); (old_raw / "src.mp4").write_text("x")
    new_raw = _mkdir(tmp_path / "raw" / "newjob")
    old_out = _mkdir(tmp_path / "output" / "oldout")
    new_out = _mkdir(tmp_path / "output" / "newout")

    stale = time.time() - 5 * 3600          # 5h ago
    week_old = time.time() - 8 * 86400      # 8 days ago
    os.utime(old_raw, (stale, stale))
    os.utime(old_out, (week_old, week_old))

    vt._sweep_storage(transient_age_h=2, output_age_days=7)

    assert not old_raw.exists()             # transient >2h → deleted
    assert new_raw.exists()                 # recent transient → kept
    assert not old_out.exists()             # output >7d → deleted
    assert new_out.exists()                 # recent output → kept


def test_purge_job_files_removes_all_traces(tmp_path, monkeypatch):
    monkeypatch.setattr(vt.settings, "storage_local_dir", tmp_path)
    for sub in ("raw", "intermediate", "output"):
        _mkdir(tmp_path / sub / "jobX")
    vt._purge_job_files("jobX")
    for sub in ("raw", "intermediate", "output"):
        assert not (tmp_path / sub / "jobX").exists()
