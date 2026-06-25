"""Job-state heartbeat + stale-job guard (orphaned jobs must fail, not spin)."""
from __future__ import annotations

import json
import time

from app.tasks import video_tasks as vt


class _FakeRedis:
    def __init__(self):
        self.d: dict[str, str] = {}

    def get(self, k):
        return self.d.get(k)

    def setex(self, k, ttl, v):
        self.d[k] = v


def _seed(fr, job_id, **state):
    base = {"job_id": job_id, "status": "rendering", "progress": 0.6, "artifacts": []}
    base.update(state)
    fr.d[vt._key(job_id)] = json.dumps(base)


def test_stale_non_terminal_job_marked_failed(monkeypatch):
    fr = _FakeRedis()
    monkeypatch.setattr(vt, "r", fr)
    _seed(fr, "stale", updated_at=time.time() - (vt._STALE_AFTER_S + 60))
    job = vt.get_job("stale")
    assert job.status.value == "failed"
    assert "again" in (job.message or "").lower()


def test_fresh_job_not_failed(monkeypatch):
    fr = _FakeRedis()
    monkeypatch.setattr(vt, "r", fr)
    _seed(fr, "fresh", updated_at=time.time() - 5)
    assert vt.get_job("fresh").status.value == "rendering"


def test_succeeded_job_never_touched(monkeypatch):
    fr = _FakeRedis()
    monkeypatch.setattr(vt, "r", fr)
    _seed(fr, "done", status="succeeded", progress=1.0,
          updated_at=time.time() - 99999)
    assert vt.get_job("done").status.value == "succeeded"


def test_update_writes_heartbeat(monkeypatch):
    fr = _FakeRedis()
    monkeypatch.setattr(vt, "r", fr)
    vt._update("x", status="rendering")
    st = json.loads(fr.d[vt._key("x")])
    assert "updated_at" in st and st["updated_at"] > 0
