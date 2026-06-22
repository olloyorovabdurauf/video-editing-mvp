"""
Upload-source pipeline tests: an uploaded R2 object enters the same pipeline
as a URL/YouTube source.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app.schemas.reel import ReelCreateRequest
from app.services import ingestion
from app.services.storage import LocalStorage


# ---------------------------------------------------------------------------
# Request: exactly one source
# ---------------------------------------------------------------------------

def test_url_source_valid():
    r = ReelCreateRequest(source_url="https://youtube.com/watch?v=x")
    assert r.is_upload is False


def test_upload_source_valid():
    r = ReelCreateRequest(upload_key="uploads/alice/abc/talk.mp4")
    assert r.is_upload is True


def test_both_sources_rejected():
    with pytest.raises(ValueError, match="exactly one"):
        ReelCreateRequest(source_url="https://x.com/v.mp4", upload_key="uploads/a/b.mp4")


def test_no_source_rejected():
    with pytest.raises(ValueError, match="exactly one"):
        ReelCreateRequest()


# ---------------------------------------------------------------------------
# Storage download_to
# ---------------------------------------------------------------------------

def test_local_storage_download_to(tmp_path: Path):
    root = tmp_path / "bucket"
    (root / "uploads/alice/x").mkdir(parents=True)
    (root / "uploads/alice/x/talk.mp4").write_bytes(b"video-bytes")
    store = LocalStorage(root)
    dst = store.download_to("uploads/alice/x/talk.mp4", tmp_path / "work" / "source.mp4")
    assert dst.read_bytes() == b"video-bytes"


# ---------------------------------------------------------------------------
# ingestion.fetch_upload
# ---------------------------------------------------------------------------

def test_fetch_upload_happy(tmp_path: Path, monkeypatch):
    root = tmp_path / "bucket"
    (root / "uploads/alice/x").mkdir(parents=True)
    (root / "uploads/alice/x/v.mp4").write_bytes(b"abc")
    monkeypatch.setattr(ingestion, "get_storage", lambda: LocalStorage(root), raising=False)
    # fetch_upload imports get_storage lazily from storage; patch there.
    import app.services.storage as storage_mod
    monkeypatch.setattr(storage_mod, "get_storage", lambda: LocalStorage(root))

    out = ingestion.fetch_upload("uploads/alice/x/v.mp4", tmp_path / "work")
    assert out.exists() and out.read_bytes() == b"abc"


def test_fetch_upload_missing_key_raises(tmp_path: Path, monkeypatch):
    import app.services.storage as storage_mod
    monkeypatch.setattr(storage_mod, "get_storage", lambda: LocalStorage(tmp_path / "empty"))
    with pytest.raises(ingestion.IngestionError):
        ingestion.fetch_upload("uploads/alice/nope.mp4", tmp_path / "work")
