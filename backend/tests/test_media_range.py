"""
Range-aware /storage serving.

Video delivery correctness: browsers scrub with `Range:` requests and iOS
Safari refuses to play media unless the server answers its `bytes=0-1` probe
with 206. StaticFiles on the pinned starlette ignores Range (200 + full body),
so app/api/media.py implements single-range serving — these tests pin the
contract: 206 + Content-Range for ranges, 416 past EOF, Accept-Ranges on full
responses, and no path traversal.
"""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.api.media import router as media_router
from app.config import get_settings

BODY = bytes(range(256)) * 4          # 1024 distinguishable bytes


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(get_settings(), "storage_local_dir", tmp_path)
    (tmp_path / "output" / "job1").mkdir(parents=True)
    (tmp_path / "output" / "job1" / "reel_0.mp4").write_bytes(BODY)
    app = FastAPI()
    app.include_router(media_router)
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://t")


URL = "/storage/output/job1/reel_0.mp4"


@pytest.mark.asyncio
async def test_full_request_advertises_ranges(client):
    r = await client.get(URL)
    assert r.status_code == 200
    assert r.headers["accept-ranges"] == "bytes"
    assert r.headers["content-type"] == "video/mp4"
    assert r.content == BODY


@pytest.mark.asyncio
async def test_range_returns_206_with_exact_bytes(client):
    r = await client.get(URL, headers={"Range": "bytes=10-19"})
    assert r.status_code == 206
    assert r.headers["content-range"] == f"bytes 10-19/{len(BODY)}"
    assert r.headers["content-length"] == "10"
    assert r.content == BODY[10:20]


@pytest.mark.asyncio
async def test_safari_probe_bytes_0_1(client):
    # iOS Safari sends this exact probe and refuses playback on a 200.
    r = await client.get(URL, headers={"Range": "bytes=0-1"})
    assert r.status_code == 206
    assert r.content == BODY[0:2]


@pytest.mark.asyncio
async def test_open_ended_range(client):
    r = await client.get(URL, headers={"Range": "bytes=1000-"})
    assert r.status_code == 206
    assert r.content == BODY[1000:]


@pytest.mark.asyncio
async def test_suffix_range(client):
    r = await client.get(URL, headers={"Range": "bytes=-16"})
    assert r.status_code == 206
    assert r.content == BODY[-16:]


@pytest.mark.asyncio
async def test_range_past_eof_is_416(client):
    r = await client.get(URL, headers={"Range": f"bytes={len(BODY)}-"})
    assert r.status_code == 416
    assert r.headers["content-range"] == f"bytes */{len(BODY)}"


@pytest.mark.asyncio
async def test_end_clamped_to_size(client):
    r = await client.get(URL, headers={"Range": "bytes=1020-99999"})
    assert r.status_code == 206
    assert r.content == BODY[1020:]


@pytest.mark.asyncio
async def test_malformed_range_serves_full_file(client):
    r = await client.get(URL, headers={"Range": "bytes=abc"})
    assert r.status_code == 200
    assert r.content == BODY


@pytest.mark.asyncio
async def test_missing_file_404(client):
    r = await client.get("/storage/output/job1/nope.mp4")
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_path_traversal_404(client):
    r = await client.get("/storage/../../etc/passwd")
    assert r.status_code == 404
