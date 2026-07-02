"""
Range-aware media serving for /storage.

Why not StaticFiles? The pinned starlette build ignores the `Range` header and
answers every request with 200 + the full body. That breaks video delivery in
two ways: seeking re-downloads from byte 0, and iOS Safari *refuses to play*
media from servers that don't answer `Range: bytes=0-1` probes with 206. This
route implements single-range byte serving (206/Content-Range/416) so faststart
MP4s stream and scrub properly in every browser.

This is still the app-server path — object storage + CDN replaces it at scale —
but it must be *correct* regardless, and it's what local/dev always uses.
"""
from __future__ import annotations

import re
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, Response, StreamingResponse

from app.config import get_settings

router = APIRouter()

_RANGE = re.compile(r"bytes=(\d*)-(\d*)$")
_CHUNK = 1024 * 256

_CT = {
    ".mp4": "video/mp4", ".webm": "video/webm", ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg", ".png": "image/png", ".ass": "text/plain",
    ".srt": "application/x-subrip", ".mp3": "audio/mpeg", ".wav": "audio/wav",
}

# Finished outputs are content-addressed by job id → safe to cache long.
_CACHE = "public, max-age=2592000"


def _resolve(rel: str) -> Path:
    root = get_settings().storage_local_dir.resolve()
    try:
        target = (root / rel).resolve()
    except (OSError, ValueError):
        raise HTTPException(status_code=404)
    # Path-traversal guard: the resolved path must stay inside the storage root.
    if root not in target.parents and target != root:
        raise HTTPException(status_code=404)
    if not target.is_file():
        raise HTTPException(status_code=404)
    return target


def _iter_range(path: Path, start: int, end: int):
    """Yield [start, end] (inclusive) in chunks. Sync generator — starlette
    drives it from a threadpool, so the event loop is never blocked."""
    remaining = end - start + 1
    with open(path, "rb") as f:
        f.seek(start)
        while remaining > 0:
            chunk = f.read(min(_CHUNK, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
            yield chunk


@router.get("/storage/{rel_path:path}")
async def serve_storage(rel_path: str, request: Request) -> Response:
    path = _resolve(rel_path)
    size = path.stat().st_size
    ct = _CT.get(path.suffix.lower(), "application/octet-stream")
    base_headers = {"Accept-Ranges": "bytes", "Cache-Control": _CACHE}

    range_header = request.headers.get("range", "")
    if not range_header:
        return FileResponse(path, media_type=ct, headers=base_headers)

    m = _RANGE.match(range_header.strip())
    if not m or (not m.group(1) and not m.group(2)):
        # Unparseable/multi-range → serve the whole file (per RFC, ignoring
        # the header is always a legal response to a malformed Range). Streamed
        # by hand: FileResponse would re-parse the request's Range and 400.
        return StreamingResponse(
            _iter_range(path, 0, size - 1), media_type=ct,
            headers={**base_headers, "Content-Length": str(size)})

    if m.group(1):
        start = int(m.group(1))
        end = int(m.group(2)) if m.group(2) else size - 1
    else:                                   # suffix form: bytes=-500 → last 500
        start = max(0, size - int(m.group(2)))
        end = size - 1
    end = min(end, size - 1)

    if start >= size or start > end:
        return Response(status_code=416, headers={"Content-Range": f"bytes */{size}"})

    return StreamingResponse(
        _iter_range(path, start, end),
        status_code=206,
        media_type=ct,
        headers={
            **base_headers,
            "Content-Range": f"bytes {start}-{end}/{size}",
            "Content-Length": str(end - start + 1),
        },
    )
