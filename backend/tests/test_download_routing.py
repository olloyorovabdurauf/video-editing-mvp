"""
Download-routing tests.

The bug this guards against: routing a direct media-file URL through yt-dlp's
generic extractor, which many CDNs answer with HTTP 403 (non-browser UA).
Direct files must go down the plain-HTTP path; only platform URLs use yt-dlp.
"""
from __future__ import annotations

import pytest

from app.tasks.video_tasks import _looks_like_direct_media


@pytest.mark.parametrize("url", [
    "https://media.w3.org/2010/05/sintel/trailer.mp4",
    "https://cdn.example.com/a/b/clip.MP4",          # case-insensitive
    "https://x.com/v.mov",
    "https://x.com/v.webm",
    "https://x.com/v.m4v",
    "https://storage.googleapis.com/bucket/talk.mkv",
    "https://x.com/path/audio.mp3?token=abc",         # query string ignored
])
def test_direct_media_urls_detected(url):
    assert _looks_like_direct_media(url) is True


@pytest.mark.parametrize("url", [
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "https://youtu.be/dQw4w9WgXcQ",
    "https://vimeo.com/123456789",
    "https://www.tiktok.com/@user/video/123",
    "https://example.com/page.html",
    "https://example.com/",
])
def test_platform_and_page_urls_not_direct(url):
    assert _looks_like_direct_media(url) is False
