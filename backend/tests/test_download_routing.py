"""
Ingestion routing + error-mapping tests.

The bug class this guards against: routing a direct media-file URL through
yt-dlp's generic extractor (CDNs answer its UA with 403). Direct files take the
plain-HTTP path; only platform URLs use yt-dlp. Plus: raw download errors must
map to clear, user-facing messages.
"""
from __future__ import annotations

import pytest

from app.services import ingestion
from app.services.ingestion import IngestionError, looks_like_direct_media


@pytest.mark.parametrize("url", [
    "https://media.w3.org/2010/05/sintel/trailer.mp4",
    "https://cdn.example.com/a/b/clip.MP4",
    "https://x.com/v.mov",
    "https://x.com/v.webm",
    "https://storage.googleapis.com/bucket/talk.mkv",
    "https://x.com/path/audio.mp3?token=abc",
])
def test_direct_media_urls_detected(url):
    assert looks_like_direct_media(url) is True


@pytest.mark.parametrize("url", [
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "https://youtu.be/dQw4w9WgXcQ",
    "https://vimeo.com/123456789",
    "https://www.tiktok.com/@user/video/123",
    "https://example.com/page.html",
])
def test_platform_urls_not_direct(url):
    assert looks_like_direct_media(url) is False


# ---------------------------------------------------------------------------
# Error mapping → user-facing messages
# ---------------------------------------------------------------------------

def test_private_video_message():
    err = ingestion._to_user_error(Exception("ERROR: Video unavailable: This video is private"))
    assert "private" in err.user_message.lower()
    assert err.retryable is False


def test_age_restricted_message():
    err = ingestion._to_user_error(Exception("ERROR: Sign in to confirm your age"))
    assert "sign-in" in err.user_message.lower() or "age" in err.user_message.lower()


def test_bot_check_is_not_mislabeled_as_age_restriction():
    # YouTube's datacenter-IP challenge contains BOTH "sign in" and "bot" —
    # it must map to the bot-check message (retryable), not the age one.
    err = ingestion._to_user_error(Exception(
        "ERROR: [youtube] g97I8Fvdkmc: Sign in to confirm you're not a bot. "
        "Use --cookies-from-browser or --cookies for the authentication."))
    assert "age" not in err.user_message.lower()
    assert "try again" in err.user_message.lower()
    assert err.retryable is True


def test_rate_limit_message_is_retryable():
    err = ingestion._to_user_error(Exception("HTTP Error 403: Forbidden"))
    assert err.retryable is True
    assert "again" in err.user_message.lower()


def test_timeout_message_is_retryable():
    err = ingestion._to_user_error(Exception("socket timed out"))
    assert err.retryable is True


def test_generic_message_is_safe():
    err = ingestion._to_user_error(Exception("some weird internal traceback xyz"))
    assert "xyz" not in err.user_message      # no internals leak to the user
    assert "url is public" in err.user_message.lower()


# ---------------------------------------------------------------------------
# Version monitoring
# ---------------------------------------------------------------------------

def test_ytdlp_version_returns_string():
    v = ingestion.ytdlp_version()
    assert isinstance(v, str) and len(v) > 0


def test_warn_if_stale_does_not_raise():
    ingestion.warn_if_ytdlp_stale()   # never throws regardless of version


# ---------------------------------------------------------------------------
# Audio-first + section downloads (the slow-residential-proxy fast path)
# ---------------------------------------------------------------------------

from app.config import get_settings


def _with_proxy(monkeypatch, value):
    monkeypatch.setattr(get_settings(), "ytdlp_proxy", value)


def test_proxy_for_session_rewrites_webshare_suffix(monkeypatch):
    _with_proxy(monkeypatch, "http://user-gb-1:pw@p.webshare.io:80")
    assert ingestion.proxy_for_session(0) == "http://user-gb-1:pw@p.webshare.io:80"
    assert ingestion.proxy_for_session(2) == "http://user-gb-3:pw@p.webshare.io:80"


def test_proxy_for_session_without_proxy_is_none(monkeypatch):
    _with_proxy(monkeypatch, None)
    assert ingestion.proxy_for_session(3) is None


def test_proxy_for_session_non_session_username_unchanged(monkeypatch):
    # No trailing "-<n>" → can't derive sessions; fall back to the base proxy.
    _with_proxy(monkeypatch, "http://plainuser:pw@proxy.example:8080")
    assert ingestion.proxy_for_session(5) == "http://plainuser:pw@proxy.example:8080"


def test_audio_opts_are_audio_only_and_small():
    opts = ingestion._audio_opts("/tmp/a.%(ext)s", proxy="http://p:1@h:80")
    assert opts["proxy"] == "http://p:1@h:80"
    assert opts["format"].startswith("ba[abr<=96]")   # bounded-bitrate audio, no video


def test_section_opts_cut_exactly_and_use_session_proxy():
    opts = ingestion._section_opts("/tmp/s.%(ext)s", 30.0, 90.0, proxy="http://u-gb-2:p@h:80")
    assert opts["force_keyframes_at_cuts"] is True    # frame-accurate [start,end]
    assert opts["download_ranges"] is not None
    assert opts["proxy"] == "http://u-gb-2:p@h:80"
    assert "height<=1080" in opts["format"]
