"""
SSRF guard tests.

DNS is monkeypatched — tests must not depend on network. We verify the four
properties that make the guard worth having:
  1. non-http(s) schemes rejected
  2. private/loopback/link-local/metadata addresses rejected
  3. split-horizon hosts (one public + one private A record) rejected
  4. honest public hosts pass
"""
from __future__ import annotations

import socket

import pytest

from app.services.url_guard import UnsafeURLError, validate_source_url


def _fake_resolver(mapping: dict[str, list[str]]):
    def fake_getaddrinfo(host, port, *args, **kwargs):
        if host not in mapping:
            raise socket.gaierror(f"unknown host {host}")
        return [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 0))
            for ip in mapping[host]
        ]
    return fake_getaddrinfo


# ---------------------------------------------------------------------------
# Scheme / shape rejections (no DNS needed)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("url", [
    "file:///etc/passwd",
    "ftp://example.com/video.mp4",
    "gopher://example.com",
    "javascript:alert(1)",
    "redis://localhost:6379",
])
def test_non_http_schemes_rejected(url):
    with pytest.raises(UnsafeURLError, match="scheme"):
        validate_source_url(url)


def test_missing_hostname_rejected():
    with pytest.raises(UnsafeURLError):
        validate_source_url("https:///path-only")


def test_embedded_credentials_rejected(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", _fake_resolver({"evil.com": ["93.184.216.34"]}))
    with pytest.raises(UnsafeURLError, match="credentials"):
        validate_source_url("https://admin:hunter2@evil.com/v.mp4")


# ---------------------------------------------------------------------------
# Address-space rejections
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ip", [
    "127.0.0.1",          # loopback
    "10.0.0.5",           # RFC1918
    "172.16.3.4",         # RFC1918
    "192.168.1.1",        # RFC1918
    "169.254.169.254",    # cloud metadata — the money shot
    "0.0.0.0",            # unspecified
    "::1",                # v6 loopback
])
def test_private_and_reserved_ips_rejected(monkeypatch, ip):
    monkeypatch.setattr(socket, "getaddrinfo", _fake_resolver({"sneaky.example": [ip]}))
    with pytest.raises(UnsafeURLError, match="non-global"):
        validate_source_url("https://sneaky.example/video.mp4")


def test_split_horizon_host_rejected(monkeypatch):
    """One public + one private A record → reject. ALL must be global."""
    monkeypatch.setattr(socket, "getaddrinfo", _fake_resolver(
        {"both.example": ["93.184.216.34", "10.0.0.5"]}
    ))
    with pytest.raises(UnsafeURLError, match="non-global"):
        validate_source_url("https://both.example/v.mp4")


def test_unresolvable_host_rejected(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", _fake_resolver({}))
    with pytest.raises(UnsafeURLError, match="does not resolve"):
        validate_source_url("https://no-such-host.invalid/v.mp4")


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_public_host_passes(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", _fake_resolver(
        {"youtube.com": ["142.250.180.14"]}
    ))
    validate_source_url("https://youtube.com/watch?v=abc")  # no raise


def test_public_host_with_port_passes(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", _fake_resolver(
        {"cdn.example": ["93.184.216.34"]}
    ))
    validate_source_url("https://cdn.example:8443/v.mp4")  # no raise
