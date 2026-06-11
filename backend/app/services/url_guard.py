"""
SSRF guard for user-supplied source URLs.

Threat model
------------
`source_url` is fetched server-side by yt-dlp. Without validation a user can
point us at:
  - cloud metadata endpoints (169.254.169.254 → instance credentials)
  - internal services (redis:6379, the Celery broker, admin panels)
  - localhost loopback on the worker itself

We reject any URL whose scheme isn't http(s) or whose hostname resolves to a
non-global address (private, loopback, link-local, reserved, multicast).

Known limitation (documented, accepted at this stage): DNS rebinding. We
resolve at validation time; yt-dlp re-resolves at fetch time, so an attacker
running a malicious DNS server could pass validation then rebind to an
internal IP. The complete fix is forcing egress through a filtering proxy —
deferred until the threat is real (PUBLISHING.md tracks it). The current
guard removes the entire drive-by / casual-abuse class.
"""
from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse

ALLOWED_SCHEMES = frozenset({"http", "https"})


class UnsafeURLError(ValueError):
    """Raised when a source URL fails SSRF validation."""


def _resolve_ips(hostname: str) -> list[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    try:
        infos = socket.getaddrinfo(hostname, None, proto=socket.IPPROTO_TCP)
    except socket.gaierror as e:
        raise UnsafeURLError(f"hostname does not resolve: {hostname!r}") from e
    ips = []
    for _family, _type, _proto, _canon, sockaddr in infos:
        ips.append(ipaddress.ip_address(sockaddr[0]))
    if not ips:
        raise UnsafeURLError(f"hostname resolved to no addresses: {hostname!r}")
    return ips


def validate_source_url(url: str) -> None:
    """
    Raise UnsafeURLError unless `url` is an http(s) URL whose host resolves
    exclusively to globally routable addresses.

    ALL resolved addresses must be global — a host that round-robins between
    a public and a private IP is rejected (classic SSRF dodge).
    """
    parsed = urlparse(url)

    if parsed.scheme.lower() not in ALLOWED_SCHEMES:
        raise UnsafeURLError(f"scheme {parsed.scheme!r} not allowed (http/https only)")

    hostname = parsed.hostname
    if not hostname:
        raise UnsafeURLError("URL has no hostname")

    # Userinfo tricks (http://internal@evil.com, http://evil.com@internal)
    # are neutralized by using parsed.hostname, but reject explicit userinfo
    # anyway — no legitimate video URL carries credentials.
    if parsed.username or parsed.password:
        raise UnsafeURLError("URLs with embedded credentials are not allowed")

    for ip in _resolve_ips(hostname):
        if not ip.is_global:
            raise UnsafeURLError(
                f"host {hostname!r} resolves to non-global address {ip} — refused"
            )
