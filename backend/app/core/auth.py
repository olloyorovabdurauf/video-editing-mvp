"""
Authentication dependency.

Three modes, controlled by `AUTH_MODE` env var:

  none    Trust the X-User-Id header. ONLY safe for local dev.
  clerk   Verify a Clerk JWT (Bearer token), extract user id from claims.
  custom  You're using your own auth; supply CUSTOM_AUTH_JWKS_URL.

Why a dependency, not middleware?
---------------------------------
Some endpoints are public (/livez, /healthz, /billing/webhook/stripe).
A middleware would force opt-out branches everywhere. A dep makes the
intent explicit: any endpoint that takes `user_id: str = Depends(auth)`
is gated. Anything else is intentionally public.

The dep ALWAYS overrides whatever user_id the client supplied in the
body — we trust the token, never the payload.
"""
from __future__ import annotations

import time
from typing import Any

import httpx
from fastapi import Depends, Header, HTTPException, Request, status
from jose import jwt
from jose.exceptions import JWTError
from loguru import logger

from app.config import get_settings


# ---------------------------------------------------------------------------
# JWKS cache — fetched once, refreshed lazily.
# JWTs are signed with rotating keys; we need to keep the key set fresh.
# Clerk rotates keys every ~6 hours.
# ---------------------------------------------------------------------------

class _JWKSCache:
    def __init__(self) -> None:
        self.keys: list[dict] = []
        self.fetched_at: float = 0.0
        self.url: str = ""

    async def get(self, url: str) -> list[dict]:
        # Refresh every hour, or immediately if the URL changed
        if self.url != url or time.time() - self.fetched_at > 3600:
            async with httpx.AsyncClient(timeout=5) as c:
                r = await c.get(url)
                r.raise_for_status()
                self.keys = r.json().get("keys", [])
                self.fetched_at = time.time()
                self.url = url
        return self.keys


_jwks = _JWKSCache()


# ---------------------------------------------------------------------------
# The dependency.
# ---------------------------------------------------------------------------

async def require_user(
    request: Request,
    authorization: str | None = Header(None),
    x_user_id: str | None = Header(None),
) -> str:
    """
    Returns the authenticated user_id. Raises 401 otherwise.

    Usage:
        @router.post("/reels")
        async def create_reel(req: ReelCreateRequest, user_id: str = Depends(require_user)):
            req.user_id = user_id          # override anything client sent
            ...
    """
    settings = get_settings()
    mode = settings.auth_mode.lower()

    if mode == "none":
        # Dev only. Refuse to start if APP_ENV=production.
        if settings.is_prod:
            raise RuntimeError(
                "AUTH_MODE=none is forbidden in production. "
                "Set AUTH_MODE=clerk + CLERK_JWKS_URL."
            )
        return x_user_id or "anonymous"

    if mode in ("clerk", "custom"):
        if not authorization or not authorization.lower().startswith("bearer "):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing or malformed Authorization header",
                headers={"WWW-Authenticate": "Bearer"},
            )
        token = authorization.split(None, 1)[1].strip()
        jwks_url = settings.clerk_jwks_url if mode == "clerk" else settings.custom_auth_jwks_url
        if not jwks_url:
            raise RuntimeError(f"AUTH_MODE={mode} but JWKS URL not set")

        try:
            keys = await _jwks.get(jwks_url)
            unverified = jwt.get_unverified_header(token)
            key = next((k for k in keys if k["kid"] == unverified.get("kid")), None)
            if key is None:
                raise JWTError("kid not in JWKS")
            claims = jwt.decode(
                token, key,
                algorithms=[unverified.get("alg", "RS256")],
                # We deliberately skip aud verification here — Clerk uses a
                # rotating azp claim instead. Add aud check if your auth
                # provider sets a stable aud.
                options={"verify_aud": False},
            )
        except JWTError as e:
            logger.warning("auth: token rejected — {}", e)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid token: {e}",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Clerk puts the user id in `sub`. Adjust here if your IDP differs.
        user_id = claims.get("sub")
        if not user_id:
            raise HTTPException(401, "Token missing sub claim")
        request.state.user_id = user_id
        return user_id

    raise RuntimeError(f"Unknown AUTH_MODE: {mode!r}")


# Convenience: a dep that doesn't 401 but provides None for unauthenticated
# requests. Used for endpoints that personalize when known, public when not.
async def optional_user(
    request: Request,
    authorization: str | None = Header(None),
    x_user_id: str | None = Header(None),
) -> str | None:
    try:
        return await require_user(request, authorization, x_user_id)
    except HTTPException:
        return None
