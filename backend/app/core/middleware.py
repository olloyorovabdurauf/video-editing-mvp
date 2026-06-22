"""
Cross-cutting middleware.

Request-ID middleware: stamps every incoming request with a UUID (or honors
an inbound `X-Request-ID` from an upstream proxy/CDN), echoes it on the
response, and binds it into loguru's contextvar so every log line in this
request's call tree carries the id automatically.

Why this matters
----------------
At 3am when a single job is misbehaving, you grep one id across:
  - the API access log
  - the worker logs
  - your APM / Sentry
That triangulation is the difference between a 5-min fix and a 5-hour fix.
"""
from __future__ import annotations

import time
import uuid

from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


REQUEST_ID_HEADER = "X-Request-ID"


class MaxBodySizeMiddleware(BaseHTTPMiddleware):
    """
    Reject oversized request bodies early (413) so a spammer can't stream a
    multi-GB body into the API. JSON API requests are tiny; real video bytes
    go straight to R2 via signed URLs, never through this server.
    """

    def __init__(self, app, *, max_bytes: int):
        super().__init__(app)
        self.max_bytes = max_bytes

    async def dispatch(self, request: Request, call_next):
        cl = request.headers.get("content-length")
        if cl is not None:
            try:
                if int(cl) > self.max_bytes:
                    return Response(
                        status_code=413,
                        content=f"request body too large (max {self.max_bytes} bytes)",
                    )
            except ValueError:
                return Response(status_code=400, content="invalid Content-Length")
        return await call_next(request)


class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        req_id = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex[:16]
        request.state.request_id = req_id
        t0 = time.perf_counter()

        # Bind into loguru's context so every log line under this request
        # carries the id. The `with` ensures it pops off on response.
        with logger.contextualize(request_id=req_id):
            try:
                response: Response = await call_next(request)
            except Exception:
                logger.exception("unhandled exception in request")
                raise

            elapsed_ms = (time.perf_counter() - t0) * 1000
            response.headers[REQUEST_ID_HEADER] = req_id
            response.headers["Server-Timing"] = f"app;dur={elapsed_ms:.1f}"
            logger.info(
                "{} {} → {} in {:.1f}ms",
                request.method, request.url.path, response.status_code, elapsed_ms,
            )
            return response
