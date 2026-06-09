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
