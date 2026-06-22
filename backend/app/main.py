"""FastAPI entry point."""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.v1.router import api_router
from app.config import get_settings
from app.core.logging import configure_logging
from app.core.middleware import MaxBodySizeMiddleware, RequestIDMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    # Sentry first so any error during the rest of bootstrapping gets reported.
    from app.core.observability import init_sentry
    init_sentry()
    settings = get_settings()
    settings.storage_local_dir.mkdir(parents=True, exist_ok=True)
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="AI Video Automation",
        version="0.1.0",
        lifespan=lifespan,
        # In prod hide /docs unless behind your VPN.
        docs_url="/docs" if not settings.is_prod else None,
        redoc_url=None,
    )

    # Middleware order matters: outermost first. RequestID first so its
    # contextvar binding wraps every other layer including CORS rejections.
    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(MaxBodySizeMiddleware, max_bytes=settings.max_request_body_bytes)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins.split(",") if settings.is_prod else ["*"],
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        # Explicit allow-list in prod (no wildcard with credentials) — browsers
        # reject "*" + credentials anyway, and it's good hygiene.
        allow_headers=["Authorization", "Content-Type", "X-Request-ID", "X-User-Id"],
        expose_headers=["X-Request-ID"],
        max_age=600,
    )
    app.include_router(api_router, prefix="/api/v1")
    app.mount(
        "/storage",
        StaticFiles(directory=settings.storage_local_dir),
        name="storage",
    )
    return app


app = create_app()
