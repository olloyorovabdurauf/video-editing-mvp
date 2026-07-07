#syntax=docker/dockerfile:1.7

# ---------------------------------------------------------------------------
# Base image — ffmpeg + Python only. No build tools.
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app

# ffmpeg is the dominant runtime dep. tini gives us a real PID 1 for clean shutdown.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    tini \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user. Storage volume will be chowned by entrypoint or by the
# host (docker-compose volumes inherit ownership).
RUN useradd --create-home --uid 1000 app
WORKDIR /app


# ---------------------------------------------------------------------------
# PO-token service builder — bgutil's Node companion for yt-dlp. YouTube
# bot-checks datacenter IPs ("Sign in to confirm you're not a bot"); this
# service mints BotGuard proof-of-origin tokens so downloads pass without
# cookies or a residential proxy.
# Version MUST match the bgutil-ytdlp-pot-provider pin in requirements.txt.
# ---------------------------------------------------------------------------
FROM node:20-slim AS potbuilder
ARG BGUTIL_VERSION=1.3.1
ADD https://github.com/Brainicism/bgutil-ytdlp-pot-provider/archive/refs/tags/${BGUTIL_VERSION}.tar.gz /tmp/bgutil.tar.gz
RUN mkdir -p /opt/bgutil && tar -xzf /tmp/bgutil.tar.gz -C /opt/bgutil --strip-components=1 \
    && cd /opt/bgutil/server && npm ci && npx tsc && npm prune --omit=dev


# ---------------------------------------------------------------------------
# Builder — install Python deps in an isolated wheel cache.
# ---------------------------------------------------------------------------
FROM base AS builder
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip wheel --wheel-dir=/wheels -r requirements.txt


# ---------------------------------------------------------------------------
# Runtime — copy wheels, install, drop privileges.
# ---------------------------------------------------------------------------
FROM base AS runtime

# Node runtime for the PO-token service: the node 20 binary from the build
# image (both bookworm → ABI-compatible) + the built server with prod deps.
COPY --from=potbuilder /usr/local/bin/node /usr/local/bin/node
COPY --from=potbuilder --chown=app:app /opt/bgutil/server /opt/bgutil/server

COPY --from=builder /wheels /wheels
COPY requirements.txt .
RUN pip install --no-index --find-links=/wheels -r requirements.txt && rm -rf /wheels

# Application code last → maximal docker layer cache hit on dep-only changes.
COPY --chown=app:app app ./app

USER app

EXPOSE 8000

# tini reaps zombies + handles SIGTERM cleanly (important for Celery workers
# in particular — Celery's default shutdown sequence respects SIGTERM).
ENTRYPOINT ["/usr/bin/tini", "--"]

# Default CMD = API. Worker services in docker-compose override this with
# `celery -A app.core.celery_app.celery_app worker ...`
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "2", \
     "--proxy-headers", \
     "--forwarded-allow-ips", "*"]

# Healthcheck — orchestrators (ECS, Kubernetes, docker swarm) use this.
HEALTHCHECK --interval=15s --timeout=3s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/api/v1/livez || exit 1
