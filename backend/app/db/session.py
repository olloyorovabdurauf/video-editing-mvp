"""
Database engine + session management.

Gated by DATABASE_URL: if it's unset, the DB layer is *disabled* and every
repository call is a safe no-op — the app runs exactly as it does today on
Redis. Set DATABASE_URL (Neon) to turn durable persistence on. This makes the
Redis→Postgres cutover a config flip, not a risky big-bang deploy.

Sync SQLAlchemy on purpose: our writes happen inside Celery worker tasks (sync)
and a couple of API handlers; async would force `greenlet`/async sessions for
no real benefit at this scale.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from loguru import logger
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import get_settings
from app.db.models import Base

_engine = None
_Session: sessionmaker | None = None


def _normalize(url: str) -> str:
    # Accept the common `postgresql://` and `postgres://` forms; pin the psycopg3 driver.
    if url.startswith("postgresql+"):
        return url
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+psycopg://", 1)
    if url.startswith("postgres://"):
        return url.replace("postgres://", "postgresql+psycopg://", 1)
    return url


def init_engine(url: str | None = None) -> None:
    """Initialise the engine. Called at startup; tests call it with a sqlite URL."""
    global _engine, _Session
    url = url or get_settings().database_url
    if not url:
        logger.info("DATABASE_URL unset — DB layer disabled (Redis-only mode)")
        return
    connect_args: dict = {"connect_timeout": 10}
    if "psycopg" in _normalize(url):
        # Fly MPG / pgbouncer use TRANSACTION pooling, which is incompatible
        # with server-side prepared statements. Disabling them (None) is the
        # documented psycopg3 + pgbouncer setting — small perf cost at our scale.
        connect_args["prepare_threshold"] = None
    _engine = create_engine(
        _normalize(url),
        pool_pre_ping=True,       # survive pooler idle-disconnects
        pool_size=5, max_overflow=5,
        connect_args=connect_args,
        future=True,
    )
    _Session = sessionmaker(bind=_engine, expire_on_commit=False, class_=Session)
    _create_all_safely()
    logger.info("DB layer enabled ({} tables)", len(Base.metadata.tables))


def _create_all_safely() -> None:
    """
    Create tables idempotently. The combined machine runs the API process AND
    several Celery worker processes, each calling init_engine() — concurrent
    create_all() races on Postgres ENUM creation (`pg_type` unique violation).
    A transaction-level advisory lock (pgbouncer-safe; auto-released at commit)
    serializes DDL so only one process builds the schema and the rest no-op.

    DDL failure is logged, not fatal: the API must never crash-loop because of
    a one-time schema step. (Move to Alembic migrations as the schema evolves.)
    """
    assert _engine is not None
    try:
        if _engine.dialect.name == "postgresql":
            with _engine.begin() as conn:
                conn.exec_driver_sql("SELECT pg_advisory_xact_lock(91273042)")
                Base.metadata.create_all(bind=conn, checkfirst=True)
        else:
            Base.metadata.create_all(_engine)
    except Exception as e:
        logger.error("schema create_all failed (continuing; DB ops may degrade): {}", e)


def db_enabled() -> bool:
    return _Session is not None


@contextmanager
def session_scope() -> Iterator[Session | None]:
    """Transactional scope. Yields None when the DB is disabled (callers no-op)."""
    if _Session is None:
        yield None
        return
    s = _Session()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()
