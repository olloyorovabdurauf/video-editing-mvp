"""
Postgres data-layer tests, run against in-memory SQLite (same SQLAlchemy code
path). Covers the money-critical ledger semantics + user/job/usage persistence.
"""
from __future__ import annotations

import pytest

from app.db import repositories as repo
from app.db import session as db_session


@pytest.fixture
def db(monkeypatch):
    """Fresh in-memory SQLite DB with tables created, wired into session_scope."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool
    from app.db.models import Base

    # StaticPool + one shared connection = the in-memory DB persists across the
    # many session_scope() calls a single test makes (else each gets a fresh DB).
    engine = create_engine("sqlite://", future=True, poolclass=StaticPool,
                           connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    Maker = sessionmaker(bind=engine, expire_on_commit=False)
    monkeypatch.setattr(db_session, "_engine", engine)
    monkeypatch.setattr(db_session, "_Session", Maker)
    yield


def _user(db) -> str:
    return repo.upsert_user("clerk_sub_1", "a@example.com")


# ---------------------------------------------------------------------------
# DB-disabled = safe no-op
# ---------------------------------------------------------------------------

def test_disabled_db_is_noop(monkeypatch):
    monkeypatch.setattr(db_session, "_Session", None)
    assert repo.upsert_user("x", "y@z.com") is None
    assert repo.get_balance("x") is None
    assert repo.hold("x", 10, job_id="j") is None        # no crash


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------

def test_upsert_user_idempotent(db):
    a = repo.upsert_user("clerk_sub_1", "a@example.com")
    b = repo.upsert_user("clerk_sub_1", "a@example.com")
    assert a is not None and a == b


def test_upsert_user_grants_signup_credits_once(db):
    # First login grants the free credits, atomically with user creation.
    uid = repo.upsert_user("clerk_sub_new", "n@example.com", signup_credits=500)
    assert repo.get_balance(uid) == 500
    # A returning login hits the existing-user branch — no second grant.
    again = repo.upsert_user("clerk_sub_new", "n@example.com", signup_credits=500)
    assert again == uid and repo.get_balance(uid) == 500


def test_upsert_user_no_grant_when_zero(db):
    uid = repo.upsert_user("clerk_sub_free", "f@example.com")  # default 0
    assert repo.get_balance(uid) == 0


# ---------------------------------------------------------------------------
# Credit ledger — money correctness
# ---------------------------------------------------------------------------

def test_credit_then_balance(db):
    u = _user(db)
    repo.credit(u, 500, source="stripe:starter", idempotency_key="cs_1")
    assert repo.get_balance(u) == 500


def test_credit_idempotent(db):
    u = _user(db)
    repo.credit(u, 500, source="stripe:starter", idempotency_key="cs_dup")
    repo.credit(u, 500, source="stripe:starter", idempotency_key="cs_dup")   # redelivery
    assert repo.get_balance(u) == 500


def test_hold_rejects_when_insufficient(db):
    u = _user(db)
    with pytest.raises(repo.InsufficientCredits):
        repo.hold(u, 100, job_id="j1")


def test_hold_then_settle_refunds_unspent(db):
    u = _user(db)
    repo.credit(u, 500, source="t", idempotency_key="cs_2")
    hold_id = repo.hold(u, 100, job_id="j2")       # balance 400
    repo.settle(u, hold_id, actual=60)   # used 60 → +40 back
    assert repo.get_balance(u) == 440


def test_hold_then_settle_charges_overage(db):
    u = _user(db)
    repo.credit(u, 500, source="t", idempotency_key="cs_3")
    hold_id = repo.hold(u, 100, job_id="j3")
    repo.settle(u, hold_id, actual=150)  # 50 over → 400 - 50
    assert repo.get_balance(u) == 350


def test_refund_returns_full_hold(db):
    u = _user(db)
    repo.credit(u, 500, source="t", idempotency_key="cs_4")
    hold_id = repo.hold(u, 100, job_id="j4")       # 400
    repo.refund(u, hold_id)              # +100 → 500
    assert repo.get_balance(u) == 500


def test_double_refund_safe(db):
    u = _user(db)
    repo.credit(u, 500, source="t", idempotency_key="cs_5")
    hold_id = repo.hold(u, 100, job_id="j5")
    repo.refund(u, hold_id)
    repo.refund(u, hold_id)              # idempotent — no double credit
    assert repo.get_balance(u) == 500


def test_double_hold_safe(db):
    u = _user(db)
    repo.credit(u, 500, source="t", idempotency_key="cs_6")
    repo.hold(u, 100, job_id="j6")
    repo.hold(u, 100, job_id="j6")                 # same job retried — one hold only
    assert repo.get_balance(u) == 400


# ---------------------------------------------------------------------------
# Durable job + usage history
# ---------------------------------------------------------------------------

def test_record_job_and_usage(db):
    u = _user(db)
    repo.record_completed_job(job_id="job-1", user_id=u, status="completed",
                              cost_usd=0.42, processing_time_s=85.0, clips=3,
                              audio_minutes=15.0)
    with db_session.session_scope() as s:
        from app.db import models
        job = s.get(models.ProcessingJob, "job-1")
        assert job.cost_usd == 0.42 and job.clips_produced == 3
        usage = s.query(models.UsageCounter).filter_by(user_id=u).one()
        assert usage.jobs_count == 1 and usage.minutes_processed == 15.0


# ---------------------------------------------------------------------------
# Auth identity resolution: Clerk sub -> internal users.id
# (the join that makes durable writes for real signed-in users actually persist)
# ---------------------------------------------------------------------------

def test_resolve_internal_user_id_creates_and_caches(db):
    from sqlalchemy import select
    from app.core import auth
    from app.db import models

    auth._uid_cache.clear()
    sub = "user_2clerkXYZ"
    uid = auth._resolve_internal_user_id(sub, {"email": "z@example.com"})

    # Returns our internal users.id, NOT the raw Clerk sub — FKs reference users.id.
    assert uid and uid != sub
    with db_session.session_scope() as s:
        row = s.scalar(select(models.User).where(models.User.auth_provider_id == sub))
        assert row is not None and row.id == uid and row.email == "z@example.com"

    # A durable write keyed by the resolved id persists (the previously-broken path).
    repo.record_completed_job(job_id="job-real", user_id=uid, status="completed",
                              cost_usd=0.1, processing_time_s=10.0, clips=1,
                              audio_minutes=2.0)
    with db_session.session_scope() as s:
        assert s.get(models.ProcessingJob, "job-real") is not None

    # Second call is cache-served and stable.
    assert auth._resolve_internal_user_id(sub, {}) == uid
    assert auth._uid_cache.get(sub) == uid


def test_resolve_internal_user_id_passthrough_when_db_disabled(monkeypatch):
    from app.core import auth
    auth._uid_cache.clear()
    monkeypatch.setattr(db_session, "_Session", None)
    # DB off (Redis-only mode): the sub IS the id, and nothing is cached/persisted.
    assert auth._resolve_internal_user_id("raw_sub", {"email": "a@b.com"}) == "raw_sub"
    assert "raw_sub" not in auth._uid_cache
