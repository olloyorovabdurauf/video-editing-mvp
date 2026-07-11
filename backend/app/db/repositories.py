"""
Repository layer — the ONLY place that reads/writes the database.

Every function is a safe no-op when the DB is disabled (DATABASE_URL unset), so
the pipeline never breaks just because Postgres isn't provisioned yet. The
credit ledger here is the durable replacement for the Redis ledger; flipping
billing onto it is a deliberate cutover (see app/db/README.md), not automatic.
"""
from __future__ import annotations

from datetime import datetime, timezone

from loguru import logger
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db import models
from app.db.session import session_scope


class InsufficientCredits(Exception):
    def __init__(self, needed: int, balance: int):
        self.needed, self.balance = needed, balance
        super().__init__(f"need {needed} credits, balance {balance}")


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

def upsert_user(auth_provider_id: str, email: str, signup_credits: int = 0) -> str | None:
    """
    Idempotent: create the user on first login, else return existing id.

    On creation, grant `signup_credits` free credits in the SAME transaction so
    a new user is never momentarily broke (the credit gate would 402 them). The
    grant is keyed `signup:<id>`, so it can apply at most once per user, ever.
    """
    with session_scope() as s:
        if s is None:
            return None
        existing = s.scalar(select(models.User).where(
            models.User.auth_provider_id == auth_provider_id))
        if existing:
            return existing.id
        user = models.User(auth_provider_id=auth_provider_id, email=email)
        s.add(user)
        try:
            s.flush()
        except IntegrityError:           # race: another request created it
            s.rollback()
            again = s.scalar(select(models.User).where(
                models.User.auth_provider_id == auth_provider_id))
            return again.id if again else None
        if signup_credits > 0:
            _append(s, user_id=user.id, delta=signup_credits, kind="credit",
                    ref="signup", idem=f"signup:{user.id}")
        return user.id


# ---------------------------------------------------------------------------
# Durable job + usage history
# ---------------------------------------------------------------------------

def record_completed_job(*, job_id: str, user_id: str, status: str, cost_usd: float,
                         processing_time_s: float | None, clips: int,
                         audio_minutes: float, error: str | None = None) -> None:
    with session_scope() as s:
        if s is None:
            return
        row = s.get(models.ProcessingJob, job_id)
        if row is None:
            row = models.ProcessingJob(id=job_id, user_id=user_id)
            s.add(row)
        try:
            row.status = models.JobStatus(status)
        except ValueError:
            row.status = models.JobStatus.COMPLETED
        row.cost_usd = cost_usd
        row.processing_time_s = processing_time_s
        row.clips_produced = clips
        row.error_message = error
        row.finished_at = datetime.now(timezone.utc)
        _bump_usage(s, user_id=user_id, minutes=audio_minutes, cost=cost_usd, add_job=1)


def _bump_usage(s: Session, *, user_id: str, minutes: float, cost: float, add_job: int) -> None:
    period = datetime.now(timezone.utc).strftime("%Y%m")
    row = s.scalar(select(models.UsageCounter).where(
        models.UsageCounter.user_id == user_id, models.UsageCounter.period == period))
    if row is None:
        row = models.UsageCounter(user_id=user_id, period=period)
        s.add(row)
        s.flush()
    row.jobs_count += add_job
    row.minutes_processed += minutes
    row.cost_usd += cost


# ---------------------------------------------------------------------------
# Credit ledger (durable, append-only, idempotent) — money source of truth
# ---------------------------------------------------------------------------

def _balance(s: Session, user_id: str) -> int:
    return int(s.scalar(select(func.coalesce(func.sum(models.LedgerEntry.delta), 0))
                        .where(models.LedgerEntry.user_id == user_id)) or 0)


def _append(s: Session, *, user_id: str, delta: int, kind: str, ref: str | None,
            idem: str | None) -> bool:
    """Append one entry. Returns False if idem key already applied (no-op)."""
    if idem:
        exists = s.scalar(select(models.LedgerEntry.id).where(
            models.LedgerEntry.idempotency_key == idem))
        if exists:
            return False
    s.add(models.LedgerEntry(user_id=user_id, delta=delta, kind=kind, ref=ref,
                             idempotency_key=idem))
    try:
        s.flush()
        return True
    except IntegrityError:               # concurrent insert with same idem
        s.rollback()
        return False


def credit(user_id: str, amount: int, *, source: str, idempotency_key: str) -> int | None:
    """Add credits (Stripe purchase). Idempotent on the Stripe session id."""
    with session_scope() as s:
        if s is None:
            return None
        _append(s, user_id=user_id, delta=abs(amount), kind="credit",
                ref=source, idem=idempotency_key)
        return _balance(s, user_id)


def get_balance(user_id: str) -> int | None:
    with session_scope() as s:
        return None if s is None else _balance(s, user_id)


def hold(user_id: str, amount: int, *, job_id: str) -> str | None:
    """Reserve credits for a job. Raises InsufficientCredits. Returns hold id."""
    with session_scope() as s:
        if s is None:
            return None
        bal = _balance(s, user_id)
        if bal < amount:
            raise InsufficientCredits(amount, bal)
        _append(s, user_id=user_id, delta=-abs(amount), kind="hold",
                ref=job_id, idem=f"hold:{job_id}")
        return f"hold:{job_id}"


def _held_for(s: Session, job_id: str) -> int:
    """How much the hold for this job reserved (the hold entry has delta=-held)."""
    delta = s.scalar(select(models.LedgerEntry.delta).where(
        models.LedgerEntry.kind == "hold", models.LedgerEntry.ref == job_id))
    return -int(delta) if delta is not None else 0


def settle(user_id: str, hold_id: str, *, actual: int) -> int | None:
    """Adjust a hold to the actual cost (refund the unspent part). Idempotent."""
    job_id = hold_id.split(":", 1)[-1]
    with session_scope() as s:
        if s is None:
            return None
        held = _held_for(s, job_id)
        _append(s, user_id=user_id, delta=(held - actual), kind="settle",
                ref=job_id, idem=f"settle:{job_id}")
        return _balance(s, user_id)


def refund(user_id: str, hold_id: str) -> int | None:
    """Return the full held amount (job failed). Idempotent."""
    job_id = hold_id.split(":", 1)[-1]
    with session_scope() as s:
        if s is None:
            return None
        held = _held_for(s, job_id)
        _append(s, user_id=user_id, delta=abs(held), kind="refund",
                ref=job_id, idem=f"refund:{job_id}")
        return _balance(s, user_id)


def record_feedback(user_id: str, *, job_id: str, clip_index: int,
                    verdict: str, reason: str | None) -> bool:
    """Store per-clip thumbs feedback. Last write wins per (user, job, clip)."""
    from app.db.models import ClipFeedback
    with session_scope() as s:
        if s is None:
            return False
        existing = (s.query(ClipFeedback)
                    .filter_by(user_id=user_id, job_id=job_id, clip_index=clip_index)
                    .one_or_none())
        if existing:
            existing.verdict = verdict
            existing.reason = reason
        else:
            s.add(ClipFeedback(user_id=user_id, job_id=job_id,
                               clip_index=clip_index, verdict=verdict, reason=reason))
        return True
