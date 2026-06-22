"""
Durable relational schema (Postgres target).

WHY a DB now, and WHY this shape
--------------------------------
Today everything lives in Redis. That's correct for *hot* job state (progress,
artifacts) — fast, ephemeral, fine to lose. It is NOT correct for things you
must never lose or must query historically:
  - who a user is and what plan they're on,
  - the money ledger (an audit flagged Redis-as-ledger as the #1 durability risk),
  - usage counters that gate quotas and feed billing.

So the migration is deliberate and minimal: **durable entities → Postgres,
hot execution state → Redis.** Don't move job *progress* into Postgres (write
amplification for no benefit). `ProcessingJob` here is the *durable record* of a
job (final status, cost, timing) written once on completion, not the live poller.

Status: schema is defined and ready; wiring is the next PR (see app/db/README).
Lean on purpose — every column earns its place. Add tables when a feature needs
them, not before (YAGNI).
"""
from __future__ import annotations

import enum
import uuid
from datetime import datetime

from sqlalchemy import (
    BigInteger, Boolean, DateTime, Enum, Float, ForeignKey, Index, Integer,
    String, Text, UniqueConstraint, func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# Auto-incrementing 64-bit PK that also works on SQLite (which only
# auto-increments INTEGER PRIMARY KEY). BigInteger on Postgres, Integer on SQLite.
AutoBigInt = BigInteger().with_variant(Integer, "sqlite")


def _uuid() -> str:
    return uuid.uuid4().hex


class Base(DeclarativeBase):
    pass


class Plan(str, enum.Enum):
    FREE = "free"
    STARTER = "starter"
    PRO = "pro"
    AGENCY = "agency"


class JobStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class SourceKind(str, enum.Enum):
    YOUTUBE = "youtube"
    URL = "url"          # arbitrary direct media URL
    UPLOAD = "upload"    # user-uploaded file in R2


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    # Clerk's `sub` — the join key between the JWT and our DB. Unique + indexed.
    auth_provider_id: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    email: Mapped[str] = mapped_column(String(320), index=True)
    plan: Mapped[Plan] = mapped_column(Enum(Plan), default=Plan.FREE)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    projects: Mapped[list["Project"]] = relationship(back_populates="user")
    videos: Mapped[list["Video"]] = relationship(back_populates="user")


class Subscription(Base):
    __tablename__ = "subscriptions"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    stripe_customer_id: Mapped[str | None] = mapped_column(String(64), index=True)
    stripe_subscription_id: Mapped[str | None] = mapped_column(String(64), unique=True)
    plan: Mapped[Plan] = mapped_column(Enum(Plan), default=Plan.FREE)
    status: Mapped[str] = mapped_column(String(32), default="active")   # active|past_due|canceled
    credits_balance: Mapped[int] = mapped_column(Integer, default=0)    # cached; ledger is source of truth
    current_period_end: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(),
                                                 onupdate=func.now())


# ---------------------------------------------------------------------------
# Content
# ---------------------------------------------------------------------------

class Project(Base):
    """Optional grouping ("my podcast season 3"). One level — no nesting (YAGNI)."""
    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    title: Mapped[str] = mapped_column(String(200), default="Untitled")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    user: Mapped[User] = relationship(back_populates="projects")
    videos: Mapped[list["Video"]] = relationship(back_populates="project")


class Video(Base):
    """A source video the user brought in (the thing clips are cut FROM)."""
    __tablename__ = "videos"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    project_id: Mapped[str | None] = mapped_column(ForeignKey("projects.id", ondelete="SET NULL"))
    title: Mapped[str | None] = mapped_column(String(300))
    source_kind: Mapped[SourceKind] = mapped_column(Enum(SourceKind))
    source_url: Mapped[str | None] = mapped_column(Text)          # null for uploads
    storage_key: Mapped[str | None] = mapped_column(String(512))  # R2 key for the original
    duration_s: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    user: Mapped[User] = relationship(back_populates="videos")
    project: Mapped[Project | None] = relationship(back_populates="videos")
    clips: Mapped[list["Clip"]] = relationship(back_populates="video")
    jobs: Mapped[list["ProcessingJob"]] = relationship(back_populates="video")


class Clip(Base):
    """A generated reel: a window of a Video plus its rendered output in R2."""
    __tablename__ = "clips"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    video_id: Mapped[str] = mapped_column(ForeignKey("videos.id", ondelete="CASCADE"), index=True)
    start_s: Mapped[float] = mapped_column(Float)
    end_s: Mapped[float] = mapped_column(Float)
    hook_score: Mapped[float] = mapped_column(Float, default=0.0)
    reason: Mapped[str | None] = mapped_column(Text)
    aspect: Mapped[str] = mapped_column(String(8), default="9:16")
    storage_key: Mapped[str | None] = mapped_column(String(512))   # R2 key for the reel mp4
    thumbnail_key: Mapped[str | None] = mapped_column(String(512))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    video: Mapped[Video] = relationship(back_populates="clips")


# ---------------------------------------------------------------------------
# Processing + usage
# ---------------------------------------------------------------------------

class ProcessingJob(Base):
    """
    Durable record of a pipeline run. Live progress stays in Redis; this row is
    written on terminal state for history, cost accounting, and support.
    """
    __tablename__ = "processing_jobs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)   # == Redis job_id
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    video_id: Mapped[str | None] = mapped_column(ForeignKey("videos.id", ondelete="SET NULL"))
    status: Mapped[JobStatus] = mapped_column(Enum(JobStatus), default=JobStatus.PENDING, index=True)
    error_message: Mapped[str | None] = mapped_column(Text)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    processing_time_s: Mapped[float | None] = mapped_column(Float)
    clips_produced: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    video: Mapped[Video | None] = relationship(back_populates="jobs")

    __table_args__ = (Index("ix_jobs_user_created", "user_id", "created_at"),)


class LedgerEntry(Base):
    """
    Append-only credit ledger — the source of truth for money. Balance is the
    sum of `delta` for a user. Never UPDATE or DELETE rows; corrections are new
    rows. `idempotency_key` makes every operation safe to retry (Stripe webhook
    redelivery, Celery task retry) without double-applying.
    """
    __tablename__ = "ledger_entries"

    id: Mapped[int] = mapped_column(AutoBigInt, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    delta: Mapped[int] = mapped_column(Integer)               # +credit / -hold / +refund / settle adj.
    kind: Mapped[str] = mapped_column(String(16))             # credit|hold|settle|refund
    ref: Mapped[str | None] = mapped_column(String(128))      # job_id or stripe session id
    idempotency_key: Mapped[str | None] = mapped_column(String(160), unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (Index("ix_ledger_user", "user_id"),)


class UsageCounter(Base):
    """
    Per-user, per-period rollup for quotas + billing analytics. One row per
    (user, YYYYMM). Incremented as jobs complete; the daily quota lives in Redis
    (hot path) and reconciles here.
    """
    __tablename__ = "usage_counters"

    id: Mapped[int] = mapped_column(AutoBigInt, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    period: Mapped[str] = mapped_column(String(6))          # YYYYMM
    jobs_count: Mapped[int] = mapped_column(Integer, default=0)
    minutes_processed: Mapped[float] = mapped_column(Float, default=0.0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)

    __table_args__ = (UniqueConstraint("user_id", "period", name="uq_usage_user_period"),)
