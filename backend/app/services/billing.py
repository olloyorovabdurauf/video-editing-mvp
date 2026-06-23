"""
Credits & billing.

Design choices
--------------
- **Credits, not dollars.** Decoupling the unit-of-account from USD lets us
  reprice (LLMs get cheaper monthly) without changing user-facing prices.
  1 credit ≈ 1 cent of inference budget today; we can rescale later.

- **Reservation pattern.** When a job is enqueued we *hold* (decrement +
  log a hold) the estimated credits. On success we settle (no change).
  On failure we refund. Prevents double-spend if a worker dies mid-job.

- **Idempotent debits.** Every debit carries an `idempotency_key` so retried
  Celery tasks don't double-charge.

- **Source of truth = Redis hash + append-only log.** Postgres comes when
  we need analytics / disputes. For now the log is in Redis stream form.

Stripe is the funding source; webhooks credit the wallet. The webhook
handler is in api/v1/endpoints/billing.py.

Pricing knobs (in credits)
--------------------------
  reel base                  : 25
  AI b-roll (per generation) : 60
  smart crop                 : 5
  premium tier multiplier    : 1.0  (kept for future use)
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Literal

import redis

from app.config import get_settings

_r = redis.from_url(get_settings().redis_url, decode_responses=True)


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------

PRICING_CREDITS = {
    "reel_base":      25,    # one reel render
    "ai_broll_gen":   60,    # one generation call (Runway-priced)
    "smart_crop":      5,    # face-tracked vertical crop
    "stock_broll":     0,    # Pexels — free; we eat the rate limit
}


def estimate_job_credits(
    *,
    target_count: int,
    use_ai_broll: bool,
    avg_brolls_per_reel: int = 2,
    use_smart_crop: bool = True,
) -> int:
    """Up-front estimate used to hold credits at enqueue time."""
    per_reel = PRICING_CREDITS["reel_base"]
    if use_smart_crop:
        per_reel += PRICING_CREDITS["smart_crop"]
    if use_ai_broll:
        per_reel += PRICING_CREDITS["ai_broll_gen"] * avg_brolls_per_reel
    return per_reel * target_count


# ---------------------------------------------------------------------------
# Wallet
# ---------------------------------------------------------------------------

class InsufficientCredits(RuntimeError):
    pass


def _wallet_key(user_id: str) -> str:
    return f"wallet:{user_id}"


def _ledger_key(user_id: str) -> str:
    return f"ledger:{user_id}"


def _idem_key(key: str) -> str:
    return f"idem:{key}"


# Durable cutover: when DATABASE_URL is set, money lives in Postgres (durable,
# append-only ledger). Until then, the Redis path below runs unchanged. There
# is no live paying money yet, so this flip is safe to land before launch.
def _use_db() -> bool:
    from app.db.session import db_enabled
    return db_enabled()


def get_balance(user_id: str) -> int:
    if _use_db():
        from app.db import repositories
        return repositories.get_balance(user_id) or 0
    return int(_r.get(_wallet_key(user_id)) or 0)


def credit(user_id: str, amount: int, *, source: str, idempotency_key: str) -> int:
    """Add credits. Idempotent on idempotency_key (e.g. Stripe charge id)."""
    if _use_db():
        from app.db import repositories
        return repositories.credit(user_id, amount, source=source,
                                   idempotency_key=idempotency_key) or 0
    if amount <= 0:
        return get_balance(user_id)
    if _r.set(_idem_key(idempotency_key), "1", nx=True, ex=86400 * 30) is None:
        # Already applied — return current balance, no double-credit.
        return get_balance(user_id)
    pipe = _r.pipeline()
    pipe.incrby(_wallet_key(user_id), amount)
    pipe.xadd(_ledger_key(user_id), {
        "ts": str(time.time()), "delta": str(amount), "source": source,
        "kind": "credit", "idem": idempotency_key,
    })
    new_balance, _ = pipe.execute()
    return int(new_balance)


def hold(user_id: str, amount: int, *, job_id: str) -> str:
    """
    Reserve `amount` credits for a job. Raises if the wallet can't cover.
    Returns a hold_id to settle/refund later.
    """
    if amount < 0:
        raise ValueError("amount must be >= 0")
    if _use_db():
        from app.db import repositories
        try:
            return repositories.hold(user_id, amount, job_id=job_id)
        except repositories.InsufficientCredits as e:
            raise InsufficientCredits(str(e)) from e
    hold_id = uuid.uuid4().hex
    # Atomic check-then-decrement via Lua to prevent race with another debit.
    script = _r.register_script("""
        local bal = tonumber(redis.call('GET', KEYS[1]) or '0')
        local amt = tonumber(ARGV[1])
        if bal < amt then return -1 end
        redis.call('DECRBY', KEYS[1], amt)
        redis.call('HSET', KEYS[2], 'amount', amt, 'job_id', ARGV[2], 'ts', ARGV[3])
        return bal - amt
    """)
    new_balance = int(script(
        keys=[_wallet_key(user_id), f"hold:{hold_id}"],
        args=[amount, job_id, time.time()],
    ))
    if new_balance < 0:
        raise InsufficientCredits(
            f"need {amount} credits, balance {get_balance(user_id)}"
        )
    _r.xadd(_ledger_key(user_id), {
        "ts": str(time.time()), "delta": str(-amount), "kind": "hold",
        "hold_id": hold_id, "job_id": job_id,
    })
    return hold_id


def settle(user_id: str, hold_id: str, *, actual_amount: int) -> None:
    """
    Convert a hold into a final debit. If actual_amount < held, refund the
    difference. If actual_amount > held, debit additionally (this should be
    rare — it means our estimate was low).
    """
    if _use_db():
        from app.db import repositories
        repositories.settle(user_id, hold_id, actual=actual_amount)
        return
    held = _r.hgetall(f"hold:{hold_id}") or {}
    if not held:
        return  # already settled or never existed
    held_amount = int(held.get("amount", 0))
    delta = held_amount - actual_amount

    if delta > 0:
        # Refund unused portion.
        _r.incrby(_wallet_key(user_id), delta)
        _r.xadd(_ledger_key(user_id), {
            "ts": str(time.time()), "delta": str(delta), "kind": "settle_refund",
            "hold_id": hold_id,
        })
    elif delta < 0:
        # Estimate was low; charge the overage. This may take wallet negative
        # for one job — acceptable; we'll block the user's *next* job.
        _r.incrby(_wallet_key(user_id), delta)  # delta is negative
        _r.xadd(_ledger_key(user_id), {
            "ts": str(time.time()), "delta": str(delta), "kind": "settle_overage",
            "hold_id": hold_id,
        })
    _r.delete(f"hold:{hold_id}")


def refund(user_id: str, hold_id: str) -> None:
    """Full refund (job failed before any work)."""
    if _use_db():
        from app.db import repositories
        repositories.refund(user_id, hold_id)
        return
    held = _r.hgetall(f"hold:{hold_id}") or {}
    if not held:
        return
    amount = int(held.get("amount", 0))
    _r.incrby(_wallet_key(user_id), amount)
    _r.xadd(_ledger_key(user_id), {
        "ts": str(time.time()), "delta": str(amount), "kind": "refund",
        "hold_id": hold_id,
    })
    _r.delete(f"hold:{hold_id}")


# ---------------------------------------------------------------------------
# Per-job cost accounting (the *actual* spend, separate from credits)
# ---------------------------------------------------------------------------

@dataclass
class CostEvent:
    job_id: str
    kind: Literal["openai_whisper", "openai_gpt4o", "runway", "higgsfield", "pexels"]
    amount_usd: float
    units: float = 1.0    # seconds of audio, tokens, generations, ...


def record_cost(event: CostEvent) -> None:
    """Append a cost event to the job's cost ledger; debit nothing."""
    _r.xadd(f"costs:{event.job_id}", {
        "ts": str(time.time()), "kind": event.kind,
        "amount_usd": f"{event.amount_usd:.4f}", "units": str(event.units),
    })


def total_cost_usd(job_id: str) -> float:
    """Sum of recorded costs for a job. Used by the eval harness."""
    total = 0.0
    for _id, fields in _r.xrange(f"costs:{job_id}"):
        total += float(fields.get("amount_usd", 0))
    return total
