"""
Credit ledger tests.

Money correctness is non-negotiable. We test:
  1. hold rejects when balance insufficient
  2. settle refunds the unspent portion
  3. settle charges overage when actual > held
  4. refund returns the whole hold
  5. credit() is idempotent on Stripe charge id (no double-credit)
"""
from __future__ import annotations

import pytest

from app.services import billing


def test_hold_rejects_when_no_balance():
    with pytest.raises(billing.InsufficientCredits):
        billing.hold("user-1", 100, job_id="job-1")


def test_credit_then_hold_works():
    bal = billing.credit("user-1", 500, source="stripe:starter", idempotency_key="cs_001")
    assert bal == 500
    hold_id = billing.hold("user-1", 100, job_id="job-1")
    assert billing.get_balance("user-1") == 400
    assert hold_id


def test_credit_is_idempotent():
    """Same Stripe checkout id arriving twice must NOT double-credit."""
    bal1 = billing.credit("user-2", 200, source="stripe:starter", idempotency_key="cs_dup")
    bal2 = billing.credit("user-2", 200, source="stripe:starter", idempotency_key="cs_dup")
    assert bal1 == bal2 == 200


def test_settle_refunds_unspent():
    billing.credit("user-3", 500, source="t", idempotency_key="cs_003")
    hold_id = billing.hold("user-3", 100, job_id="j3")  # held 100
    billing.settle("user-3", hold_id, actual_amount=60)  # used only 60
    # Refund of 40 → balance back to 440
    assert billing.get_balance("user-3") == 440


def test_settle_charges_overage():
    """If estimate was low, actual > held → wallet takes the hit."""
    billing.credit("user-4", 500, source="t", idempotency_key="cs_004")
    hold_id = billing.hold("user-4", 100, job_id="j4")
    billing.settle("user-4", hold_id, actual_amount=150)  # 50 over
    # 400 (after hold) - 50 (overage) = 350
    assert billing.get_balance("user-4") == 350


def test_refund_returns_full_hold():
    billing.credit("user-5", 500, source="t", idempotency_key="cs_005")
    hold_id = billing.hold("user-5", 100, job_id="j5")
    billing.refund("user-5", hold_id)
    assert billing.get_balance("user-5") == 500


def test_double_refund_is_safe():
    """Refunding the same hold twice must not double-credit."""
    billing.credit("user-6", 500, source="t", idempotency_key="cs_006")
    hold_id = billing.hold("user-6", 100, job_id="j6")
    billing.refund("user-6", hold_id)
    billing.refund("user-6", hold_id)  # second time = no-op
    assert billing.get_balance("user-6") == 500


def test_estimate_scales_with_target_count():
    a = billing.estimate_job_credits(target_count=1, use_ai_broll=False)
    b = billing.estimate_job_credits(target_count=3, use_ai_broll=False)
    assert b > a
    # Roughly 3x (base + smart_crop costs per reel)
    assert 2.5 * a <= b <= 3.5 * a


def test_ai_broll_costs_more():
    base = billing.estimate_job_credits(target_count=3, use_ai_broll=False)
    premium = billing.estimate_job_credits(target_count=3, use_ai_broll=True)
    assert premium > base
