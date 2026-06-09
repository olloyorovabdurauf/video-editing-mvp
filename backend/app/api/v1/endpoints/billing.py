"""
Stripe webhook + wallet query endpoints.

The webhook handler verifies the Stripe signature, idempotently credits the
wallet, and acks. Everything else (subscription UI, customer portal) is a
plain Stripe Checkout redirect — we don't build a billing UI ourselves.
"""
from __future__ import annotations

from fastapi import APIRouter, Header, HTTPException, Request, status
from loguru import logger
from pydantic import BaseModel

from app.config import get_settings
from app.services import billing

router = APIRouter(prefix="/billing", tags=["billing"])


class BalanceResponse(BaseModel):
    user_id: str
    balance: int


@router.get("/balance/{user_id}", response_model=BalanceResponse)
async def get_balance(user_id: str) -> BalanceResponse:
    return BalanceResponse(user_id=user_id, balance=billing.get_balance(user_id))


# Credit pack catalog — usually behind feature flag in prod, exposed publicly
# here for simplicity. Real Stripe products/prices live in dashboard.
CREDIT_PACKS = {
    "starter":  {"credits": 500,  "usd": 5},
    "pro":      {"credits": 1200, "usd": 10},   # ~20% bonus
    "agency":   {"credits": 6000, "usd": 40},   # ~33% bonus
}


@router.get("/packs")
async def list_packs() -> dict:
    return CREDIT_PACKS


@router.post("/webhook/stripe", status_code=status.HTTP_200_OK)
async def stripe_webhook(
    request: Request,
    stripe_signature: str = Header(None, alias="Stripe-Signature"),
) -> dict:
    """
    Stripe → us. Verifies the signature and credits the wallet on
    `checkout.session.completed` for credit-pack purchases.

    We only handle the events we care about; everything else is acked-and-ignored
    so Stripe doesn't retry forever.
    """
    settings = get_settings()
    body = await request.body()

    try:
        import stripe                                                  # type: ignore
    except ImportError:
        raise HTTPException(500, "stripe SDK not installed")

    if not settings.stripe_webhook_secret:
        # Soft-fail in dev: parse without verifying.
        try:
            event = stripe.Event.construct_from(
                stripe.util.json.loads(body), settings.stripe_api_key or "sk_dev"
            )
            logger.warning("STRIPE_WEBHOOK_SECRET unset — accepting unverified event {}",
                           event.get("type"))
        except Exception as e:
            raise HTTPException(400, f"unparseable event: {e}")
    else:
        try:
            event = stripe.Webhook.construct_event(
                body, stripe_signature, settings.stripe_webhook_secret,
            )
        except stripe.error.SignatureVerificationError:
            raise HTTPException(400, "invalid signature")
        except Exception as e:
            raise HTTPException(400, f"webhook parse failed: {e}")

    if event["type"] == "checkout.session.completed":
        sess = event["data"]["object"]
        user_id = (sess.get("metadata") or {}).get("user_id")
        pack = (sess.get("metadata") or {}).get("pack")
        if not user_id or not pack or pack not in CREDIT_PACKS:
            logger.warning("stripe event missing metadata: {}", sess.get("id"))
            return {"ok": True, "skipped": "missing metadata"}
        amount = CREDIT_PACKS[pack]["credits"]
        new_balance = billing.credit(
            user_id, amount,
            source=f"stripe:{pack}",
            idempotency_key=sess["id"],          # Stripe session id = natural idem key
        )
        logger.info("stripe → credited {} {} credits (balance {})", user_id, amount, new_balance)
        return {"ok": True, "credited": amount, "balance": new_balance}

    return {"ok": True, "ignored": event["type"]}
