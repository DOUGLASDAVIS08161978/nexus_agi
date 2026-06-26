"""
Nexus Earn — Stripe Payments
Handles subscription checkout, webhooks, and tier management.
"""

import os
import json
import hmac
import hashlib
import httpx
from typing import Optional

STRIPE_API = "https://api.stripe.com/v1"


def _sk() -> str:
    key = os.getenv("STRIPE_SECRET_KEY", "")
    if not key:
        raise RuntimeError("STRIPE_SECRET_KEY not set")
    return key


def _post(path: str, data: dict) -> dict:
    with httpx.Client(timeout=30) as client:
        r = client.post(
            f"{STRIPE_API}/{path}",
            data=data,
            auth=(_sk(), ""),
        )
        r.raise_for_status()
        return r.json()


def _get(path: str) -> dict:
    with httpx.Client(timeout=30) as client:
        r = client.get(f"{STRIPE_API}/{path}", auth=(_sk(), ""))
        r.raise_for_status()
        return r.json()


# ── Checkout ──────────────────────────────────────────────────────────────────

def create_checkout_session(price_id: str, customer_email: str,
                             success_url: str, cancel_url: str) -> str:
    """Create a Stripe Checkout session and return the URL."""
    result = _post("checkout/sessions", {
        "mode":                        "subscription",
        "customer_email":              customer_email,
        "line_items[0][price]":        price_id,
        "line_items[0][quantity]":     "1",
        "success_url":                 success_url,
        "cancel_url":                  cancel_url,
        "metadata[source]":            "nexus_earn",
    })
    return result["url"]


# ── Webhook verification ───────────────────────────────────────────────────────

def verify_webhook(payload: bytes, sig_header: str) -> Optional[dict]:
    """Verify Stripe webhook signature and return parsed event, or None if invalid."""
    secret = os.getenv("STRIPE_WEBHOOK_SECRET", "")
    if not secret:
        return None
    try:
        parts = {k: v for k, v in (p.split("=", 1) for p in sig_header.split(","))}
        timestamp = parts.get("t", "")
        signature = parts.get("v1", "")
        signed_payload = f"{timestamp}.{payload.decode()}"
        expected = hmac.new(
            secret.encode(), signed_payload.encode(), hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(expected, signature):
            return None
        return json.loads(payload)
    except Exception:
        return None


# ── Tier mapping ──────────────────────────────────────────────────────────────

def price_id_to_tier(price_id: str) -> str:
    """Map a Stripe price ID to a tier name."""
    from database import STRIPE_PRICE_IDS
    for tier, pid in STRIPE_PRICE_IDS.items():
        if pid and pid == price_id:
            return tier
    return "free"
