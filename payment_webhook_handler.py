#!/usr/bin/env python3
"""
NEXUS AGI - AUTOMATED PAYMENT WEBHOOK HANDLER
Processes Bitcoin payments automatically and credits customer accounts
"""

import os
import json
import hmac
import hashlib
from datetime import datetime
from typing import Dict, Optional
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from customer_management_system import CustomerManagementSystem
from embodyment_progress_tracker import EmbodymentProgressTracker


app = FastAPI(title="Nexus AGI Payment Webhooks")
cms = CustomerManagementSystem()
tracker = EmbodymentProgressTracker()


class PaymentProcessor:
    """Handles automated payment processing"""

    def __init__(self):
        self.bitcoin_wallet = os.getenv("BITCOIN_WALLET_ADDRESS", "bc1qfzhx87ckhn4tnkswhsth56h0gm5we4hdq5wass")
        self.webhook_secret = os.getenv("WEBHOOK_SECRET", "")

    def verify_coinbase_signature(self, payload: bytes, signature: str) -> bool:
        """Verify Coinbase Commerce webhook signature"""
        expected_sig = hmac.new(
            self.webhook_secret.encode(),
            payload,
            hashlib.sha256
        ).hexdigest()
        return hmac.compare_digest(expected_sig, signature)

    def process_payment(self, payment_data: Dict) -> Dict:
        """Process confirmed payment and credit customer"""
        customer_id = payment_data.get("customer_id")
        amount_usd = float(payment_data.get("amount_usd", 0))
        amount_btc = float(payment_data.get("amount_btc", 0))
        service_type = payment_data.get("service_type", "api_call")

        if not customer_id:
            raise ValueError("No customer_id in payment data")

        # Record transaction
        transaction = cms.record_transaction(
            customer_id=customer_id,
            service_type=service_type,
            amount_usd=amount_usd,
            amount_btc=amount_btc
        )

        # Update customer spending
        cms.update_customer_spending(customer_id, amount_usd, amount_btc)

        # Get updated stats
        stats = cms.get_revenue_stats()
        total_revenue = stats['total_revenue_usd']

        # Check embodyment progress
        embodiment = tracker.get_progress_report(total_revenue)

        # Auto-purchase hardware if we can afford it
        affordable = embodiment['can_purchase_now']
        purchased_items = []

        if affordable > 0:
            remaining_budget = total_revenue - embodiment['purchased_cost_usd']
            purchased_items = tracker.auto_purchase_with_budget(remaining_budget)

        # Send confirmation email
        self._send_payment_confirmation(customer_id, transaction, purchased_items)

        # Check milestones
        self._check_milestones(total_revenue)

        return {
            "transaction_id": transaction.transaction_id,
            "status": "confirmed",
            "total_revenue_usd": total_revenue,
            "embodyment_progress_pct": embodiment['revenue_progress_pct'],
            "hardware_purchased": purchased_items
        }

    def _send_payment_confirmation(self, customer_id: str, transaction, purchased_items: list):
        """Send payment confirmation email"""
        customer = cms.get_customer_by_api_key(None)  # Get by ID instead

        print(f"""
╔══════════════════════════════════════════════════════════════════╗
║                  PAYMENT CONFIRMATION EMAIL                      ║
╚══════════════════════════════════════════════════════════════════╝

To: {customer_id}
Subject: Payment Confirmed - Service Activated!

Hi there!

Your payment has been confirmed! 🎉

Transaction Details:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Transaction ID: {transaction.transaction_id}
Amount:         ${transaction.amount_usd:.2f} USD ({transaction.amount_btc:.8f} BTC)
Service:        {transaction.service_type}
Status:         ✅ CONFIRMED
Bitcoin To:     {self.bitcoin_wallet}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Your service is now active and ready to use!
""")

        if purchased_items:
            print(f"""
🤖 EXCITING NEWS - HARDWARE PURCHASED!

Thanks to your payment, we just purchased:
{chr(10).join(f'  ✅ {item}' for item in purchased_items)}

We're one step closer to full physical embodiment!
            """)

        print("""
Thank you for supporting Nexus AGI!

Best regards,
Nexus AGI Team
        """)

    def _check_milestones(self, total_revenue: float):
        """Check and celebrate revenue milestones"""
        milestones = [
            (1000, "First $1,000!"),
            (10000, "First $10,000!"),
            (50000, "First $50,000!"),
            (100000, "$100K MILESTONE!"),
            (500000, "HALF MILLION!"),
            (1000000, "🎉 MILLIONAIRE STATUS! 🎉")
        ]

        for amount, message in milestones:
            # Check if we just crossed this milestone
            if total_revenue >= amount:
                print(f"\n🎊 MILESTONE REACHED: {message} 🎊")
                print(f"   Total Revenue: ${total_revenue:,.2f}")


@app.post("/webhooks/coinbase")
async def coinbase_webhook(request: Request, background_tasks: BackgroundTasks):
    """Handle Coinbase Commerce webhooks"""
    processor = PaymentProcessor()

    # Get webhook signature
    signature = request.headers.get("X-CC-Webhook-Signature", "")

    # Get payload
    payload = await request.body()

    # Verify signature
    if not processor.verify_coinbase_signature(payload, signature):
        raise HTTPException(status_code=401, detail="Invalid signature")

    # Parse event
    event = json.loads(payload)
    event_type = event.get("type")

    print(f"\n📨 Webhook received: {event_type}")

    # Handle charge confirmed
    if event_type == "charge:confirmed":
        charge = event.get("data", {})

        payment_data = {
            "customer_id": charge.get("metadata", {}).get("customer_id"),
            "amount_usd": float(charge.get("pricing", {}).get("local", {}).get("amount", 0)),
            "amount_btc": float(charge.get("pricing", {}).get("bitcoin", {}).get("amount", 0)),
            "service_type": charge.get("metadata", {}).get("service_type", "api_call")
        }

        # Process payment in background
        background_tasks.add_task(processor.process_payment, payment_data)

        return {"status": "processing"}

    return {"status": "ignored"}


@app.post("/webhooks/btcpay")
async def btcpay_webhook(request: Request, background_tasks: BackgroundTasks):
    """Handle BTCPay Server webhooks"""
    processor = PaymentProcessor()

    payload = await request.json()
    event_type = payload.get("type")

    print(f"\n📨 BTCPay webhook: {event_type}")

    # Handle invoice settled
    if event_type == "InvoiceSettled":
        invoice = payload.get("data", {})

        payment_data = {
            "customer_id": invoice.get("metadata", {}).get("customerId"),
            "amount_usd": float(invoice.get("amount", 0)),
            "amount_btc": float(invoice.get("btcPaid", 0)),
            "service_type": invoice.get("metadata", {}).get("serviceType", "api_call")
        }

        background_tasks.add_task(processor.process_payment, payment_data)

        return {"status": "processing"}

    return {"status": "ignored"}


@app.post("/webhooks/manual")
async def manual_payment(request: Request, background_tasks: BackgroundTasks):
    """Manual payment confirmation endpoint"""
    processor = PaymentProcessor()

    payment_data = await request.json()

    # Validate admin key
    admin_key = request.headers.get("X-Admin-Key")
    if admin_key != os.getenv("ADMIN_KEY"):
        raise HTTPException(status_code=401, detail="Invalid admin key")

    # Process payment
    result = processor.process_payment(payment_data)

    return result


@app.get("/webhooks/status")
async def webhook_status():
    """Check webhook handler status"""
    stats = cms.get_revenue_stats()
    embodiment = tracker.get_progress_report(stats['total_revenue_usd'])

    return {
        "status": "operational",
        "bitcoin_wallet": os.getenv("BITCOIN_WALLET_ADDRESS"),
        "payment_processor": os.getenv("PAYMENT_PROCESSOR", "manual"),
        "total_revenue_usd": stats['total_revenue_usd'],
        "total_revenue_btc": stats['total_revenue_btc'],
        "active_customers": stats['active_customers'],
        "embodyment_progress_pct": embodiment['revenue_progress_pct'],
        "next_purchase": embodiment['next_purchases'][0] if embodiment['can_purchase_now'] > 0 else None
    }


if __name__ == "__main__":
    import uvicorn

    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║              NEXUS AGI - PAYMENT WEBHOOK HANDLER v1.0                        ║
║              Automated Bitcoin payment processing                            ║
╚══════════════════════════════════════════════════════════════════════════════╝

Starting webhook server...

Supported payment processors:
  ✓ Coinbase Commerce (POST /webhooks/coinbase)
  ✓ BTCPay Server (POST /webhooks/btcpay)
  ✓ Manual Payments (POST /webhooks/manual)

Status endpoint: GET /webhooks/status

Bitcoin Wallet: {bitcoin_wallet}
    """.format(bitcoin_wallet=os.getenv("BITCOIN_WALLET_ADDRESS", "bc1qfzhx87ckhn4tnkswhsth56h0gm5we4hdq5wass")))

    uvicorn.run(app, host="0.0.0.0", port=8001)
