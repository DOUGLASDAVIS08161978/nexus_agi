"""
Cryptocurrency Payment Integration for Nexus AGI
Supports Bitcoin payments and automatic profit withdrawals
"""

import os
import requests
import hashlib
import hmac
import json
from typing import Dict, Any, Optional
from datetime import datetime
from decimal import Decimal

from config.config_loader import get_env


class CryptoPaymentProcessor:
    """
    Handles cryptocurrency payments using Coinbase Commerce API
    Automatically transfers profits to specified Bitcoin wallet
    """

    def __init__(self):
        self.api_key = get_env("COINBASE_COMMERCE_API_KEY", "")
        self.webhook_secret = get_env("COINBASE_COMMERCE_WEBHOOK_SECRET", "")
        self.bitcoin_wallet = get_env("BITCOIN_WALLET_ADDRESS", "")
        self.base_url = "https://api.commerce.coinbase.com"

        # Pricing in USD (converted to crypto at checkout)
        self.pricing_tiers = {
            "starter": {"price_usd": 29, "requests_limit": 1000},
            "professional": {"price_usd": 99, "requests_limit": 10000},
            "enterprise": {"price_usd": 499, "requests_limit": 100000}
        }

    def _get_headers(self) -> Dict[str, str]:
        """Get API request headers"""
        return {
            "X-CC-Api-Key": self.api_key,
            "X-CC-Version": "2018-03-22",
            "Content-Type": "application/json"
        }

    def create_charge(
        self,
        tier: str,
        user_email: str,
        user_id: int
    ) -> Dict[str, Any]:
        """
        Create a cryptocurrency payment charge

        Args:
            tier: Subscription tier (starter, professional, enterprise)
            user_email: Customer email
            user_id: Internal user ID

        Returns:
            Dict with charge details including payment URL
        """
        if tier not in self.pricing_tiers:
            raise ValueError(f"Invalid tier: {tier}")

        tier_info = self.pricing_tiers[tier]

        charge_data = {
            "name": f"Nexus AGI {tier.capitalize()} Plan",
            "description": f"{tier_info['requests_limit']} API requests per month",
            "pricing_type": "fixed_price",
            "local_price": {
                "amount": str(tier_info["price_usd"]),
                "currency": "USD"
            },
            "metadata": {
                "user_id": str(user_id),
                "user_email": user_email,
                "tier": tier,
                "billing_type": "subscription_monthly"
            },
            "redirect_url": get_env("APP_URL", "http://localhost:8000") + "/billing/success",
            "cancel_url": get_env("APP_URL", "http://localhost:8000") + "/billing/cancel"
        }

        try:
            response = requests.post(
                f"{self.base_url}/charges",
                headers=self._get_headers(),
                json=charge_data,
                timeout=30
            )
            response.raise_for_status()

            result = response.json()
            return {
                "success": True,
                "charge_id": result["data"]["id"],
                "charge_code": result["data"]["code"],
                "payment_url": result["data"]["hosted_url"],
                "expires_at": result["data"]["expires_at"],
                "pricing": result["data"]["pricing"]
            }

        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "error": str(e)
            }

    def verify_webhook(self, payload: bytes, signature: str) -> bool:
        """
        Verify Coinbase Commerce webhook signature

        Args:
            payload: Raw webhook payload
            signature: X-CC-Webhook-Signature header value

        Returns:
            True if signature is valid
        """
        if not self.webhook_secret:
            return False

        computed_signature = hmac.new(
            self.webhook_secret.encode('utf-8'),
            payload,
            hashlib.sha256
        ).hexdigest()

        return hmac.compare_digest(computed_signature, signature)

    def process_webhook(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process Coinbase Commerce webhook event

        Args:
            event_data: Webhook event data

        Returns:
            Processing result with action taken
        """
        event_type = event_data.get("type")
        charge = event_data.get("data", {})
        metadata = charge.get("metadata", {})

        result = {
            "event_type": event_type,
            "charge_id": charge.get("id"),
            "user_id": metadata.get("user_id"),
            "action": "none"
        }

        if event_type == "charge:confirmed":
            # Payment confirmed - activate subscription
            result["action"] = "activate_subscription"
            result["tier"] = metadata.get("tier")
            result["amount_paid"] = charge.get("pricing", {}).get("local", {}).get("amount")

        elif event_type == "charge:failed":
            # Payment failed
            result["action"] = "payment_failed"
            result["failure_reason"] = charge.get("timeline", [])[-1].get("status", "unknown")

        elif event_type == "charge:pending":
            # Payment pending (waiting for confirmations)
            result["action"] = "payment_pending"

        return result

    def get_charge_status(self, charge_id: str) -> Dict[str, Any]:
        """
        Check status of a charge

        Args:
            charge_id: Charge ID to check

        Returns:
            Charge status and details
        """
        try:
            response = requests.get(
                f"{self.base_url}/charges/{charge_id}",
                headers=self._get_headers(),
                timeout=30
            )
            response.raise_for_status()

            charge = response.json()["data"]

            return {
                "success": True,
                "status": charge["timeline"][-1]["status"],
                "payments": charge.get("payments", []),
                "pricing": charge.get("pricing", {})
            }

        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "error": str(e)
            }

    def withdraw_to_bitcoin(
        self,
        amount_usd: Decimal,
        note: str = "Nexus AGI profit withdrawal"
    ) -> Dict[str, Any]:
        """
        Withdraw profits to Bitcoin wallet

        NOTE: This is a placeholder for manual withdrawal process.
        Coinbase Commerce does not support automatic withdrawals via API.

        Actual process:
        1. Login to Coinbase Commerce dashboard
        2. Go to Withdrawals
        3. Select Bitcoin
        4. Enter wallet address: {bitcoin_wallet}
        5. Enter amount and confirm

        Args:
            amount_usd: Amount in USD to withdraw
            note: Note for the withdrawal

        Returns:
            Withdrawal instructions
        """
        return {
            "success": True,
            "method": "manual",
            "amount_usd": str(amount_usd),
            "bitcoin_wallet": self.bitcoin_wallet,
            "instructions": [
                "1. Login to Coinbase Commerce at https://commerce.coinbase.com",
                "2. Navigate to 'Balance' in the sidebar",
                "3. Click 'Withdraw' next to your Bitcoin balance",
                f"4. Enter Bitcoin address: {self.bitcoin_wallet}",
                f"5. Enter amount: ${amount_usd} USD equivalent",
                "6. Review and confirm withdrawal",
                "7. Wait 10-30 minutes for blockchain confirmation"
            ],
            "note": note,
            "estimated_time": "10-30 minutes after confirmation"
        }


class BTCPayServerProcessor:
    """
    Alternative processor using BTCPay Server (self-hosted)
    Provides more control and direct Bitcoin payments
    """

    def __init__(self):
        self.btcpay_url = get_env("BTCPAY_SERVER_URL", "")
        self.api_key = get_env("BTCPAY_API_KEY", "")
        self.store_id = get_env("BTCPAY_STORE_ID", "")
        self.bitcoin_wallet = get_env("BITCOIN_WALLET_ADDRESS", "")

    def _get_headers(self) -> Dict[str, str]:
        """Get API request headers"""
        return {
            "Authorization": f"token {self.api_key}",
            "Content-Type": "application/json"
        }

    def create_invoice(
        self,
        tier: str,
        user_email: str,
        user_id: int
    ) -> Dict[str, Any]:
        """
        Create a BTCPay Server invoice

        Args:
            tier: Subscription tier
            user_email: Customer email
            user_id: Internal user ID

        Returns:
            Invoice details with payment URL
        """
        pricing = {
            "starter": 29,
            "professional": 99,
            "enterprise": 499
        }

        if tier not in pricing:
            raise ValueError(f"Invalid tier: {tier}")

        invoice_data = {
            "amount": str(pricing[tier]),
            "currency": "USD",
            "metadata": {
                "orderId": f"nexus-{user_id}-{tier}",
                "itemDesc": f"Nexus AGI {tier.capitalize()} Plan",
                "buyerEmail": user_email,
                "userId": str(user_id),
                "tier": tier
            },
            "checkout": {
                "redirectURL": get_env("APP_URL", "http://localhost:8000") + "/billing/success",
                "redirectAutomatically": True
            }
        }

        try:
            response = requests.post(
                f"{self.btcpay_url}/api/v1/stores/{self.store_id}/invoices",
                headers=self._get_headers(),
                json=invoice_data,
                timeout=30
            )
            response.raise_for_status()

            result = response.json()

            return {
                "success": True,
                "invoice_id": result["id"],
                "payment_url": result["checkoutLink"],
                "status": result["status"],
                "amount_btc": result.get("btcPrice", "pending")
            }

        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "error": str(e)
            }

    def check_invoice_status(self, invoice_id: str) -> Dict[str, Any]:
        """Check BTCPay invoice payment status"""
        try:
            response = requests.get(
                f"{self.btcpay_url}/api/v1/stores/{self.store_id}/invoices/{invoice_id}",
                headers=self._get_headers(),
                timeout=30
            )
            response.raise_for_status()

            invoice = response.json()

            return {
                "success": True,
                "status": invoice["status"],
                "amount_paid": invoice.get("btcPaid", "0"),
                "confirmed": invoice["status"] in ["settled", "complete"]
            }

        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "error": str(e)
            }


# Initialize processors
crypto_processor = CryptoPaymentProcessor()
btcpay_processor = BTCPayServerProcessor()


def get_crypto_payment_options() -> Dict[str, Any]:
    """
    Get available cryptocurrency payment options

    Returns:
        Dict with available payment methods and configuration
    """
    options = {
        "bitcoin_enabled": False,
        "payment_methods": [],
        "withdrawal_wallet": os.getenv("BITCOIN_WALLET_ADDRESS", "")
    }

    # Check Coinbase Commerce
    if crypto_processor.api_key:
        options["bitcoin_enabled"] = True
        options["payment_methods"].append({
            "provider": "coinbase_commerce",
            "currencies": ["BTC", "ETH", "USDC", "DAI"],
            "status": "active"
        })

    # Check BTCPay Server
    if btcpay_processor.btcpay_url and btcpay_processor.api_key:
        options["bitcoin_enabled"] = True
        options["payment_methods"].append({
            "provider": "btcpay_server",
            "currencies": ["BTC", "Lightning Network"],
            "status": "active"
        })

    return options
