"""
Stripe Payment Integration for Nexus AGI API
Handles subscriptions, payments, and webhooks
"""

import stripe
from typing import Dict, Any, Optional
from datetime import datetime
import sys
from pathlib import Path
import os

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import get_stripe_keys

# Initialize Stripe
stripe_keys = get_stripe_keys()
stripe.api_key = stripe_keys.get('secret_key')

# Pricing tiers configuration
PRICING_TIERS = {
    "free": {
        "name": "Free",
        "price": 0,
        "requests_limit": 100,
        "features": ["100 requests/month", "Basic support", "All tools access"]
    },
    "starter": {
        "name": "Starter",
        "price": 29,
        "requests_limit": 1000,
        "price_id": os.getenv('STRIPE_PRICE_STARTER'),
        "features": ["1,000 requests/month", "Email support", "All tools access", "Priority processing"]
    },
    "professional": {
        "name": "Professional",
        "price": 99,
        "requests_limit": 10000,
        "price_id": os.getenv('STRIPE_PRICE_PROFESSIONAL'),
        "features": ["10,000 requests/month", "Priority support", "All tools access", "Advanced analytics", "Custom integrations"]
    },
    "enterprise": {
        "name": "Enterprise",
        "price": 499,
        "requests_limit": 100000,
        "price_id": os.getenv('STRIPE_PRICE_ENTERPRISE'),
        "features": ["100,000 requests/month", "24/7 dedicated support", "All tools access", "Advanced analytics", "Custom integrations", "SLA guarantee", "White-label option"]
    }
}


class StripePaymentProcessor:
    """Handle Stripe payment operations"""

    def __init__(self):
        """Initialize Stripe payment processor"""
        self.api_key = stripe_keys.get('secret_key')
        if not self.api_key or self.api_key == 'your_stripe_key':
            print("⚠️  Warning: Stripe API key not configured. Payment processing disabled.")
            self.enabled = False
        else:
            self.enabled = True
            stripe.api_key = self.api_key

    def create_customer(self, email: str, name: Optional[str] = None, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Create a Stripe customer

        Args:
            email: Customer email
            name: Customer name
            metadata: Additional metadata

        Returns:
            Stripe customer object
        """
        if not self.enabled:
            raise Exception("Stripe is not configured")

        try:
            customer = stripe.Customer.create(
                email=email,
                name=name,
                metadata=metadata or {}
            )
            return customer
        except stripe.error.StripeError as e:
            raise Exception(f"Failed to create customer: {str(e)}")

    def create_subscription(
        self,
        customer_id: str,
        price_id: str,
        trial_period_days: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Create a subscription for a customer

        Args:
            customer_id: Stripe customer ID
            price_id: Stripe price ID
            trial_period_days: Optional trial period in days

        Returns:
            Stripe subscription object
        """
        if not self.enabled:
            raise Exception("Stripe is not configured")

        try:
            subscription_params = {
                "customer": customer_id,
                "items": [{"price": price_id}],
                "payment_behavior": "default_incomplete",
                "expand": ["latest_invoice.payment_intent"],
            }

            if trial_period_days:
                subscription_params["trial_period_days"] = trial_period_days

            subscription = stripe.Subscription.create(**subscription_params)
            return subscription
        except stripe.error.StripeError as e:
            raise Exception(f"Failed to create subscription: {str(e)}")

    def cancel_subscription(self, subscription_id: str, at_period_end: bool = True) -> Dict[str, Any]:
        """
        Cancel a subscription

        Args:
            subscription_id: Stripe subscription ID
            at_period_end: Cancel at end of billing period (vs immediately)

        Returns:
            Updated subscription object
        """
        if not self.enabled:
            raise Exception("Stripe is not configured")

        try:
            if at_period_end:
                subscription = stripe.Subscription.modify(
                    subscription_id,
                    cancel_at_period_end=True
                )
            else:
                subscription = stripe.Subscription.delete(subscription_id)
            return subscription
        except stripe.error.StripeError as e:
            raise Exception(f"Failed to cancel subscription: {str(e)}")

    def update_subscription(self, subscription_id: str, new_price_id: str) -> Dict[str, Any]:
        """
        Update subscription to a different plan

        Args:
            subscription_id: Stripe subscription ID
            new_price_id: New price ID

        Returns:
            Updated subscription object
        """
        if not self.enabled:
            raise Exception("Stripe is not configured")

        try:
            subscription = stripe.Subscription.retrieve(subscription_id)

            stripe.Subscription.modify(
                subscription_id,
                cancel_at_period_end=False,
                items=[{
                    'id': subscription['items']['data'][0].id,
                    'price': new_price_id,
                }]
            )

            return stripe.Subscription.retrieve(subscription_id)
        except stripe.error.StripeError as e:
            raise Exception(f"Failed to update subscription: {str(e)}")

    def get_subscription(self, subscription_id: str) -> Dict[str, Any]:
        """Get subscription details"""
        if not self.enabled:
            raise Exception("Stripe is not configured")

        try:
            return stripe.Subscription.retrieve(subscription_id)
        except stripe.error.StripeError as e:
            raise Exception(f"Failed to retrieve subscription: {str(e)}")

    def create_checkout_session(
        self,
        price_id: str,
        customer_email: str,
        success_url: str,
        cancel_url: str,
        trial_period_days: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Create a Stripe Checkout session

        Args:
            price_id: Stripe price ID
            customer_email: Customer email
            success_url: URL to redirect after successful payment
            cancel_url: URL to redirect after cancelled payment
            trial_period_days: Optional trial period

        Returns:
            Checkout session object with URL
        """
        if not self.enabled:
            raise Exception("Stripe is not configured")

        try:
            session_params = {
                "payment_method_types": ["card"],
                "line_items": [{
                    "price": price_id,
                    "quantity": 1,
                }],
                "mode": "subscription",
                "success_url": success_url,
                "cancel_url": cancel_url,
                "customer_email": customer_email,
            }

            if trial_period_days:
                session_params["subscription_data"] = {
                    "trial_period_days": trial_period_days
                }

            session = stripe.checkout.Session.create(**session_params)
            return session
        except stripe.error.StripeError as e:
            raise Exception(f"Failed to create checkout session: {str(e)}")

    def create_billing_portal_session(self, customer_id: str, return_url: str) -> Dict[str, Any]:
        """
        Create a billing portal session for customer to manage subscription

        Args:
            customer_id: Stripe customer ID
            return_url: URL to return to after managing subscription

        Returns:
            Billing portal session with URL
        """
        if not self.enabled:
            raise Exception("Stripe is not configured")

        try:
            session = stripe.billing_portal.Session.create(
                customer=customer_id,
                return_url=return_url,
            )
            return session
        except stripe.error.StripeError as e:
            raise Exception(f"Failed to create billing portal session: {str(e)}")

    def verify_webhook_signature(self, payload: bytes, signature: str, webhook_secret: str) -> Dict[str, Any]:
        """
        Verify Stripe webhook signature

        Args:
            payload: Request body bytes
            signature: Stripe signature header
            webhook_secret: Webhook signing secret

        Returns:
            Verified event object
        """
        try:
            event = stripe.Webhook.construct_event(
                payload, signature, webhook_secret
            )
            return event
        except ValueError:
            raise Exception("Invalid payload")
        except stripe.error.SignatureVerificationError:
            raise Exception("Invalid signature")


def get_tier_from_price_id(price_id: str) -> Optional[str]:
    """Get tier name from Stripe price ID"""
    for tier, config in PRICING_TIERS.items():
        if config.get('price_id') == price_id:
            return tier
    return None


def get_price_id_from_tier(tier: str) -> Optional[str]:
    """Get Stripe price ID from tier name"""
    return PRICING_TIERS.get(tier, {}).get('price_id')


if __name__ == "__main__":
    print("=== Testing Stripe Payment Integration ===\n")

    processor = StripePaymentProcessor()

    if not processor.enabled:
        print("⚠️  Stripe is not configured. Set STRIPE_SECRET_KEY in .env to test.\n")
        print("Pricing tiers configuration:")
        for tier, config in PRICING_TIERS.items():
            print(f"\n{config['name']} - ${config['price']}/month")
            print(f"  Limit: {config['requests_limit']} requests")
            print(f"  Features: {', '.join(config['features'])}")
        print("\n✅ Pricing configuration loaded")
    else:
        print("✅ Stripe configured and ready")

        # Test customer creation (commented out to avoid actual API calls)
        # customer = processor.create_customer(
        #     email="test@example.com",
        #     name="Test User",
        #     metadata={"user_id": "123"}
        # )
        # print(f"Customer created: {customer.id}")

    print("\nPricing tiers:")
    for tier, config in PRICING_TIERS.items():
        price_id = config.get('price_id', 'Not configured')
        print(f"  {tier}: ${config['price']}/mo - {config['requests_limit']} req/mo - Price ID: {price_id}")
