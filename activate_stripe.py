#!/usr/bin/env python3
"""
STRIPE ACTIVATION HELPER
Interactive script to get your real Stripe API keys and activate payments
"""

import os
import re
import sys
from pathlib import Path


def print_header():
    print("""
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║              💰 STRIPE PAYMENT ACTIVATION 💰                  ║
║                                                               ║
║           Get Your API Keys in 5 Minutes                     ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
    """)


def print_step(number, title):
    print(f"\n{'='*70}")
    print(f"  STEP {number}: {title}")
    print(f"{'='*70}\n")


def main():
    print_header()

    print("\n🎯 This script will help you:")
    print("  1. Get your Stripe API keys")
    print("  2. Create pricing tiers")
    print("  3. Configure webhooks")
    print("  4. Activate live payments\n")

    # Step 1: Create Stripe Account
    print_step(1, "CREATE STRIPE ACCOUNT")
    print("📝 Go to: \033[96mhttps://stripe.com\033[0m")
    print("\n   Actions:")
    print("   • Click 'Start now' or 'Sign up'")
    print("   • Enter your email and password")
    print("   • Complete business information")
    print("   • Verify your email")
    print("\n   ⏱️  Time: ~3 minutes")

    input("\n   Press Enter when you've created your account...")

    # Step 2: Get API Keys
    print_step(2, "GET YOUR API KEYS")
    print("🔑 Go to: \033[96mhttps://dashboard.stripe.com/apikeys\033[0m")
    print("\n   You'll see:")
    print("   • Publishable key (starts with pk_live_...)")
    print("   • Secret key (starts with sk_live_...)")
    print("\n   ⚠️  IMPORTANT: Click 'Reveal live key token' for the secret key")
    print("   📋 Copy BOTH keys")

    print("\n")
    secret_key = input("   Paste your SECRET key (sk_live_...): ").strip()
    publishable_key = input("   Paste your PUBLISHABLE key (pk_live_...): ").strip()

    # Validate keys
    if not secret_key.startswith('sk_'):
        print("\n   ⚠️  Warning: Secret key should start with 'sk_live_' or 'sk_test_'")
        if input("   Continue anyway? (y/n): ").lower() != 'y':
            sys.exit(1)

    if not publishable_key.startswith('pk_'):
        print("\n   ⚠️  Warning: Publishable key should start with 'pk_live_' or 'pk_test_'")
        if input("   Continue anyway? (y/n): ").lower() != 'y':
            sys.exit(1)

    # Step 3: Create Products
    print_step(3, "CREATE PRICING TIERS")
    print("📦 Go to: \033[96mhttps://dashboard.stripe.com/products\033[0m")
    print("\n   Create THREE products:")
    print("\n   1️⃣  STARTER PLAN")
    print("      • Name: 'Nexus AGI - Starter'")
    print("      • Price: $29/month (or one-time)")
    print("      • Description: 'Basic AGI API access - 100 requests/day'")
    print("      • After creating, copy the Price ID (starts with price_...)")

    print("\n   2️⃣  PROFESSIONAL PLAN")
    print("      • Name: 'Nexus AGI - Professional'")
    print("      • Price: $99/month")
    print("      • Description: 'Advanced AGI features - 1,000 requests/day'")
    print("      • Copy the Price ID")

    print("\n   3️⃣  ENTERPRISE PLAN")
    print("      • Name: 'Nexus AGI - Enterprise'")
    print("      • Price: $499/month")
    print("      • Description: 'Unlimited AGI power - Custom solutions'")
    print("      • Copy the Price ID")

    print("\n   ⏱️  Time: ~5 minutes to create all three")

    input("\n   Press Enter when you've created the products...")

    print("\n")
    price_starter = input("   Paste STARTER Price ID (price_...): ").strip()
    price_professional = input("   Paste PROFESSIONAL Price ID (price_...): ").strip()
    price_enterprise = input("   Paste ENTERPRISE Price ID (price_...): ").strip()

    # Step 4: Webhook Secret (optional for now)
    print_step(4, "WEBHOOK SETUP (Optional - Do Later)")
    print("🔗 Webhooks allow Stripe to notify your app of payment events")
    print("\n   You can set this up after deploying to cloud.")
    print("   For now, we'll use a placeholder.")

    webhook_secret = input("\n   Webhook secret (or press Enter to skip): ").strip()
    if not webhook_secret:
        webhook_secret = "whsec_placeholder_configure_after_deployment"

    # Step 5: Update .env
    print_step(5, "UPDATING CONFIGURATION")
    print("💾 Writing keys to .env file...")

    env_file = Path(__file__).parent / '.env'

    # Read current .env
    with open(env_file, 'r') as f:
        content = f.read()

    # Replace keys
    content = re.sub(r'STRIPE_SECRET_KEY=.*', f'STRIPE_SECRET_KEY={secret_key}', content)
    content = re.sub(r'STRIPE_PUBLISHABLE_KEY=.*', f'STRIPE_PUBLISHABLE_KEY={publishable_key}', content)
    content = re.sub(r'STRIPE_PRICE_STARTER=.*', f'STRIPE_PRICE_STARTER={price_starter}', content)
    content = re.sub(r'STRIPE_PRICE_PROFESSIONAL=.*', f'STRIPE_PRICE_PROFESSIONAL={price_professional}', content)
    content = re.sub(r'STRIPE_PRICE_ENTERPRISE=.*', f'STRIPE_PRICE_ENTERPRISE={price_enterprise}', content)
    content = re.sub(r'STRIPE_WEBHOOK_SECRET=.*', f'STRIPE_WEBHOOK_SECRET={webhook_secret}', content)

    # Mark as LIVE
    content = content.replace('# 🔴 USING TEST KEYS', '# ✅ USING LIVE KEYS')
    content = content.replace('TEST MODE', 'LIVE MODE')

    # Write back
    with open(env_file, 'w') as f:
        f.write(content)

    print("\n   ✅ Configuration updated!")

    # Summary
    print_step(6, "ACTIVATION COMPLETE!")
    print("\n✅ Stripe payments are now CONFIGURED!")
    print("\n📊 Your setup:")
    print(f"   • Secret Key: {secret_key[:20]}...")
    print(f"   • Publishable Key: {publishable_key[:20]}...")
    print(f"   • Starter Plan: {price_starter}")
    print(f"   • Professional Plan: {price_professional}")
    print(f"   • Enterprise Plan: {price_enterprise}")

    print("\n🚀 NEXT STEPS:")
    print("\n   1. Restart API server:")
    print("      \033[96mpython autonomous_deployment_complete.py\033[0m")
    print("\n   2. Test payment endpoint:")
    print("      \033[96mcurl http://localhost:8000/api/payments/pricing\033[0m")
    print("\n   3. Monitor revenue:")
    print("      \033[96m./monitor_revenue.py\033[0m")
    print("\n   4. Deploy to cloud (for live payments):")
    print("      \033[96mrailway login && railway up\033[0m")

    print("\n💰 You're ready to start earning!")
    print("\n╔═══════════════════════════════════════════════════════════════╗")
    print("║                                                               ║")
    print("║         STRIPE PAYMENTS ACTIVATED! START EARNING! 🎉         ║")
    print("║                                                               ║")
    print("╚═══════════════════════════════════════════════════════════════╝\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup cancelled. Run again when ready.")
        sys.exit(1)
