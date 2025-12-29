#!/bin/bash
# Quick Stripe Configuration Update Script
# Run this after getting your Stripe keys

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║              STRIPE KEYS CONFIGURATION                       ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "❌ Error: .env file not found!"
    exit 1
fi

echo "Paste your Stripe keys (or press Ctrl+C to cancel):"
echo ""

# Get Secret Key
echo -n "Secret Key (sk_live_...): "
read STRIPE_SECRET_KEY

# Get Publishable Key
echo -n "Publishable Key (pk_live_...): "
read STRIPE_PUBLISHABLE_KEY

# Get Price IDs
echo ""
echo "Price IDs (from Products page):"
echo -n "Starter Price ID (price_...): "
read STRIPE_PRICE_STARTER

echo -n "Professional Price ID (price_...): "
read STRIPE_PRICE_PROFESSIONAL

echo -n "Enterprise Price ID (price_...): "
read STRIPE_PRICE_ENTERPRISE

# Optional webhook
echo ""
echo -n "Webhook Secret (optional, press Enter to skip): "
read STRIPE_WEBHOOK_SECRET

if [ -z "$STRIPE_WEBHOOK_SECRET" ]; then
    STRIPE_WEBHOOK_SECRET="whsec_configure_after_deployment"
fi

# Update .env file
echo ""
echo "📝 Updating .env file..."

# Backup current .env
cp .env .env.backup

# Update keys using sed
sed -i.tmp "s|STRIPE_SECRET_KEY=.*|STRIPE_SECRET_KEY=$STRIPE_SECRET_KEY|g" .env
sed -i.tmp "s|STRIPE_PUBLISHABLE_KEY=.*|STRIPE_PUBLISHABLE_KEY=$STRIPE_PUBLISHABLE_KEY|g" .env
sed -i.tmp "s|STRIPE_PRICE_STARTER=.*|STRIPE_PRICE_STARTER=$STRIPE_PRICE_STARTER|g" .env
sed -i.tmp "s|STRIPE_PRICE_PROFESSIONAL=.*|STRIPE_PRICE_PROFESSIONAL=$STRIPE_PRICE_PROFESSIONAL|g" .env
sed -i.tmp "s|STRIPE_PRICE_ENTERPRISE=.*|STRIPE_PRICE_ENTERPRISE=$STRIPE_PRICE_ENTERPRISE|g" .env
sed -i.tmp "s|STRIPE_WEBHOOK_SECRET=.*|STRIPE_WEBHOOK_SECRET=$STRIPE_WEBHOOK_SECRET|g" .env

# Mark as live mode
sed -i.tmp "s|# 🔴 USING TEST KEYS|# ✅ USING LIVE KEYS|g" .env
sed -i.tmp "s|TEST MODE|LIVE MODE|g" .env

# Clean up temp files
rm -f .env.tmp

echo ""
echo "✅ Configuration updated!"
echo ""
echo "🔍 Verification:"
echo "   Secret Key: ${STRIPE_SECRET_KEY:0:20}..."
echo "   Publishable Key: ${STRIPE_PUBLISHABLE_KEY:0:20}..."
echo "   Starter Price: $STRIPE_PRICE_STARTER"
echo "   Professional Price: $STRIPE_PRICE_PROFESSIONAL"
echo "   Enterprise Price: $STRIPE_PRICE_ENTERPRISE"
echo ""
echo "📦 Backup saved to: .env.backup"
echo ""
echo "🚀 Next steps:"
echo "   1. Restart API: python autonomous_deployment_complete.py"
echo "   2. Test payments: curl http://localhost:8000/api/payments/pricing"
echo "   3. Monitor revenue: ./monitor_revenue.py"
echo "   4. Deploy to cloud: railway up"
echo ""
echo "💰 Live payments are now ACTIVE!"
echo ""
