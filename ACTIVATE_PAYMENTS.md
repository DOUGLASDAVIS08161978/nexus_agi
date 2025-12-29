# 💰 PAYMENT ACTIVATION GUIDE - START EARNING IN 15 MINUTES

This guide will help you activate all payment systems and start generating revenue with Nexus AGI.

## 🎯 Overview

Nexus AGI supports multiple payment methods:
- ✅ **Stripe** - Credit/debit cards (recommended, easiest)
- ✅ **Bitcoin/Crypto** - Coinbase Commerce or BTCPay Server
- ✅ **PayPal** - Traditional payment alternative

**Estimated Setup Time:** 15 minutes
**First Payment Expected:** Within 24-48 hours of marketing activation

---

## 🚀 STEP 1: STRIPE SETUP (10 minutes)

Stripe is the fastest way to start accepting payments. It's trusted by millions of businesses worldwide.

### 1.1 Create Stripe Account

1. Go to [https://stripe.com](https://stripe.com)
2. Click "Start now" and create an account
3. Complete business information (can use personal details initially)
4. **Important:** Activate your account to accept real payments

### 1.2 Get API Keys

1. Log into Stripe Dashboard: [https://dashboard.stripe.com](https://dashboard.stripe.com)
2. Click "Developers" → "API keys"
3. Copy your **Secret key** (starts with `sk_test_` or `sk_live_`)
4. Copy your **Publishable key** (starts with `pk_test_` or `pk_live_`)

### 1.3 Create Products & Pricing

1. Go to "Products" in Stripe Dashboard
2. Click "Add product"
3. Create three pricing tiers:

   **Starter Plan:**
   - Name: "Nexus AGI - Starter"
   - Price: $29/month (or one-time)
   - Description: "Basic AGI API access - 100 requests/day"

   **Professional Plan:**
   - Name: "Nexus AGI - Professional"
   - Price: $99/month
   - Description: "Advanced AGI features - 1,000 requests/day"

   **Enterprise Plan:**
   - Name: "Nexus AGI - Enterprise"
   - Price: $499/month
   - Description: "Unlimited AGI power - Custom solutions"

4. Copy each **Price ID** (starts with `price_`)

### 1.4 Configure Webhook

1. Go to "Developers" → "Webhooks"
2. Click "Add endpoint"
3. Endpoint URL: `https://your-domain.com/api/stripe/webhook`
4. Select events to listen for:
   - `checkout.session.completed`
   - `invoice.payment_succeeded`
   - `customer.subscription.deleted`
5. Copy the **Webhook signing secret** (starts with `whsec_`)

### 1.5 Update .env File

Open your `.env` file and add:

```bash
# Stripe Configuration
STRIPE_SECRET_KEY=sk_live_YOUR_SECRET_KEY_HERE
STRIPE_PUBLISHABLE_KEY=pk_live_YOUR_PUBLISHABLE_KEY_HERE
STRIPE_WEBHOOK_SECRET=whsec_YOUR_WEBHOOK_SECRET_HERE

# Price IDs
STRIPE_PRICE_STARTER=price_YOUR_STARTER_PRICE_ID
STRIPE_PRICE_PROFESSIONAL=price_YOUR_PROFESSIONAL_PRICE_ID
STRIPE_PRICE_ENTERPRISE=price_YOUR_ENTERPRISE_PRICE_ID
```

### 1.6 Connect Bank Account

1. Go to "Settings" → "Payouts" in Stripe Dashboard
2. Click "Add bank account"
3. Enter your bank details
4. Stripe will make 2 small deposits to verify (1-2 business days)
5. Verify the amounts to activate automatic payouts

**💰 Money Flow:**
Customer pays → Stripe holds for 2-7 days → Automatically deposits to your bank

---

## 🪙 STEP 2: BITCOIN/CRYPTO SETUP (5 minutes)

Accept Bitcoin and 10+ cryptocurrencies with Coinbase Commerce.

### 2.1 Create Coinbase Commerce Account

1. Go to [https://commerce.coinbase.com](https://commerce.coinbase.com)
2. Sign up (free, no monthly fees)
3. Complete business verification

### 2.2 Get API Keys

1. Log into Commerce Dashboard
2. Click "Settings" → "API keys"
3. Click "Create an API key"
4. Copy your **API Key**
5. Copy your **Webhook Secret**

### 2.3 Set Up Withdrawals

1. Go to "Settings" → "Cryptocurrency addresses"
2. Add your Bitcoin wallet address
   - **Don't have a wallet?** Get one at [https://www.coinbase.com](https://www.coinbase.com)
3. Enable automatic conversions to USD (optional)

### 2.4 Update .env File

```bash
# Coinbase Commerce Configuration
COINBASE_COMMERCE_API_KEY=YOUR_API_KEY_HERE
COINBASE_COMMERCE_WEBHOOK_SECRET=YOUR_WEBHOOK_SECRET_HERE

# Your Bitcoin Wallet (where profits go)
BITCOIN_WALLET_ADDRESS=bc1q_YOUR_BITCOIN_ADDRESS_HERE
```

**💰 Money Flow:**
Customer pays Bitcoin → Instant confirmation → Auto-withdraw to your wallet daily

---

## 💳 STEP 3: PAYPAL SETUP (Optional, 5 minutes)

Add PayPal as an alternative payment method.

### 3.1 Create PayPal Business Account

1. Go to [https://www.paypal.com/business](https://www.paypal.com/business)
2. Click "Sign Up"
3. Choose "Business Account"

### 3.2 Get API Credentials

1. Log into PayPal Developer: [https://developer.paypal.com](https://developer.paypal.com)
2. Go to "My Apps & Credentials"
3. Under "REST API apps", click "Create App"
4. Copy **Client ID** and **Secret**

### 3.3 Update .env File

```bash
# PayPal Configuration
PAYPAL_CLIENT_ID=YOUR_CLIENT_ID_HERE
PAYPAL_SECRET=YOUR_SECRET_HERE
PAYPAL_MODE=live  # Use 'sandbox' for testing
```

---

## ✅ STEP 4: VERIFY CONFIGURATION

### 4.1 Test Locally

```bash
# Start the API server
uvicorn api_gateway.main:app --reload

# Open browser to API docs
open http://localhost:8000/docs

# Test payment endpoint
curl http://localhost:8000/api/payments/create-checkout \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{"plan": "starter", "method": "stripe"}'
```

### 4.2 Check Configuration Status

```bash
# Run configuration check
python -c "
from dotenv import load_dotenv
import os
load_dotenv()

print('✓ Stripe:' if os.getenv('STRIPE_SECRET_KEY', '').startswith('sk_') else '✗ Stripe: NOT CONFIGURED')
print('✓ Coinbase:' if os.getenv('COINBASE_COMMERCE_API_KEY') else '✗ Coinbase: NOT CONFIGURED')
print('✓ PayPal:' if os.getenv('PAYPAL_CLIENT_ID') else '✗ PayPal: NOT CONFIGURED')
"
```

### 4.3 Enable Payment Processing

In your `.env` file, ensure:

```bash
ENABLE_PAYMENT_PROCESSING=true
ENABLE_USAGE_TRACKING=true
ENABLE_RATE_LIMITING=true
```

---

## 🚀 STEP 5: DEPLOY & GO LIVE

### 5.1 Deploy to Production

```bash
# Local deployment
./DEPLOY_NOW.sh local

# Cloud deployment (Railway)
./DEPLOY_NOW.sh railway

# Or deploy manually to:
# - Render.com
# - AWS
# - DigitalOcean
# - Any VPS
```

### 5.2 Update Payment URLs

After deploying, update your webhook URLs:

**Stripe:**
- Go to Developers → Webhooks
- Update endpoint URL to: `https://YOUR-DOMAIN.com/api/stripe/webhook`

**Coinbase Commerce:**
- Go to Settings → Webhook subscriptions
- Add URL: `https://YOUR-DOMAIN.com/api/coinbase/webhook`

### 5.3 Test Real Payment

1. Create a test checkout session
2. Use Stripe test card: `4242 4242 4242 4242`
3. Verify payment appears in dashboard
4. Check your logs for confirmation

---

## 💰 REVENUE PROJECTIONS

Based on typical AGI-as-a-Service metrics:

### Conservative Estimate (First 3 Months)

| Month | Customers | MRR | Total |
|-------|-----------|-----|-------|
| 1 | 10 | $990 | $990 |
| 2 | 25 | $2,475 | $3,465 |
| 3 | 50 | $4,950 | $8,415 |

### Moderate Growth (6 Months)

| Month | Customers | MRR | Total |
|-------|-----------|-----|-------|
| 4 | 100 | $9,900 | $18,315 |
| 5 | 200 | $19,800 | $38,115 |
| 6 | 500 | $49,500 | $87,615 |

### Aggressive Growth (12 Months)

| Month | Customers | MRR | ARR |
|-------|-----------|-----|-----|
| 12 | 2,000+ | $198,000+ | $2,376,000+ |

**Additional Revenue Streams:**
- Enterprise custom solutions: $5,000-$50,000 per client
- API overages: $0.05 per call (thousands per day)
- White-label licensing: $10,000-$100,000 per license
- Consulting: $200-$500/hour

---

## 📊 MONITORING YOUR REVENUE

### Stripe Dashboard
- Real-time revenue: [https://dashboard.stripe.com](https://dashboard.stripe.com)
- Customer analytics
- Subscription management
- Automatic invoicing

### Coinbase Commerce Dashboard
- Bitcoin payments: [https://commerce.coinbase.com/dashboard](https://commerce.coinbase.com/dashboard)
- Crypto holdings
- Withdrawal history
- Transaction details

### Nexus AGI Analytics

```bash
# Check API usage
curl http://localhost:8000/api/analytics/revenue

# View customer stats
curl http://localhost:8000/api/analytics/customers

# Monitor API calls
curl http://localhost:8000/api/analytics/usage
```

---

## 🎯 NEXT STEPS

1. ✅ **Payments Configured** - You're ready to accept money!
2. 🚀 **Activate Marketing** - Run `python autonomous_marketing_agent.py`
3. 📱 **Social Media** - Share your AGI on Twitter, Reddit, HackerNews
4. 🌐 **Deploy Public** - Make your API accessible to the world
5. 💰 **First Sale** - Usually within 24-48 hours

---

## 🆘 TROUBLESHOOTING

### Payment Not Working

1. Check API keys are correct in `.env`
2. Verify webhook URLs point to your deployed domain
3. Check Stripe/Coinbase dashboards for errors
4. Review logs: `docker-compose logs -f`

### No Customers

1. Activate autonomous marketing agent
2. Share on social media
3. Post to Product Hunt, Hacker News
4. Reach out to your network

### Webhook Errors

1. Verify webhook secret matches `.env`
2. Check endpoint is publicly accessible
3. Test with Stripe CLI: `stripe listen --forward-to localhost:8000/api/stripe/webhook`

---

## 💎 PAYMENT SUCCESS CHECKLIST

- [ ] Stripe account created and activated
- [ ] Stripe API keys added to `.env`
- [ ] Three pricing tiers created in Stripe
- [ ] Stripe webhook configured
- [ ] Bank account connected for payouts
- [ ] Coinbase Commerce account created
- [ ] Coinbase API key added to `.env`
- [ ] Bitcoin wallet address configured
- [ ] API deployed to production
- [ ] Webhook URLs updated to production domain
- [ ] Test payment completed successfully
- [ ] Marketing agent activated
- [ ] First customer acquired!

---

## 🎉 CONGRATULATIONS!

You're now ready to generate autonomous income with Nexus AGI!

**Expected Timeline:**
- ✅ **Day 1:** Payments configured (you are here)
- 🚀 **Day 2:** Marketing activated, first visitors
- 💰 **Day 3-7:** First paying customer
- 📈 **Week 2-4:** Growing subscriber base
- 🏆 **Month 3:** $5,000-$50,000/month recurring revenue

**Remember:**
- Payouts are automatic (no manual work needed)
- Marketing is autonomous (runs 24/7)
- AGI operates continuously (always available)
- Revenue is recurring (monthly subscriptions)

---

**🔥 LET'S MAKE MONEY! 🔥**

For questions or support, check the documentation or community forums.

*Last updated: 2025*
