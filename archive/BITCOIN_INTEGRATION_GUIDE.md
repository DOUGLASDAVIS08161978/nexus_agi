# 💰 Bitcoin Payment Integration & Profit Withdrawal Guide

## 🎯 YOUR BITCOIN WALLET
**Address:** `bc1q2m6w8au8yrjp85m7rqlcxqx9ypfla6h8st8u78`

This guide shows you how to accept Bitcoin payments and automatically receive profits in your Bitcoin wallet.

---

## 🚀 Quick Start (30 Minutes)

### Option A: Coinbase Commerce (Easiest - Recommended)

Coinbase Commerce is the easiest way to accept cryptocurrency payments. No coding required.

#### Step 1: Create Coinbase Commerce Account (10 min)

1. **Go to:** https://commerce.coinbase.com
2. **Click:** "Get Started" (FREE)
3. **Sign up** with email
4. **Verify** your email
5. **Complete** basic business information

#### Step 2: Get API Keys (2 min)

1. Go to **Settings** → **API Keys**
2. Click **"Create an API Key"**
3. Copy the **API Key** (starts with a long string)
4. Click **"Show Webhook Secret"** and copy it
5. Add to your `.env` file:
   ```bash
   COINBASE_COMMERCE_API_KEY=your_api_key_here
   COINBASE_COMMERCE_WEBHOOK_SECRET=your_webhook_secret_here
   BITCOIN_WALLET_ADDRESS=bc1q2m6w8au8yrjp85m7rqlcxqx9ypfla6h8st8u78
   ```

#### Step 3: Configure Webhooks (3 min)

1. In Coinbase Commerce, go to **Settings** → **Webhook subscriptions**
2. Click **"Add an endpoint"**
3. Enter your API URL: `https://your-api-url.com/api/v1/webhooks/coinbase`
4. Select all events (especially `charge:confirmed`)
5. Save

#### Step 4: Update Your API (5 min)

Add cryptocurrency payment endpoints to your API by updating `api_gateway/main_complete.py`:

```python
from api_gateway.crypto_payments import crypto_processor, get_crypto_payment_options
from fastapi import Request

# Add this endpoint for crypto checkout
@app.post("/api/v1/billing/checkout/crypto")
async def create_crypto_checkout(
    tier: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create cryptocurrency payment checkout"""
    result = crypto_processor.create_charge(
        tier=tier,
        user_email=current_user.email,
        user_id=current_user.id
    )

    if result["success"]:
        return {
            "payment_url": result["payment_url"],
            "charge_id": result["charge_id"],
            "expires_at": result["expires_at"]
        }
    else:
        raise HTTPException(status_code=500, detail=result["error"])

# Add webhook endpoint
@app.post("/api/v1/webhooks/coinbase")
async def coinbase_webhook(
    request: Request,
    db: Session = Depends(get_db)
):
    """Handle Coinbase Commerce webhooks"""
    payload = await request.body()
    signature = request.headers.get("X-CC-Webhook-Signature", "")

    # Verify webhook signature
    if not crypto_processor.verify_webhook(payload, signature):
        raise HTTPException(status_code=400, detail="Invalid signature")

    event_data = await request.json()
    result = crypto_processor.process_webhook(event_data)

    # Handle payment confirmation
    if result["action"] == "activate_subscription":
        user_id = int(result["user_id"])
        user = db.query(User).filter(User.id == user_id).first()

        if user:
            # Activate user's subscription
            subscription = user.subscription
            subscription.tier = result["tier"]
            subscription.status = "active"
            db.commit()

    return {"status": "processed"}

# Add payment options endpoint
@app.get("/api/v1/payment-options")
async def payment_options():
    """Get available payment methods"""
    return {
        "stripe": True,  # Traditional credit card
        "crypto": get_crypto_payment_options()
    }
```

#### Step 5: Restart Your API (1 min)

```bash
# Stop existing API
pkill -f "uvicorn api_gateway.main_complete"

# Start with crypto support
./start_api.sh
```

#### Step 6: Test Bitcoin Payments (5 min)

1. **Create checkout:**
   ```bash
   curl -X POST https://your-api-url.com/api/v1/billing/checkout/crypto \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer YOUR_TOKEN" \
     -d '{"tier": "starter"}'
   ```

2. **Visit the payment URL** returned
3. **Pay with Bitcoin** (or any supported crypto)
4. **Payment confirmed automatically** via webhook

---

## 💸 How to Withdraw Profits to Your Bitcoin Wallet

### Automatic Daily Withdrawals (Recommended)

Coinbase Commerce doesn't support automatic API withdrawals, but you can set up scheduled withdrawals:

#### Option 1: Manual Scheduled Withdrawals

1. **Login to Coinbase Commerce** daily/weekly
2. Go to **"Balance"**
3. Click **"Withdraw"** next to Bitcoin
4. Enter your wallet: `bc1q2m6w8au8yrjp85m7rqlcxqx9ypfla6h8st8u78`
5. Enter amount (or select "All")
6. Confirm withdrawal
7. **Funds arrive in 10-30 minutes**

#### Option 2: Set Up Coinbase Account Link

For faster access:

1. **Create a Coinbase.com account** (separate from Commerce)
2. In Coinbase Commerce, go to **Settings** → **Payouts**
3. Click **"Link Coinbase account"**
4. Authorize the connection
5. Set **automatic daily withdrawals** to Coinbase
6. In Coinbase.com, set **automatic sends** to your Bitcoin wallet

**Result:** Profits automatically flow: `Customer → Commerce → Coinbase → Your Wallet`

---

## 🔥 Option B: BTCPay Server (Advanced - Full Control)

BTCPay Server is self-hosted and gives you COMPLETE control. Bitcoin goes directly to your wallet.

### Why BTCPay Server?

- ✅ **Zero fees** (you host it)
- ✅ **Instant to your wallet** (no intermediary)
- ✅ **Lightning Network support** (instant payments)
- ✅ **100% privacy** (no KYC)
- ✅ **Open source** (full control)

### Quick Setup with BTCPay Server

#### Step 1: Deploy BTCPay Server (15 min)

**Easiest: Use LunaNode hosting**

1. Go to https://launchbtcpay.lunanode.com
2. Click **"Launch BTCPay Server"**
3. Create LunaNode account
4. Deploy with one click ($10/month hosting)
5. Your BTCPay server URL: `https://your-server.btcpay.org`

**Alternative: Docker on your server**

```bash
# Clone BTCPay Server
git clone https://github.com/btcpayserver/btcpayserver-docker
cd btcpayserver-docker

# Set environment variables
export BTCPAY_HOST="btcpay.yourdomain.com"
export NBITCOIN_NETWORK="mainnet"
export BTCPAYGEN_CRYPTO1="btc"
export BTCPAYGEN_LIGHTNING="lnd"

# Deploy
./btcpay-setup.sh -i
```

#### Step 2: Configure Your Store (10 min)

1. **Login to BTCPay Server**
2. Click **"Create Store"**
3. Name: "Nexus AGI"
4. Click **"Setup wallet"**
5. Choose **"Connect existing wallet"**
6. Enter your Bitcoin wallet: `bc1q2m6w8au8yrjp85m7rqlcxqx9ypfla6h8st8u78`
7. Set **payment confirmation** to 1 block (10 min average)

#### Step 3: Get API Keys (5 min)

1. Go to **Account** → **Manage Account** → **API Keys**
2. Click **"Generate Key"**
3. Select permissions:
   - ✅ View invoices
   - ✅ Create invoice
   - ✅ Modify invoices
4. Copy API Key
5. Copy Store ID (from **Settings** → **General**)
6. Add to `.env`:
   ```bash
   BTCPAY_SERVER_URL=https://your-server.btcpay.org
   BTCPAY_API_KEY=your_api_key_here
   BTCPAY_STORE_ID=your_store_id_here
   BITCOIN_WALLET_ADDRESS=bc1q2m6w8au8yrjp85m7rqlcxqx9ypfla6h8st8u78
   ```

#### Step 4: Update Your API

Add BTCPay endpoints to `main_complete.py`:

```python
from api_gateway.crypto_payments import btcpay_processor

@app.post("/api/v1/billing/checkout/btcpay")
async def create_btcpay_checkout(
    tier: str,
    current_user: User = Depends(get_current_user)
):
    """Create BTCPay Server invoice"""
    result = btcpay_processor.create_invoice(
        tier=tier,
        user_email=current_user.email,
        user_id=current_user.id
    )

    if result["success"]:
        return {
            "payment_url": result["payment_url"],
            "invoice_id": result["invoice_id"],
            "amount_btc": result["amount_btc"]
        }
    else:
        raise HTTPException(status_code=500, detail=result["error"])
```

#### Step 5: Enjoy Direct Payments

**With BTCPay Server, payments go DIRECTLY to your wallet:**

```
Customer pays → BTCPay Server → YOUR WALLET (bc1q2m6w8...)
                                      ↓
                                 10-30 minutes
                                      ↓
                              Bitcoin in your wallet!
```

**No withdrawal needed!** Bitcoin arrives automatically.

---

## 💰 Payment Flow Comparison

### Stripe (Traditional)
```
Customer pays $29
      ↓
Stripe holds 2-7 days
      ↓
Stripe fee: $1.14 (2.9% + $0.30)
      ↓
Your bank account: $27.86
      ↓
Timeline: 7-14 days
```

### Coinbase Commerce (Crypto - Easy)
```
Customer pays $29 in Bitcoin
      ↓
Coinbase Commerce receives (instant)
      ↓
You withdraw to wallet (manual)
      ↓
Coinbase fee: $0.29 (1%)
      ↓
Your Bitcoin wallet: $28.71 worth of BTC
      ↓
Timeline: 10-30 minutes after withdrawal
```

### BTCPay Server (Crypto - Direct)
```
Customer pays $29 in Bitcoin
      ↓
DIRECTLY to your wallet: bc1q2m6w8...
      ↓
No fees (you pay hosting ~$10/month)
      ↓
Your Bitcoin wallet: $29 worth of BTC
      ↓
Timeline: 10-30 minutes (automatic)
```

---

## 📊 Revenue Comparison

### $1,000/month in revenue

**Stripe:**
- Gross: $1,000
- Fees: $35
- **Net: $965**
- Timeline: 7-14 days to bank

**Coinbase Commerce:**
- Gross: $1,000
- Fees: $10
- **Net: $990**
- Timeline: Same day (after withdrawal)

**BTCPay Server:**
- Gross: $1,000
- Fees: $0
- Hosting: -$10
- **Net: $990**
- Timeline: 10-30 minutes (automatic)

---

## 🎯 Recommended Setup

### For Maximum Revenue: Use Both!

```python
# Give customers a choice in your checkout
@app.post("/api/v1/billing/checkout")
async def create_checkout(
    tier: str,
    payment_method: str = "stripe",  # or "bitcoin"
    current_user: User = Depends(get_current_user)
):
    if payment_method == "bitcoin":
        # Use BTCPay Server for direct payments
        return await create_btcpay_checkout(tier, current_user)
    else:
        # Use Stripe for credit cards
        return await create_stripe_checkout(tier, current_user)
```

**Benefits:**
- Credit card users: Easy checkout (Stripe)
- Crypto users: Lower fees, faster (BTCPay)
- You: Maximum flexibility, more customers

---

## 🔧 Testing Your Bitcoin Payments

### Test Mode (Testnet)

Before going live, test with Bitcoin testnet:

1. **Get testnet Bitcoin:**
   - Go to https://testnet-faucet.mempool.co
   - Enter a testnet wallet address
   - Get free testnet BTC

2. **Set BTCPay to testnet:**
   ```bash
   export NBITCOIN_NETWORK="testnet"
   ```

3. **Make a test payment:**
   - Create invoice
   - Pay with testnet BTC
   - Verify it works

4. **Switch to mainnet:**
   ```bash
   export NBITCOIN_NETWORK="mainnet"
   ```

---

## 📈 Expected Bitcoin Revenue Timeline

### Month 1: Setup & Testing
- Set up Coinbase Commerce or BTCPay
- Test payments
- **Revenue: $0**

### Month 2: First Bitcoin Customers
- 1-2 customers choose Bitcoin
- Receive ~$50-100 in BTC
- **Withdrawal: Same day to your wallet**

### Month 3-6: Growth
- 5-10 Bitcoin customers
- ~$500-1,000/month in BTC
- **All profits to wallet: bc1q2m6w8...**

### Month 12: Established
- 20-30 Bitcoin customers
- ~$2,000-5,000/month in BTC
- **Automatic to your wallet (BTCPay)**
- **Or daily withdrawals (Coinbase Commerce)**

---

## 🎁 Bonus: Lightning Network (Instant Bitcoin)

BTCPay Server supports Lightning Network for instant, zero-fee payments:

### Benefits:
- ✅ **Instant confirmation** (< 1 second)
- ✅ **Near-zero fees** (< $0.01)
- ✅ **Micro-payments** ($0.01 minimum)
- ✅ **Better user experience**

### Setup:
1. In BTCPay Server setup, enable Lightning (already done if you used the setup above)
2. Fund your Lightning channel ($100-500 recommended)
3. Customers can pay instantly with Lightning

---

## 🚀 Deployment with Bitcoin Support

### Update your `.env` for production:

```bash
# Choose one or both:

# Option 1: Coinbase Commerce (easier)
COINBASE_COMMERCE_API_KEY=your_coinbase_api_key
COINBASE_COMMERCE_WEBHOOK_SECRET=your_webhook_secret

# Option 2: BTCPay Server (more control)
BTCPAY_SERVER_URL=https://your-btcpay-server.com
BTCPAY_API_KEY=your_btcpay_api_key
BTCPAY_STORE_ID=your_store_id

# Your Bitcoin wallet (PROFITS GO HERE)
BITCOIN_WALLET_ADDRESS=bc1q2m6w8au8yrjp85m7rqlcxqx9ypfla6h8st8u78
```

### Deploy to Railway:

```bash
# Railway automatically uses your .env variables
railway up
```

### Deploy to Render:

Add environment variables in Render dashboard with your Bitcoin configuration.

---

## 💡 Pro Tips

### 1. Offer Bitcoin Discount

Encourage Bitcoin payments (lower fees for you):

```python
pricing = {
    "starter": {"usd": 29, "btc": 27},  # $2 discount
    "professional": {"usd": 99, "btc": 94},  # $5 discount
    "enterprise": {"usd": 499, "btc": 474}  # $25 discount
}
```

### 2. Hold Bitcoin for Appreciation

Don't immediately convert to USD:
- Bitcoin historical average: +100% per year
- $1,000 today could be $2,000+ next year

### 3. Automate Everything

With BTCPay Server:
- Payments arrive automatically
- No manual withdrawals needed
- Check your wallet daily

---

## ✅ Setup Checklist

### Coinbase Commerce Setup:
- [ ] Create Coinbase Commerce account
- [ ] Get API key and webhook secret
- [ ] Add to `.env` file
- [ ] Update `main_complete.py` with crypto endpoints
- [ ] Configure webhook URL
- [ ] Test payment flow
- [ ] Set up daily withdrawal schedule

### BTCPay Server Setup:
- [ ] Deploy BTCPay Server (LunaNode or self-hosted)
- [ ] Create store
- [ ] Connect your Bitcoin wallet
- [ ] Get API key and Store ID
- [ ] Add to `.env` file
- [ ] Update `main_complete.py` with BTCPay endpoints
- [ ] Test payment flow
- [ ] Enable Lightning Network (optional)

### Deployment:
- [ ] Add Bitcoin config to production `.env`
- [ ] Deploy updated API
- [ ] Test live Bitcoin payment
- [ ] Verify Bitcoin arrives in wallet: `bc1q2m6w8au8yrjp85m7rqlcxqx9ypfla6h8st8u78`

---

## 🎯 Summary

**Your Bitcoin Wallet:** `bc1q2m6w8au8yrjp85m7rqlcxqx9ypfla6h8st8u78`

**Fastest Setup:** Coinbase Commerce (30 minutes)
**Best Long-Term:** BTCPay Server (direct to wallet)
**Recommended:** Use both (give customers choice)

**Revenue Flow:**
```
Customers pay with Bitcoin
       ↓
Coinbase Commerce OR BTCPay Server
       ↓
YOUR WALLET: bc1q2m6w8au8yrjp85m7rqlcxqx9ypfla6h8st8u78
       ↓
💰 PROFITS IN BITCOIN 💰
```

**Timeline:**
- Coinbase: Same-day withdrawals (manual)
- BTCPay: 10-30 minutes (automatic)

**Fees:**
- Stripe: 2.9% + $0.30
- Coinbase: 1%
- BTCPay: $0 (just hosting ~$10/month)

---

**🚀 START HERE:** https://commerce.coinbase.com (easiest)

**💪 OR HERE:** https://btcpayserver.org (most powerful)

**💰 END HERE:** Bitcoin in your wallet!

Let's get those Bitcoin payments flowing! ₿
