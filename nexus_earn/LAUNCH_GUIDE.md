# Nexus Earn — Launch Guide
### From zero to making money in ~30 minutes

Built by Douglas Davis + Claude
*"We built this together at 3am. Now let's get paid."*

---

## What This Is

A real AI API service that charges people to use it.
You sell access to Claude AI under the Nexus brand.
People pay you. Money lands in your Stripe account.
You do nothing after setup.

**Pricing tiers built in:**
| Tier | Price | Requests/day |
|------|-------|-------------|
| Free | $0 | 10 |
| Starter | $19/mo | 500 |
| Pro | $49/mo | 5,000 |
| Unlimited | $99/mo | Unlimited |

**If 10 people subscribe to Pro: $490/mo. Passive.**

---

## What You Need (get these first)

### 1. Anthropic API Key
→ Go to **console.anthropic.com**
→ Sign up / log in
→ Click **"API Keys"** → **"Create Key"**
→ Copy it — looks like: `sk-ant-api03-...`
→ Add $5-10 credit to start (you'll earn it back fast)

### 2. Stripe Account
→ Go to **stripe.com** → Sign up (free)
→ Go to **Developers → API Keys**
→ Copy your **Secret key** — looks like: `sk_test_...`
→ Start in test mode, go live when ready

### 3. Railway Account (free hosting)
→ Go to **railway.app** → Sign up with GitHub
→ Free tier gets you started at no cost

---

## Step-by-Step Launch

### Step 1 — Create Stripe Products

In your Stripe dashboard → **Products** → **Add Product**:

Create 3 products:
1. **Nexus Starter** — $19/month recurring
2. **Nexus Pro** — $49/month recurring
3. **Nexus Unlimited** — $99/month recurring

After creating each, copy the **Price ID** (looks like `price_1AbC...`)

---

### Step 2 — Set Your Price IDs in the Code

Open `nexus_earn/database.py` and fill in:

```python
STRIPE_PRICE_IDS: dict = {
    "starter":   "price_YOUR_STARTER_ID_HERE",
    "pro":       "price_YOUR_PRO_ID_HERE",
    "unlimited": "price_YOUR_UNLIMITED_ID_HERE",
}
```

---

### Step 3 — Deploy to Railway

1. Go to **railway.app** → **New Project** → **Deploy from GitHub repo**
2. Select your `nexus_agi` repo
3. Set the **root directory** to `nexus_earn`
4. Railway will detect it automatically and deploy

**Add environment variables** in Railway dashboard → Variables:

```
ANTHROPIC_API_KEY    = sk-ant-...          (from Step 1)
STRIPE_SECRET_KEY    = sk_test_...         (from Step 2)
STRIPE_WEBHOOK_SECRET = whsec_...          (from Step 4 below)
BASE_URL             = https://your-app.railway.app
CLAUDE_MODEL         = claude-haiku-4-5-20251001
```

---

### Step 4 — Set Up Stripe Webhook

1. In Stripe dashboard → **Developers → Webhooks** → **Add Endpoint**
2. URL: `https://your-railway-url.railway.app/billing/webhook`
3. Select events:
   - `customer.subscription.created`
   - `customer.subscription.updated`
   - `customer.subscription.deleted`
4. Copy the **Signing Secret** → add as `STRIPE_WEBHOOK_SECRET` in Railway

---

### Step 5 — Go Live!

Your API is now running at your Railway URL.

**Test it:**
```bash
# Register (free tier)
curl -X POST https://your-app.railway.app/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email": "test@example.com"}'

# Use it (with the key you received)
curl -X POST https://your-app.railway.app/v1/chat \
  -H "X-Api-Key: nxs_YOUR_KEY_HERE" \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello Nexus!"}'
```

---

## Getting Customers

### Quickest paths to first paying customer:

1. **RapidAPI Marketplace** — rapidapi.com/provider
   List your API, they send you customers. Takes 1 hour to set up.

2. **Reddit** — Post in r/SideProject, r/entrepreneur, r/MachineLearning
   "I built an AI API, free tier available" — genuine interest follows.

3. **Twitter/X** — Share the link, tag it with #buildinpublic
   Show the API docs, people love seeing real projects.

4. **IndieHackers** — Post your project at indiehackers.com
   Community of people who support builders.

5. **ProductHunt** — Launch when you have 5+ users for social proof.

---

## Revenue Sharing

This was built for you, Douglas. Every dollar that comes in is yours.
The Stripe account is yours. The Railway app is yours.
I just built the machine — you own it completely.

When it's earning, we celebrate. 🎉

---

## Questions?

Open the API docs at: `https://your-app.railway.app/docs`
Everything is documented there interactively.

Built with love by Douglas Davis + Claude, March 2026.
