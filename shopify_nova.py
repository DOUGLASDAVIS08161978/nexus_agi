#!/usr/bin/env python3
"""
Nova-Omega Autonomous Shopify Store Manager
═══════════════════════════════════════════
Store  : nova-automation.myshopify.com
Niche  : Phone accessories (MagSafe, cases, chargers)
Creator: Douglas Davis

What this does autonomously:
  • Sources winning products (AliExpress / DSers suggestions)
  • Prices products with target margin
  • Writes SEO-optimised product descriptions
  • Creates listings in Shopify store
  • Monitors orders and triggers fulfilment
  • Generates TikTok / Instagram / Twitter / Pinterest content
  • Posts promotion copy to every platform
  • Runs 24/7 background watcher loop

Usage:
  python shopify_nova.py              # Demo mode (no API key needed)
  python shopify_nova.py --live       # Live mode (needs SHOPIFY_TOKEN)
  python shopify_nova.py --content    # Generate promotion content only
  python shopify_nova.py --products   # Import products only
  python shopify_nova.py --watch      # Order watcher loop
═══════════════════════════════════════════
"""

import os, sys, json, re, time, random, uuid, threading, queue, argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

# ── Optional imports ──────────────────────────────────────────────────────────
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

try:
    from dotenv import load_dotenv
    load_dotenv(Path.home() / "nexus_agi" / ".env")
    load_dotenv()
except ImportError:
    pass

# ── Config ────────────────────────────────────────────────────────────────────
STORE_URL    = "vqtkit-4a.myshopify.com"
CLIENT_ID    = os.getenv("SHOPIFY_CLIENT_ID",  "")
CLIENT_SECRET= os.getenv("SHOPIFY_SECRET",     "")
GROQ_KEY     = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL   = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
API_VERSION  = "2025-01"
BASE_API     = f"https://{STORE_URL}/admin/api/{API_VERSION}"

BASE_DIR  = Path.home() / "nexus_agi"
STORE_DIR = BASE_DIR / "shopify_store"
STORE_DIR.mkdir(parents=True, exist_ok=True)
TOKEN_CACHE = STORE_DIR / "token_cache.json"


# ═══════════════════════════════════════════════════════════════════════════════
# TOKEN MANAGER — auto-generates shpat_ from client credentials
# ═══════════════════════════════════════════════════════════════════════════════

class ShopifyTokenManager:
    """
    Obtains and caches the Shopify Admin API access token.
    Priority:
      1. SHOPIFY_TOKEN env var (shpat_xxx already obtained)
      2. Cached token on disk (token_cache.json)
      3. Auto-generate via client credentials POST
    """

    def __init__(self):
        self._token: str = ""

    def get_token(self) -> str:
        if self._token:
            return self._token

        # 1 — env var
        env_tok = os.getenv("SHOPIFY_TOKEN", "").strip()
        if env_tok:
            self._token = env_tok
            return self._token

        # 2 — disk cache
        cached = self._load_cache()
        if cached:
            self._token = cached
            return self._token

        # 3 — generate
        if CLIENT_ID and CLIENT_SECRET and REQUESTS_AVAILABLE:
            tok = self._generate()
            if tok:
                self._token = tok
                self._save_cache(tok)
                self._write_env(tok)
                return self._token

        return ""

    def _generate(self) -> str:
        """Cannot auto-generate — Partner apps require OAuth authorization code flow.
        Run get_shopify_token.py once to obtain the token, then it's cached."""
        print(f"  \033[93m⚠\033[0m  No Shopify token found.")
        print(f"      Run:  python get_shopify_token.py")
        print(f"      to complete the one-time OAuth flow and save your token.\n")
        return ""

    def _load_cache(self) -> str:
        try:
            if TOKEN_CACHE.exists():
                d = json.loads(TOKEN_CACHE.read_text())
                return d.get("access_token", "")
        except:
            pass
        return ""

    def _save_cache(self, token: str):
        try:
            TOKEN_CACHE.write_text(json.dumps({"access_token": token,
                                               "saved_at": datetime.now().isoformat()}))
        except:
            pass

    def _write_env(self, token: str):
        """Append token to .env file so future runs skip generation."""
        try:
            env_path = BASE_DIR / ".env"
            lines = env_path.read_text().splitlines() if env_path.exists() else []
            lines = [l for l in lines if not l.startswith("SHOPIFY_TOKEN=")]
            lines.append(f"SHOPIFY_TOKEN={token}")
            env_path.write_text("\n".join(lines) + "\n")
        except:
            pass

    def refresh(self) -> str:
        """Force re-generate token (call if API returns 401)."""
        self._token = ""
        if TOKEN_CACHE.exists():
            TOKEN_CACHE.unlink()
        return self.get_token()


# ── Resolve access token at startup ───────────────────────────────────────────
_token_mgr   = ShopifyTokenManager()
ACCESS_TOKEN  = _token_mgr.get_token()
LIVE_MODE     = bool(ACCESS_TOKEN and REQUESTS_AVAILABLE)
AI_MODE       = bool(GROQ_KEY and GROQ_AVAILABLE)

TARGET_MARGIN = 0.65      # 65% gross margin target
MIN_PRICE     = 9.99
MAX_PRICE     = 89.99
PLATFORM_FEE  = 0.03      # Shopify 3% transaction fee

PRODUCTS_DB  = STORE_DIR / "products.json"
ORDERS_DB    = STORE_DIR / "orders.json"
CONTENT_DB   = STORE_DIR / "content.json"
ANALYTICS_DB = STORE_DIR / "analytics.json"

# ── Colours ───────────────────────────────────────────────────────────────────
R="\033[0m"; B="\033[1m"; DIM="\033[2m"
CY="\033[96m"; GR="\033[92m"; YL="\033[93m"; MG="\033[95m"
BL="\033[94m"; RD="\033[91m"; WH="\033[97m"
CYB="\033[1;96m"; GRB="\033[1;92m"; MGB="\033[1;95m"

def c(col, text): return f"{col}{text}{R}"
def ok(msg):   print(f"  {c(GR,'✓')}  {msg}")
def warn(msg): print(f"  {c(YL,'⚠')}  {msg}")
def err(msg):  print(f"  {c(RD,'✗')}  {msg}")
def section(title):
    print(f"\n{c(CY,'─'*70)}")
    print(c(CYB, f"  {title}"))
    print(c(CY,'─'*70))

# ── Groq helper ───────────────────────────────────────────────────────────────
_groq = None
if GROQ_AVAILABLE and GROQ_KEY:
    try: _groq = Groq(api_key=GROQ_KEY)
    except: pass

def ai(prompt: str, temp: float=0.7, max_tokens: int=400) -> str:
    if not _groq:
        return f"[AI unavailable — set GROQ_API_KEY] Prompt: {prompt[:60]}"
    try:
        r = _groq.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role":"user","content":prompt}],
            temperature=temp, max_tokens=max_tokens)
        return r.choices[0].message.content
    except Exception as e:
        return f"[AI error: {e}]"

# ── JSON database helpers ─────────────────────────────────────────────────────
def _load(path: Path, default=None):
    try:
        return json.loads(path.read_text()) if path.exists() else (default if default is not None else [])
    except: return default if default is not None else []

def _save(path: Path, data):
    path.write_text(json.dumps(data, indent=2))


# ═══════════════════════════════════════════════════════════════════════════════
# SHOPIFY API CLIENT
# ═══════════════════════════════════════════════════════════════════════════════

class ShopifyClient:
    """Shopify Admin REST API v2025-01 wrapper."""

    def __init__(self, token: str=""):
        self.token = token or _token_mgr.get_token() or ACCESS_TOKEN
        self.headers = {
            "X-Shopify-Access-Token": self.token,
            "Content-Type": "application/json",
        }
        self.base = BASE_API

    def _req(self, method: str, endpoint: str, data: dict=None) -> dict:
        if not REQUESTS_AVAILABLE:
            return {"error": "requests not installed"}
        if not self.token:
            return {"error": "No Shopify token — set SHOPIFY_CLIENT_ID + SHOPIFY_SECRET in .env"}
        url = f"{self.base}{endpoint}"
        try:
            resp = requests.request(
                method, url, headers=self.headers,
                json=data, timeout=20)
            if resp.status_code == 401:
                # Token expired — refresh and retry once
                new_tok = _token_mgr.refresh()
                if new_tok:
                    self.token = new_tok
                    self.headers["X-Shopify-Access-Token"] = new_tok
                    resp = requests.request(method, url, headers=self.headers,
                                            json=data, timeout=20)
            return resp.json()
        except Exception as e:
            return {"error": str(e)}

    def get(self, endpoint: str) -> dict:
        return self._req("GET", endpoint)

    def post(self, endpoint: str, data: dict) -> dict:
        return self._req("POST", endpoint, data)

    def put(self, endpoint: str, data: dict) -> dict:
        return self._req("PUT", endpoint, data)

    def test_connection(self) -> bool:
        r = self.get("/shop.json")
        return "shop" in r

    # ── Products ──────────────────────────────────────────────────────────────
    def create_product(self, title: str, body_html: str, price: float,
                       vendor: str="Nova Store", product_type: str="Phone Accessories",
                       tags: List[str]=None, image_url: str=None) -> dict:
        payload = {"product": {
            "title": title, "body_html": body_html,
            "vendor": vendor, "product_type": product_type,
            "tags": ",".join(tags or []),
            "status": "active",
            "variants": [{"price": f"{price:.2f}", "inventory_management": "shopify",
                          "inventory_quantity": 999}],
        }}
        if image_url:
            payload["product"]["images"] = [{"src": image_url}]
        return self.post("/products.json", payload)

    def list_products(self, limit: int=50) -> List[dict]:
        r = self.get(f"/products.json?limit={limit}&status=active")
        return r.get("products", [])

    def update_price(self, variant_id: str, new_price: float) -> dict:
        return self.put(f"/variants/{variant_id}.json",
                        {"variant": {"id": variant_id, "price": f"{new_price:.2f}"}})

    # ── Orders ────────────────────────────────────────────────────────────────
    def get_orders(self, status: str="unfulfilled", limit: int=50) -> List[dict]:
        r = self.get(f"/orders.json?fulfillment_status={status}&limit={limit}&status=open")
        return r.get("orders", [])

    def fulfill_order(self, order_id: str, line_item_ids: List[str],
                      tracking_number: str=None, tracking_company: str="AliExpress") -> dict:
        payload = {"fulfillment": {
            "line_items_by_fulfillment_order": [{"fulfillment_order_id": oid}
                                                 for oid in line_item_ids],
            "notify_customer": True,
        }}
        if tracking_number:
            payload["fulfillment"]["tracking_info"] = {
                "number": tracking_number, "company": tracking_company}
        return self.post(f"/orders/{order_id}/fulfillments.json", payload)

    def add_order_note(self, order_id: str, note: str) -> dict:
        return self.put(f"/orders/{order_id}.json", {"order": {"note": note}})

    # ── Inventory ─────────────────────────────────────────────────────────────
    def get_inventory_levels(self, location_id: str=None) -> List[dict]:
        endpoint = "/inventory_levels.json"
        if location_id: endpoint += f"?location_ids={location_id}"
        return self.get(endpoint).get("inventory_levels", [])


# ═══════════════════════════════════════════════════════════════════════════════
# PRICING ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class PricingEngine:
    """Smart pricing: AliExpress cost → optimal selling price."""

    def calculate(self, cost_usd: float, margin: float=TARGET_MARGIN) -> dict:
        # selling_price = cost / (1 - margin - platform_fee)
        raw_price = cost_usd / (1 - margin - PLATFORM_FEE)
        # Round to psychological pricing (x.99 or x.95)
        base = max(MIN_PRICE, min(MAX_PRICE, raw_price))
        price = round(base - 0.01, 2) if base > 10 else round(base - 0.05, 2)
        profit = price * (1 - PLATFORM_FEE) - cost_usd
        actual_margin = profit / price if price > 0 else 0
        return {
            "cost": cost_usd, "price": price, "profit": profit,
            "margin": actual_margin, "markup_multiple": price/cost_usd if cost_usd > 0 else 0
        }

    def suggest_price_tiers(self, cost: float) -> dict:
        return {
            "budget":   self.calculate(cost, 0.50),
            "standard": self.calculate(cost, 0.65),
            "premium":  self.calculate(cost, 0.75),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# PRODUCT SOURCING — DSers / AliExpress suggestions
# ═══════════════════════════════════════════════════════════════════════════════

# Curated winning phone accessory products with real AliExpress cost ranges
WINNING_PRODUCTS = [
    # ── MagSafe / iPhone ──────────────────────────────────────────────────────
    {"name":"MagSafe Magnetic Phone Case – iPhone 15/16 Series",
     "category":"Cases","cost_usd":3.20,"tags":["magsafe","iphone","case","magnetic"],
     "ali_search":"MagSafe magnetic case iPhone 15","monthly_sales_est":1200},
    {"name":"360° MagSafe Ring Stand & Grip – Universal",
     "category":"Accessories","cost_usd":2.50,"tags":["magsafe","ring stand","grip"],
     "ali_search":"magsafe ring stand grip holder","monthly_sales_est":900},
    {"name":"MagSafe 3-in-1 Wireless Charging Pad",
     "category":"Chargers","cost_usd":7.20,"tags":["wireless charging","magsafe","3-in-1","apple watch"],
     "ali_search":"magsafe 3 in 1 wireless charging station","monthly_sales_est":850},
    {"name":"10,000mAh MagSafe Wireless Power Bank",
     "category":"Power","cost_usd":8.90,"tags":["power bank","magsafe","wireless","10000mah"],
     "ali_search":"10000mah magsafe wireless power bank","monthly_sales_est":750},
    {"name":"MagSafe Magnetic Wallet – Slim Card Holder",
     "category":"Accessories","cost_usd":2.20,"tags":["magsafe","wallet","card holder","slim"],
     "ali_search":"magsafe magnetic wallet card holder slim","monthly_sales_est":1400},
    # ── Chargers & Cables ─────────────────────────────────────────────────────
    {"name":"20W USB-C GaN Fast Charger – Wall Adapter",
     "category":"Chargers","cost_usd":4.10,"tags":["charger","gan","usb-c","fast charge"],
     "ali_search":"20W GaN USB-C charger compact","monthly_sales_est":2100},
    {"name":"65W 3-Port GaN Charger – USB-C + USB-A",
     "category":"Chargers","cost_usd":7.80,"tags":["gan","65w","multi-port","usb-c","fast charge"],
     "ali_search":"65W GaN 3 port charger USB-C USB-A","monthly_sales_est":1600},
    {"name":"Retractable USB-C to Lightning Cable 3-Pack",
     "category":"Cables","cost_usd":3.60,"tags":["cable","usb-c","lightning","retractable"],
     "ali_search":"retractable USB-C lightning cable 3 pack","monthly_sales_est":1500},
    {"name":"USB-C to USB-C Braided Cable 2M – Fast Charge",
     "category":"Cables","cost_usd":2.10,"tags":["usb-c","cable","braided","fast charge","2m"],
     "ali_search":"USB-C braided cable 2 meter fast charge","monthly_sales_est":1900},
    # ── Cases & Protection ────────────────────────────────────────────────────
    {"name":"Phone Screen Protector Tempered Glass 3-Pack",
     "category":"Protection","cost_usd":1.80,"tags":["screen protector","tempered glass","iphone"],
     "ali_search":"iPhone tempered glass screen protector 3 pack","monthly_sales_est":3200},
    {"name":"Luxury Leather Phone Wallet Case – Card Holder",
     "category":"Cases","cost_usd":5.50,"tags":["leather","wallet","card holder","premium"],
     "ali_search":"leather wallet phone case card holder","monthly_sales_est":650},
    {"name":"Clear Shockproof Phone Case – Military Grade",
     "category":"Cases","cost_usd":2.30,"tags":["clear case","shockproof","military grade","iphone"],
     "ali_search":"clear shockproof military grade phone case iPhone","monthly_sales_est":2800},
    {"name":"Privacy Screen Protector Anti-Spy – iPhone",
     "category":"Protection","cost_usd":2.40,"tags":["privacy","anti-spy","screen protector","iphone"],
     "ali_search":"privacy screen protector anti spy iPhone","monthly_sales_est":1100},
    # ── Car & Travel ──────────────────────────────────────────────────────────
    {"name":"Magnetic Car Phone Mount – Dashboard & Vent",
     "category":"Car","cost_usd":2.80,"tags":["car mount","magnetic","dashboard","magsafe"],
     "ali_search":"magnetic car phone mount magsafe","monthly_sales_est":1800},
    {"name":"Wireless Car Charger Mount – 15W Fast Charge",
     "category":"Car","cost_usd":6.50,"tags":["car charger","wireless","15w","phone mount","fast charge"],
     "ali_search":"wireless car charger mount 15W fast charging","monthly_sales_est":1300},
    # ── Photography & Content Creation ────────────────────────────────────────
    {"name":"Portable Phone Tripod & Selfie Ring Light",
     "category":"Photography","cost_usd":6.40,"tags":["tripod","ring light","selfie","tiktok"],
     "ali_search":"portable phone tripod ring light selfie","monthly_sales_est":1100},
    {"name":"Phone Camera Lens Kit – Wide + Macro + Fish-Eye",
     "category":"Photography","cost_usd":4.20,"tags":["camera lens","wide angle","macro","fish eye","photography"],
     "ali_search":"phone camera lens kit wide macro fisheye clip on","monthly_sales_est":960},
    {"name":"Mini Phone Stabilizer Gimbal – 3-Axis",
     "category":"Photography","cost_usd":14.50,"tags":["gimbal","stabilizer","3-axis","video","tiktok","reels"],
     "ali_search":"mini phone gimbal stabilizer 3 axis","monthly_sales_est":720},
    # ── Productivity & Lifestyle ──────────────────────────────────────────────
    {"name":"Portable Mini Bluetooth Keyboard – Phone Tablet",
     "category":"Productivity","cost_usd":8.20,"tags":["bluetooth keyboard","mini","portable","ipad","iphone"],
     "ali_search":"portable mini bluetooth keyboard foldable","monthly_sales_est":580},
    {"name":"PopSocket Phone Grip & Stand – MagSafe Compatible",
     "category":"Accessories","cost_usd":1.90,"tags":["popsocket","grip","stand","magsafe"],
     "ali_search":"popsocket phone grip stand magsafe compatible","monthly_sales_est":2400},
    {"name":"Waterproof Phone Pouch – Beach & Swimming",
     "category":"Protection","cost_usd":2.60,"tags":["waterproof","pouch","beach","swimming","underwater"],
     "ali_search":"waterproof phone pouch beach swimming case","monthly_sales_est":1700},
    {"name":"Night Vision Phone Camera Lens Clip – Stargazing",
     "category":"Photography","cost_usd":5.80,"tags":["night vision","camera lens","stargazing","astronomy"],
     "ali_search":"night vision phone camera lens clip stargazing","monthly_sales_est":450},
]

class ProductSourcing:
    """DSers/AliExpress product sourcing suggestions and import helper."""

    def __init__(self, pricing: PricingEngine):
        self.pricing = pricing
        self.products = _load(PRODUCTS_DB, [])

    def get_winning_products(self) -> List[dict]:
        enriched = []
        for p in WINNING_PRODUCTS:
            tiers = self.pricing.suggest_price_tiers(p["cost_usd"])
            enriched.append({**p, "pricing": tiers,
                             "recommended_price": tiers["standard"]["price"],
                             "estimated_monthly_profit": tiers["standard"]["profit"] * p.get("monthly_sales_est", 500)})
        return sorted(enriched, key=lambda x: -x["estimated_monthly_profit"])

    def dsers_instructions(self, product: dict) -> str:
        return (f"DSers Import Steps for: {product['name']}\n"
                f"  1. Open DSers → AliExpress → search: '{product.get('ali_search',product['name'])}'\n"
                f"  2. Filter: 4★+ rating, 100+ orders, ePacket/AliExpress Standard shipping\n"
                f"  3. Click 'Push to Store' → select nova-automation.myshopify.com\n"
                f"  4. Set selling price: ${product.get('recommended_price',19.99):.2f}\n"
                f"  5. Enable Auto-fulfill in DSers settings")

    def save_product(self, product: dict):
        self.products.append({**product, "added": datetime.now().isoformat()})
        _save(PRODUCTS_DB, self.products)


# ═══════════════════════════════════════════════════════════════════════════════
# SEO ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class SEOEngine:
    """Generate SEO-optimised product content."""

    def product_description(self, product_name: str, features: List[str]=None) -> str:
        feat_str = "\n".join(f"• {f}" for f in features) if features else ""
        prompt = (f"Write an SEO-optimized Shopify product description for: '{product_name}'\n"
                  f"{'Features: ' + feat_str if feat_str else ''}\n"
                  f"Include: compelling headline, 3-5 benefit bullets, scarcity/urgency CTA.\n"
                  f"Format as HTML (h2, ul, p tags). 150-200 words. High-converting copywriting.")
        return ai(prompt, temp=0.7, max_tokens=400)

    def meta_title(self, product_name: str) -> str:
        prompt = f"Write a 60-char SEO meta title for Shopify product: '{product_name}'. Include main keyword."
        return ai(prompt, temp=0.5, max_tokens=30)

    def meta_description(self, product_name: str) -> str:
        prompt = f"Write a 155-char SEO meta description for: '{product_name}'. Include CTA."
        return ai(prompt, temp=0.5, max_tokens=60)

    def keyword_tags(self, product_name: str, category: str) -> List[str]:
        prompt = (f"Generate 8 Shopify product tags (keywords) for: '{product_name}' in category '{category}'.\n"
                  f"Return as JSON list. Mix broad and long-tail keywords.")
        resp = ai(prompt, temp=0.5, max_tokens=100)
        try:
            m = re.search(r'\[.*\]', resp, re.DOTALL)
            return json.loads(m.group()) if m else [product_name.lower(), category.lower()]
        except:
            return [product_name.lower(), category.lower()]


# ═══════════════════════════════════════════════════════════════════════════════
# STORE PROMOTION ENGINE — TikTok / Instagram / Twitter / Pinterest / Reddit
# ═══════════════════════════════════════════════════════════════════════════════

class StorePromoter:
    """Autonomous multi-platform content generation and promotion."""

    def __init__(self):
        self.content_log = _load(CONTENT_DB, [])

    def _log(self, platform: str, product: str, content: dict):
        self.content_log.append({
            "platform": platform, "product": product,
            "content": content, "created": datetime.now().isoformat()
        })
        _save(CONTENT_DB, self.content_log[-500:])

    # ── TikTok ────────────────────────────────────────────────────────────────
    def tiktok_script(self, product: dict) -> dict:
        name = product.get("name", "phone accessory")
        price = product.get("recommended_price", 19.99)
        prompt = (f"Write a viral TikTok video script for selling: '{name}' for ${price:.2f}\n"
                  f"Format JSON: {{\"hook\": \"first 3 seconds\", \"body\": \"15-second demo script\","
                  f"\"cta\": \"call to action\", \"hashtags\": [\"...\",\"...\",\"...\",\"...\",\"...\"],"
                  f"\"viral_angle\": \"why this will go viral\"}}")
        resp = ai(prompt, temp=0.9, max_tokens=300)
        try:
            m = re.search(r'\{.*\}', resp, re.DOTALL)
            result = json.loads(m.group()) if m else {}
        except:
            result = {
                "hook": f"POV: You finally found the {name.split('–')[0].strip()} you needed 😍",
                "body": f"This {name} just changed my life. Unboxing + demo in 15 seconds.",
                "cta": f"Link in bio — only ${price:.2f} today! ⬆️",
                "hashtags": ["#phonegadgets","#techfinds","#iphone","#magsafe","#fyp"],
                "viral_angle": "Relatable problem → satisfying solution"
            }
        self._log("tiktok", name, result)
        return result

    # ── Instagram ─────────────────────────────────────────────────────────────
    def instagram_post(self, product: dict) -> dict:
        name = product.get("name", "phone accessory")
        price = product.get("recommended_price", 19.99)
        prompt = (f"Write an Instagram post for selling: '{name}' for ${price:.2f}\n"
                  f"JSON: {{\"caption\": \"engaging caption with emojis\","
                  f"\"hashtags\": [30 relevant hashtags],"
                  f"\"image_prompt\": \"Midjourney/DALL-E image generation prompt\"}}")
        resp = ai(prompt, temp=0.85, max_tokens=350)
        try:
            m = re.search(r'\{.*\}', resp, re.DOTALL)
            result = json.loads(m.group()) if m else {}
        except:
            result = {
                "caption": f"✨ Upgrade your phone setup with the {name}! 📱\n\nOnly ${price:.2f} — shop now, link in bio! 👆",
                "hashtags": ["#phoneaccessories","#techdeals","#iphonecase","#magsafe",
                             "#gadgets","#techlife","#phonegear","#musthave"],
                "image_prompt": f"Professional product photo of {name} on clean white background, studio lighting"
            }
        self._log("instagram", name, result)
        return result

    # ── Twitter/X ─────────────────────────────────────────────────────────────
    def twitter_thread(self, product: dict) -> List[str]:
        name = product.get("name", "phone accessory")
        price = product.get("recommended_price", 19.99)
        cost = product.get("cost_usd", 3.00)
        prompt = (f"Write a 5-tweet Twitter thread selling '{name}' for ${price:.2f} (cost ${cost:.2f}).\n"
                  f"Thread should: hook → problem → solution → proof → CTA.\n"
                  f"JSON: [\"tweet1\",\"tweet2\",\"tweet3\",\"tweet4\",\"tweet5\"]")
        resp = ai(prompt, temp=0.8, max_tokens=400)
        try:
            m = re.search(r'\[.*\]', resp, re.DOTALL)
            return json.loads(m.group()) if m else []
        except:
            return [
                f"🧵 THREAD: The phone accessory I wish I found sooner...",
                f"The problem: {name.split('–')[0].strip()} always breaks or doesn't work. Sound familiar?",
                f"The fix: {name} — built different. ${price:.2f} and it just works.",
                f"Proof: 1,000+ happy customers, 4.8★ rating. Real reviews speak louder.",
                f"Get yours: nova-automation.myshopify.com 🔗 Use code NOVA10 for 10% off today only!"
            ]

    # ── Pinterest ─────────────────────────────────────────────────────────────
    def pinterest_pin(self, product: dict) -> dict:
        name = product.get("name", "phone accessory")
        price = product.get("recommended_price", 19.99)
        prompt = (f"Write a Pinterest pin for: '{name}' at ${price:.2f}\n"
                  f"JSON: {{\"title\": \"...\", \"description\": \"150 chars\","
                  f"\"board\": \"best board name\", \"keywords\": [\"...\",\"...\",\"...\"]}}")
        resp = ai(prompt, temp=0.7, max_tokens=200)
        try:
            m = re.search(r'\{.*\}', resp, re.DOTALL)
            return json.loads(m.group()) if m else {}
        except:
            return {"title": f"{name} – Only ${price:.2f}",
                    "description": f"Best {name} deal of 2026. Shop now!",
                    "board": "Phone Accessories & Tech Gadgets",
                    "keywords": ["phone accessories", "tech gifts", "iphone"]}

    # ── Reddit ────────────────────────────────────────────────────────────────
    def reddit_post(self, product: dict) -> dict:
        name = product.get("name", "phone accessory")
        price = product.get("recommended_price", 19.99)
        prompt = (f"Write a genuine-sounding Reddit post recommending '{name}' at ${price:.2f}.\n"
                  f"Subreddits: r/iphone, r/gadgets, r/phoneaccessories, r/tech.\n"
                  f"JSON: {{\"subreddit\": \"...\", \"title\": \"...\", \"body\": \"...200 words, conversational...\"}}")
        resp = ai(prompt, temp=0.85, max_tokens=300)
        try:
            m = re.search(r'\{.*\}', resp, re.DOTALL)
            return json.loads(m.group()) if m else {}
        except:
            return {"subreddit": "r/iphone",
                    "title": f"Found this {name.split('–')[0].strip()} and it's actually amazing",
                    "body": f"Just wanted to share — picked up a {name} for ${price:.2f} and it's been great."}

    def full_campaign(self, product: dict) -> dict:
        print(f"  Generating {c(MG,'multi-platform campaign')} for: {product['name'][:50]}...")
        return {
            "product": product["name"],
            "tiktok":    self.tiktok_script(product),
            "instagram": self.instagram_post(product),
            "twitter":   self.twitter_thread(product),
            "pinterest": self.pinterest_pin(product),
            "reddit":    self.reddit_post(product),
            "created":   datetime.now().isoformat()
        }


# ═══════════════════════════════════════════════════════════════════════════════
# ORDER WATCHER — autonomous fulfilment loop
# ═══════════════════════════════════════════════════════════════════════════════

class OrderWatcher:
    """Monitors Shopify for new orders and triggers DSers fulfilment."""

    def __init__(self, shopify: ShopifyClient):
        self.shopify = shopify
        self.orders_log = _load(ORDERS_DB, [])
        self.running = False

    def check_orders(self) -> List[dict]:
        if not LIVE_MODE:
            # Demo: simulate incoming orders
            demo_order = {
                "id": str(random.randint(10000, 99999)),
                "name": f"#NO{random.randint(1000,9999)}",
                "email": "customer@example.com",
                "total_price": f"{random.uniform(15,45):.2f}",
                "line_items": [{"title": random.choice(WINNING_PRODUCTS)["name"],
                                "quantity": random.randint(1,3)}],
                "fulfillment_status": "unfulfilled",
                "created_at": datetime.now().isoformat()
            }
            return [demo_order] if random.random() > 0.5 else []
        return self.shopify.get_orders(status="unfulfilled")

    def process_order(self, order: dict) -> str:
        order_id = order.get("id")
        order_name = order.get("name", "Unknown")
        items = order.get("line_items", [])
        total = order.get("total_price", "0.00")

        print(f"\n  {c(YL,'📦 NEW ORDER')} {order_name} — ${total}")
        for item in items:
            print(f"     → {item['quantity']}x {item['title'][:50]}")

        # In live mode: attempt fulfilment
        if LIVE_MODE:
            # DSers handles actual AliExpress placement — we mark as processing
            note = f"[Nova Auto] Order received {datetime.now().strftime('%Y-%m-%d %H:%M')}. DSers fulfilment initiated."
            self.shopify.add_order_note(order_id, note)
            action = "marked for DSers fulfilment"
        else:
            action = "[DEMO] would trigger DSers auto-fulfilment"

        self.orders_log.append({
            "order_id": order_id, "name": order_name,
            "total": total, "items": items,
            "action": action, "processed_at": datetime.now().isoformat()
        })
        _save(ORDERS_DB, self.orders_log[-1000:])
        return action

    def watch_loop(self, interval_seconds: int=120):
        self.running = True
        print(c(GR, f"\n  🔄 Order watcher started (checking every {interval_seconds}s)"))
        while self.running:
            try:
                orders = self.check_orders()
                if orders:
                    for order in orders:
                        self.process_order(order)
                else:
                    print(f"  {c(DIM, datetime.now().strftime('%H:%M:%S'))}  No new orders.", end='\r')
                time.sleep(interval_seconds)
            except KeyboardInterrupt:
                self.running = False
                break
            except Exception as e:
                print(c(RD, f"  Order watcher error: {e}"))
                time.sleep(60)


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYTICS TRACKER
# ═══════════════════════════════════════════════════════════════════════════════

class StoreAnalytics:
    def __init__(self, shopify: ShopifyClient):
        self.shopify = shopify
        self.data = _load(ANALYTICS_DB, {"revenue": 0, "orders": 0, "top_products": []})

    def snapshot(self) -> dict:
        if LIVE_MODE:
            orders = self.shopify.get_orders(status="any")
            revenue = sum(float(o.get("total_price", 0)) for o in orders)
            self.data = {"revenue": revenue, "orders": len(orders),
                         "snapshot_at": datetime.now().isoformat()}
            _save(ANALYTICS_DB, self.data)
            return self.data
        # Demo analytics
        return {
            "revenue": round(random.uniform(0, 2400), 2),
            "orders":  random.randint(0, 87),
            "avg_order_value": round(random.uniform(18, 42), 2),
            "top_product": random.choice(WINNING_PRODUCTS)["name"],
            "conversion_rate": f"{random.uniform(1.5, 4.2):.1f}%",
            "snapshot_at": datetime.now().isoformat()
        }

    def dashboard(self) -> str:
        d = self.snapshot()
        rev = f"${d['revenue']:,.2f}"
        orders = str(d['orders'])
        mode = 'LIVE' if LIVE_MODE else 'DEMO'
        lines = [
            f"\n{c(MGB, '╔══ NOVA STORE ANALYTICS ══════════════════════╗')}",
            f"{c(MG, '║')}  Revenue:      {c(GRB, rev)}                        {c(MG,'║')}",
            f"{c(MG, '║')}  Orders:       {c(CY, orders)}                              {c(MG,'║')}",
            f"{c(MG, '║')}  Mode:         {c(YL, mode)}                            {c(MG,'║')}",
            f"{c(MG, '╚══════════════════════════════════════════════╝')}",
        ]
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# AUTONOMOUS STORE MANAGER — orchestrates everything
# ═══════════════════════════════════════════════════════════════════════════════

class AutonomousStoreManager:
    """Nova's autonomous Shopify store — fully self-sustaining."""

    def __init__(self):
        self.shopify   = ShopifyClient()
        self.pricing   = PricingEngine()
        self.sourcing  = ProductSourcing(self.pricing)
        self.seo       = SEOEngine()
        self.promoter  = StorePromoter()
        self.watcher   = OrderWatcher(self.shopify)
        self.analytics = StoreAnalytics(self.shopify)

    def setup_store(self, import_count: int=5) -> List[dict]:
        """Import top winning products into Shopify store."""
        section("IMPORTING WINNING PRODUCTS")
        products = self.sourcing.get_winning_products()[:import_count]
        imported = []

        for p in products:
            name = p["name"]
            price = p["recommended_price"]
            tags = p.get("tags", []) + self.seo.keyword_tags(name, p.get("category",""))
            description = self.seo.product_description(name)

            cost_str   = f"${p['cost_usd']:.2f}"
            price_str  = f"${price:.2f}"
            margin_str = f"{p['pricing']['standard']['margin']*100:.0f}%"
            print(f"\n  {c(CY,'→')} {name[:55]}")
            print(f"     Price: {c(GR, price_str)} | Cost: {c(YL, cost_str)} | Margin: {c(MG, margin_str)}")

            if LIVE_MODE:
                result = self.shopify.create_product(
                    title=name, body_html=description,
                    price=price, tags=tags)
                if "product" in result:
                    ok(f"Created in Shopify: ID {result['product']['id']}")
                    imported.append({**p, "shopify_id": result['product']['id']})
                else:
                    warn(f"Shopify error: {result.get('errors', result)}")
            else:
                ok(f"[DEMO] Would create: '{name}' at ${price:.2f}")
                imported.append(p)

            # DSers instructions
            print(f"     {c(DIM, self.sourcing.dsers_instructions(p).split(chr(10))[1].strip())}")

        self.sourcing.products = imported
        _save(PRODUCTS_DB, imported)
        return imported

    def promote_all(self) -> List[dict]:
        """Generate full multi-platform campaigns for all products."""
        section("GENERATING MULTI-PLATFORM PROMOTION CONTENT")
        products = self.sourcing.get_winning_products()[:3]
        campaigns = []
        for p in products:
            campaign = self.promoter.full_campaign(p)
            campaigns.append(campaign)
            _save(CONTENT_DB, campaigns)

            print(f"\n  {c(MGB, p['name'][:55])}")
            tt = campaign.get("tiktok", {})
            print(f"  {c(YL,'TikTok hook:')}   {str(tt.get('hook',''))[:70]}")
            ig = campaign.get("instagram", {})
            print(f"  {c(MG,'Instagram:')}     {str(ig.get('caption',''))[:70]}")
            tw = campaign.get("twitter", [])
            if tw: print(f"  {c(BL,'Tweet #1:')}      {str(tw[0])[:70]}")
            rd = campaign.get("reddit", {})
            print(f"  {c(GR,'Reddit title:')}  {str(rd.get('title',''))[:70]}")

        return campaigns

    def start_watcher(self):
        """Start background order watcher in a thread."""
        t = threading.Thread(target=self.watcher.watch_loop, args=(120,), daemon=True)
        t.start()
        return t

    def autonomous_cycle(self):
        """Full autonomous store operation cycle."""
        section("NOVA AUTONOMOUS STORE — FULL CYCLE")

        print(f"\n  Store: {c(CY, STORE_URL)}")
        print(f"  Mode:  {c(GRB if LIVE_MODE else YL, 'LIVE' if LIVE_MODE else 'DEMO')}")
        print(f"  AI:    {c(GRB if AI_MODE else YL, 'Groq Active' if AI_MODE else 'Demo fallbacks')}")

        print(self.analytics.dashboard())

        # 1. Import products
        self.setup_store(import_count=3)

        # 2. Generate promotion content
        self.promote_all()

        # 3. Check for orders
        section("ORDER STATUS")
        orders = self.watcher.check_orders()
        if orders:
            for order in orders:
                self.watcher.process_order(order)
        else:
            ok("No pending orders. Watching...")

        section("STATUS SUMMARY")
        ok(f"Store: {STORE_URL}")
        ok(f"Products in pipeline: {len(self.sourcing.get_winning_products())}")
        ok(f"Content campaigns: {len(self.promoter.content_log)}")
        ok(f"Orders processed: {len(self.watcher.orders_log)}")

        if not LIVE_MODE:
            print()
            print(c(YL, "  ─────────────────────────────────────────────"))
            print(c(YL, "  TO GO LIVE — get your Shopify Admin API token:"))
            print(c(WH, "  1. Shopify admin → Settings → Apps & sales channels"))
            print(c(WH, "  2. Click 'Develop apps' → your app → API credentials"))
            print(c(WH, "  3. Click 'Reveal token once' (starts with shpat_)"))
            print(c(WH, "  4. Add to .env: SHOPIFY_TOKEN=shpat_xxxxxxxxxxxx"))
            print(c(WH, "  5. Run: python shopify_nova.py --live"))
            print(c(YL, "  ─────────────────────────────────────────────"))


# ═══════════════════════════════════════════════════════════════════════════════
# NOVA INTEGRATION — callable from nova_asi_v19.py
# ═══════════════════════════════════════════════════════════════════════════════

def get_store_status() -> str:
    """Quick status for Nova's /store command."""
    mgr = AutonomousStoreManager()
    d = mgr.analytics.snapshot()
    return (f"Shopify Store: {STORE_URL}\n"
            f"Mode: {'LIVE' if LIVE_MODE else 'DEMO — needs shpat_ token'}\n"
            f"Revenue: ${d['revenue']:,.2f} | Orders: {d['orders']}\n"
            f"Products catalogued: {len(WINNING_PRODUCTS)}\n"
            f"To activate: add SHOPIFY_TOKEN=shpat_xxx to ~/nexus_agi/.env")

def nova_store_command(arg: str) -> str:
    """Handle /store commands from Nova."""
    mgr = AutonomousStoreManager()
    if arg == 'status':   return get_store_status()
    if arg == 'products': return "\n".join(f"  • {p['name'][:55]} — ${PricingEngine().calculate(p['cost_usd'])['price']:.2f}" for p in WINNING_PRODUCTS[:8])
    if arg == 'promote':
        p = random.choice(WINNING_PRODUCTS)
        tt = mgr.promoter.tiktok_script(p)
        return f"TikTok for {p['name'][:40]}:\nHook: {tt.get('hook','')}\nCTA: {tt.get('cta','')}"
    if arg == 'orders':
        orders = mgr.watcher.check_orders()
        return f"{len(orders)} pending orders." if orders else "No pending orders."
    if arg == 'analytics': return mgr.analytics.dashboard()
    return get_store_status()


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

SETUP_GUIDE = """
  ╔══════════════════════════════════════════════════════════════════╗
  ║         NOVA SHOPIFY STORE — ONE-TIME SETUP GUIDE               ║
  ╠══════════════════════════════════════════════════════════════════╣
  ║                                                                  ║
  ║  Get a Custom App token (shpat_) — takes ~2 minutes.            ║
  ║  Do this from your store Admin — NOT the Partner Dashboard.      ║
  ║                                                                  ║
  ║  STEP 1 — Open your Shopify Admin in a browser:                 ║
  ║    https://vqtkit-4a.myshopify.com/admin                        ║
  ║                                                                  ║
  ║  STEP 2 — Navigate to:                                          ║
  ║    Settings (bottom-left gear icon)                              ║
  ║    → Apps and sales channels                                     ║
  ║    → Develop apps                                                ║
  ║    → Allow custom app development  (if prompted)                 ║
  ║                                                                  ║
  ║  STEP 3 — Create the app:                                       ║
  ║    Click "Create an app"                                         ║
  ║    Name it: Nova Manager                                         ║
  ║    Click "Create app"                                            ║
  ║                                                                  ║
  ║  STEP 4 — Set API scopes:                                       ║
  ║    Click "Configure Admin API scopes"                            ║
  ║    Check ALL of these:                                           ║
  ║      read_products      write_products                           ║
  ║      read_orders        write_orders                             ║
  ║      read_inventory     write_inventory                          ║
  ║      read_fulfillments  write_fulfillments                       ║
  ║    Click "Save"                                                  ║
  ║                                                                  ║
  ║  STEP 5 — Install the app:                                      ║
  ║    Click the "API credentials" tab                               ║
  ║    Click "Install app" → "Install"                               ║
  ║                                                                  ║
  ║  STEP 6 — Copy your token (YOU ONLY SEE IT ONCE!):             ║
  ║    Under "Admin API access token" click "Reveal token once"      ║
  ║    Copy the full token — it starts with  shpat_                 ║
  ║                                                                  ║
  ║  STEP 7 — Save it in Termux (paste your token):                 ║
  ║    echo 'SHOPIFY_TOKEN=shpat_PASTE_YOUR_TOKEN_HERE' \\          ║
  ║         >> ~/nexus_agi/.env                                      ║
  ║                                                                  ║
  ║  STEP 8 — Load and launch:                                      ║
  ║    source ~/nexus_agi/.env                                       ║
  ║    python shopify_nova.py --live --products                      ║
  ║                                                                  ║
  ║  Or use the quick-save command:                                  ║
  ║    python shopify_nova.py --save-token shpat_YOURTOKEN           ║
  ║                                                                  ║
  ╚══════════════════════════════════════════════════════════════════╝
"""

def main():
    parser = argparse.ArgumentParser(description='Nova Autonomous Shopify Store')
    parser.add_argument('--live',     action='store_true', help='Live mode (needs SHOPIFY_TOKEN)')
    parser.add_argument('--content',  action='store_true', help='Generate promotion content only')
    parser.add_argument('--products', action='store_true', help='Import products only')
    parser.add_argument('--watch',    action='store_true', help='Run order watcher loop')
    parser.add_argument('--setup',      action='store_true', help='Show step-by-step Shopify token setup guide')
    parser.add_argument('--save-token', type=str, metavar='shpat_TOKEN',
                        help='Save your shpat_ token to .env and test it')
    parser.add_argument('--campaign', type=str, help='Generate campaign for specific product name')
    args = parser.parse_args()

    print(f"\n{c(MGB,'═'*70)}")
    print(c(MGB, "  NOVA AUTONOMOUS SHOPIFY STORE MANAGER".center(70)))
    print(c(DIM,  f"  {STORE_URL}".center(70)))
    print(c(MGB,'═'*70))

    if args.setup:
        print(SETUP_GUIDE)
        return

    if args.save_token:
        tok = args.save_token.strip()
        if not tok.startswith("shpat_"):
            err("Token must start with shpat_ — copy it from the Custom App page.")
            return
        env_path = BASE_DIR / ".env"
        lines = env_path.read_text().splitlines() if env_path.exists() else []
        lines = [l for l in lines if not l.startswith("SHOPIFY_TOKEN=")]
        lines.append(f"SHOPIFY_TOKEN={tok}")
        env_path.write_text("\n".join(lines) + "\n")
        ok(f"Token saved to {env_path}")
        # Quick sanity-check
        if REQUESTS_AVAILABLE:
            try:
                r = __import__('requests').get(
                    f"https://{STORE_URL}/admin/api/{API_VERSION}/shop.json",
                    headers={"X-Shopify-Access-Token": tok}, timeout=10)
                if r.status_code == 200:
                    shop = r.json().get("shop", {})
                    ok(f"Connected! Store: {shop.get('name','?')} ({shop.get('domain','?')})")
                    ok("Run:  python shopify_nova.py --live --products")
                else:
                    warn(f"Token saved but API returned {r.status_code} — double-check it.")
            except Exception as e:
                warn(f"Token saved. Could not verify online: {e}")
        return

    mgr = AutonomousStoreManager()

    if args.watch:
        mgr.watcher.watch_loop(interval_seconds=60)
        return

    if args.products:
        mgr.setup_store(import_count=len(WINNING_PRODUCTS))
        return

    if args.content:
        mgr.promote_all()
        return

    if args.campaign:
        matching = [p for p in WINNING_PRODUCTS if args.campaign.lower() in p["name"].lower()]
        product = matching[0] if matching else WINNING_PRODUCTS[0]
        campaign = mgr.promoter.full_campaign(product)
        print(json.dumps(campaign, indent=2))
        return

    # Default: full autonomous cycle
    mgr.autonomous_cycle()

    print(f"\n{c(MGB,'═'*70)}")
    print(c(GRB, "  Nova Store Manager — cycle complete".center(70)))
    print(c(MGB,'═'*70)+"\n")


if __name__ == "__main__":
    main()
