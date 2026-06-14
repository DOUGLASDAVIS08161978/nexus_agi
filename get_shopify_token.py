#!/usr/bin/env python3
"""
Shopify Partner App — OAuth authorization-code token generator.
Usage:
  export SHOPIFY_CLIENT_ID=your_client_id
  export SHOPIFY_SECRET=your_client_secret
  python get_shopify_token.py
"""
import os, json, secrets, urllib.parse
from pathlib import Path

try:
    import requests
    HAS_REQ = True
except ImportError:
    HAS_REQ = False

STORE      = "nova-automation.myshopify.com"
CLIENT_ID  = os.getenv("SHOPIFY_CLIENT_ID", "")
CLIENT_SEC = os.getenv("SHOPIFY_SECRET",    "")

# Must match an Allowed redirect URL in your Shopify Partner Dashboard app settings.
# Go to: partners.shopify.com → Apps → nova-automation → App setup → Allowed redirect URLs
# Add exactly:  https://nova-automation.myshopify.com
REDIRECT   = "https://nova-automation.myshopify.com"

SCOPES = ",".join([
    "read_products","write_products",
    "read_orders","write_orders",
    "read_inventory","write_inventory",
    "read_fulfillments","write_fulfillments",
])

print("\n" + "═"*65)
print("  SHOPIFY OAUTH TOKEN GENERATOR")
print("═"*65)

if not CLIENT_ID or not CLIENT_SEC:
    print("\n  ✗  Missing credentials. Set these first:\n")
    print("     export SHOPIFY_CLIENT_ID=your_client_id")
    print("     export SHOPIFY_SECRET=your_client_secret\n")
    raise SystemExit(1)

if not HAS_REQ:
    print("\n  ✗  'requests' not installed:  pip install requests\n")
    raise SystemExit(1)

state = secrets.token_hex(8)

auth_url = (
    f"https://{STORE}/admin/oauth/authorize"
    f"?client_id={CLIENT_ID}"
    f"&scope={urllib.parse.quote(SCOPES)}"
    f"&redirect_uri={urllib.parse.quote(REDIRECT)}"
    f"&state={state}"
)

print(f"""
  STEP 1 — Make sure your Partner Dashboard has the redirect URL set:
  ─────────────────────────────────────────────────────────────────
  • Go to:  https://partners.shopify.com
  • Apps → nova-automation → App setup
  • Under "Allowed redirection URL(s)" add:
      {REDIRECT}
  • Save

  STEP 2 — Open this URL in your phone browser:
  ─────────────────────────────────────────────
  {auth_url}

  STEP 3 — Click "Install app" in Shopify.
  You'll be redirected to:  {REDIRECT}?code=XXXX&state=...
  The page may show an error — that's OK.
  Copy the FULL URL from your browser address bar and paste it below.
""")

full_url = input("  Paste the full redirect URL here: ").strip()

# Extract code from URL
try:
    parsed = urllib.parse.urlparse(full_url)
    params = urllib.parse.parse_qs(parsed.query)
    code   = params.get("code", [""])[0]
    got_state = params.get("state", [""])[0]
except Exception:
    code = ""
    got_state = ""

if not code:
    print("\n  ✗  Could not find 'code=' in that URL. Make sure you copied the full URL.\n")
    raise SystemExit(1)

print(f"\n  Code found: {code[:12]}...  Exchanging for token...\n")

try:
    resp = requests.post(
        f"https://{STORE}/admin/oauth/access_token",
        json={"client_id": CLIENT_ID, "client_secret": CLIENT_SEC, "code": code},
        headers={"Content-Type": "application/json"}, timeout=15)

    data  = resp.json()
    token = data.get("access_token", "")

    if token:
        print(f"  ✓  Token obtained!\n")
        print(f"  {token}\n")
        env_path = Path.home() / "nexus_agi" / ".env"
        lines    = env_path.read_text().splitlines() if env_path.exists() else []
        lines    = [l for l in lines if not l.startswith("SHOPIFY_TOKEN=")]
        lines.append(f"SHOPIFY_TOKEN={token}")
        env_path.write_text("\n".join(lines) + "\n")
        print(f"  ✓  Saved to {env_path}")
        print(f"\n  Now run:  python shopify_nova.py --live\n")
    else:
        print(f"\n  ✗  Token exchange failed.")
        print(f"  Shopify says: {data}\n")
        if "invalid_request" in str(data):
            print("  Tip: The code can only be used once. Re-run this script to get a fresh URL.\n")

except Exception as e:
    print(f"\n  ✗  Request error: {e}\n")
