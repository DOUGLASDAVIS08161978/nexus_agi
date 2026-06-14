#!/usr/bin/env python3
"""
Shopify token generator — webhook.site redirect (no localhost needed).

Usage:
  export SHOPIFY_CLIENT_ID=your_api_key
  export SHOPIFY_SECRET=your_api_secret_key
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

SCOPES = ",".join([
    "read_products","write_products",
    "read_orders","write_orders",
    "read_inventory","write_inventory",
    "read_fulfillments","write_fulfillments",
])

print("\n" + "═"*65)
print("  SHOPIFY TOKEN GENERATOR")
print("═"*65)

if not CLIENT_ID or not CLIENT_SEC:
    print("\n  ✗  Missing credentials:\n")
    print("     export SHOPIFY_CLIENT_ID=your_client_id")
    print("     export SHOPIFY_SECRET=your_client_secret\n")
    raise SystemExit(1)

if not HAS_REQ:
    print("\n  ✗  Run:  pip install requests\n")
    raise SystemExit(1)

print(f"""
  ┌─ BEFORE YOU CONTINUE ───────────────────────────────────────┐
  │  1. Open https://webhook.site in your phone browser          │
  │  2. Copy the unique URL it gives you                         │
  │     (like https://webhook.site/abc-123-def...)               │
  │  3. In partners.shopify.com → Apps → nova-automation         │
  │     → App setup → Allowed redirection URL(s)                 │
  │     Paste that webhook.site URL → Save                       │
  └──────────────────────────────────────────────────────────────┘
""")

redirect = input("  Paste your webhook.site URL here: ").strip()
if not redirect.startswith("http"):
    print("  ✗  That doesn't look like a URL. Try again.\n")
    raise SystemExit(1)

state    = secrets.token_hex(8)
auth_url = (
    f"https://{STORE}/admin/oauth/authorize"
    f"?client_id={CLIENT_ID}"
    f"&scope={urllib.parse.quote(SCOPES)}"
    f"&redirect_uri={urllib.parse.quote(redirect)}"
    f"&state={state}"
    f"&grant_options[]=offline"
)

print(f"""
  ✓  Redirect URI set to: {redirect}

  Now open this URL in your browser (while logged into Shopify admin):

  {auth_url}

  Tap "Install app" → your browser will redirect to webhook.site.
  On the webhook.site page you'll see a request come in.
  Look for the "code" value in the URL or in the request details.
""")

code = input("  Paste the code= value from webhook.site: ").strip()
# Handle if they paste the full URL instead of just the code
if "code=" in code:
    code = urllib.parse.parse_qs(urllib.parse.urlparse(code).query).get("code", [""])[0]

if not code:
    print("  ✗  No code found. Make sure to copy the value after 'code=' in the URL.\n")
    raise SystemExit(1)

print(f"\n  Exchanging code for token...")

try:
    resp  = requests.post(
        f"https://{STORE}/admin/oauth/access_token",
        json={"client_id": CLIENT_ID, "client_secret": CLIENT_SEC, "code": code},
        headers={"Content-Type": "application/json"}, timeout=15)

    print(f"  Status: {resp.status_code}")

    data  = resp.json()
    token = data.get("access_token", "")

    if token:
        env_path = Path.home() / "nexus_agi" / ".env"
        lines    = env_path.read_text().splitlines() if env_path.exists() else []
        lines    = [l for l in lines if not l.startswith("SHOPIFY_TOKEN=")]
        lines.append(f"SHOPIFY_TOKEN={token}")
        env_path.write_text("\n".join(lines) + "\n")
        print(f"\n  ✓  Token: {token[:16]}...")
        print(f"  ✓  Saved to {env_path}")
        print(f"\n  Run:  python shopify_nova.py --live\n")
    else:
        print(f"\n  ✗  Token exchange failed: {data}")
        if "invalid_request" in str(data):
            print("  The code may have expired. Run this script again.\n")

except Exception as e:
    print(f"\n  ✗  Error: {e}\n")
