#!/usr/bin/env python3
"""
Shopify token generator — uses example.com as redirect (no server needed).

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
REDIRECT   = "https://example.com"

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
    print("\n  Missing credentials:\n")
    print("     export SHOPIFY_CLIENT_ID=your_client_id")
    print("     export SHOPIFY_SECRET=your_client_secret\n")
    raise SystemExit(1)

if not HAS_REQ:
    print("\n  Run:  pip install requests\n")
    raise SystemExit(1)

state    = secrets.token_hex(8)
auth_url = (
    f"https://{STORE}/admin/oauth/authorize"
    f"?client_id={CLIENT_ID}"
    f"&scope={urllib.parse.quote(SCOPES)}"
    f"&redirect_uri={urllib.parse.quote(REDIRECT)}"
    f"&state={state}"
    f"&grant_options[]=offline"
)

print(f"""
  ══ REQUIRED: Add this to Partner Dashboard first ══════════════
  partners.shopify.com → Apps → nova-automation → App setup
  Under "Allowed redirection URL(s)" add:  https://example.com
  Save.
  ═══════════════════════════════════════════════════════════════

  STEP 1 — Open this URL in your phone browser:

  {auth_url}

  STEP 2 — Tap "Install app" on the Shopify page.

  STEP 3 — Your browser will go to example.com.
           The address bar will look like:
           https://example.com/?code=XXXXXXXXXX&state=...
           Copy the ENTIRE URL from the address bar.

  STEP 4 — Paste it below.
""")

raw = input("  Paste the full example.com URL from your address bar: ").strip()

# Extract code whether they paste full URL or just the code
code = ""
if "code=" in raw:
    try:
        parsed = urllib.parse.urlparse(raw)
        code   = urllib.parse.parse_qs(parsed.query).get("code", [""])[0]
    except Exception:
        code = ""
    if not code:
        # Try splitting manually
        for part in raw.split("&"):
            if part.startswith("code=") or "?code=" in part:
                code = part.split("code=")[-1].split("&")[0]
                break
else:
    code = raw  # They pasted just the code value

if not code:
    print("\n  Could not find code= in that URL.")
    print("  Make sure to copy from the ADDRESS BAR after example.com loads.\n")
    raise SystemExit(1)

print(f"\n  Code found: {code[:16]}...  Getting token...")

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
        print(f"\n  ✓  Token obtained and saved!")
        print(f"  Token starts with: {token[:20]}...")
        print(f"\n  Run:  python shopify_nova.py --live\n")
    else:
        print(f"\n  Failed: {data}")
        if "invalid" in str(data).lower():
            print("  Code may have expired (they last ~10 min).")
            print("  Run this script again to get a fresh auth URL.\n")

except Exception as e:
    print(f"\n  Error: {e}\n")
