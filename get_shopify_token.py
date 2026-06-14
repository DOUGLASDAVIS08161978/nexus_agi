#!/usr/bin/env python3
"""
Shopify Admin API token generator — client credentials flow.
Usage:
  export SHOPIFY_CLIENT_ID=your_client_id
  export SHOPIFY_SECRET=your_client_secret
  python get_shopify_token.py
"""
import os, json
from pathlib import Path

try:
    import requests
    HAS_REQ = True
except ImportError:
    HAS_REQ = False

STORE      = "nova-automation.myshopify.com"
CLIENT_ID  = os.getenv("SHOPIFY_CLIENT_ID", "")
CLIENT_SEC = os.getenv("SHOPIFY_SECRET",    "")

print("\n" + "═"*65)
print("  SHOPIFY TOKEN GENERATOR  (client credentials flow)")
print("═"*65)

if not CLIENT_ID or not CLIENT_SEC:
    print("\n  ✗  Missing credentials. Set these env vars first:\n")
    print("     export SHOPIFY_CLIENT_ID=your_client_id")
    print("     export SHOPIFY_SECRET=your_client_secret\n")
    raise SystemExit(1)

if not HAS_REQ:
    print("\n  ✗  'requests' not installed.  Run: pip install requests\n")
    raise SystemExit(1)

print(f"\n  Store     : {STORE}")
print(f"  Client ID : {CLIENT_ID[:8]}{'*'*8}")
print(f"\n  Requesting token...\n")

url  = f"https://{STORE}/admin/oauth/access_token"
data = {"client_id": CLIENT_ID, "client_secret": CLIENT_SEC,
        "grant_type": "client_credentials"}

try:
    # Try JSON body first
    resp = requests.post(url, json=data,
                         headers={"Content-Type": "application/json"}, timeout=15)
    result = resp.json()
    token = result.get("access_token", "")

    if not token:
        # Fallback: form-encoded body
        resp2  = requests.post(url, data=data, timeout=15)
        result = resp2.json()
        token  = result.get("access_token", "")

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
        print(f"  ✗  Failed. Shopify response:\n  {result}\n")
        print("  Check that your Client ID and Secret match the nova-automation app")
        print("  in Shopify Admin → Settings → Apps → nova-automation → API credentials\n")

except Exception as e:
    print(f"  ✗  Request error: {e}\n")
