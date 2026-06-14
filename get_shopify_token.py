#!/usr/bin/env python3
"""
One-time Shopify OAuth token generator.
Run this, visit the URL it prints, paste back the callback URL.
"""
import os, json, hashlib, secrets, urllib.parse
try:
    import requests
    HAS_REQ = True
except ImportError:
    HAS_REQ = False
try:
    from flask import Flask, request, redirect
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False

STORE       = "nova-automation.myshopify.com"
CLIENT_ID   = os.getenv("SHOPIFY_CLIENT_ID", "")
CLIENT_SEC  = os.getenv("SHOPIFY_SECRET",    "")
REDIRECT    = "https://nova-automation.myshopify.com/admin"  # placeholder

SCOPES = ",".join([
    "read_products","write_products",
    "read_orders","write_orders",
    "read_inventory","write_inventory",
    "read_fulfillments","write_fulfillments",
])

state = secrets.token_hex(8)

auth_url = (
    f"https://{STORE}/admin/oauth/authorize"
    f"?client_id={CLIENT_ID}"
    f"&scope={urllib.parse.quote(SCOPES)}"
    f"&redirect_uri={urllib.parse.quote(REDIRECT)}"
    f"&state={state}"
)

print("\n" + "═"*70)
print("  SHOPIFY TOKEN GENERATOR")
print("═"*70)
print(f"\n  Client ID : {CLIENT_ID}")
print(f"  Store     : {STORE}")
print(f"\n  1. Visit this URL in your browser:")
print(f"\n     {auth_url}\n")
print(f"  2. Click 'Install app' in Shopify")
print(f"  3. After install, you'll be redirected — copy the 'code=' value from the URL")
print(f"  4. Paste the code below\n")

if HAS_REQ:
    code = input("  Paste the code= value from the redirect URL: ").strip()
    if code:
        resp = requests.post(
            f"https://{STORE}/admin/oauth/access_token",
            json={"client_id": CLIENT_ID, "client_secret": CLIENT_SEC, "code": code},
            headers={"Content-Type": "application/json"}, timeout=15)
        data = resp.json()
        token = data.get("access_token", "")
        if token:
            print(f"\n  ✓ SUCCESS! Your token:\n\n  {token}\n")
            env_path = os.path.expanduser("~/nexus_agi/.env")
            with open(env_path, "a") as f:
                f.write(f"\nSHOPIFY_TOKEN={token}\n")
            print(f"  ✓ Saved to {env_path}")
            print(f"  ✓ Now run: python shopify_nova.py --live")
        else:
            print(f"\n  Error: {data}")
else:
    print("  (Install 'requests' to auto-exchange: pip install requests)")
