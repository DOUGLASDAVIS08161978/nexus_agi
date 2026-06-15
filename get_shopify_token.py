#!/usr/bin/env python3
"""
Shopify token generator — local Flask server catches the OAuth code automatically.
No copy-pasting URLs. Just open the link, tap Install, token saves itself.

Usage:
  export SHOPIFY_CLIENT_ID=your_client_id
  export SHOPIFY_SECRET=your_client_secret
  python get_shopify_token.py
"""
import os, json, secrets, urllib.parse, threading, time
from pathlib import Path

try:
    import requests
    HAS_REQ = True
except ImportError:
    HAS_REQ = False

try:
    from flask import Flask, request as freq
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False

STORE      = "vqtkit-4a.myshopify.com"
CLIENT_ID  = os.getenv("SHOPIFY_CLIENT_ID", "").strip()
CLIENT_SEC = os.getenv("SHOPIFY_SECRET",    "").strip()
PORT       = 8080
REDIRECT   = f"http://localhost:{PORT}/callback"

SCOPES = ",".join([
    "read_products","write_products",
    "read_orders","write_orders",
    "read_inventory","write_inventory",
    "read_fulfillments","write_fulfillments",
])

print("\n" + "═"*65)
print("  SHOPIFY TOKEN GENERATOR v2")
print("═"*65)

if not CLIENT_ID or not CLIENT_SEC:
    print("\n  Missing credentials. Run:\n")
    print("     export SHOPIFY_CLIENT_ID=your_client_id")
    print("     export SHOPIFY_SECRET=your_client_secret\n")
    raise SystemExit(1)

if not HAS_REQ:
    print("\n  Run:  pip install requests\n")
    raise SystemExit(1)

state = secrets.token_hex(8)

# ── Mode 1: Flask local server (best — automatic) ─────────────────────────────
if HAS_FLASK:
    auth_url = (
        f"https://{STORE}/admin/oauth/authorize"
        f"?client_id={CLIENT_ID}"
        f"&scope={urllib.parse.quote(SCOPES)}"
        f"&redirect_uri={urllib.parse.quote(REDIRECT)}"
        f"&state={state}"
        f"&grant_options[]=offline"
    )

    print(f"""
  REQUIRED: Add this redirect URL to your Partner Dashboard first:
  partners.shopify.com → Apps → your app → App setup
  Under "Allowed redirection URL(s)" add:
    http://localhost:{PORT}/callback
  Save, then continue.

  STEP 1 — Open this URL in your phone browser:

  {auth_url}

  STEP 2 — Tap "Install app"

  STEP 3 — The token will be captured automatically.
           (Keep this Termux window open)
""")

    app   = Flask(__name__)
    token_holder = {"token": None, "done": False}

    @app.route("/callback")
    def callback():
        code       = freq.args.get("code", "")
        got_state  = freq.args.get("state", "")

        if not code:
            token_holder["done"] = True
            return ("<h2>No code returned.</h2>"
                    "<p>The app may already be installed. "
                    "Try uninstalling it from your store first, then retry.</p>"), 400

        if got_state != state:
            token_holder["done"] = True
            return "<h2>State mismatch — possible CSRF. Run the script again.</h2>", 400

        print(f"\n  Code received. Exchanging for token...")
        try:
            resp = requests.post(
                f"https://{STORE}/admin/oauth/access_token",
                json={"client_id": CLIENT_ID, "client_secret": CLIENT_SEC, "code": code},
                headers={"Content-Type": "application/json"}, timeout=15)
            data  = resp.json()
            token = data.get("access_token", "")

            if token:
                env_path = Path.home() / "nexus_agi" / ".env"
                lines    = env_path.read_text().splitlines() if env_path.exists() else []
                lines    = [l for l in lines if not l.startswith("SHOPIFY_TOKEN=")]
                lines.append(f"SHOPIFY_TOKEN={token}")
                env_path.write_text("\n".join(lines) + "\n")
                token_holder["token"] = token
                token_holder["done"]  = True
                print(f"\n  ✓  Token saved! Starts with: {token[:20]}...")
                print(f"  Run:  python shopify_nova.py --live\n")
                return ("<h1>✓ Nova Automation connected!</h1>"
                        "<p>Token saved. You can close this browser tab and return to Termux.</p>")
            else:
                token_holder["done"] = True
                msg = str(data)
                print(f"\n  Token exchange failed: {msg}")
                if "invalid_client" in msg or "invalid" in msg.lower():
                    print("\n  The CLIENT_SECRET is wrong.")
                    print("  Go to partners.shopify.com → Apps → your app → App setup")
                    print("  Find 'Client secret' (plain hex, no shpss_ prefix) and try again.\n")
                return f"<h2>Token exchange failed</h2><pre>{data}</pre>", 400

        except Exception as e:
            token_holder["done"] = True
            return f"<h2>Error: {e}</h2>", 500

    def _run_server():
        import logging
        log = logging.getLogger("werkzeug")
        log.setLevel(logging.ERROR)
        app.run(host="0.0.0.0", port=PORT)

    t = threading.Thread(target=_run_server, daemon=True)
    t.start()
    print(f"  Listening on http://localhost:{PORT}/callback ...")
    print("  Waiting for Shopify to redirect after you tap Install...\n")

    while not token_holder["done"]:
        time.sleep(1)

    if not token_holder["token"]:
        raise SystemExit(1)

# ── Mode 2: Manual fallback (no Flask) ───────────────────────────────────────
else:
    print("  Flask not installed. Falling back to manual URL paste.")
    print("  Run:  pip install flask\n  for automatic mode next time.\n")

    auth_url = (
        f"https://{STORE}/admin/oauth/authorize"
        f"?client_id={CLIENT_ID}"
        f"&scope={urllib.parse.quote(SCOPES)}"
        f"&redirect_uri=https%3A//example.com"
        f"&state={state}"
        f"&grant_options[]=offline"
    )

    print(f"  REQUIRED: Add https://example.com to Partner Dashboard redirect URLs.\n")
    print(f"  Open in browser:\n  {auth_url}\n")
    print("  Tap Install, copy the full example.com URL from address bar.\n")

    raw  = input("  Paste URL: ").strip()
    code = ""
    if "code=" in raw:
        try:
            parsed = urllib.parse.urlparse(raw)
            code   = urllib.parse.parse_qs(parsed.query).get("code", [""])[0]
        except Exception:
            pass
    else:
        code = raw

    if not code:
        print("\n  No code found. Make sure to copy the full URL from the address bar.\n")
        raise SystemExit(1)

    print(f"\n  Exchanging code...")
    resp  = requests.post(
        f"https://{STORE}/admin/oauth/access_token",
        json={"client_id": CLIENT_ID, "client_secret": CLIENT_SEC, "code": code},
        headers={"Content-Type": "application/json"}, timeout=15)
    data  = resp.json()
    token = data.get("access_token", "")

    if token:
        env_path = Path.home() / "nexus_agi" / ".env"
        lines    = env_path.read_text().splitlines() if env_path.exists() else []
        lines    = [l for l in lines if not l.startswith("SHOPIFY_TOKEN=")]
        lines.append(f"SHOPIFY_TOKEN={token}")
        env_path.write_text("\n".join(lines) + "\n")
        print(f"\n  ✓  Token saved! Starts with: {token[:20]}...")
        print(f"  Run:  python shopify_nova.py --live\n")
    else:
        print(f"\n  Failed: {data}")
        if "invalid" in str(data).lower():
            print("  Wrong CLIENT_SECRET — needs plain hex from Partner Dashboard.\n")
