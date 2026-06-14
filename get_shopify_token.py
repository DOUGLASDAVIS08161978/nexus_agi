#!/usr/bin/env python3
"""
Shopify OAuth token generator — local callback server method.
Starts a tiny HTTP server in Termux, catches the code automatically.

Usage:
  export SHOPIFY_CLIENT_ID=your_client_id
  export SHOPIFY_SECRET=your_client_secret
  python get_shopify_token.py
"""
import os, json, secrets, urllib.parse, threading, time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

try:
    import requests
    HAS_REQ = True
except ImportError:
    HAS_REQ = False

STORE      = "nova-automation.myshopify.com"
CLIENT_ID  = os.getenv("SHOPIFY_CLIENT_ID", "")
CLIENT_SEC = os.getenv("SHOPIFY_SECRET",    "")
PORT       = 8080
REDIRECT   = f"http://localhost:{PORT}/callback"

SCOPES = ",".join([
    "read_products","write_products",
    "read_orders","write_orders",
    "read_inventory","write_inventory",
    "read_fulfillments","write_fulfillments",
])

# Shared state between server thread and main thread
_result = {"code": None, "state": None, "error": None}
_server_done = threading.Event()

print("\n" + "═"*65)
print("  SHOPIFY TOKEN GENERATOR  (auto-capture mode)")
print("═"*65)

if not CLIENT_ID or not CLIENT_SEC:
    print("\n  ✗  Missing credentials. Set these first:\n")
    print("     export SHOPIFY_CLIENT_ID=your_client_id")
    print("     export SHOPIFY_SECRET=your_client_secret\n")
    raise SystemExit(1)

if not HAS_REQ:
    print("\n  ✗  Run this first:  pip install requests\n")
    raise SystemExit(1)

state = secrets.token_hex(8)

auth_url = (
    f"https://{STORE}/admin/oauth/authorize"
    f"?client_id={CLIENT_ID}"
    f"&scope={urllib.parse.quote(SCOPES)}"
    f"&redirect_uri={urllib.parse.quote(REDIRECT)}"
    f"&state={state}"
    f"&grant_options[]=offline"
)


class CallbackHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # silence server logs

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)
        code   = params.get("code",  [""])[0]
        error  = params.get("error", [""])[0]

        if code:
            _result["code"]  = code
            _result["state"] = params.get("state", [""])[0]
            body = b"<h2>Got it! Token is being generated. You can close this tab.</h2>"
        else:
            _result["error"] = error or "No code in callback"
            body = f"<h2>Error: {error}</h2>".encode()

        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(body)
        _server_done.set()


# Start local server
httpd = HTTPServer(("127.0.0.1", PORT), CallbackHandler)
server_thread = threading.Thread(target=httpd.handle_request, daemon=True)
server_thread.start()

print(f"""
  Listening on http://localhost:{PORT}/callback

  ┌─ IMPORTANT (one-time setup) ────────────────────────────────┐
  │  In Shopify Partner Dashboard:                               │
  │  Apps → nova-automation → App setup                         │
  │  Add to "Allowed redirection URL(s)":                       │
  │    http://localhost:{PORT}/callback                            │
  │  Save before continuing.                                     │
  └──────────────────────────────────────────────────────────────┘

  Open this URL in your phone browser:

  {auth_url}

  After tapping Install, your browser will show "Got it!" and
  the token will be saved automatically.

  Waiting for Shopify callback...
""")

# Wait up to 5 minutes for the callback
_server_done.wait(timeout=300)
httpd.server_close()

if _result["error"]:
    print(f"  ✗  OAuth error: {_result['error']}\n")
    raise SystemExit(1)

if not _result["code"]:
    print("  ✗  Timed out — no callback received within 5 minutes.\n")
    raise SystemExit(1)

print(f"  ✓  Callback received. Exchanging code for token...\n")

try:
    resp  = requests.post(
        f"https://{STORE}/admin/oauth/access_token",
        json={"client_id": CLIENT_ID, "client_secret": CLIENT_SEC,
              "code": _result["code"]},
        headers={"Content-Type": "application/json"}, timeout=15)
    data  = resp.json()
    token = data.get("access_token", "")

    if token:
        print(f"  ✓  Token obtained:  {token[:12]}...\n")
        env_path = Path.home() / "nexus_agi" / ".env"
        lines    = env_path.read_text().splitlines() if env_path.exists() else []
        lines    = [l for l in lines if not l.startswith("SHOPIFY_TOKEN=")]
        lines.append(f"SHOPIFY_TOKEN={token}")
        env_path.write_text("\n".join(lines) + "\n")
        print(f"  ✓  Saved to {env_path}")
        print(f"\n  Run:  python shopify_nova.py --live\n")
    else:
        print(f"  ✗  Token exchange failed: {data}\n")
        if "invalid_request" in str(data) or "used_token" in str(data):
            print("  The code was already used. Run this script again for a fresh code.\n")

except Exception as e:
    print(f"  ✗  Request error: {e}\n")
