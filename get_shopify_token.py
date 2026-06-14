#!/usr/bin/env python3
"""
Shopify token generator — tries client credentials first, then OAuth fallback.
Works for both custom apps (Admin-created) and Partner apps.

Usage:
  export SHOPIFY_CLIENT_ID=your_api_key
  export SHOPIFY_SECRET=your_api_secret_key
  python get_shopify_token.py
"""
import os, json, secrets, urllib.parse, threading
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
    print("\n  ✗  Missing credentials. Set these first:\n")
    print("     export SHOPIFY_CLIENT_ID=your_api_key")
    print("     export SHOPIFY_SECRET=your_api_secret_key\n")
    raise SystemExit(1)

if not HAS_REQ:
    print("\n  ✗  Run:  pip install requests\n")
    raise SystemExit(1)

print(f"\n  Store     : {STORE}")
print(f"  Client ID : {CLIENT_ID[:8]}{'*'*8}\n")


def save_token(token: str):
    env_path = Path.home() / "nexus_agi" / ".env"
    lines    = env_path.read_text().splitlines() if env_path.exists() else []
    lines    = [l for l in lines if not l.startswith("SHOPIFY_TOKEN=")]
    lines.append(f"SHOPIFY_TOKEN={token}")
    env_path.write_text("\n".join(lines) + "\n")
    print(f"\n  ✓  Token: {token[:16]}...")
    print(f"  ✓  Saved to {env_path}")
    print(f"\n  Run:  python shopify_nova.py --live\n")


# ── METHOD 1: Client Credentials (works for Admin custom apps) ────────────────
print("  Trying client credentials flow...")
url = f"https://{STORE}/admin/oauth/access_token"

for ct, body in [
    ("application/json",                  {"client_id": CLIENT_ID, "client_secret": CLIENT_SEC, "grant_type": "client_credentials"}),
    ("application/x-www-form-urlencoded", {"client_id": CLIENT_ID, "client_secret": CLIENT_SEC, "grant_type": "client_credentials"}),
]:
    try:
        if ct == "application/json":
            resp = requests.post(url, json=body, headers={"Content-Type": ct}, timeout=15)
        else:
            resp = requests.post(url, data=body, headers={"Content-Type": ct}, timeout=15)

        print(f"    Status: {resp.status_code}  Content-Type: {resp.headers.get('content-type','?')[:40]}")

        try:
            data  = resp.json()
            token = data.get("access_token", "")
            if token:
                print("  ✓  Client credentials succeeded!")
                save_token(token)
                raise SystemExit(0)
            else:
                print(f"    Response: {json.dumps(data)[:120]}")
        except (ValueError, KeyError):
            print(f"    Raw response: {resp.text[:120]}")
    except SystemExit:
        raise
    except Exception as e:
        print(f"    Error: {e}")

print("\n  Client credentials not supported for this app type.")
print("  Falling back to OAuth authorization-code flow...\n")


# ── METHOD 2: OAuth Authorization Code (works for Partner apps) ───────────────
_result = {"code": None, "error": None}
_done   = threading.Event()


class CallbackHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args): pass
    def do_GET(self):
        params = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        _result["code"]  = params.get("code",  [""])[0]
        _result["error"] = params.get("error", [""])[0]
        body = (b"<h2>Got it! Return to Termux.</h2>"
                if _result["code"] else
                f"<h2>Error: {_result['error']}</h2>".encode())
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(body)
        _done.set()


REDIRECT = f"http://localhost:{PORT}/callback"
state    = secrets.token_hex(8)
auth_url = (
    f"https://{STORE}/admin/oauth/authorize"
    f"?client_id={CLIENT_ID}"
    f"&scope={urllib.parse.quote(SCOPES)}"
    f"&redirect_uri={urllib.parse.quote(REDIRECT)}"
    f"&state={state}"
    f"&grant_options[]=offline"
)

try:
    httpd = HTTPServer(("127.0.0.1", PORT), CallbackHandler)
except OSError:
    print(f"  ✗  Port {PORT} is busy. Kill the old process and retry:\n")
    print(f"     pkill -f get_shopify_token\n")
    raise SystemExit(1)

threading.Thread(target=httpd.handle_request, daemon=True).start()

print(f"""  ┌─ ONE-TIME PARTNER DASHBOARD SETUP ──────────────────────────┐
  │  partners.shopify.com → Apps → nova-automation → App setup   │
  │  Allowed redirection URL(s): http://localhost:{PORT}/callback    │
  │  Save, then continue below.                                   │
  └──────────────────────────────────────────────────────────────┘

  Open this URL in your phone browser (while logged into Admin):

  {auth_url}

  Tap Install → browser shows "Got it!" → token saved automatically.
  Waiting (5 min timeout)...
""")

_done.wait(timeout=300)
httpd.server_close()

if not _result["code"]:
    msg = _result["error"] or "Timed out — no callback received"
    print(f"  ✗  {msg}\n")
    raise SystemExit(1)

print("  ✓  Callback received. Exchanging code...")
try:
    resp  = requests.post(url,
        json={"client_id": CLIENT_ID, "client_secret": CLIENT_SEC, "code": _result["code"]},
        headers={"Content-Type": "application/json"}, timeout=15)
    data  = resp.json()
    token = data.get("access_token", "")
    if token:
        save_token(token)
    else:
        print(f"  ✗  Exchange failed: {data}\n")
except Exception as e:
    print(f"  ✗  {e}\n")
