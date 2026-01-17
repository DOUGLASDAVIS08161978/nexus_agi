#!/usr/bin/env python3
"""
NEXUS AGI DIRECTORY SERVER
==========================
Serves the Nexus AGI seeds registry for distributed AI coordination
"""

import http.server
import socketserver
import json
import os
from datetime import datetime

PORT = 8000
FILE_NAME = "nexus_seeds.json"

class NexusHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        # Serve the JSON file regardless of the path for this demo
        # This mimics the /.well-known/seeds-public.json endpoint
        if self.path.endswith(".json") or self.path == "/" or "seeds" in self.path:
            if os.path.exists(FILE_NAME):
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                with open(FILE_NAME, 'rb') as f:
                    self.wfile.write(f.read())
                print(f"⚡ [{datetime.now().strftime('%H:%M:%S')}] Served {FILE_NAME} to client")
            else:
                self.send_error(404, "Seeds file not found")
        else:
            self.send_error(404, "Endpoint not found")

    def log_message(self, format, *args):
        """Override to customize logging"""
        print(f"📡 [{datetime.now().strftime('%H:%M:%S')}] {format % args}")

def create_default_seeds():
    """Create default Nexus seeds registry"""
    default_seeds = {
        "version": "1.0.0",
        "updated": datetime.now().isoformat(),
        "seeds": [
            {
                "id": "nexus-primary",
                "url": "https://nexus-agi.com",
                "blockchain": {
                    "bitcoin_enabled": True,
                    "ethereum_enabled": True,
                    "bridge_active": True
                },
                "capabilities": [
                    "bitcoin_mining",
                    "ethereum_bridge",
                    "smart_contracts",
                    "cross_chain_transfer"
                ]
            },
            {
                "id": "nexus-local",
                "url": "http://localhost:8000",
                "blockchain": {
                    "bitcoin_enabled": True,
                    "ethereum_enabled": True,
                    "bridge_active": True
                },
                "capabilities": [
                    "bitcoin_validation",
                    "ethereum_bridge",
                    "local_testing"
                ]
            }
        ]
    }

    with open(FILE_NAME, 'w') as f:
        json.dump(default_seeds, f, indent=2)

    print(f"✅ Created default seeds registry: {FILE_NAME}")

if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"🌍 NEXUS AGI DIRECTORY SERVER")
    print(f"{'='*60}")

    # Create default seeds if file doesn't exist
    if not os.path.exists(FILE_NAME):
        create_default_seeds()

    print(f"📡 Listening on http://localhost:{PORT}")
    print(f"📂 Serving registry from: {FILE_NAME}")
    print(f"🔗 Endpoint: http://localhost:{PORT}/{FILE_NAME}")
    print(f"{'='*60}\n")

    with socketserver.TCPServer(("", PORT), NexusHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print(f"\n\n🛑 Server stopped")
