"""
NEXUS AGI DIRECTORY SERVER
===========================

Local server for Nexus AGI seed distribution
Provides .well-known/seeds-public.json endpoint
"""

import http.server
import socketserver
import json
import os
from datetime import datetime

PORT = 8000
FILE_NAME = "nexus_seeds.json"


class NexusHandler(http.server.SimpleHTTPRequestHandler):
    """Custom HTTP handler for Nexus AGI directory"""

    def do_GET(self):
        """Handle GET requests"""

        # Serve the JSON file for any .json request or root
        if self.path.endswith(".json") or self.path == "/" or "seeds" in self.path:
            if os.path.exists(FILE_NAME):
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()

                with open(FILE_NAME, 'rb') as f:
                    self.wfile.write(f.read())

                print(f"⚡ [{datetime.now().strftime('%H:%M:%S')}] Served {FILE_NAME}")

            else:
                self.send_error(404, "Seeds file not found")
        else:
            self.send_error(404, "Endpoint not found")

    def log_message(self, format, *args):
        """Suppress default logging"""
        pass


def create_default_seeds():
    """Create default nexus_seeds.json if it doesn't exist"""

    if not os.path.exists(FILE_NAME):
        seeds_data = {
            "version": "1.0",
            "timestamp": datetime.now().isoformat(),
            "network": "nexus_agi",
            "nodes": [
                {
                    "id": "omega_node_1",
                    "type": "mining",
                    "capabilities": ["bitcoin_mining", "ethereum_bridge", "predictive_ai"],
                    "endpoint": "http://localhost:8001"
                },
                {
                    "id": "bridge_node_1",
                    "type": "bridge",
                    "capabilities": ["btc_eth_bridge", "token_minting"],
                    "endpoint": "http://localhost:8002",
                    "contracts": {
                        "wTBTC": "0x324befe00354823df73691e37ed4f7b19ad74f63",
                        "HashProof": "0x0000000000000000000000000000000000000000"
                    }
                }
            ],
            "mining_pools": [
                {
                    "name": "OmegaInfinitePool",
                    "url": "stratum+tcp://pool.omega.io:3333",
                    "fee": 0.01
                }
            ],
            "bridge_contracts": {
                "sepolia": {
                    "wTBTC": "0x324befe00354823df73691e37ed4f7b19ad74f63",
                    "network_id": 11155111
                }
            }
        }

        with open(FILE_NAME, 'w') as f:
            json.dump(seeds_data, f, indent=2)

        print(f"📝 Created default {FILE_NAME}")


def main():
    """Start Nexus AGI directory server"""

    print("\n" + "="*80)
    print("🌍 NEXUS AGI DIRECTORY SERVER")
    print("="*80)

    # Create default seeds if needed
    create_default_seeds()

    print(f"\n📡 Server Configuration:")
    print(f"   Port:        {PORT}")
    print(f"   Endpoint:    http://localhost:{PORT}")
    print(f"   Seeds File:  {FILE_NAME}")

    print(f"\n🚀 Starting server...")

    try:
        with socketserver.TCPServer(("", PORT), NexusHandler) as httpd:
            print(f"✅ Server online at http://localhost:{PORT}")
            print(f"📊 Serving: {FILE_NAME}")
            print(f"\n⚡ Ready to serve Nexus clients...\n")
            print("Press Ctrl+C to stop\n")

            httpd.serve_forever()

    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Server error: {e}")


if __name__ == "__main__":
    main()
