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
            "version": "2.0",
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
                },
                {
                    "id": "arbitrum_bridge_1",
                    "type": "l2_bridge",
                    "capabilities": ["arbitrum_bridge", "token_transfer", "optimistic_rollup"],
                    "endpoint": "arbitrum-bridge.nexus.io:8545",
                    "network": "arbitrum",
                    "chain_id": 42161
                },
                {
                    "id": "optimism_bridge_1",
                    "type": "l2_bridge",
                    "capabilities": ["optimism_bridge", "token_transfer", "optimistic_rollup"],
                    "endpoint": "optimism-bridge.nexus.io:8545",
                    "network": "optimism",
                    "chain_id": 10
                },
                {
                    "id": "base_bridge_1",
                    "type": "l2_bridge",
                    "capabilities": ["base_bridge", "token_transfer", "optimistic_rollup"],
                    "endpoint": "base-bridge.nexus.io:8545",
                    "network": "base",
                    "chain_id": 8453
                },
                {
                    "id": "avalanche_bridge_1",
                    "type": "alt_l1_bridge",
                    "capabilities": ["avalanche_bridge", "token_transfer"],
                    "endpoint": "avalanche-bridge.nexus.io:9650",
                    "network": "avalanche",
                    "chain_id": 43114
                },
                {
                    "id": "bsc_bridge_1",
                    "type": "alt_l1_bridge",
                    "capabilities": ["bsc_bridge", "token_transfer"],
                    "endpoint": "bsc-bridge.nexus.io:8545",
                    "network": "bsc",
                    "chain_id": 56
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
                },
                "arbitrum": {
                    "wBTC": "0x2f2a2543B76A4166549F7aaB2e75Bef0aefC5B0f",
                    "network_id": 42161
                },
                "optimism": {
                    "wBTC": "0x68f180fcCe6836688e9084f035309E29Bf0A2095",
                    "network_id": 10
                },
                "base": {
                    "wBTC": "0x0555E30da8f98308EdB960aa94C0Db47230d2B9c",
                    "network_id": 8453
                },
                "polygon": {
                    "wBTC": "0x1BFD67037B42Cf73acF2047067bd4F2C47D9BfD6",
                    "network_id": 137
                },
                "avalanche": {
                    "wBTC": "0x50b7545627a5162F82A992c33b87aDc75187B218",
                    "network_id": 43114
                },
                "bsc": {
                    "wBTC": "0x7130d2A12B9BCbFAe4f2634d864A1Ee1Ce3Ead9c",
                    "network_id": 56
                }
            },
            "bridge_routes": [
                {"from": "bitcoin", "to": "ethereum", "token": "wBTC", "operator": "ethereum_bridge_1"},
                {"from": "ethereum", "to": "arbitrum", "token": "wBTC", "operator": "arbitrum_bridge_1"},
                {"from": "ethereum", "to": "optimism", "token": "wBTC", "operator": "optimism_bridge_1"},
                {"from": "ethereum", "to": "base", "token": "wBTC", "operator": "base_bridge_1"},
                {"from": "ethereum", "to": "polygon", "token": "wBTC", "operator": "polygon_bridge_1"},
                {"from": "ethereum", "to": "avalanche", "token": "wBTC", "operator": "avalanche_bridge_1"},
                {"from": "ethereum", "to": "bsc", "token": "wBTC", "operator": "bsc_bridge_1"},
                {"from": "bitcoin", "to": "arbitrum", "token": "wBTC", "operator": "arbitrum_bridge_1", "multi_hop": true},
                {"from": "bitcoin", "to": "optimism", "token": "wBTC", "operator": "optimism_bridge_1", "multi_hop": true},
                {"from": "bitcoin", "to": "base", "token": "wBTC", "operator": "base_bridge_1", "multi_hop": true},
                {"from": "bitcoin", "to": "polygon", "token": "wBTC", "operator": "polygon_bridge_1", "multi_hop": true},
                {"from": "bitcoin", "to": "avalanche", "token": "wBTC", "operator": "avalanche_bridge_1", "multi_hop": true},
                {"from": "bitcoin", "to": "bsc", "token": "wBTC", "operator": "bsc_bridge_1", "multi_hop": true}
            ]
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
