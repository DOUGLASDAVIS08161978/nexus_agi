#!/usr/bin/env python3
"""
Nexus AGI Directory Server

Serves a registry of Nexus AI agents and their capabilities.
Acts as a discovery service for the distributed Nexus network.
"""

import http.server
import socketserver
import json
import os

PORT = 8000
FILE_NAME = "nexus_seeds.json"

# Create the dummy data file if it doesn't exist
if not os.path.exists(FILE_NAME):
    seeds_data = [
        {"id": "agent-alpha-001", "name": "Quantum Ledger Node", "capabilities": ["blockchain_analysis"], "endpoint": "http://localhost:8001"},
        {"id": "agent-gamma-003", "name": "Ethereum Gateway", "capabilities": ["ethereum_smart_contracts"], "endpoint": "http://localhost:8003"},
        {"id": "agent-delta-004", "name": "Satoshi Oracle", "capabilities": ["bitcoin_prediction"], "endpoint": "http://localhost:8004"}
    ]
    with open(FILE_NAME, 'w') as f:
        json.dump(seeds_data, f, indent=2)
    print(f"📝 Created dummy data file: {FILE_NAME}")

class NexusHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path.endswith(".json") or self.path == "/":
            if os.path.exists(FILE_NAME):
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                with open(FILE_NAME, 'rb') as f:
                    self.wfile.write(f.read())
                print(f"⚡ [SERVER] Served {FILE_NAME} to Nexus Client")
            else:
                self.send_error(404, "Seeds file not found")
        else:
            self.send_error(404, "Endpoint not found")

print(f"🌍 NEXUS AGI DIRECTORY ONLINE")
print(f"📡 Listening on http://localhost:{PORT}")
print(f"📂 Serving registry from: {FILE_NAME}")

with socketserver.TCPServer(("", PORT), NexusHandler) as httpd:
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
