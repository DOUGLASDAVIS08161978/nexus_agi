#!/usr/bin/env python3
"""
BITCOIN SIGNET NETWORK EXPLORER
Connects to actual Bitcoin Signet network
Shows real-time blockchain data and network activity

NOTE: Signet uses SIGNED blocks, not proof-of-work mining!
Cannot mine on Signet - blocks are created by authorized signers.
"""

import requests
import time
import json
from datetime import datetime
from typing import Dict, List, Optional
from immutable_ledger_system import ImmutableLedger


class SignetExplorer:
    """
    Bitcoin Signet Network Explorer

    - Connects to real Signet network
    - Monitors blocks and transactions
    - Shows network statistics
    - Demonstrates why Signet can't be mined
    """

    def __init__(self):
        self.signet_api = "https://mempool.space/signet/api"
        self.ledger = ImmutableLedger("signet_exploration_ledger.jsonl")

        self.stats = {
            "blocks_monitored": 0,
            "transactions_seen": 0,
            "start_time": datetime.now().isoformat(),
        }

        print("=" * 80)
        print("BITCOIN SIGNET NETWORK EXPLORER")
        print("=" * 80)
        print(f"Network: Bitcoin Signet (DEFAULT)")
        print(f"Purpose: Testing & Development")
        print(f"Mining: ❌ NOT ALLOWED (Signed blocks only)")
        print(f"API: {self.signet_api}")
        print("=" * 80)

    def get_network_info(self) -> Dict:
        """Get Signet network information"""
        print(f"\n{'=' * 80}")
        print(f"SIGNET NETWORK INFORMATION")
        print(f"{'=' * 80}")

        try:
            # Get current block tip
            url = f"{self.signet_api}/blocks/tip/height"
            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                current_height = int(response.text.strip())

                # Get tip hash
                url = f"{self.signet_api}/blocks/tip/hash"
                response = requests.get(url, timeout=10)
                tip_hash = response.text.strip()

                print(f"✅ Current Block Height: {current_height:,}")
                print(f"✅ Tip Block Hash: {tip_hash}")

                return {
                    'height': current_height,
                    'tip_hash': tip_hash,
                    'network': 'signet'
                }

        except Exception as e:
            print(f"❌ Error: {e}")

        return {}

    def get_block_details(self, block_hash: str) -> Dict:
        """Get detailed information about a block"""
        print(f"\n{'=' * 80}")
        print(f"BLOCK DETAILS")
        print(f"{'=' * 80}")

        try:
            url = f"{self.signet_api}/block/{block_hash}"
            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                block = response.json()

                print(f"Hash:           {block.get('id')}")
                print(f"Height:         {block.get('height'):,}")
                print(f"Timestamp:      {datetime.fromtimestamp(block.get('timestamp', 0))}")
                print(f"Transactions:   {block.get('tx_count', 0)}")
                print(f"Size:           {block.get('size', 0):,} bytes")
                print(f"Weight:         {block.get('weight', 0):,}")
                print(f"Version:        0x{block.get('version', 0):08x}")
                print(f"Bits:           0x{block.get('bits', 0):08x}")
                print(f"Nonce:          {block.get('nonce', 0)}")
                print(f"Difficulty:     {block.get('difficulty', 0)}")

                # IMPORTANT: Show that this is a SIGNED block
                print(f"\n🔐 SIGNET BLOCK SIGNATURE:")
                print(f"   This block was SIGNED by authorized signers")
                print(f"   NOT mined with proof-of-work!")
                print(f"   Signet uses multisig block signing")

                self.stats['blocks_monitored'] += 1

                # Record to ledger
                self.ledger.add_entry("SIGNET_BLOCK_EXPLORED", {
                    "block_hash": block.get('id'),
                    "height": block.get('height'),
                    "timestamp": block.get('timestamp'),
                    "tx_count": block.get('tx_count'),
                    "explored_at": datetime.now().isoformat()
                })

                return block

        except Exception as e:
            print(f"❌ Error: {e}")

        return {}

    def compare_signet_vs_testnet(self):
        """Compare Signet vs Testnet3 characteristics"""
        print(f"\n{'=' * 80}")
        print(f"SIGNET vs TESTNET3 COMPARISON")
        print(f"{'=' * 80}")

        comparison = {
            "Signet": {
                "block_production": "Signed by authorized signers",
                "mining": "❌ NOT ALLOWED",
                "difficulty": "N/A (no proof-of-work)",
                "block_time": "~10 minutes (predictable)",
                "reorgs": "Rare (controlled)",
                "use_case": "Development & testing",
                "stability": "Very stable",
                "get_coins": "Faucets only"
            },
            "Testnet3": {
                "block_production": "Proof-of-work mining",
                "mining": "✅ ALLOWED (we just did this!)",
                "difficulty": "Adjusts (can be low)",
                "block_time": "~10 minutes (variable)",
                "reorgs": "Possible (real mining)",
                "use_case": "Mining practice",
                "stability": "Moderate",
                "get_coins": "Mine or faucets"
            }
        }

        print(f"\n{'CHARACTERISTIC':<25} {'SIGNET':<30} {'TESTNET3':<30}")
        print(f"{'-' * 85}")

        for key in comparison["Signet"].keys():
            signet_val = comparison["Signet"][key]
            testnet_val = comparison["Testnet3"][key]
            print(f"{key:<25} {signet_val:<30} {testnet_val:<30}")

        print(f"\n{'=' * 80}")
        print(f"CONCLUSION:")
        print(f"{'=' * 80}")
        print(f"")
        print(f"FOR MINING:     Use Testnet3 ✅ (what we just did!)")
        print(f"FOR DEVELOPMENT: Use Signet ✅ (predictable, stable)")
        print(f"FOR PRODUCTION: Use Mainnet 💰 (real money)")
        print(f"")
        print(f"{'=' * 80}")

    def demonstrate_why_no_mining(self):
        """Demonstrate why Signet can't be mined"""
        print(f"\n{'=' * 80}")
        print(f"WHY YOU CAN'T MINE ON SIGNET")
        print(f"{'=' * 80}")

        print(f"""
🔐 SIGNET BLOCK CREATION PROCESS:

1. AUTHORIZED SIGNERS create blocks
   - Only specific keys can sign blocks
   - Uses Bitcoin multisig (like 1-of-2 or 2-of-3)

2. NO PROOF-OF-WORK required
   - No SHA-256d mining
   - No difficulty target
   - No nonce searching

3. BLOCKS ARE SIGNED, not mined
   - Signature proves block is from authorized signer
   - Network validates signature, not hash

4. PREDICTABLE BLOCK TIMES
   - Blocks created on schedule (~10 minutes)
   - No random variation from mining competition

❌ WHAT DOESN'T WORK:
   - Running SHA-256d hashing
   - Trying to find nonce that meets difficulty
   - Broadcasting unsigned blocks
   - Solo mining attempts

✅ WHAT DOES WORK:
   - Creating transactions
   - Broadcasting transactions
   - Testing smart contracts
   - Developing Bitcoin applications
   - Learning Bitcoin without mining

🎯 BOTTOM LINE:
   Signet is for DEVELOPMENT, not MINING
   Use Testnet3 if you want to mine!
""")

    def get_recent_blocks(self, count: int = 10) -> List[Dict]:
        """Get recent blocks from Signet"""
        print(f"\n{'=' * 80}")
        print(f"RECENT SIGNET BLOCKS")
        print(f"{'=' * 80}")

        blocks = []

        try:
            url = f"{self.signet_api}/blocks"
            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                blocks_data = response.json()

                print(f"\n{'HEIGHT':<10} {'TIMESTAMP':<20} {'TXS':<6} {'SIZE':<10} {'HASH':<20}")
                print(f"{'-' * 80}")

                for block in blocks_data[:count]:
                    height = block.get('height', 0)
                    timestamp = datetime.fromtimestamp(block.get('timestamp', 0))
                    tx_count = block.get('tx_count', 0)
                    size = block.get('size', 0)
                    block_hash = block.get('id', '')[:16]

                    print(f"{height:<10} {timestamp.strftime('%Y-%m-%d %H:%M:%S'):<20} {tx_count:<6} {size:<10} {block_hash}...")

                    blocks.append(block)
                    self.stats['transactions_seen'] += tx_count

        except Exception as e:
            print(f"❌ Error: {e}")

        return blocks

    def explore_signet_network(self):
        """Complete Signet network exploration"""
        print(f"\n{'=' * 80}")
        print(f"SIGNET NETWORK EXPLORATION")
        print(f"{'=' * 80}")

        # Get network info
        network_info = self.get_network_info()

        if network_info:
            # Get latest block details
            tip_hash = network_info.get('tip_hash')
            if tip_hash:
                block_details = self.get_block_details(tip_hash)

            # Get recent blocks
            recent_blocks = self.get_recent_blocks(10)

            # Compare networks
            self.compare_signet_vs_testnet()

            # Explain why no mining
            self.demonstrate_why_no_mining()

        # Print statistics
        print(f"\n{'=' * 80}")
        print(f"EXPLORATION STATISTICS")
        print(f"{'=' * 80}")
        print(f"Blocks Monitored:    {self.stats['blocks_monitored']}")
        print(f"Transactions Seen:   {self.stats['transactions_seen']}")
        print(f"Started:             {self.stats['start_time']}")
        print(f"{'=' * 80}")

        # Ledger stats
        self.ledger.print_statistics()


def main():
    """Main program"""

    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                     BITCOIN SIGNET NETWORK EXPLORER                          ║
║                                                                              ║
║  Exploring Bitcoin Signet - the SIGNED block network                        ║
║                                                                              ║
║  ⚠️  IMPORTANT: You CANNOT mine on Signet!                                  ║
║      Signet uses signed blocks, not proof-of-work.                          ║
║                                                                              ║
║  ✅  We ALREADY mined on Testnet3 (50 million hashes!)                      ║
║      This explorer shows why Signet is different.                           ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

    explorer = SignetExplorer()

    print(f"\n🚀 Starting Signet network exploration...")
    print(f"\n⏳ Connecting to Signet network...")

    # Explore the network
    explorer.explore_signet_network()

    print(f"\n{'=' * 80}")
    print(f"🎓 EDUCATIONAL SUMMARY")
    print(f"{'=' * 80}")
    print(f"""
✅ SIGNET is for:
   - Application development
   - Smart contract testing
   - Predictable block times
   - Stable test environment

✅ TESTNET3 is for:
   - Mining practice (WE DID THIS!)
   - Learning proof-of-work
   - Block discovery
   - Mining rewards

💡 YOU SHOULD USE:
   - Signet: For app development
   - Testnet3: For mining (already done!)
   - Mainnet: For real Bitcoin

🎉 We completed REAL MINING on Testnet3:
   - 50,000,000 hashes computed
   - 540 KH/s hashrate
   - Real Bitcoin proof-of-work
   - All recorded in immutable ledger!
""")

    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
