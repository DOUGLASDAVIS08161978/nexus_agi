"""
NEXUS AGI - INTEGRATED MINING & MULTI-NETWORK BRIDGE
====================================================

Integrates Bitcoin mining with automatic multi-network token distribution
Uses all configured blockchain networks and public APIs

Features:
- Bitcoin mining simulation
- Automatic bridging to multiple networks
- Real-time price data integration
- GitHub result publication
- Multi-network token distribution
"""

import sys
import os
import json
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.network_config import get_network_manager
from tools.multi_network_bridge_orchestrator import MultiNetworkBridgeOrchestrator, BridgeTransaction
from tools.public_api_integrator import PublicAPIIntegrator
from tools.github_api_integration import GitHubAPIClient, NexusGitHubIntegration


@dataclass
class MiningSession:
    """Mining session data"""
    session_id: str
    start_time: str
    duration_seconds: int
    blocks_mined: int
    total_btc: float
    hash_rate: float
    networks_bridged: List[str]
    bridge_transactions: List[str]  # Bridge IDs
    total_fees_usd: float
    btc_price_usd: float


class IntegratedMiningBridge:
    """
    Integrated mining and bridging system
    Mines Bitcoin and automatically distributes to multiple networks
    """

    def __init__(
        self,
        target_networks: Optional[List[str]] = None,
        github_token: Optional[str] = None
    ):
        """
        Initialize integrated mining bridge

        Args:
            target_networks: List of networks to bridge to
            github_token: GitHub token for result publication
        """
        self.network_manager = get_network_manager()
        self.bridge_orchestrator = MultiNetworkBridgeOrchestrator(self.network_manager)
        self.api_integrator = PublicAPIIntegrator()

        # Default target networks (all mainnets)
        self.target_networks = target_networks or [
            "ethereum_mainnet",
            "arbitrum_mainnet",
            "polygon_mainnet",
            "base_mainnet",
            "avalanche_mainnet"
        ]

        # GitHub integration
        self.github_token = github_token or os.environ.get("GITHUB_TOKEN")
        if self.github_token:
            self.github_client = GitHubAPIClient(token=self.github_token)
            self.github_integration = NexusGitHubIntegration(
                self.github_client,
                "DOUGLASDAVIS08161978",
                "nexus_agi"
            )
        else:
            self.github_client = None
            self.github_integration = None

        # Mining configuration
        self.block_reward = 6.25  # BTC per block
        self.blocks_per_bridge = 3  # Bridge every N blocks

        # Session tracking
        self.mining_sessions: List[MiningSession] = []

    def simulate_mining_block(self, block_number: int) -> Dict[str, Any]:
        """
        Simulate mining a single block

        Args:
            block_number: Block number

        Returns:
            Block data
        """
        # Simple mining simulation
        nonce = 0
        target_prefix = "0000"

        while True:
            block_data = f"{block_number}-{int(time.time())}-{nonce}"
            block_hash = f"{nonce:064x}"  # Simplified

            if block_hash.startswith(target_prefix):
                return {
                    "block_number": block_number,
                    "block_hash": block_hash,
                    "nonce": nonce,
                    "reward": self.block_reward,
                    "timestamp": datetime.now().isoformat()
                }

            nonce += 1

            # Limit iterations for demo
            if nonce > 10000:
                return {
                    "block_number": block_number,
                    "block_hash": f"{nonce:064x}",
                    "nonce": nonce,
                    "reward": self.block_reward,
                    "timestamp": datetime.now().isoformat()
                }

    def run_mining_session(
        self,
        duration_seconds: int = 60,
        recipient_address: str = "0x24f6b1ce11c57d40b542f91ac85fa9eb61f78771"
    ) -> MiningSession:
        """
        Run complete mining session with automatic bridging

        Args:
            duration_seconds: Mining duration in seconds
            recipient_address: Address to receive bridged tokens

        Returns:
            Mining session data
        """
        session_id = f"MINING-{int(time.time())}"
        start_time = datetime.now()

        print("\n" + "=" * 80)
        print("NEXUS AGI - INTEGRATED MINING & BRIDGE SESSION")
        print("=" * 80)

        print(f"\n🚀 Session Configuration:")
        print(f"   Session ID: {session_id}")
        print(f"   Duration: {duration_seconds}s")
        print(f"   Target Networks: {len(self.target_networks)}")
        print(f"   Bridge Frequency: Every {self.blocks_per_bridge} blocks")
        print(f"   Recipient: {recipient_address}")

        # Get BTC price
        btc_price = self.api_integrator.get_token_price_by_symbol("BTC") or 0
        print(f"   BTC Price: ${btc_price:,.2f} USD")

        blocks_mined = 0
        total_btc = 0
        bridge_ids = []
        total_fees = 0
        end_time = time.time() + duration_seconds

        print(f"\n⛏️  Mining started...")

        while time.time() < end_time:
            # Mine block
            blocks_mined += 1
            block = self.simulate_mining_block(blocks_mined)
            total_btc += block["reward"]

            print(f"\n   Block #{blocks_mined} mined!")
            print(f"   Hash: {block['block_hash'][:16]}...")
            print(f"   Reward: {block['reward']} BTC")
            print(f"   Total BTC: {total_btc:.4f}")

            # Bridge every N blocks
            if blocks_mined % self.blocks_per_bridge == 0:
                print(f"\n   🌉 Bridging {block['reward']} BTC to {len(self.target_networks)} networks...")

                amount_per_network = block['reward'] / len(self.target_networks)

                for network in self.target_networks:
                    try:
                        tx = self.bridge_orchestrator.execute_bridge(
                            from_network="bitcoin",
                            to_network=network,
                            token="WBTC",
                            amount=amount_per_network,
                            from_address="bc1qfzhx87ckhn4tnkswhsth56h0gm5we4hdq5wass",
                            to_address=recipient_address
                        )

                        bridge_ids.append(tx.bridge_id)
                        total_fees += tx.fee_usd or 0

                    except Exception as e:
                        print(f"      ❌ Error bridging to {network}: {e}")

        # Calculate hash rate
        actual_duration = (datetime.now() - start_time).total_seconds()
        hash_rate = (blocks_mined * 10000) / actual_duration if actual_duration > 0 else 0

        # Create session
        session = MiningSession(
            session_id=session_id,
            start_time=start_time.isoformat(),
            duration_seconds=int(actual_duration),
            blocks_mined=blocks_mined,
            total_btc=total_btc,
            hash_rate=hash_rate,
            networks_bridged=self.target_networks,
            bridge_transactions=bridge_ids,
            total_fees_usd=total_fees,
            btc_price_usd=btc_price
        )

        self.mining_sessions.append(session)

        # Display results
        print("\n" + "=" * 80)
        print("✨ MINING SESSION COMPLETE")
        print("=" * 80)

        print(f"\n📊 Mining Statistics:")
        print(f"   Blocks Mined: {blocks_mined}")
        print(f"   Total BTC: {total_btc:.4f} BTC (${total_btc * btc_price:,.2f} USD)")
        print(f"   Hash Rate: {hash_rate:,.2f} H/s")
        print(f"   Duration: {actual_duration:.1f}s")

        print(f"\n🌉 Bridge Statistics:")
        print(f"   Networks: {len(self.target_networks)}")
        print(f"   Transactions: {len(bridge_ids)}")
        print(f"   Total Fees: ${total_fees:.2f} USD")

        # Publish to GitHub if enabled
        if self.github_integration:
            print(f"\n📤 Publishing results to GitHub...")
            self.publish_to_github(session)

        print("\n" + "=" * 80)

        return session

    def publish_to_github(self, session: MiningSession):
        """
        Publish mining session to GitHub

        Args:
            session: Mining session data
        """
        if not self.github_integration:
            print("   ⚠️  GitHub integration not enabled (no token)")
            return

        try:
            # Convert session to dict
            session_data = {
                "session_id": session.session_id,
                "start_time": session.start_time,
                "duration_seconds": session.duration_seconds,
                "mining_stats": {
                    "blocks_mined": session.blocks_mined,
                    "total_btc": session.total_btc,
                    "hash_rate": session.hash_rate,
                    "btc_price_usd": session.btc_price_usd,
                    "total_value_usd": session.total_btc * session.btc_price_usd
                },
                "bridge_stats": {
                    "networks_count": len(session.networks_bridged),
                    "networks": session.networks_bridged,
                    "transaction_count": len(session.bridge_transactions),
                    "transactions": session.bridge_transactions,
                    "total_fees_usd": session.total_fees_usd
                }
            }

            # Publish as gist
            gist = self.github_integration.publish_mining_results(session_data)
            print(f"   ✅ Published to GitHub Gist: {gist['html_url']}")

        except Exception as e:
            print(f"   ❌ Error publishing to GitHub: {e}")

    def export_session(self, session: MiningSession, filename: Optional[str] = None) -> str:
        """
        Export mining session to JSON file

        Args:
            session: Mining session
            filename: Output filename (optional)

        Returns:
            File path
        """
        if not filename:
            filename = f"mining_session_{session.session_id}.json"

        data = {
            "session": asdict(session),
            "bridge_history": [
                {
                    "bridge_id": tx.bridge_id,
                    "from_network": tx.from_network,
                    "to_network": tx.to_network,
                    "amount": tx.amount,
                    "fee_usd": tx.fee_usd,
                    "status": tx.status
                }
                for tx in self.bridge_orchestrator.bridge_history
                if tx.bridge_id in session.bridge_transactions
            ]
        }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

        return filename

    def get_all_sessions_summary(self) -> Dict[str, Any]:
        """Get summary of all mining sessions"""
        if not self.mining_sessions:
            return {"total_sessions": 0}

        total_blocks = sum(s.blocks_mined for s in self.mining_sessions)
        total_btc = sum(s.total_btc for s in self.mining_sessions)
        total_bridges = sum(len(s.bridge_transactions) for s in self.mining_sessions)
        total_fees = sum(s.total_fees_usd for s in self.mining_sessions)

        avg_btc_price = sum(s.btc_price_usd for s in self.mining_sessions) / len(self.mining_sessions)

        return {
            "total_sessions": len(self.mining_sessions),
            "total_blocks_mined": total_blocks,
            "total_btc_earned": total_btc,
            "total_value_usd": total_btc * avg_btc_price,
            "total_bridge_transactions": total_bridges,
            "total_fees_usd": total_fees,
            "networks_used": len(set(
                net for session in self.mining_sessions for net in session.networks_bridged
            ))
        }


def main():
    """Main execution"""
    print("\n" + "=" * 80)
    print("NEXUS AGI - INTEGRATED MINING & MULTI-NETWORK BRIDGE")
    print("=" * 80)

    # Initialize system
    mining_bridge = IntegratedMiningBridge(
        target_networks=[
            "ethereum_mainnet",
            "arbitrum_mainnet",
            "polygon_mainnet",
            "base_mainnet",
            "avalanche_mainnet"
        ]
    )

    # Run mining session
    session = mining_bridge.run_mining_session(
        duration_seconds=30,
        recipient_address="0x24f6b1ce11c57d40b542f91ac85fa9eb61f78771"
    )

    # Export results
    export_file = mining_bridge.export_session(session)
    print(f"\n💾 Session exported to: {export_file}")

    # Show summary
    summary = mining_bridge.get_all_sessions_summary()

    print(f"\n📈 All-Time Summary:")
    print(f"   Total Sessions: {summary['total_sessions']}")
    print(f"   Total Blocks: {summary['total_blocks_mined']}")
    print(f"   Total BTC: {summary['total_btc_earned']:.4f}")
    print(f"   Total Value: ${summary['total_value_usd']:,.2f} USD")
    print(f"   Bridge Transactions: {summary['total_bridge_transactions']}")
    print(f"   Networks Used: {summary['networks_used']}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
