"""
NEXUS AGI - MULTI-NETWORK BRIDGE ORCHESTRATOR
==============================================

Orchestrates cross-chain bridges across all supported networks
Integrates Bitcoin mining with multi-chain token distribution

Supported Bridge Routes:
- Bitcoin → Ethereum/L2s/Alt-L1s
- Ethereum ↔ All L2s
- Cross-L2 bridges
- Alt-L1 bridges

Features:
- Automatic route finding
- Fee optimization
- Multi-hop bridging
- Price oracle integration
- Transaction tracking
"""

import sys
import os
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import json
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.network_config import NetworkManager, get_network_manager, BLOCKCHAIN_NETWORKS
from tools.multi_network_bridge_config import (
    BRIDGE_ROUTES,
    TOKENS,
    get_token_address,
    find_bridge_route,
    estimate_bridge_cost
)


@dataclass
class BridgeTransaction:
    """Bridge transaction data"""
    bridge_id: str
    from_network: str
    to_network: str
    token: str
    amount: float
    from_address: str
    to_address: str
    status: str = "pending"
    from_tx_hash: Optional[str] = None
    to_tx_hash: Optional[str] = None
    fee_usd: Optional[float] = None
    estimated_time_minutes: Optional[int] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


class MultiNetworkBridgeOrchestrator:
    """
    Orchestrates cross-chain bridges across all networks
    Integrates with Nexus AGI mining and token systems
    """

    def __init__(self, network_manager: Optional[NetworkManager] = None):
        """
        Initialize bridge orchestrator

        Args:
            network_manager: Network manager instance
        """
        self.network_manager = network_manager or get_network_manager()
        self.bridge_history: List[BridgeTransaction] = []
        self.total_volume_usd = 0.0

    def find_optimal_route(
        self,
        from_network: str,
        to_network: str,
        token: str = "WBTC"
    ) -> Optional[Dict[str, Any]]:
        """
        Find optimal bridge route between networks

        Args:
            from_network: Source network
            to_network: Destination network
            token: Token to bridge

        Returns:
            Route information or None
        """
        try:
            # Try direct route first
            route = find_bridge_route(from_network, to_network, token)

            return {
                "route_type": "direct",
                "from": from_network,
                "to": to_network,
                "token": token,
                "bridge_contract": route.bridge_contract,
                "operator_type": route.operator_type,
                "estimated_time": route.estimated_time,
                "estimated_fee": route.estimated_fee_usd,
                "hops": 1
            }

        except ValueError:
            # Try multi-hop route (e.g., Bitcoin → Ethereum → Target)
            if from_network == "bitcoin":
                try:
                    # Bitcoin → Ethereum → Target
                    eth_route = find_bridge_route("bitcoin", "ethereum", token)
                    target_route = find_bridge_route("ethereum", to_network, token)

                    return {
                        "route_type": "multi_hop",
                        "from": from_network,
                        "to": to_network,
                        "token": token,
                        "via": "ethereum",
                        "bridge_contracts": [
                            eth_route.bridge_contract,
                            target_route.bridge_contract
                        ],
                        "estimated_time": eth_route.estimated_time + target_route.estimated_time,
                        "estimated_fee": eth_route.estimated_fee_usd + target_route.estimated_fee_usd,
                        "hops": 2
                    }
                except ValueError:
                    pass

            return None

    def estimate_bridge_fees(
        self,
        from_network: str,
        to_network: str,
        token: str,
        amount: float
    ) -> Dict[str, Any]:
        """
        Estimate total bridge fees including gas

        Args:
            from_network: Source network
            to_network: Destination network
            token: Token to bridge
            amount: Amount to bridge

        Returns:
            Fee breakdown
        """
        route = self.find_optimal_route(from_network, to_network, token)

        if not route:
            return {
                "error": "No route found",
                "total_fee_usd": 0,
                "estimated_time_minutes": 0
            }

        # Get token price
        price_usd = self.get_token_price(token)
        value_usd = amount * price_usd if price_usd else 0

        # Calculate fees
        bridge_fee_usd = route["estimated_fee"]
        bridge_fee_percent = (bridge_fee_usd / value_usd * 100) if value_usd > 0 else 0

        return {
            "route_type": route["route_type"],
            "hops": route["hops"],
            "bridge_fee_usd": bridge_fee_usd,
            "bridge_fee_percent": bridge_fee_percent,
            "total_fee_usd": bridge_fee_usd,
            "value_usd": value_usd,
            "estimated_time_minutes": route["estimated_time"],
            "token_price_usd": price_usd
        }

    def execute_bridge(
        self,
        from_network: str,
        to_network: str,
        token: str,
        amount: float,
        from_address: str,
        to_address: str
    ) -> BridgeTransaction:
        """
        Execute cross-chain bridge (simulation)

        Args:
            from_network: Source network
            to_network: Destination network
            token: Token to bridge
            amount: Amount to bridge
            from_address: Source address
            to_address: Destination address

        Returns:
            Bridge transaction
        """
        bridge_id = f"BRIDGE-{int(time.time())}-{from_network[:3].upper()}-{to_network[:3].upper()}"

        # Find route
        route = self.find_optimal_route(from_network, to_network, token)

        if not route:
            raise ValueError(f"No bridge route found: {from_network} → {to_network}")

        # Estimate fees
        fees = self.estimate_bridge_fees(from_network, to_network, token, amount)

        # Create transaction
        tx = BridgeTransaction(
            bridge_id=bridge_id,
            from_network=from_network,
            to_network=to_network,
            token=token,
            amount=amount,
            from_address=from_address,
            to_address=to_address,
            status="initiated",
            fee_usd=fees["total_fee_usd"],
            estimated_time_minutes=fees["estimated_time_minutes"],
            started_at=datetime.now().isoformat()
        )

        # Simulate bridge execution
        print(f"\n🌉 Executing Bridge: {bridge_id}")
        print(f"   Route: {from_network} → {to_network}")
        print(f"   Token: {token}")
        print(f"   Amount: {amount}")
        print(f"   Fee: ${fees['total_fee_usd']:.2f} USD")
        print(f"   Estimated Time: {fees['estimated_time_minutes']} minutes")
        print(f"   Route Type: {route['route_type']} ({route['hops']} hop{'s' if route['hops'] > 1 else ''})")

        # Simulate transaction hashes
        tx.from_tx_hash = f"0x{'a' * 64}"
        tx.to_tx_hash = f"0x{'b' * 64}"
        tx.status = "completed"
        tx.completed_at = datetime.now().isoformat()

        # Track
        self.bridge_history.append(tx)
        self.total_volume_usd += fees["value_usd"]

        print(f"   ✅ Bridge completed!")
        print(f"   From TX: {tx.from_tx_hash[:16]}...")
        print(f"   To TX: {tx.to_tx_hash[:16]}...")

        return tx

    def bridge_mining_rewards(
        self,
        btc_amount: float,
        target_networks: List[str],
        recipient_address: str
    ) -> List[BridgeTransaction]:
        """
        Bridge Bitcoin mining rewards to multiple networks

        Args:
            btc_amount: Amount of BTC to bridge
            target_networks: List of target network names
            recipient_address: Recipient address on target networks

        Returns:
            List of bridge transactions
        """
        print("\n" + "=" * 80)
        print("BRIDGING MINING REWARDS TO MULTIPLE NETWORKS")
        print("=" * 80)

        print(f"\n📊 Mining Rewards Distribution:")
        print(f"   Total BTC: {btc_amount}")
        print(f"   Target Networks: {len(target_networks)}")
        print(f"   Per Network: {btc_amount / len(target_networks):.4f} BTC")

        transactions = []
        amount_per_network = btc_amount / len(target_networks)

        for network in target_networks:
            print(f"\n🔄 Bridging to {network}...")

            try:
                tx = self.execute_bridge(
                    from_network="bitcoin",
                    to_network=network,
                    token="WBTC",
                    amount=amount_per_network,
                    from_address="bc1qfzhx87ckhn4tnkswhsth56h0gm5we4hdq5wass",
                    to_address=recipient_address
                )
                transactions.append(tx)

            except Exception as e:
                print(f"   ❌ Error bridging to {network}: {e}")

        print("\n" + "=" * 80)
        print(f"✅ Bridged to {len(transactions)}/{len(target_networks)} networks")
        print("=" * 80)

        return transactions

    def get_token_price(self, token_symbol: str) -> Optional[float]:
        """
        Get token price in USD from CoinGecko

        Args:
            token_symbol: Token symbol (WBTC, ETH, etc.)

        Returns:
            Price in USD or None
        """
        coin_id_map = {
            "WBTC": "wrapped-bitcoin",
            "BTC": "bitcoin",
            "ETH": "ethereum",
            "MATIC": "matic-network",
            "AVAX": "avalanche-2",
            "BNB": "binancecoin"
        }

        coin_id = coin_id_map.get(token_symbol.upper())
        if not coin_id:
            return None

        return self.network_manager.get_price_from_coingecko(coin_id)

    def get_bridge_statistics(self) -> Dict[str, Any]:
        """Get bridge statistics"""
        total_bridges = len(self.bridge_history)
        successful = sum(1 for tx in self.bridge_history if tx.status == "completed")

        networks_used = set()
        for tx in self.bridge_history:
            networks_used.add(tx.from_network)
            networks_used.add(tx.to_network)

        total_fees = sum(tx.fee_usd or 0 for tx in self.bridge_history)

        return {
            "total_bridges": total_bridges,
            "successful_bridges": successful,
            "failed_bridges": total_bridges - successful,
            "networks_used": len(networks_used),
            "total_volume_usd": self.total_volume_usd,
            "total_fees_usd": total_fees,
            "average_fee_usd": total_fees / total_bridges if total_bridges > 0 else 0
        }

    def export_bridge_history(self, filename: str = "bridge_history.json") -> str:
        """
        Export bridge history to JSON file

        Args:
            filename: Output filename

        Returns:
            File path
        """
        data = {
            "exported_at": datetime.now().isoformat(),
            "statistics": self.get_bridge_statistics(),
            "transactions": [
                {
                    "bridge_id": tx.bridge_id,
                    "from_network": tx.from_network,
                    "to_network": tx.to_network,
                    "token": tx.token,
                    "amount": tx.amount,
                    "status": tx.status,
                    "from_tx_hash": tx.from_tx_hash,
                    "to_tx_hash": tx.to_tx_hash,
                    "fee_usd": tx.fee_usd,
                    "estimated_time_minutes": tx.estimated_time_minutes,
                    "started_at": tx.started_at,
                    "completed_at": tx.completed_at
                }
                for tx in self.bridge_history
            ]
        }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

        return filename


def main():
    """Main execution"""
    print("\n" + "=" * 80)
    print("NEXUS AGI - MULTI-NETWORK BRIDGE ORCHESTRATOR")
    print("=" * 80)

    orchestrator = MultiNetworkBridgeOrchestrator()

    # Example: Bridge 1 BTC from Bitcoin to multiple networks
    print("\n📊 Example: Distribute 1 BTC across multiple networks\n")

    target_networks = [
        "ethereum_mainnet",
        "arbitrum_mainnet",
        "polygon_mainnet",
        "base_mainnet",
        "avalanche_mainnet"
    ]

    recipient = "0x24f6b1ce11c57d40b542f91ac85fa9eb61f78771"

    transactions = orchestrator.bridge_mining_rewards(
        btc_amount=1.0,
        target_networks=target_networks,
        recipient_address=recipient
    )

    # Display statistics
    stats = orchestrator.get_bridge_statistics()

    print("\n📈 Bridge Statistics:")
    print(f"   Total Bridges: {stats['total_bridges']}")
    print(f"   Successful: {stats['successful_bridges']}")
    print(f"   Networks Used: {stats['networks_used']}")
    print(f"   Total Volume: ${stats['total_volume_usd']:.2f} USD")
    print(f"   Total Fees: ${stats['total_fees_usd']:.2f} USD")
    print(f"   Average Fee: ${stats['average_fee_usd']:.2f} USD")

    # Export history
    export_file = orchestrator.export_bridge_history()
    print(f"\n💾 Bridge history exported to: {export_file}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
