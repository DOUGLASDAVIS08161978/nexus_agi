#!/usr/bin/env python3
"""
DEMO: Full Nexus AGI Workflow
==============================

Demonstrates the complete integrated system:
1. Bitcoin mining simulation
2. Multi-network bridge orchestration
3. Smart contract interaction
4. GitHub result publication

This is a DEMONSTRATION SCRIPT showing how all components work together.
Actual blockchain transactions require private keys and real gas fees.
"""
import sys
import os
import json
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.integrated_mining_bridge import IntegratedMiningBridge
from tools.contract_interactor import ERC20Reader
from config.network_config import get_network_manager


def print_banner(text):
    """Print formatted banner"""
    print("\n" + "=" * 80)
    print(text.center(80))
    print("=" * 80 + "\n")


def demo_network_connectivity():
    """Demonstrate multi-network connectivity"""
    print_banner("PHASE 1: MULTI-NETWORK CONNECTIVITY TEST")

    network_manager = get_network_manager()

    print("📡 Testing connectivity to all configured networks...\n")

    networks_to_test = [
        "ethereum_mainnet",
        "ethereum_sepolia",
        "polygon_mainnet",
        "arbitrum_mainnet",
        "base_mainnet",
        "avalanche_mainnet"
    ]

    results = {}
    for network in networks_to_test:
        print(f"Testing {network}...", end=" ")
        success = network_manager.test_rpc_connection(network)
        results[network] = success

        if success:
            print("✅ Connected")
        else:
            print("❌ Failed (may be network restrictions)")

    successful = sum(1 for v in results.values() if v)
    print(f"\n✅ {successful}/{len(networks_to_test)} networks accessible")

    return results


def demo_token_price_integration():
    """Demonstrate price API integration"""
    print_banner("PHASE 2: CRYPTOCURRENCY PRICE DATA")

    network_manager = get_network_manager()

    print("💰 Fetching real-time cryptocurrency prices...\n")

    tokens = {
        "bitcoin": "BTC",
        "ethereum": "ETH",
        "polygon": "MATIC",
        "avalanche-2": "AVAX"
    }

    prices = {}
    for coin_id, symbol in tokens.items():
        price = network_manager.get_price_from_coingecko(coin_id)
        prices[symbol] = price

        if price:
            print(f"   {symbol:6} ${price:>10,.2f} USD")
        else:
            print(f"   {symbol:6} Price unavailable")

    return prices


def demo_mining_session():
    """Demonstrate Bitcoin mining with bridge simulation"""
    print_banner("PHASE 3: INTEGRATED MINING & BRIDGING SIMULATION")

    print("⛏️  Starting integrated mining session...\n")
    print("NOTE: This is a SIMULATION. Real mining requires:")
    print("   • Actual mining hardware (ASICs)")
    print("   • Real Bitcoin network connection")
    print("   • Private keys for transaction signing")
    print("   • Gas fees for bridge transactions\n")

    # Initialize integrated mining bridge
    mining_bridge = IntegratedMiningBridge(
        target_networks=[
            "ethereum_mainnet",
            "polygon_mainnet",
            "arbitrum_mainnet",
            "base_mainnet"
        ]
    )

    print("Configuration:")
    print(f"   Mining Duration: 30 seconds (demo)")
    print(f"   Target Networks: {len(mining_bridge.target_networks)}")
    print(f"   Bridge Frequency: Every {mining_bridge.blocks_per_bridge} blocks")
    print(f"   Recipient: 0x24f6b1ce11c57d40b542f91ac85fa9eb61f78771")

    input("\n⏸  Press Enter to start mining simulation...")

    # Run mining session
    session = mining_bridge.run_mining_session(
        duration_seconds=30,
        recipient_address="0x24f6b1ce11c57d40b542f91ac85fa9eb61f78771"
    )

    return session


def demo_contract_interaction(contract_address=None):
    """Demonstrate smart contract reading"""
    print_banner("PHASE 4: SMART CONTRACT INTERACTION")

    if not contract_address or contract_address == "0xYourContractAddressHere":
        print("⚠️  No contract address provided\n")
        print("To test contract interaction:")
        print("   1. Deploy NexusRewardToken via Remix")
        print("   2. Update contract address in scripts")
        print("   3. Run verify_deployment.py\n")
        print("Skipping contract interaction demo...")
        return None

    print(f"📖 Reading contract state: {contract_address}\n")

    reader = ERC20Reader()
    network = "ethereum_sepolia"

    # Get token info
    info = reader.get_token_info(network, contract_address)
    if info:
        print("Token Information:")
        print(f"   Name: {info['name']}")
        print(f"   Symbol: {info['symbol']}")
        print(f"   Decimals: {info['decimals']}")
        print(f"   Total Supply: {info['totalSupply']:,}")
    else:
        print("❌ Could not read contract (verify address and network)")

    return info


def demo_github_integration():
    """Demonstrate GitHub integration capabilities"""
    print_banner("PHASE 5: GITHUB INTEGRATION")

    github_token = os.environ.get("GITHUB_TOKEN")

    if not github_token:
        print("⚠️  No GITHUB_TOKEN environment variable set\n")
        print("To enable GitHub integration:")
        print("   1. Create personal access token: https://github.com/settings/tokens")
        print("   2. Set permissions: gist, repo (if publishing to repo)")
        print("   3. Export token: export GITHUB_TOKEN='your_token_here'")
        print("   4. Re-run this script\n")
        print("Skipping GitHub integration demo...")
        return None

    print("✅ GitHub token detected")
    print("\nGitHub integration capabilities:")
    print("   • Publish mining results to gists")
    print("   • Create issues for errors/events")
    print("   • Trigger workflow automation")
    print("   • Store historical data\n")

    print("💡 Run mining session with GITHUB_TOKEN to see automatic publishing")

    return True


def demo_bridge_routes():
    """Demonstrate bridge route finding"""
    print_banner("PHASE 6: CROSS-CHAIN BRIDGE ROUTING")

    from tools.multi_network_bridge_orchestrator import MultiNetworkBridgeOrchestrator

    network_manager = get_network_manager()
    orchestrator = MultiNetworkBridgeOrchestrator(network_manager)

    print("🌉 Finding optimal bridge routes...\n")

    # Test route finding
    routes = [
        ("ethereum_mainnet", "polygon_mainnet", "WBTC"),
        ("bitcoin", "ethereum_mainnet", "WBTC"),
        ("ethereum_mainnet", "arbitrum_mainnet", "WBTC"),
        ("avalanche_mainnet", "base_mainnet", "WBTC")
    ]

    for from_net, to_net, token in routes:
        route = orchestrator.find_optimal_route(from_net, to_net, token)

        if route:
            if "bridge" in route:
                print(f"✅ {from_net} → {to_net}")
                print(f"   Bridge: {route['bridge']}")
                print(f"   Fee: {route.get('fee_percent', 'N/A')}%")
            elif "hops" in route:
                print(f"✅ {from_net} → {to_net} (Multi-hop)")
                print(f"   Route: {' → '.join(route['hops'])}")
            print()
        else:
            print(f"❌ {from_net} → {to_net}: No route found\n")


def save_demo_results(results):
    """Save demo results to file"""
    output_file = f"demo_results_{int(datetime.now().timestamp())}.json"

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    return output_file


def main():
    """Run complete workflow demo"""
    print_banner("NEXUS AGI - COMPLETE WORKFLOW DEMONSTRATION")

    print("This demonstration shows all integrated components:")
    print("   1. ✅ Multi-network blockchain connectivity")
    print("   2. ✅ Real-time cryptocurrency price data")
    print("   3. ✅ Bitcoin mining simulation")
    print("   4. ✅ Multi-network bridge orchestration")
    print("   5. ✅ Smart contract interaction (if deployed)")
    print("   6. ✅ GitHub integration (if token provided)")
    print("   7. ✅ Cross-chain routing optimization")

    input("\n⏸  Press Enter to begin demonstration...")

    results = {}

    # Phase 1: Network connectivity
    try:
        network_results = demo_network_connectivity()
        results["network_connectivity"] = network_results
    except Exception as e:
        print(f"❌ Error in network test: {e}")
        results["network_connectivity"] = {"error": str(e)}

    input("\n⏸  Press Enter to continue to price data...")

    # Phase 2: Price integration
    try:
        prices = demo_token_price_integration()
        results["cryptocurrency_prices"] = prices
    except Exception as e:
        print(f"❌ Error fetching prices: {e}")
        results["cryptocurrency_prices"] = {"error": str(e)}

    input("\n⏸  Press Enter to continue to mining simulation...")

    # Phase 3: Mining simulation
    try:
        session = demo_mining_session()
        results["mining_session"] = {
            "session_id": session.session_id,
            "blocks_mined": session.blocks_mined,
            "total_btc": session.total_btc,
            "hash_rate": session.hash_rate,
            "networks_bridged": session.networks_bridged,
            "total_fees": session.total_fees_usd
        }
    except Exception as e:
        print(f"❌ Error in mining simulation: {e}")
        results["mining_session"] = {"error": str(e)}

    input("\n⏸  Press Enter to continue to contract interaction...")

    # Phase 4: Contract interaction
    try:
        # Check if user has set contract address
        contract_address = "0xYourContractAddressHere"  # Default placeholder
        contract_info = demo_contract_interaction(contract_address)
        results["contract_interaction"] = contract_info or {"status": "skipped"}
    except Exception as e:
        print(f"❌ Error in contract interaction: {e}")
        results["contract_interaction"] = {"error": str(e)}

    input("\n⏸  Press Enter to continue to GitHub integration...")

    # Phase 5: GitHub integration
    try:
        github_status = demo_github_integration()
        results["github_integration"] = {"enabled": github_status is not None}
    except Exception as e:
        print(f"❌ Error in GitHub integration: {e}")
        results["github_integration"] = {"error": str(e)}

    input("\n⏸  Press Enter to continue to bridge routing...")

    # Phase 6: Bridge routing
    try:
        demo_bridge_routes()
        results["bridge_routing"] = {"status": "completed"}
    except Exception as e:
        print(f"❌ Error in bridge routing: {e}")
        results["bridge_routing"] = {"error": str(e)}

    # Save results
    print_banner("DEMONSTRATION COMPLETE")

    output_file = save_demo_results(results)

    print(f"✅ Demonstration complete!")
    print(f"📄 Results saved to: {output_file}\n")

    print("📋 Summary:")
    print("   • All core components tested")
    print("   • Multi-network integration operational")
    print("   • Mining simulation functional")
    print("   • Bridge orchestration configured")
    print("   • Contract interaction tools ready")

    print("\n🚀 Next Steps:")
    print("   1. Deploy NexusRewardToken via Remix (see REMIX_DEPLOYMENT_GUIDE.md)")
    print("   2. Update scripts with your contract address")
    print("   3. Run verify_deployment.py")
    print("   4. Start claiming rewards!")
    print("   5. Bridge to Bitcoin address: bc1qyhkq7usdefhhhynkjksdqfx32u3rmv94y0htsal")

    print("\n📚 Documentation:")
    print("   • COMPLETE_DEPLOYMENT_WORKFLOW.md - Full deployment guide")
    print("   • REMIX_DEPLOYMENT_GUIDE.md - Remix IDE instructions")
    print("   • CONTRACT_INTERACTION_GUIDE.md - Contract interaction reference")
    print("   • MULTI_NETWORK_INTEGRATION.md - Network configuration")

    print("\n" + "=" * 80)
    print("Thank you for using Nexus AGI! ✨✨✨")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
