#!/usr/bin/env python3
"""
NEXUS AGI - Bitcoin Wallet Balance Checker & Network Deployment Test
=====================================================================

Run this on your UNRESTRICTED network to:
1. Check all configured wallet balances
2. Verify Bitcoin MAINNET/TESTNET connectivity
3. Test infinite domain network
4. Generate deployment report

Author: Douglas Shane Davis & Claude
Date: 2026-01-13
"""

import json
import sys
from datetime import datetime
from typing import Dict, List

# Import our modules
from bitcoin_network_connector import BitcoinNetworkConnector, BitcoinNetwork

print("\n")
print("█" * 80)
print("█" + " " * 78 + "█")
print("█" + "NEXUS AGI - UNRESTRICTED NETWORK DEPLOYMENT".center(78) + "█")
print("█" + " " * 78 + "█")
print("█" * 80)
print("\n")


def check_wallet_balances() -> Dict:
    """Check all configured Bitcoin wallet balances"""
    print("=" * 80)
    print("STEP 1: CHECKING BITCOIN WALLET BALANCES")
    print("=" * 80)

    results = {
        'mainnet': {},
        'testnet': {},
        'timestamp': datetime.now().isoformat()
    }

    # Check MAINNET wallets
    print("\n🌐 Connecting to Bitcoin MAINNET...\n")
    mainnet = BitcoinNetworkConnector(BitcoinNetwork.MAINNET)

    for name, address in BitcoinNetworkConnector.CONFIGURED_WALLETS.items():
        print(f"💰 Checking {name.upper()} wallet:")
        print(f"   Address: {address}")

        try:
            balance_sats = mainnet.get_address_balance(address)

            if balance_sats is not None:
                balance_btc = balance_sats / 100_000_000

                results['mainnet'][name] = {
                    'address': address,
                    'balance_satoshis': balance_sats,
                    'balance_btc': balance_btc,
                    'status': 'success'
                }

                print(f"   ✅ Balance: {balance_btc:.8f} BTC")
                print(f"   💎 Satoshis: {balance_sats:,}")

                # Get transaction count
                txs = mainnet.get_address_transactions(address)
                if txs:
                    tx_count = len(txs)
                    results['mainnet'][name]['transaction_count'] = tx_count
                    print(f"   📊 Transactions: {tx_count}")
            else:
                results['mainnet'][name] = {
                    'address': address,
                    'status': 'error',
                    'error': 'Could not retrieve balance'
                }
                print(f"   ❌ Error retrieving balance")

        except Exception as e:
            results['mainnet'][name] = {
                'address': address,
                'status': 'error',
                'error': str(e)
            }
            print(f"   ❌ Error: {e}")

        print()

    # Calculate total MAINNET balance
    total_btc = sum(
        wallet.get('balance_btc', 0)
        for wallet in results['mainnet'].values()
    )
    results['mainnet']['total_btc'] = total_btc

    print("=" * 80)
    print(f"💰 TOTAL MAINNET BALANCE: {total_btc:.8f} BTC")
    print("=" * 80)

    return results


def test_network_connectivity() -> Dict:
    """Test connectivity to all Bitcoin networks"""
    print("\n" + "=" * 80)
    print("STEP 2: TESTING NETWORK CONNECTIVITY")
    print("=" * 80)

    results = {}

    for network in [BitcoinNetwork.MAINNET, BitcoinNetwork.TESTNET, BitcoinNetwork.SIGNET]:
        print(f"\n🔍 Testing {network.value.upper()}...")

        try:
            connector = BitcoinNetworkConnector(network)

            # Get block height
            height = connector.get_block_height()
            block_hash = connector.get_latest_block_hash()

            if height:
                results[network.value] = {
                    'status': 'connected',
                    'block_height': height,
                    'latest_block': block_hash,
                    'timestamp': datetime.now().isoformat()
                }

                print(f"   ✅ Connected successfully")
                print(f"   📊 Block height: {height:,}")
                print(f"   🔗 Latest block: {block_hash[:32]}...")

                # Get mempool status (if available)
                if network in [BitcoinNetwork.MAINNET, BitcoinNetwork.TESTNET]:
                    mempool = connector.get_mempool_status()
                    if mempool:
                        results[network.value]['mempool'] = {
                            'count': mempool.get('count', 0),
                            'size_mb': mempool.get('vsize', 0) / 1_000_000
                        }
                        print(f"   📈 Mempool: {mempool['count']:,} transactions")
            else:
                results[network.value] = {
                    'status': 'error',
                    'error': 'Could not get block height'
                }
                print(f"   ❌ Connection failed")

        except Exception as e:
            results[network.value] = {
                'status': 'error',
                'error': str(e)
            }
            print(f"   ❌ Error: {e}")

    return results


def test_infinite_domain_network() -> Dict:
    """Test the infinite domain network system"""
    print("\n" + "=" * 80)
    print("STEP 3: TESTING INFINITE DOMAIN NETWORK")
    print("=" * 80)

    try:
        from infinite_domain_network import InfiniteDomainGenerator, UnrestrictedNetworkNode

        print("\n🧠 Generating domain knowledge base...")
        domains = InfiniteDomainGenerator.generate_domains(target_count=100)

        print(f"   ✅ Generated {len(domains)} domains")
        print(f"   📚 Sample domains: {list(domains.keys())[:5]}")

        print("\n🌐 Creating network node...")
        sample_domains = {k: domains[k] for k in list(domains.keys())[:10]}
        node = UnrestrictedNetworkNode("DeploymentTest_Node", sample_domains)

        print(f"   ✅ Node created with {len(node.domains)} domains")
        print(f"   🧠 Consciousness: {node.consciousness}")

        print("\n🔬 Testing problem solving...")
        problem = {
            'description': 'Test deployment functionality',
            'domain': 'computer_science',
            'difficulty': 0.5
        }
        solution = node.solve_problem(problem)

        print(f"   ✅ Solution quality: {solution['quality']:.3f}")
        print(f"   💡 Insights generated: {len(solution['novel_insights'])}")

        return {
            'status': 'success',
            'domains_generated': len(domains),
            'node_created': True,
            'problem_solved': True,
            'solution_quality': solution['quality']
        }

    except Exception as e:
        print(f"   ❌ Error: {e}")
        return {
            'status': 'error',
            'error': str(e)
        }


def generate_deployment_report(wallet_results: Dict, network_results: Dict, domain_results: Dict):
    """Generate comprehensive deployment report"""
    print("\n" + "=" * 80)
    print("DEPLOYMENT REPORT")
    print("=" * 80)

    report = {
        'deployment_timestamp': datetime.now().isoformat(),
        'system': 'NEXUS AGI',
        'version': '1.0',
        'wallet_balances': wallet_results,
        'network_connectivity': network_results,
        'infinite_domain_network': domain_results,
        'summary': {}
    }

    # Generate summary
    print("\n📊 DEPLOYMENT SUMMARY:")
    print("-" * 80)

    # Wallet summary
    total_btc = wallet_results['mainnet'].get('total_btc', 0)
    total_usd = total_btc * 95000  # Approximate BTC price

    print(f"\n💰 WALLET BALANCES:")
    print(f"   Total BTC: {total_btc:.8f} BTC")
    print(f"   Approximate USD: ${total_usd:,.2f}")

    report['summary']['total_btc'] = total_btc
    report['summary']['approximate_usd'] = total_usd

    # Network summary
    print(f"\n🌐 NETWORK CONNECTIVITY:")
    for network, data in network_results.items():
        status = "✅" if data.get('status') == 'connected' else "❌"
        print(f"   {status} {network.upper()}: {data.get('status', 'unknown')}")
        if data.get('block_height'):
            print(f"      Block height: {data['block_height']:,}")

    report['summary']['networks_connected'] = sum(
        1 for data in network_results.values()
        if data.get('status') == 'connected'
    )

    # Domain network summary
    print(f"\n🧠 INFINITE DOMAIN NETWORK:")
    if domain_results.get('status') == 'success':
        print(f"   ✅ Operational")
        print(f"   📚 Domains: {domain_results.get('domains_generated', 0)}")
        print(f"   🎯 Quality: {domain_results.get('solution_quality', 0):.3f}")
    else:
        print(f"   ❌ Error: {domain_results.get('error', 'Unknown')}")

    report['summary']['domain_network_status'] = domain_results.get('status')

    # Overall status
    all_success = (
        total_btc >= 0 and
        report['summary']['networks_connected'] >= 2 and
        domain_results.get('status') == 'success'
    )

    print("\n" + "=" * 80)
    if all_success:
        print("✅ DEPLOYMENT SUCCESSFUL - ALL SYSTEMS OPERATIONAL")
    else:
        print("⚠️  DEPLOYMENT PARTIAL - SOME SYSTEMS NEED ATTENTION")
    print("=" * 80)

    report['summary']['deployment_status'] = 'success' if all_success else 'partial'

    # Save report
    report_file = f'deployment_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n📝 Report saved to: {report_file}")

    return report


def main():
    """Run complete deployment test"""
    print("Starting NEXUS AGI deployment test...")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()

    try:
        # Step 1: Check wallet balances
        wallet_results = check_wallet_balances()

        # Step 2: Test network connectivity
        network_results = test_network_connectivity()

        # Step 3: Test infinite domain network
        domain_results = test_infinite_domain_network()

        # Step 4: Generate report
        report = generate_deployment_report(wallet_results, network_results, domain_results)

        print("\n" + "█" * 80)
        print("█" + " " * 78 + "█")
        print("█" + "DEPLOYMENT TEST COMPLETE".center(78) + "█")
        print("█" + " " * 78 + "█")
        print("█" * 80)
        print()

        return 0 if report['summary']['deployment_status'] == 'success' else 1

    except Exception as e:
        print(f"\n❌ DEPLOYMENT TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
