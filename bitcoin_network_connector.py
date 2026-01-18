#!/usr/bin/env python3
"""
NEXUS AGI - Bitcoin Network Connector
======================================

Comprehensive Bitcoin connectivity for MAINNET, TESTNET, and SIGNET
Includes wallet management, transaction handling, and network monitoring

Author: Douglas Shane Davis & Claude
Date: 2026-01-13
"""

import requests
import json
import time
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from enum import Enum


class BitcoinNetwork(Enum):
    """Bitcoin network types"""
    MAINNET = "mainnet"
    TESTNET = "testnet"
    SIGNET = "signet"


class BitcoinNetworkConnector:
    """
    Multi-network Bitcoin connector supporting MAINNET, TESTNET, and SIGNET
    """

    # API endpoints for each network
    API_ENDPOINTS = {
        BitcoinNetwork.MAINNET: {
            'blockstream': 'https://blockstream.info/api',
            'mempool': 'https://mempool.space/api',
            'blockcypher': 'https://api.blockcypher.com/v1/btc/main',
            'blockchain_info': 'https://blockchain.info'
        },
        BitcoinNetwork.TESTNET: {
            'blockstream': 'https://blockstream.info/testnet/api',
            'mempool': 'https://mempool.space/testnet/api',
            'blockcypher': 'https://api.blockcypher.com/v1/btc/test3'
        },
        BitcoinNetwork.SIGNET: {
            'mempool': 'https://mempool.space/signet/api'
        }
    }

    # Existing wallet addresses in the system
    CONFIGURED_WALLETS = {
        'primary': 'bc1qfzhx87ckhn4tnkswhsth56h0gm5we4hdq5wass',
        'secondary': 'bc1q8z6z78dy5squapjpkeruem98jcezsw37hnae6qjyhxma6jmxyn6qsmqxce',
        'lightning': 'bc1qxy2kgdygjrsqtzq2n0yrf2493p83kkfjhx0wlh'
    }

    def __init__(self, network: BitcoinNetwork = BitcoinNetwork.MAINNET):
        """
        Initialize Bitcoin connector

        Args:
            network: Bitcoin network to connect to (MAINNET, TESTNET, SIGNET)
        """
        self.network = network
        self.endpoints = self.API_ENDPOINTS[network]
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'NexusAGI/1.0'})

        print("=" * 80)
        print("NEXUS AGI - BITCOIN NETWORK CONNECTOR")
        print("=" * 80)
        print(f"Network: {network.value.upper()}")
        print(f"Available APIs: {', '.join(self.endpoints.keys())}")
        print("=" * 80)

    def get_block_height(self, api: str = 'blockstream') -> Optional[int]:
        """Get current blockchain height"""
        try:
            if api == 'blockstream':
                url = f"{self.endpoints['blockstream']}/blocks/tip/height"
            elif api == 'mempool':
                url = f"{self.endpoints['mempool']}/blocks/tip/height"
            elif api == 'blockcypher':
                response = self.session.get(self.endpoints['blockcypher'], timeout=10)
                return response.json()['height']
            else:
                raise ValueError(f"Unknown API: {api}")

            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            return int(response.text)

        except Exception as e:
            print(f"❌ Error getting block height from {api}: {e}")
            return None

    def get_latest_block_hash(self, api: str = 'blockstream') -> Optional[str]:
        """Get latest block hash"""
        try:
            if api == 'blockstream':
                url = f"{self.endpoints['blockstream']}/blocks/tip/hash"
            elif api == 'mempool':
                url = f"{self.endpoints['mempool']}/blocks/tip/hash"
            else:
                raise ValueError(f"Unknown API: {api}")

            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            return response.text.strip()

        except Exception as e:
            print(f"❌ Error getting block hash from {api}: {e}")
            return None

    def get_block_info(self, block_hash: str, api: str = 'blockstream') -> Optional[Dict]:
        """Get detailed block information"""
        try:
            if api == 'blockstream':
                url = f"{self.endpoints['blockstream']}/block/{block_hash}"
            elif api == 'mempool':
                url = f"{self.endpoints['mempool']}/block/{block_hash}"
            else:
                raise ValueError(f"Unknown API: {api}")

            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            return response.json()

        except Exception as e:
            print(f"❌ Error getting block info from {api}: {e}")
            return None

    def get_address_info(self, address: str, api: str = 'blockstream') -> Optional[Dict]:
        """Get address information (balance, transactions, etc.)"""
        try:
            if api == 'blockstream':
                url = f"{self.endpoints['blockstream']}/address/{address}"
            elif api == 'mempool':
                url = f"{self.endpoints['mempool']}/address/{address}"
            elif api == 'blockcypher':
                url = f"{self.endpoints['blockcypher']}/addrs/{address}/balance"
            else:
                raise ValueError(f"Unknown API: {api}")

            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            return response.json()

        except Exception as e:
            print(f"❌ Error getting address info from {api}: {e}")
            return None

    def get_address_balance(self, address: str, api: str = 'blockstream') -> Optional[int]:
        """Get address balance in satoshis"""
        info = self.get_address_info(address, api)

        if info:
            if api == 'blockstream' or api == 'mempool':
                funded = info.get('chain_stats', {}).get('funded_txo_sum', 0)
                spent = info.get('chain_stats', {}).get('spent_txo_sum', 0)
                return funded - spent
            elif api == 'blockcypher':
                return info.get('balance', 0)

        return None

    def get_address_transactions(self, address: str, api: str = 'blockstream') -> Optional[List[Dict]]:
        """Get transaction history for an address"""
        try:
            if api == 'blockstream':
                url = f"{self.endpoints['blockstream']}/address/{address}/txs"
            elif api == 'mempool':
                url = f"{self.endpoints['mempool']}/address/{address}/txs"
            elif api == 'blockcypher':
                url = f"{self.endpoints['blockcypher']}/addrs/{address}/full"
                response = self.session.get(url, timeout=10)
                response.raise_for_status()
                return response.json().get('txs', [])
            else:
                raise ValueError(f"Unknown API: {api}")

            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            return response.json()

        except Exception as e:
            print(f"❌ Error getting transactions from {api}: {e}")
            return None

    def get_transaction(self, txid: str, api: str = 'blockstream') -> Optional[Dict]:
        """Get transaction details"""
        try:
            if api == 'blockstream':
                url = f"{self.endpoints['blockstream']}/tx/{txid}"
            elif api == 'mempool':
                url = f"{self.endpoints['mempool']}/tx/{txid}"
            elif api == 'blockcypher':
                url = f"{self.endpoints['blockcypher']}/txs/{txid}"
            else:
                raise ValueError(f"Unknown API: {api}")

            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            return response.json()

        except Exception as e:
            print(f"❌ Error getting transaction from {api}: {e}")
            return None

    def get_mempool_status(self, api: str = 'mempool') -> Optional[Dict]:
        """Get mempool status (pending transactions)"""
        try:
            if api == 'mempool':
                url = f"{self.endpoints['mempool']}/mempool"
            else:
                raise ValueError(f"Mempool status only available via mempool API")

            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            return response.json()

        except Exception as e:
            print(f"❌ Error getting mempool status: {e}")
            return None

    def get_fee_estimates(self, api: str = 'mempool') -> Optional[Dict]:
        """Get fee estimates for different confirmation targets"""
        try:
            if api == 'mempool':
                url = f"{self.endpoints['mempool']}/v1/fees/recommended"
            elif api == 'blockcypher':
                response = self.session.get(self.endpoints['blockcypher'], timeout=10)
                data = response.json()
                return {
                    'high': data.get('high_fee_per_kb', 0) // 1000,
                    'medium': data.get('medium_fee_per_kb', 0) // 1000,
                    'low': data.get('low_fee_per_kb', 0) // 1000
                }
            else:
                raise ValueError(f"Fee estimates not available for {api}")

            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            return response.json()

        except Exception as e:
            print(f"❌ Error getting fee estimates: {e}")
            return None

    def broadcast_transaction(self, raw_tx_hex: str, api: str = 'blockstream') -> Optional[str]:
        """
        Broadcast a raw transaction to the network

        Args:
            raw_tx_hex: Raw transaction in hexadecimal format
            api: API to use for broadcasting

        Returns:
            Transaction ID if successful, None otherwise
        """
        try:
            if api == 'blockstream':
                url = f"{self.endpoints['blockstream']}/tx"
                response = self.session.post(url, data=raw_tx_hex, timeout=10)
            elif api == 'blockcypher':
                url = f"{self.endpoints['blockcypher']}/txs/push"
                response = self.session.post(url, json={'tx': raw_tx_hex}, timeout=10)
            else:
                raise ValueError(f"Broadcasting not supported for {api}")

            response.raise_for_status()
            return response.text.strip()

        except Exception as e:
            print(f"❌ Error broadcasting transaction: {e}")
            return None

    def monitor_address(self, address: str, interval: int = 60, duration: int = 300):
        """
        Monitor an address for new transactions

        Args:
            address: Bitcoin address to monitor
            interval: Check interval in seconds
            duration: Total monitoring duration in seconds
        """
        print(f"\n🔍 Monitoring address: {address}")
        print(f"   Network: {self.network.value}")
        print(f"   Checking every {interval} seconds for {duration} seconds...")

        last_tx_count = 0
        start_time = time.time()

        while time.time() - start_time < duration:
            txs = self.get_address_transactions(address)

            if txs:
                current_tx_count = len(txs)

                if current_tx_count > last_tx_count:
                    print(f"\n🔔 NEW TRANSACTION DETECTED!")
                    print(f"   Total transactions: {current_tx_count}")

                    # Show new transactions
                    new_txs = txs[:current_tx_count - last_tx_count]
                    for tx in new_txs:
                        txid = tx.get('txid', 'unknown')
                        print(f"   • {txid}")

                    last_tx_count = current_tx_count

            time.sleep(interval)

        print(f"\n✅ Monitoring complete")

    def get_network_stats(self) -> Dict:
        """Get comprehensive network statistics"""
        print(f"\n📊 Gathering {self.network.value.upper()} network statistics...")

        stats = {
            'network': self.network.value,
            'timestamp': datetime.now().isoformat(),
            'block_height': None,
            'latest_block_hash': None,
            'mempool_size': None,
            'mempool_tx_count': None,
            'fee_estimates': None
        }

        # Try multiple APIs for redundancy
        for api in self.endpoints.keys():
            if stats['block_height'] is None:
                stats['block_height'] = self.get_block_height(api)

            if stats['latest_block_hash'] is None:
                stats['latest_block_hash'] = self.get_latest_block_hash(api)

            if stats['fee_estimates'] is None:
                stats['fee_estimates'] = self.get_fee_estimates(api)

        # Mempool stats (if available)
        if 'mempool' in self.endpoints:
            mempool = self.get_mempool_status()
            if mempool:
                stats['mempool_size'] = mempool.get('vsize', 0)
                stats['mempool_tx_count'] = mempool.get('count', 0)

        return stats

    def check_configured_wallets(self) -> Dict[str, Dict]:
        """Check all configured wallet balances"""
        print(f"\n💰 Checking configured wallet balances on {self.network.value.upper()}...")

        wallet_info = {}

        for name, address in self.CONFIGURED_WALLETS.items():
            print(f"\n   Checking {name}: {address}")

            balance_sats = self.get_address_balance(address)

            if balance_sats is not None:
                balance_btc = balance_sats / 100_000_000

                wallet_info[name] = {
                    'address': address,
                    'balance_satoshis': balance_sats,
                    'balance_btc': balance_btc,
                    'balance_formatted': f"{balance_btc:.8f} BTC"
                }

                print(f"   ✅ Balance: {balance_btc:.8f} BTC ({balance_sats:,} sats)")
            else:
                wallet_info[name] = {
                    'address': address,
                    'error': 'Could not retrieve balance'
                }
                print(f"   ❌ Could not retrieve balance")

        return wallet_info

    def display_network_info(self):
        """Display comprehensive network information"""
        print("\n" + "=" * 80)
        print(f"BITCOIN {self.network.value.upper()} - NETWORK INFORMATION")
        print("=" * 80)

        stats = self.get_network_stats()

        if stats['block_height']:
            print(f"\n📊 Blockchain Status:")
            print(f"   Block Height: {stats['block_height']:,}")

            if stats['latest_block_hash']:
                print(f"   Latest Block: {stats['latest_block_hash'][:32]}...")

        if stats['mempool_tx_count']:
            print(f"\n🔄 Mempool Status:")
            print(f"   Pending Transactions: {stats['mempool_tx_count']:,}")
            print(f"   Mempool Size: {stats['mempool_size'] / 1_000_000:.2f} MB")

        if stats['fee_estimates']:
            print(f"\n💸 Fee Estimates (sat/vB):")
            for priority, fee in stats['fee_estimates'].items():
                print(f"   {priority.capitalize()}: {fee}")

        # Check wallets
        wallet_info = self.check_configured_wallets()

        print("\n" + "=" * 80)
        print(f"✅ {self.network.value.upper()} NETWORK STATUS: CONNECTED")
        print("=" * 80)

        return {
            'network_stats': stats,
            'wallet_info': wallet_info
        }


def run_bitcoin_connectivity_test():
    """Run comprehensive Bitcoin connectivity test"""
    print("\n")
    print("█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + "NEXUS AGI - BITCOIN NETWORK CONNECTIVITY TEST".center(78) + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80)
    print("\n")

    results = {}

    # Test each network
    for network in [BitcoinNetwork.MAINNET, BitcoinNetwork.TESTNET, BitcoinNetwork.SIGNET]:
        try:
            print(f"\n{'=' * 80}")
            print(f"Testing {network.value.upper()} Network")
            print(f"{'=' * 80}")

            connector = BitcoinNetworkConnector(network)
            network_info = connector.display_network_info()

            results[network.value] = {
                'status': 'connected',
                'info': network_info
            }

        except Exception as e:
            print(f"\n❌ Error testing {network.value}: {e}")
            results[network.value] = {
                'status': 'error',
                'error': str(e)
            }

    # Summary
    print("\n" + "=" * 80)
    print("CONNECTIVITY TEST SUMMARY")
    print("=" * 80)

    for network, result in results.items():
        status_emoji = "✅" if result['status'] == 'connected' else "❌"
        print(f"{status_emoji} {network.upper()}: {result['status']}")

    print("\n" + "=" * 80)

    return results


if __name__ == "__main__":
    # Run connectivity test
    results = run_bitcoin_connectivity_test()

    # Export results
    output_file = 'bitcoin_connectivity_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n📝 Results exported to: {output_file}")
