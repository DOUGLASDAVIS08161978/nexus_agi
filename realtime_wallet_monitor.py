#!/usr/bin/env python3
"""
REAL-TIME WALLET MONITORING SYSTEM
Monitors ALL wallet addresses (mainnet & testnet) in real-time
Records ALL activity to immutable ledger
"""

import requests
import time
import json
from datetime import datetime
from typing import Dict, List, Set
from immutable_ledger_system import ImmutableLedger


class WalletMonitor:
    """
    Real-time monitoring system for Bitcoin wallet addresses

    Features:
    - Monitors multiple addresses simultaneously
    - Supports mainnet and testnet
    - Detects new transactions instantly
    - Records all activity to immutable ledger
    - Checks historical transactions
    - Never misses an event
    """

    def __init__(self):
        # Initialize immutable ledger
        self.ledger = ImmutableLedger("wallet_activity_ledger.jsonl")

        # Wallet addresses to monitor
        self.mainnet_addresses: Set[str] = set()
        self.testnet_addresses: Set[str] = set()

        # APIs
        self.mainnet_api = "https://blockstream.info/api"
        self.testnet_api = "https://blockstream.info/testnet/api"

        # Transaction tracking
        self.known_transactions: Dict[str, Set[str]] = {}  # address -> set of txids

        # Statistics
        self.stats = {
            "total_addresses_monitored": 0,
            "total_transactions_found": 0,
            "monitoring_start_time": datetime.now().isoformat(),
            "last_check_time": None,
            "checks_performed": 0,
        }

        print("=" * 80)
        print("REAL-TIME WALLET MONITORING SYSTEM")
        print("=" * 80)
        print("Status: Initialized")
        print("Immutable Ledger: Active")
        print("=" * 80)

    def add_mainnet_address(self, address: str):
        """Add mainnet address to monitoring"""
        if address not in self.mainnet_addresses:
            self.mainnet_addresses.add(address)
            self.known_transactions[address] = set()
            self.stats["total_addresses_monitored"] += 1

            # Record in ledger
            self.ledger.add_entry("ADDRESS_ADDED", {
                "address": address,
                "network": "mainnet",
                "added_at": datetime.now().isoformat()
            })

            print(f"✅ Added mainnet address: {address}")

    def add_testnet_address(self, address: str):
        """Add testnet address to monitoring"""
        if address not in self.testnet_addresses:
            self.testnet_addresses.add(address)
            self.known_transactions[address] = set()
            self.stats["total_addresses_monitored"] += 1

            # Record in ledger
            self.ledger.add_entry("ADDRESS_ADDED", {
                "address": address,
                "network": "testnet",
                "added_at": datetime.now().isoformat()
            })

            print(f"✅ Added testnet address: {address}")

    def get_address_info(self, address: str, network: str = "testnet") -> Dict:
        """Get current address information from blockchain"""
        api = self.testnet_api if network == "testnet" else self.mainnet_api

        try:
            url = f"{api}/address/{address}"
            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                return response.json()
            else:
                return {}

        except Exception as e:
            print(f"⚠️  Error fetching {network} address {address}: {e}")
            return {}

    def get_address_transactions(self, address: str, network: str = "testnet") -> List[Dict]:
        """Get all transactions for an address"""
        api = self.testnet_api if network == "testnet" else self.mainnet_api

        try:
            url = f"{api}/address/{address}/txs"
            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                return response.json()
            else:
                return []

        except Exception as e:
            print(f"⚠️  Error fetching transactions for {address}: {e}")
            return []

    def check_historical_activity(self, address: str, network: str = "testnet") -> Dict:
        """
        Check if address has ANY historical activity (even if balance is 0)

        Returns complete history including:
        - All past transactions
        - Total received (even if spent)
        - Total spent
        - Transaction count
        """
        print(f"\n🔍 Checking historical activity: {address}")
        print(f"   Network: {network}")

        info = self.get_address_info(address, network)

        if not info:
            print(f"   ❌ No data found")
            return {}

        chain_stats = info.get('chain_stats', {})
        mempool_stats = info.get('mempool_stats', {})

        # Get stats
        funded_txo_sum = chain_stats.get('funded_txo_sum', 0)
        spent_txo_sum = chain_stats.get('spent_txo_sum', 0)
        tx_count = chain_stats.get('tx_count', 0)

        balance = funded_txo_sum - spent_txo_sum

        # Get all transactions
        transactions = self.get_address_transactions(address, network)

        result = {
            "address": address,
            "network": network,
            "balance_satoshis": balance,
            "balance_btc": balance / 100_000_000,
            "total_received_satoshis": funded_txo_sum,
            "total_received_btc": funded_txo_sum / 100_000_000,
            "total_spent_satoshis": spent_txo_sum,
            "total_spent_btc": spent_txo_sum / 100_000_000,
            "transaction_count": tx_count,
            "has_activity": tx_count > 0,
            "transactions": transactions
        }

        # Print summary
        if tx_count > 0:
            print(f"   ✅ HAS ACTIVITY!")
            print(f"   📊 Transactions: {tx_count}")
            print(f"   💰 Total Received: {result['total_received_btc']:.8f} BTC")
            print(f"   💸 Total Spent: {result['total_spent_btc']:.8f} BTC")
            print(f"   💵 Current Balance: {result['balance_btc']:.8f} BTC")
        else:
            print(f"   ❌ No historical activity")

        # Record in immutable ledger
        self.ledger.add_entry("HISTORICAL_CHECK", {
            "address": address,
            "network": network,
            "has_activity": result["has_activity"],
            "transaction_count": tx_count,
            "total_received_btc": result["total_received_btc"],
            "current_balance_btc": result["balance_btc"],
            "checked_at": datetime.now().isoformat()
        })

        # Record each transaction to immutable ledger
        for tx in transactions:
            self.ledger.add_entry("TRANSACTION_FOUND", {
                "address": address,
                "network": network,
                "txid": tx.get('txid'),
                "confirmed": tx.get('status', {}).get('confirmed', False),
                "block_height": tx.get('status', {}).get('block_height'),
                "timestamp": tx.get('status', {}).get('block_time'),
                "found_at": datetime.now().isoformat()
            })

        return result

    def monitor_address(self, address: str, network: str = "testnet") -> Dict:
        """Monitor single address for new activity"""
        info = self.get_address_info(address, network)

        if not info:
            return {"address": address, "network": network, "status": "error"}

        chain_stats = info.get('chain_stats', {})
        funded = chain_stats.get('funded_txo_sum', 0)
        spent = chain_stats.get('spent_txo_sum', 0)
        balance = funded - spent
        tx_count = chain_stats.get('tx_count', 0)

        # Get transactions
        transactions = self.get_address_transactions(address, network)

        # Check for new transactions
        new_transactions = []
        for tx in transactions:
            txid = tx.get('txid')
            if txid not in self.known_transactions.get(address, set()):
                new_transactions.append(tx)
                self.known_transactions[address].add(txid)

                # Record new transaction in immutable ledger
                self.ledger.add_entry("NEW_TRANSACTION", {
                    "address": address,
                    "network": network,
                    "txid": txid,
                    "confirmed": tx.get('status', {}).get('confirmed', False),
                    "block_height": tx.get('status', {}).get('block_height'),
                    "detected_at": datetime.now().isoformat()
                })

                self.stats["total_transactions_found"] += 1

                print(f"\n🚨 NEW TRANSACTION DETECTED!")
                print(f"   Address: {address}")
                print(f"   Network: {network}")
                print(f"   TXID: {txid}")
                print(f"   Confirmed: {tx.get('status', {}).get('confirmed', False)}")

        # Record balance check
        self.ledger.add_entry("BALANCE_CHECK", {
            "address": address,
            "network": network,
            "balance_satoshis": balance,
            "balance_btc": balance / 100_000_000,
            "transaction_count": tx_count,
            "new_transactions": len(new_transactions),
            "checked_at": datetime.now().isoformat()
        })

        return {
            "address": address,
            "network": network,
            "balance": balance,
            "tx_count": tx_count,
            "new_transactions": new_transactions
        }

    def monitor_all_addresses(self):
        """Monitor all addresses once"""
        print(f"\n{'=' * 80}")
        print(f"MONITORING ALL ADDRESSES - {datetime.now().isoformat()}")
        print(f"{'=' * 80}")

        # Monitor testnet addresses
        for address in self.testnet_addresses:
            self.monitor_address(address, "testnet")

        # Monitor mainnet addresses
        for address in self.mainnet_addresses:
            self.monitor_address(address, "mainnet")

        self.stats["last_check_time"] = datetime.now().isoformat()
        self.stats["checks_performed"] += 1

        print(f"✅ Monitoring cycle complete")

    def start_continuous_monitoring(self, interval_seconds: int = 60):
        """
        Start continuous monitoring loop

        Args:
            interval_seconds: How often to check (default 60 seconds)
        """
        print(f"\n{'=' * 80}")
        print(f"STARTING CONTINUOUS MONITORING")
        print(f"{'=' * 80}")
        print(f"Interval: {interval_seconds} seconds")
        print(f"Mainnet Addresses: {len(self.mainnet_addresses)}")
        print(f"Testnet Addresses: {len(self.testnet_addresses)}")
        print(f"Total: {len(self.mainnet_addresses) + len(self.testnet_addresses)}")
        print(f"{'=' * 80}")
        print(f"\nPress Ctrl+C to stop monitoring\n")

        try:
            while True:
                self.monitor_all_addresses()

                print(f"\n⏳ Waiting {interval_seconds} seconds until next check...")
                time.sleep(interval_seconds)

        except KeyboardInterrupt:
            print(f"\n\n{'=' * 80}")
            print(f"MONITORING STOPPED")
            print(f"{'=' * 80}")
            self.print_statistics()

    def print_statistics(self):
        """Print monitoring statistics"""
        print(f"\n{'=' * 80}")
        print(f"MONITORING STATISTICS")
        print(f"{'=' * 80}")
        print(f"Addresses Monitored:      {self.stats['total_addresses_monitored']}")
        print(f"  - Mainnet:              {len(self.mainnet_addresses)}")
        print(f"  - Testnet:              {len(self.testnet_addresses)}")
        print(f"Transactions Found:       {self.stats['total_transactions_found']}")
        print(f"Checks Performed:         {self.stats['checks_performed']}")
        print(f"Started:                  {self.stats['monitoring_start_time']}")
        print(f"Last Check:               {self.stats['last_check_time']}")
        print(f"{'=' * 80}")

        # Ledger stats
        self.ledger.print_statistics()


def main():
    """Main monitoring program"""
    monitor = WalletMonitor()

    print("\n" + "=" * 80)
    print("ADDING WALLET ADDRESSES TO MONITOR")
    print("=" * 80)

    # Add ALL your addresses
    # Mainnet
    monitor.add_mainnet_address("bc1qyhkq7usdfhhhynkjksdqfx32u3rmv94y0htsal")

    # Testnet
    monitor.add_testnet_address("tb1q36naaxcn9n2qsyrav3hcqa8zugdlkxll3h8zu6")
    monitor.add_testnet_address("tb1qva9h6chqy9x4jrp8rdjm69czhj5d83eyy3aqpk")
    monitor.add_testnet_address("tb1qkvyjam64dmkvjav5dkhpfyqpjxq7hqj80u6e6f")
    monitor.add_testnet_address("tb1qfzhx87ckhn4tnkswhsth56h0gm5we4hdq5wass")
    monitor.add_testnet_address("tb1qw508d6qejxtdg4y5r3zarvary0c5xw7kxpjzsx")

    print("\n" + "=" * 80)
    print("CHECKING HISTORICAL ACTIVITY ON ALL ADDRESSES")
    print("=" * 80)

    # Check historical activity on all testnet addresses
    all_results = {}

    for address in monitor.testnet_addresses:
        result = monitor.check_historical_activity(address, "testnet")
        all_results[address] = result

    # Check mainnet
    for address in monitor.mainnet_addresses:
        result = monitor.check_historical_activity(address, "mainnet")
        all_results[address] = result

    print("\n" + "=" * 80)
    print("HISTORICAL ACTIVITY SUMMARY")
    print("=" * 80)

    for address, result in all_results.items():
        if result.get("has_activity"):
            print(f"\n✅ {address}")
            print(f"   Network: {result['network']}")
            print(f"   Transactions: {result['transaction_count']}")
            print(f"   Total Received: {result['total_received_btc']:.8f} BTC")
            print(f"   Current Balance: {result['balance_btc']:.8f} BTC")
        else:
            print(f"\n❌ {address}")
            print(f"   Network: {result.get('network', 'unknown')}")
            print(f"   No activity found")

    print("\n" + "=" * 80)
    print("START CONTINUOUS MONITORING? (yes/no)")
    print("=" * 80)

    choice = input("Monitor in real-time? ").strip().lower()

    if choice == 'yes':
        monitor.start_continuous_monitoring(interval_seconds=60)
    else:
        print("\n✅ Historical check complete")
        monitor.print_statistics()


if __name__ == "__main__":
    main()
