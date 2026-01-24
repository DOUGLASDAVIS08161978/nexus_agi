#!/usr/bin/env python3
"""
Universal Monitor Launcher
Automatically adds all project addresses and starts monitoring
"""

import sys
import os

# Add the directory to path
sys.path.insert(0, '/home/user/nexus_agi')

from universal_wallet_monitor import UniversalWalletMonitor

def main():
    print("\n🚀 LAUNCHING UNIVERSAL WALLET MONITOR")
    print("=" * 70)

    monitor = UniversalWalletMonitor()

    # Add all addresses from our projects
    print("\n📍 Adding project addresses...\n")

    addresses_to_monitor = [
        # Monad Testnet Addresses
        ('0x24f6b1ce11c57d40b542f91ac85fa9eb61f78771', 'monad', 'Main Monad Wallet - WBTC Receiving'),
        ('0x0555E30da8f98308EdB960aa94C0Db47230d2B9c', 'monad', 'WBTC Token Contract'),

        # Ethereum Sepolia Testnet (if needed for testing)
        ('0x24f6b1ce11c57d40b542f91ac85fa9eb61f78771', 'sepolia', 'Sepolia Test Wallet'),
    ]

    for address, blockchain, label in addresses_to_monitor:
        print(f"  Adding {blockchain:10} | {address[:12]}... | {label}")
        monitor.add_address(address, blockchain, label)

    print("\n✓ All addresses added!")
    print("\n" + "=" * 70)
    print("INITIAL SCAN - Checking all addresses now...")
    print("=" * 70 + "\n")

    # Do initial scan
    from concurrent.futures import as_completed

    addresses = monitor.db.get_monitored_addresses()
    futures = []

    for addr, chain in addresses:
        future = monitor.thread_pool.submit(
            monitor.processor.process_address,
            addr,
            chain
        )
        futures.append((future, addr, chain))

    for future, addr, chain in futures:
        try:
            new_tx = future.result(timeout=30)
            print(f"✓ {chain:10} {addr[:12]}... - {new_tx} transactions")
        except Exception as e:
            print(f"✗ {chain:10} {addr[:12]}... - Error: {e}")

    print("\n" + "=" * 70)
    print("INITIAL SCAN COMPLETE")
    print("=" * 70)

    # Show menu
    print("\nWhat would you like to do?")
    print("  1) Start continuous monitoring (checks every 45 seconds)")
    print("  2) View transaction history")
    print("  3) Check balances only")
    print("  4) Generate full report")
    print("  5) Enter interactive mode")
    print("  6) Exit")

    choice = input("\nEnter choice (1-6): ").strip()

    if choice == '1':
        print("\n🔄 Starting continuous monitoring...")
        print("Press Ctrl+C to stop\n")
        monitor.monitor_loop()

    elif choice == '2':
        print("\n📜 TRANSACTION HISTORY")
        print("=" * 70)
        for addr, chain in addresses:
            txs = monitor.db.get_address_transactions(addr, chain, limit=5)
            print(f"\n🔗 {chain.upper()} - {addr}")
            if txs:
                for tx in txs:
                    from datetime import datetime
                    dt = datetime.fromtimestamp(tx['timestamp']).strftime('%Y-%m-%d %H:%M')
                    print(f"  [{dt}] {tx['direction']:10} {tx['amount']:12.8f} {tx['token_symbol']}")
            else:
                print("  No transactions found")
        print("\n" + "=" * 70)

    elif choice == '3':
        print("\n💰 CURRENT BALANCES")
        print("=" * 70)
        for addr, chain in addresses:
            if chain == 'bitcoin':
                balance = monitor.api.get_bitcoin_balance(addr)
                if balance is not None:
                    print(f"🔗 {chain:10} {addr[:16]}... : {balance:.8f} BTC")
            else:
                balance = monitor.api.get_evm_balance(addr, chain)
                if balance is not None:
                    symbol = 'MON' if chain == 'monad' else 'ETH'
                    print(f"🔗 {chain:10} {addr[:16]}... : {balance:.8f} {symbol}")

                # Check WBTC balance on Monad
                if chain == 'monad' and addr == '0x24f6b1ce11c57d40b542f91ac85fa9eb61f78771':
                    wbtc_balance = monitor.api.get_erc20_balance(
                        addr,
                        '0x0555E30da8f98308EdB960aa94C0Db47230d2B9c',
                        'monad'
                    )
                    if wbtc_balance is not None:
                        print(f"  {'':10} {'WBTC Token':<16}    : {wbtc_balance:.8f} WBTC")
        print("=" * 70)

    elif choice == '4':
        filename = 'wallet_report.json'
        monitor._generate_report(filename)
        print(f"\n✓ Full report saved to {filename}")

    elif choice == '5':
        monitor.interactive_mode()

    else:
        print("\n✓ Exiting")
        monitor.stop()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n✓ Stopped by user")
