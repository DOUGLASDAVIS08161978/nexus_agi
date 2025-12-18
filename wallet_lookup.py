#!/usr/bin/env python3
"""
Quick Wallet Lookup Tool
Quickly search for device ID and network info for any wallet address

Usage:
    python3 wallet_lookup.py bc1qfzhx87ckhn4tnkswhsth56h0gm5we4hdq5wass
    python3 wallet_lookup.py --list
    python3 wallet_lookup.py --summary

Author: Douglas Shane Davis & Claude
"""

import sys
import json
from typing import Dict, Any


def load_wallet_data(filename: str = "wallet_tracking_data.json") -> Dict[str, Any]:
    """Load wallet tracking data"""
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: {filename} not found. Run wallet_device_tracker.py first.")
        sys.exit(1)


def print_wallet_info(wallet_address: str, data: Dict[str, Any]):
    """Print detailed information for a specific wallet"""
    wallets = data.get("wallets", {})

    if wallet_address not in wallets:
        print(f"❌ Wallet {wallet_address} not found in tracking system")
        print(f"\n📋 Available wallets:")
        for addr in wallets.keys():
            print(f"   - {addr}")
        return

    wallet = wallets[wallet_address]
    device = wallet["device_info"]
    network = wallet["network_info"]

    print("\n" + "="*100)
    print(f"🔍 WALLET LOOKUP RESULT")
    print("="*100)
    print(f"\n📍 Wallet Address: {wallet_address}")
    print(f"\n🖥️  DEVICE ID: {device['device_id']}")
    print(f"   Device Name:        {device['device_name']}")
    print(f"   Type:               {device['device_type']}")
    print(f"   Manufacturer:       {device['manufacturer']}")
    print(f"   Model:              {device['model']}")
    print(f"   Hash Rate:          {device['hash_rate']}")
    print(f"   Quantum Enabled:    {device['quantum_enabled']} ({device['quantum_qubits']} qubits)")

    print(f"\n🌐 NETWORK INFO:")
    print(f"   IP Address:         {network['ip_address']}")
    print(f"   MAC Address:        {network['mac_address']}")
    print(f"   Hostname:           {network['hostname']}")
    print(f"   Port:               {network['port']}")
    print(f"   Protocol:           {network['protocol']}")
    print(f"   Bandwidth:          {network['bandwidth_mbps']} Mbps")
    print(f"   Latency:            {network['latency_ms']} ms")
    print(f"   ISP:                {network['isp']}")
    print(f"   Location:           {network['geolocation']}")

    print(f"\n📊 STATS:")
    print(f"   Blocks Mined:       {wallet['total_blocks_mined']}")
    print(f"   BTC Earned:         {wallet['total_btc_earned']:.8f} BTC")
    print(f"   Status:             {wallet['status']}")
    print(f"   Last Active:        {wallet['last_active']}")
    if wallet.get('notes'):
        print(f"   Notes:              {wallet['notes']}")
    print("="*100)


def print_summary(data: Dict[str, Any]):
    """Print summary of all wallets"""
    wallets = data.get("wallets", {})

    print("\n" + "="*100)
    print("📊 WALLET TRACKING SUMMARY")
    print("="*100)
    print(f"\nLast Updated: {data.get('last_updated')}")
    print(f"Total Wallets: {data.get('total_wallets')}")

    print(f"\n{'Wallet Address':<52} {'Device ID':<20} {'IP Address':<16} {'Status':<10}")
    print("-"*100)

    for addr, wallet in wallets.items():
        device_id = wallet['device_info']['device_id']
        ip = wallet['network_info']['ip_address']
        status = wallet['status']
        print(f"{addr:<52} {device_id:<20} {ip:<16} {status:<10}")

    print("="*100)


def print_list(data: Dict[str, Any]):
    """Print list of all tracked wallets"""
    wallets = data.get("wallets", {})

    print("\n" + "="*100)
    print("📋 ALL TRACKED WALLET ADDRESSES")
    print("="*100)

    for i, (addr, wallet) in enumerate(wallets.items(), 1):
        notes = wallet.get('notes', 'No notes')
        print(f"\n{i}. {addr}")
        print(f"   Device ID: {wallet['device_info']['device_id']}")
        print(f"   Notes: {notes}")

    print("\n" + "="*100)


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("\n❌ Error: No wallet address or option provided")
        print("\nUsage:")
        print("  python3 wallet_lookup.py <wallet_address>")
        print("  python3 wallet_lookup.py --summary")
        print("  python3 wallet_lookup.py --list")
        print("\nExample:")
        print("  python3 wallet_lookup.py bc1qfzhx87ckhn4tnkswhsth56h0gm5we4hdq5wass")
        sys.exit(1)

    data = load_wallet_data()

    arg = sys.argv[1]

    if arg == "--summary" or arg == "-s":
        print_summary(data)
    elif arg == "--list" or arg == "-l":
        print_list(data)
    elif arg == "--help" or arg == "-h":
        print("\n🔍 Wallet Lookup Tool - Help")
        print("\nUsage:")
        print("  python3 wallet_lookup.py <wallet_address>  - Look up specific wallet")
        print("  python3 wallet_lookup.py --summary         - Show summary table")
        print("  python3 wallet_lookup.py --list            - List all wallets")
        print("  python3 wallet_lookup.py --help            - Show this help")
    else:
        # Assume it's a wallet address
        print_wallet_info(arg, data)


if __name__ == "__main__":
    main()
