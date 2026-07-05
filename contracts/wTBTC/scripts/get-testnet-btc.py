#!/usr/bin/env python3
"""
Smart Testnet BTC Acquisition Tool
Gets you testnet BTC FAST - no mining required!

For testing your wTBTC bridge.
"""

import requests
import json
import time
from datetime import datetime


class TestnetBTCGetter:
    """
    Smart tool to get testnet BTC through multiple methods
    """

    FAUCETS = [
        {
            "name": "Testnet Faucet",
            "url": "https://testnet-faucet.com/btc-testnet/",
            "amount": "0.01 tBTC",
            "speed": "Instant",
            "api": False
        },
        {
            "name": "CoinFaucet",
            "url": "https://coinfaucet.eu/en/btc-testnet/",
            "amount": "0.001 tBTC",
            "speed": "Fast",
            "api": False
        },
        {
            "name": "Bitcoin Faucet",
            "url": "https://bitcoinfaucet.uo1.net/",
            "amount": "0.001 tBTC",
            "speed": "Fast",
            "api": False
        },
        {
            "name": "Testnet QC",
            "url": "https://testnet.qc.to/",
            "amount": "Variable",
            "speed": "Medium",
            "api": False
        }
    ]

    def __init__(self):
        self.total_received = 0.0

    def print_header(self):
        """Print welcome header"""
        print("=" * 70)
        print("💰 TESTNET BTC ACQUISITION TOOL")
        print("=" * 70)
        print()
        print("Purpose: Get testnet BTC for wTBTC bridge testing")
        print()

    def print_faucet_list(self):
        """Display available faucets"""
        print("📋 Available Testnet BTC Faucets:")
        print("-" * 70)
        print()

        for i, faucet in enumerate(self.FAUCETS, 1):
            print(f"{i}. {faucet['name']}")
            print(f"   URL: {faucet['url']}")
            print(f"   Amount: {faucet['amount']}")
            print(f"   Speed: {faucet['speed']}")
            print()

    def get_testnet_address(self):
        """Get user's testnet address"""
        print("🔑 Enter Your Testnet Bitcoin Address:")
        print("-" * 70)
        print("Don't have one? Use a wallet like:")
        print("  • Electrum (testnet mode)")
        print("  • Bitcoin Core (testnet)")
        print("  • Sparrow Wallet (testnet)")
        print()

        while True:
            address = input("Testnet address (starts with 'tb1' or 'm/n'): ").strip()

            if not address:
                print("❌ Address cannot be empty")
                continue

            # Basic validation
            if address.startswith('tb1') or address.startswith('m') or address.startswith('n'):
                print(f"✅ Address: {address}")
                print()
                return address
            else:
                print("⚠️  Warning: Address doesn't look like testnet format")
                confirm = input("Use anyway? (y/n): ")
                if confirm.lower() == 'y':
                    return address

    def open_faucets(self, address):
        """Open all faucets in browser"""
        print("🌐 Opening Faucets:")
        print("-" * 70)
        print()
        print(f"Your address: {address}")
        print()
        print("📋 Instructions:")
        print("1. Browser will open each faucet")
        print("2. Paste your address")
        print("3. Complete any CAPTCHA")
        print("4. Click 'Send' or 'Get BTC'")
        print("5. Wait for confirmation")
        print()

        try:
            import webbrowser

            for i, faucet in enumerate(self.FAUCETS, 1):
                print(f"Opening {faucet['name']}...")
                webbrowser.open(faucet['url'])
                time.sleep(2)  # Delay between opens

            print()
            print("✅ All faucets opened!")
            print()

        except Exception as e:
            print(f"❌ Error opening browsers: {e}")
            print()
            print("Manual URLs:")
            for faucet in self.FAUCETS:
                print(f"  • {faucet['url']}")
            print()

    def check_balance(self, address):
        """Check testnet BTC balance"""
        print("💰 Checking Balance:")
        print("-" * 70)
        print()

        try:
            # Use blockchain.info testnet API
            url = f"https://blockstream.info/testnet/api/address/{address}"
            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                data = response.json()

                funded = data['chain_stats']['funded_txo_sum']
                spent = data['chain_stats']['spent_txo_sum']
                balance = (funded - spent) / 100000000  # Convert satoshis to BTC

                print(f"Address: {address}")
                print(f"Balance: {balance:.8f} tBTC")
                print(f"Transactions: {data['chain_stats']['tx_count']}")
                print()

                if balance > 0:
                    print(f"✅ You have {balance:.8f} tBTC!")
                else:
                    print("⏳ No balance yet - faucets may take a few minutes")

                print()
                print(f"View on explorer:")
                print(f"https://blockstream.info/testnet/address/{address}")
                print()

                return balance

            else:
                print(f"❌ API error: {response.status_code}")
                print("Check manually:")
                print(f"https://blockstream.info/testnet/address/{address}")
                print()
                return 0

        except Exception as e:
            print(f"❌ Error checking balance: {e}")
            print()
            return 0

    def bridge_integration(self, address, amount):
        """Show how to use with wTBTC bridge"""
        print("=" * 70)
        print("🌉 READY TO TEST wTBTC BRIDGE!")
        print("=" * 70)
        print()
        print(f"You have: {amount:.8f} tBTC")
        print(f"Address: {address}")
        print()
        print("📋 Next Steps:")
        print("-" * 70)
        print()
        print("1. LOCK TESTNET BTC")
        print("   Send tBTC to bridge's testnet address")
        print("   Include your Ethereum address in OP_RETURN or memo")
        print()
        print("2. BRIDGE OPERATOR DETECTS")
        print("   Automated system watches for your transaction")
        print("   Waits for confirmations (usually 6)")
        print()
        print("3. MINT wTBTC ON ETHEREUM")
        print("   Bridge operator calls:")
        print(f"   wTBTC.mint(yourAddress, {amount:.8f} ether, 'btc_tx_id')")
        print()
        print("4. USE wTBTC IN DEFI")
        print("   Trade on Uniswap")
        print("   Lend on Aave")
        print("   Provide liquidity")
        print()
        print("5. BURN wTBTC TO GET BTC BACK")
        print("   Call: wTBTC.burn(amount, 'your_btc_address')")
        print("   Bridge operator sends tBTC back")
        print()

    def run(self):
        """Main program flow"""
        self.print_header()

        # Show available faucets
        self.print_faucet_list()

        # Get user's address
        address = self.get_testnet_address()

        # Check current balance
        balance = self.check_balance(address)

        if balance >= 0.01:
            print(f"✅ You already have {balance:.8f} tBTC!")
            print()
            use_anyway = input("Get more from faucets? (y/n): ")
            if use_anyway.lower() != 'y':
                self.bridge_integration(address, balance)
                return

        # Open faucets
        self.open_faucets(address)

        # Wait and check
        print("⏳ Waiting for Faucet Transactions:")
        print("-" * 70)
        print("Usually takes 1-5 minutes...")
        print()

        for i in range(12):  # Check for 6 minutes
            time.sleep(30)  # Every 30 seconds

            print(f"Checking... ({i+1}/12)")
            new_balance = self.check_balance(address)

            if new_balance > balance:
                print(f"🎉 Received {new_balance - balance:.8f} tBTC!")
                balance = new_balance
                print()
                break

        # Show final status
        print()
        print("=" * 70)
        print("📊 FINAL STATUS")
        print("=" * 70)
        print()
        print(f"Address: {address}")
        print(f"Balance: {balance:.8f} tBTC")
        print()

        if balance >= 0.001:
            self.bridge_integration(address, balance)
        else:
            print("⏳ Still waiting for faucets?")
            print()
            print("Check status:")
            print(f"https://blockstream.info/testnet/address/{address}")
            print()
            print("Try more faucets or wait a few more minutes.")
            print()


def main():
    """Entry point"""
    print()
    getter = TestnetBTCGetter()
    getter.run()


if __name__ == "__main__":
    main()
