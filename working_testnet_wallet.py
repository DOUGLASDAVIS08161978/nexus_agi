#!/usr/bin/env python3
"""
WORKING Bitcoin Testnet Wallet & Transaction System
Uses bitcoinlib for REAL testnet transactions
"""

import sys
import json
import requests
from datetime import datetime
from bitcoinlib.wallets import Wallet, WalletError
from bitcoinlib.keys import Key

# ============================================================================
# CONFIGURATION
# ============================================================================

TESTNET_API = "https://blockstream.info/testnet/api"
WALLET_NAME = "nexus_testnet_wallet"

# Faucet URLs for getting FREE testnet Bitcoin
FAUCETS = [
    "https://testnet-faucet.mempool.co/",
    "https://coinfaucet.eu/en/btc-testnet/",
    "https://bitcoinfaucet.uo1.net/",
]


class TestnetWalletManager:
    """Complete testnet wallet management system"""

    def __init__(self, wallet_name: str = WALLET_NAME):
        self.wallet_name = wallet_name
        self.wallet = None

        print("=" * 80)
        print("NEXUS AGI - BITCOIN TESTNET WALLET SYSTEM")
        print("=" * 80)
        print(f"Wallet: {wallet_name}")
        print(f"Network: Bitcoin Testnet")
        print("=" * 80)

    def create_or_load_wallet(self) -> Wallet:
        """Create new wallet or load existing one"""
        try:
            # Try to load existing wallet
            self.wallet = Wallet(self.wallet_name)
            print(f"✅ Loaded existing wallet: {self.wallet_name}")

        except WalletError:
            # Create new wallet
            print(f"📝 Creating new testnet wallet: {self.wallet_name}")
            self.wallet = Wallet.create(
                self.wallet_name,
                network='testnet',
                witness_type='segwit'  # Use SegWit addresses (tb1...)
            )
            print(f"✅ Wallet created successfully!")

        return self.wallet

    def get_receiving_address(self) -> str:
        """Get address to receive testnet coins"""
        if not self.wallet:
            self.create_or_load_wallet()

        # Get new receiving address
        key = self.wallet.new_key()
        address = key.address

        print(f"\n💰 TESTNET RECEIVING ADDRESS:")
        print(f"   {address}")
        print(f"\n📥 Get FREE testnet coins from these faucets:")
        for i, faucet in enumerate(FAUCETS, 1):
            print(f"   {i}. {faucet}")

        return address

    def check_balance(self) -> dict:
        """Check wallet balance"""
        if not self.wallet:
            self.create_or_load_wallet()

        print(f"\n🔍 Scanning blockchain for transactions...")
        self.wallet.scan()
        self.wallet.utxos_update()

        balance_sats = self.wallet.balance()
        balance_btc = balance_sats / 100_000_000

        print(f"\n💰 WALLET BALANCE:")
        print(f"   {balance_btc:.8f} tBTC")
        print(f"   ({balance_sats:,} satoshis)")

        # Get UTXOs
        utxos = self.wallet.utxos()
        print(f"\n📊 Unspent Outputs (UTXOs): {len(utxos)}")

        for i, utxo in enumerate(utxos[:5], 1):  # Show first 5
            print(f"   {i}. {utxo['value']:,} sats - {utxo['txid'][:16]}...")

        if len(utxos) > 5:
            print(f"   ... and {len(utxos) - 5} more")

        return {
            'balance_sats': balance_sats,
            'balance_btc': balance_btc,
            'utxo_count': len(utxos),
            'utxos': utxos
        }

    def send_transaction(self, to_address: str, amount_sats: int, fee_per_kb: int = 5000):
        """
        Send testnet Bitcoin transaction

        Args:
            to_address: Testnet address to send to (must start with tb1, n, m, or 2)
            amount_sats: Amount in satoshis
            fee_per_kb: Fee per kilobyte in satoshis
        """
        if not self.wallet:
            self.create_or_load_wallet()

        # Validate it's a testnet address
        testnet_prefixes = ['tb1', 'n', 'm', '2']
        if not any(to_address.startswith(prefix) for prefix in testnet_prefixes):
            print(f"\n❌ ERROR: {to_address} appears to be a MAINNET address!")
            print(f"   Testnet addresses must start with: {testnet_prefixes}")
            print(f"   Cannot proceed - this would fail on testnet network")
            return None

        print(f"\n🔨 CREATING TRANSACTION:")
        print(f"   To: {to_address}")
        print(f"   Amount: {amount_sats:,} sats ({amount_sats/100_000_000:.8f} tBTC)")
        print(f"   Fee rate: {fee_per_kb:,} sats/kB")

        # Check balance
        self.wallet.scan()
        balance = self.wallet.balance()

        if balance < amount_sats:
            print(f"\n❌ INSUFFICIENT BALANCE!")
            print(f"   Have: {balance:,} sats")
            print(f"   Need: {amount_sats:,} sats")
            print(f"\n💡 Get testnet coins from faucets:")
            for faucet in FAUCETS:
                print(f"      {faucet}")
            return None

        try:
            # Create transaction (don't broadcast yet)
            tx = self.wallet.send_to(
                to_address,
                amount_sats,
                fee=fee_per_kb,
                broadcast=False
            )

            print(f"\n✅ TRANSACTION CREATED:")
            print(f"   TXID: {tx.txid}")
            print(f"   Size: {tx.size} bytes")
            print(f"   Fee: {tx.fee:,} sats")
            print(f"   Inputs: {len(tx.inputs)}")
            print(f"   Outputs: {len(tx.outputs)}")

            # Show transaction details
            print(f"\n📋 TRANSACTION DETAILS:")
            for i, output in enumerate(tx.outputs):
                print(f"   Output {i}: {output.value:,} sats -> {output.address}")

            # Ask for confirmation
            print(f"\n⚠️  READY TO BROADCAST TO TESTNET")
            confirm = input("   Broadcast this transaction? (yes/no): ").strip().lower()

            if confirm == 'yes':
                # Broadcast transaction
                tx.send()

                print(f"\n✅ TRANSACTION BROADCAST SUCCESSFUL!")
                print(f"   TXID: {tx.txid}")
                print(f"   View on explorer:")
                print(f"   https://blockstream.info/testnet/tx/{tx.txid}")

                # Save transaction info
                tx_info = {
                    'timestamp': datetime.now().isoformat(),
                    'txid': tx.txid,
                    'to_address': to_address,
                    'amount_sats': amount_sats,
                    'fee': tx.fee,
                    'size': tx.size,
                    'status': 'broadcast'
                }

                filename = f"testnet_tx_{int(datetime.now().timestamp())}.json"
                with open(filename, 'w') as f:
                    json.dump(tx_info, f, indent=2)

                print(f"   Transaction saved to: {filename}")

                return tx
            else:
                print(f"\n❌ Transaction not broadcast")
                return None

        except Exception as e:
            print(f"\n❌ ERROR creating transaction: {e}")
            return None

    def get_transaction_history(self):
        """Get transaction history"""
        if not self.wallet:
            self.create_or_load_wallet()

        self.wallet.scan()
        transactions = self.wallet.transactions()

        print(f"\n📜 TRANSACTION HISTORY: {len(transactions)} transactions")

        for i, tx in enumerate(transactions[:10], 1):  # Show last 10
            status = "✅ Confirmed" if tx.confirmations > 0 else "⏳ Pending"
            tx_type = "📥 Received" if tx.value > 0 else "📤 Sent"

            print(f"\n   {i}. {tx_type} - {status}")
            print(f"      TXID: {tx.txid}")
            print(f"      Amount: {abs(tx.value):,} sats")
            print(f"      Confirmations: {tx.confirmations}")
            print(f"      Date: {tx.date}")

        return transactions

    def export_wallet_info(self) -> dict:
        """Export wallet information"""
        if not self.wallet:
            self.create_or_load_wallet()

        info = {
            'wallet_name': self.wallet_name,
            'network': 'testnet',
            'addresses': [],
            'balance_sats': self.wallet.balance(),
            'utxo_count': len(self.wallet.utxos())
        }

        # Get all addresses
        for key in self.wallet.keys():
            info['addresses'].append({
                'address': key.address,
                'path': key.path,
                'used': key.used
            })

        print(f"\n📊 WALLET EXPORT:")
        print(json.dumps(info, indent=2))

        filename = f"wallet_export_{int(datetime.now().timestamp())}.json"
        with open(filename, 'w') as f:
            json.dump(info, f, indent=2)

        print(f"\n💾 Wallet info saved to: {filename}")

        return info


# ============================================================================
# MAIN PROGRAM
# ============================================================================

def main():
    """Main program"""

    manager = TestnetWalletManager()

    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                     NEXUS AGI TESTNET WALLET MENU                          ║
╚════════════════════════════════════════════════════════════════════════════╝

1. Create/Load Wallet
2. Get Receiving Address (for faucets)
3. Check Balance
4. Send Transaction
5. View Transaction History
6. Export Wallet Info
7. Exit

""")

    while True:
        choice = input("Select option (1-7): ").strip()

        if choice == '1':
            manager.create_or_load_wallet()
            print(f"\n✅ Wallet ready: {manager.wallet_name}")

        elif choice == '2':
            address = manager.get_receiving_address()

        elif choice == '3':
            manager.check_balance()

        elif choice == '4':
            to_address = input("\nEnter TESTNET address to send to: ").strip()
            amount_input = input("Enter amount in satoshis: ").strip()

            try:
                amount_sats = int(amount_input)
                manager.send_transaction(to_address, amount_sats)
            except ValueError:
                print("❌ Invalid amount")

        elif choice == '5':
            manager.get_transaction_history()

        elif choice == '6':
            manager.export_wallet_info()

        elif choice == '7':
            print("\n👋 Goodbye!")
            break

        else:
            print("❌ Invalid option")

        print("\n" + "=" * 80)


if __name__ == "__main__":
    # Quick start mode
    if len(sys.argv) > 1:
        manager = TestnetWalletManager()

        if sys.argv[1] == 'address':
            manager.get_receiving_address()
        elif sys.argv[1] == 'balance':
            manager.check_balance()
        elif sys.argv[1] == 'history':
            manager.get_transaction_history()
        else:
            print(f"Unknown command: {sys.argv[1]}")
            print("Available: address, balance, history")
    else:
        # Interactive mode
        main()
