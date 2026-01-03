#!/usr/bin/env python3
"""
Bitcoin TESTNET Transaction Creator & Broadcaster
TESTNET ONLY - No real money involved
"""

import hashlib
import requests
import json
from typing import Dict, List, Optional
from dataclasses import dataclass

# ============================================================================
# CONFIGURATION - TESTNET ONLY
# ============================================================================

TESTNET_API = "https://blockstream.info/testnet/api"
TESTNET_BROADCAST_URL = f"{TESTNET_API}/tx"

@dataclass
class TestnetWallet:
    """Your testnet wallet info - YOU MUST FILL THIS IN"""
    # Get testnet coins from: https://testnet-faucet.mempool.co/
    # Or: https://coinfaucet.eu/en/btc-testnet/

    private_key_wif: str = "YOUR_TESTNET_PRIVATE_KEY_HERE"  # WIF format
    address: str = "YOUR_TESTNET_ADDRESS_HERE"  # tb1q... format

    # Find your UTXOs at: https://blockstream.info/testnet/address/YOUR_ADDRESS
    utxos: List[Dict] = None  # Will be fetched automatically


class BitcoinTestnetTransfer:
    """
    Creates and broadcasts Bitcoin TESTNET transactions

    SAFETY FEATURES:
    - Only works with testnet
    - Validates all addresses are testnet format
    - Checks balance before creating transaction
    - Provides detailed logging
    """

    def __init__(self):
        self.api_base = TESTNET_API
        print("=" * 80)
        print("BITCOIN TESTNET TRANSACTION SYSTEM")
        print("=" * 80)
        print("⚠️  TESTNET ONLY - No real Bitcoin involved")
        print("⚠️  Get free testnet coins from faucets")
        print("=" * 80)

    def validate_testnet_address(self, address: str) -> bool:
        """Ensure address is testnet format"""
        testnet_prefixes = ['tb1', 'n', 'm', '2']  # Common testnet formats

        if not any(address.startswith(prefix) for prefix in testnet_prefixes):
            print(f"❌ ERROR: {address} appears to be MAINNET address!")
            print(f"   Testnet addresses start with: {testnet_prefixes}")
            return False
        return True

    def get_address_utxos(self, address: str) -> List[Dict]:
        """Fetch unspent outputs for address"""
        print(f"\n📡 Fetching UTXOs for: {address}")

        try:
            url = f"{self.api_base}/address/{address}/utxo"
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            utxos = response.json()
            print(f"✅ Found {len(utxos)} UTXOs")

            total_sats = sum(utxo['value'] for utxo in utxos)
            total_btc = total_sats / 100_000_000
            print(f"💰 Total balance: {total_btc:.8f} tBTC ({total_sats} satoshis)")

            return utxos

        except Exception as e:
            print(f"❌ Error fetching UTXOs: {e}")
            return []

    def get_address_balance(self, address: str) -> int:
        """Get total balance in satoshis"""
        utxos = self.get_address_utxos(address)
        return sum(utxo['value'] for utxo in utxos)

    def create_simple_transaction(self,
                                  from_address: str,
                                  to_address: str,
                                  amount_sats: int,
                                  fee_sats: int = 1000) -> Optional[str]:
        """
        Create a simple P2WPKH transaction

        NOTE: This is a SIMPLIFIED example showing the concept.
        For production, use libraries like:
        - bitcoinlib
        - python-bitcoinlib
        - bit
        """

        print(f"\n🔨 Creating transaction...")
        print(f"   From: {from_address}")
        print(f"   To: {to_address}")
        print(f"   Amount: {amount_sats} sats ({amount_sats/100_000_000:.8f} tBTC)")
        print(f"   Fee: {fee_sats} sats")

        # Validate addresses
        if not self.validate_testnet_address(from_address):
            return None
        if not self.validate_testnet_address(to_address):
            return None

        # Get UTXOs
        utxos = self.get_address_utxos(from_address)
        if not utxos:
            print("❌ No UTXOs found - address has no funds!")
            print("   Get testnet coins from: https://testnet-faucet.mempool.co/")
            return None

        # Check sufficient balance
        total_balance = sum(utxo['value'] for utxo in utxos)
        total_needed = amount_sats + fee_sats

        if total_balance < total_needed:
            print(f"❌ Insufficient balance!")
            print(f"   Have: {total_balance} sats")
            print(f"   Need: {total_needed} sats")
            return None

        print(f"✅ Balance check passed")
        print(f"\n⚠️  ACTUAL TRANSACTION CREATION REQUIRES:")
        print(f"   1. Private key to sign inputs")
        print(f"   2. Proper transaction serialization")
        print(f"   3. Signature generation (ECDSA)")
        print(f"   4. Witness data for SegWit")
        print(f"\n💡 Use a proper library like 'bitcoinlib' or 'python-bitcoinlib'")

        return None  # Placeholder

    def broadcast_transaction(self, signed_tx_hex: str) -> Dict:
        """
        Broadcast signed transaction to testnet
        """
        print(f"\n📤 Broadcasting transaction to testnet...")

        try:
            response = requests.post(
                TESTNET_BROADCAST_URL,
                data=signed_tx_hex,
                headers={'Content-Type': 'text/plain'},
                timeout=10
            )

            if response.status_code == 200:
                txid = response.text
                print(f"✅ Transaction broadcast successful!")
                print(f"📋 TXID: {txid}")
                print(f"🔍 View at: https://blockstream.info/testnet/tx/{txid}")
                return {'success': True, 'txid': txid}
            else:
                print(f"❌ Broadcast failed: {response.text}")
                return {'success': False, 'error': response.text}

        except Exception as e:
            print(f"❌ Broadcast error: {e}")
            return {'success': False, 'error': str(e)}


# ============================================================================
# COMPLETE EXAMPLE USING BITCOINLIB
# ============================================================================

def example_with_bitcoinlib():
    """
    Complete working example using bitcoinlib library

    Install: pip install bitcoinlib
    """

    print("\n" + "=" * 80)
    print("WORKING EXAMPLE USING BITCOINLIB")
    print("=" * 80)

    print("""
To use this properly, install bitcoinlib:

    pip install bitcoinlib

Then run this code:
    """)

    code = '''
from bitcoinlib.wallets import Wallet
from bitcoinlib.transactions import Transaction

# 1. Create or load testnet wallet
wallet = Wallet.create('my_testnet_wallet', network='testnet')
print(f"Your testnet address: {wallet.get_key().address}")

# 2. Get testnet coins from faucet
print("Get free coins: https://testnet-faucet.mempool.co/")
input("Press Enter after receiving coins...")

# 3. Create and send transaction
to_address = "tb1qyhkq7usdfhhhynkjksdqfx32u3rmv94ydummyx"  # TESTNET address
amount = 50000  # 0.0005 tBTC in satoshis

# Update wallet UTXOs
wallet.scan()

# Create transaction
tx = wallet.send_to(to_address, amount, fee=1000, broadcast=False)
print(f"Transaction created: {tx.txid}")

# Review transaction
print(f"Inputs: {len(tx.inputs)}")
print(f"Outputs: {len(tx.outputs)}")
print(f"Fee: {tx.fee} sats")

# Broadcast (only if you confirm it's correct)
confirm = input("Broadcast? (yes/no): ")
if confirm.lower() == 'yes':
    tx.send()
    print(f"✅ Sent! TXID: {tx.txid}")
    print(f"View: https://blockstream.info/testnet/tx/{tx.txid}")
else:
    print("Transaction not broadcast")
'''

    print(code)


# ============================================================================
# MAIN DEMO
# ============================================================================

def main():
    """Run demonstration"""

    transfer = BitcoinTestnetTransfer()

    print("\n" + "=" * 80)
    print("STEP-BY-STEP GUIDE")
    print("=" * 80)

    print("""
1. GET TESTNET WALLET:
   - Download Electrum: https://electrum.org
   - Create new wallet, select "Testnet" network
   - OR install bitcoinlib: pip install bitcoinlib

2. GET TESTNET COINS (FREE):
   - Visit: https://testnet-faucet.mempool.co/
   - Or: https://coinfaucet.eu/en/btc-testnet/
   - Enter your testnet address
   - Wait for confirmation (10-30 minutes)

3. CHECK YOUR BALANCE:
   """)

    # Example: Check a testnet address balance
    example_address = "tb1qw508d6qejxtdg4y5r3zarvary0c5xw7kxpjzsx"  # Example testnet address

    print(f"\nExample - Checking balance for: {example_address}")
    balance = transfer.get_address_balance(example_address)
    print(f"Balance: {balance} satoshis ({balance/100_000_000:.8f} tBTC)")

    print("""
4. CREATE TRANSACTION:
   - Use the bitcoinlib example above
   - Or use Electrum wallet GUI
   - Or use Bitcoin Core with testnet flag

5. VERIFY BEFORE BROADCASTING:
   - Check addresses are correct
   - Verify amounts
   - Confirm fee is reasonable
   - Double-check it's TESTNET not mainnet!
    """)

    # Show the working example
    example_with_bitcoinlib()

    print("\n" + "=" * 80)
    print("SAFETY REMINDERS")
    print("=" * 80)
    print("✅ Always use TESTNET for learning")
    print("✅ Never share private keys")
    print("✅ Test with small amounts first")
    print("✅ Verify addresses before sending")
    print("✅ Check explorer to confirm transactions")
    print("=" * 80)


if __name__ == "__main__":
    main()
