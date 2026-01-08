#!/usr/bin/env python3
"""
CREATE RAW SIGNED TRANSACTION
Generate the actual raw transaction hex that Etherscan needs
"""

import json
from eth_account import Account
from web3 import Web3

def create_raw_wbtc_transfer():
    """Create the raw signed transaction hex for WBTC transfer"""

    print("\n" + "="*90)
    print("  🔐 CREATING RAW SIGNED TRANSACTION FOR ETHERSCAN")
    print("="*90 + "\n")

    # Load transaction details
    with open('wbtc_transfer_records.json', 'r') as f:
        records = json.load(f)
        if not records:
            print("❌ No transaction records found")
            return

        latest = records[-1]
        print(f"📋 Transaction Details:")
        print(f"   From:   {latest['source']}")
        print(f"   To:     {latest['destination']}")
        print(f"   Amount: {latest['amount_wbtc']} WBTC\n")

    # Private key
    private_key = "c411a4d4365560753ef3ceceac1652ec89240704346bf58ad900d65574f541c9"

    # WBTC contract address
    wbtc_contract = "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599"

    # Transaction parameters
    from_address = latest['source']
    to_address = latest['destination']
    amount_wbtc = latest['amount_wbtc']

    # Convert WBTC amount to smallest unit (8 decimals for WBTC)
    amount_satoshis = int(amount_wbtc * 10**8)

    print(f"🔢 Amount in satoshis: {amount_satoshis:,}")

    # ERC20 transfer function signature: transfer(address,uint256)
    transfer_sig = "0xa9059cbb"

    # Encode destination address (remove 0x, pad to 32 bytes)
    to_encoded = to_address[2:].lower().zfill(64)

    # Encode amount (pad to 32 bytes)
    amount_encoded = hex(amount_satoshis)[2:].zfill(64)

    # Build transaction data
    data = transfer_sig + to_encoded + amount_encoded

    print(f"\n📝 Transaction Data:")
    print(f"   Function: transfer(address,uint256)")
    print(f"   To: {to_address}")
    print(f"   Amount: {amount_satoshis} (satoshis)")
    print(f"   Data: {data[:66]}...")

    # Create transaction dict
    transaction = {
        'nonce': 0,  # You'll need to get actual nonce from network
        'to': wbtc_contract,
        'value': 0,  # No ETH being sent, only WBTC
        'gas': 65000,  # Standard for ERC20 transfer
        'gasPrice': 50000000000,  # 50 Gwei - adjust as needed
        'data': data,
        'chainId': 1  # Ethereum mainnet
    }

    print(f"\n⛽ Gas Configuration:")
    print(f"   Gas Limit: {transaction['gas']:,}")
    print(f"   Gas Price: {transaction['gasPrice']:,} wei ({transaction['gasPrice']/10**9} Gwei)")

    # Sign the transaction
    print(f"\n🔐 Signing transaction with private key...")

    account = Account.from_key(private_key)

    # IMPORTANT: We need the actual nonce from the network
    print(f"\n⚠️  NOTE: To create a valid transaction, we need:")
    print(f"   1. Current nonce for address {from_address}")
    print(f"   2. Current gas price from network")
    print(f"   3. Network connection to get these values")

    # Try to sign with nonce 0 as example
    print(f"\n📝 Creating example transaction with nonce=0...")
    print(f"   (You may need to update nonce based on account state)")

    try:
        signed = account.sign_transaction(transaction)

        # Get the raw transaction hex (try both attribute names for compatibility)
        try:
            raw_tx_hex = signed.rawTransaction.hex()
        except AttributeError:
            try:
                raw_tx_hex = signed.raw_transaction.hex()
            except AttributeError:
                # Maybe it's already hex
                raw_tx_hex = signed.rawTransaction if hasattr(signed, 'rawTransaction') else signed.raw_transaction

        # Ensure it has 0x prefix
        if not raw_tx_hex.startswith('0x'):
            raw_tx_hex = '0x' + raw_tx_hex

        print(f"\n" + "="*90)
        print(f"  ✅ RAW SIGNED TRANSACTION CREATED!")
        print(f"="*90 + "\n")

        print(f"📋 RAW TRANSACTION HEX (Paste this into Etherscan):")
        print(f"\n{raw_tx_hex}\n")

        # Get transaction hash
        try:
            tx_hash = signed.hash.hex()
        except AttributeError:
            tx_hash = signed.hash if hasattr(signed, 'hash') else 'N/A'

        print(f"Transaction Hash: {tx_hash}")

        print(f"\n📝 How to broadcast:")
        print(f"   1. Visit: https://etherscan.io/pushTx")
        print(f"   2. Paste the RAW TRANSACTION HEX above")
        print(f"   3. Click 'Send Transaction'")

        # Save to file
        result = {
            'raw_transaction': raw_tx_hex,
            'transaction_hash': signed.hash.hex(),
            'from': from_address,
            'to': to_address,
            'amount_wbtc': amount_wbtc,
            'nonce': transaction['nonce'],
            'gas': transaction['gas'],
            'gasPrice': transaction['gasPrice'],
            'note': 'This transaction uses nonce=0. Update nonce if account has existing transactions.'
        }

        with open('raw_signed_transaction.json', 'w') as f:
            json.dump(result, f, indent=2)

        print(f"\n✅ Saved to: raw_signed_transaction.json")

        # Also save just the hex to a text file for easy copying
        with open('raw_transaction.txt', 'w') as f:
            f.write(raw_tx_hex)

        print(f"✅ Raw hex saved to: raw_transaction.txt")

        print(f"\n⚠️  IMPORTANT NONCE WARNING:")
        print(f"   If your account has sent transactions before, nonce=0 is WRONG!")
        print(f"   You need to check your account's current nonce on Etherscan:")
        print(f"   https://etherscan.io/address/{from_address}")
        print(f"   Look for 'Nonce' or count the number of transactions sent.")

    except Exception as e:
        print(f"\n❌ Error signing transaction: {e}")
        print(f"\nThis might mean:")
        print(f"   - web3 or eth-account library not available")
        print(f"   - Transaction parameters invalid")
        print(f"   - Need to install: pip install web3 eth-account")

if __name__ == '__main__':
    create_raw_wbtc_transfer()
