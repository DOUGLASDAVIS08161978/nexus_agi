#!/usr/bin/env python3
"""
Send 1 test ETH from Hardhat account to user's wallet
"""

from web3 import Web3

# Connect to local Hardhat network
w3 = Web3(Web3.HTTPProvider('http://localhost:8545'))

if not w3.is_connected():
    print("❌ Cannot connect to Hardhat network")
    print("   Make sure Hardhat node is running at localhost:8545")
    exit(1)

print("✅ Connected to Hardhat local blockchain\n")

# Get Hardhat test accounts
accounts = w3.eth.accounts

# Sender (Hardhat test account #0 with 10,000 ETH)
sender = accounts[0]
sender_balance = w3.eth.get_balance(sender)

print(f"📤 Sender: {sender}")
print(f"   Balance: {w3.from_wei(sender_balance, 'ether')} ETH\n")

# Recipient (your wallet) - convert to checksum address
recipient = w3.to_checksum_address("0x24f6b1ce11c57d40b542f91ac85fa9eb61f78771")
recipient_balance_before = w3.eth.get_balance(recipient)

print(f"📥 Recipient: {recipient}")
print(f"   Balance before: {w3.from_wei(recipient_balance_before, 'ether')} ETH\n")

# Send 1 ETH
amount_wei = w3.to_wei(1, 'ether')

print("💸 Sending 1 test ETH...")

try:
    tx_hash = w3.eth.send_transaction({
        'from': sender,
        'to': recipient,
        'value': amount_wei,
        'gas': 21000
    })

    # Wait for transaction
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)

    print(f"✅ Transaction successful!")
    print(f"   Tx Hash: {tx_hash.hex()}")
    print(f"   Block: {receipt['blockNumber']}")
    print(f"   Gas Used: {receipt['gasUsed']}\n")

    # Check new balance
    recipient_balance_after = w3.eth.get_balance(recipient)

    print(f"💰 New Balance: {w3.from_wei(recipient_balance_after, 'ether')} ETH")
    print(f"   Received: {w3.from_wei(recipient_balance_after - recipient_balance_before, 'ether')} ETH\n")

    print("🎉 Success! Your wallet now has 1 test ETH on the local blockchain!")

except Exception as e:
    print(f"❌ Error: {e}")
