#!/usr/bin/env python3
"""
Verify that deployed contracts have bytecode on the blockchain
"""

from web3 import Web3
import json

# Connect to Geth
w3 = Web3(Web3.HTTPProvider('http://localhost:8545'))

# Check connection
if not w3.is_connected():
    print("❌ Cannot connect to Geth at localhost:8545")
    print("   Make sure Geth is running!")
    exit(1)

print("✅ Connected to Geth blockchain")
print(f"   Chain ID: {w3.eth.chain_id}")
print(f"   Block number: {w3.eth.block_number}\n")

# Load deployed addresses
try:
    with open('deployment_addresses.json', 'r') as f:
        deployment = json.load(f)
except FileNotFoundError:
    print("❌ deployment_addresses.json not found")
    exit(1)

print("=" * 80)
print("VERIFYING CONTRACT BYTECODE")
print("=" * 80)
print()

all_have_bytecode = True

for contract in deployment['contracts']:
    name = contract['name']
    address = Web3.to_checksum_address(contract['address'])

    print(f"📋 {name}")
    print(f"   Address: {address}")

    # Get bytecode
    bytecode = w3.eth.get_code(address)
    bytecode_length = len(bytecode)

    if bytecode_length > 2:  # "0x" is 2 characters
        print(f"   ✅ HAS BYTECODE: {bytecode_length} bytes")
        print(f"   MetaMask SHOULD recognize this as a CONTRACT")
        all_have_bytecode = all_have_bytecode and True
    else:
        print(f"   ❌ NO BYTECODE FOUND!")
        print(f"   This is a personal address (EOA)")
        all_have_bytecode = False

    print()

print("=" * 80)

if all_have_bytecode:
    print("✅ ALL CONTRACTS HAVE BYTECODE ON-CHAIN!")
    print()
    print("If MetaMask shows 'personal address', check:")
    print("  1. MetaMask is connected to correct network:")
    print("     Network: Geth Local")
    print("     RPC URL: http://localhost:8545")
    print("     Chain ID: 1337")
    print()
    print("  2. You're looking at the contract address (not your wallet)")
    print()
    print("  3. Try refreshing MetaMask or reconnecting to the network")
else:
    print("❌ SOME CONTRACTS MISSING BYTECODE!")
    print()
    print("This means deployment didn't actually work.")
    print("The contracts need to be redeployed.")

print()
print("💡 To check in MetaMask:")
print("   1. Make sure you're on 'Geth Local' network (Chain ID 1337)")
print("   2. Go to 'Assets' → 'Import tokens'")
print("   3. Paste the contract address")
print("   4. If it has bytecode, MetaMask should show contract info")
print()
