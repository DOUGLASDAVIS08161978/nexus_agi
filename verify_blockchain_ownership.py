#!/usr/bin/env python3
"""
Blockchain Ownership and Value Verification Tool

This script helps you understand:
1. What you own vs what you're using
2. Whether funding adds value or costs money
3. How to verify your blockchain access

Author: Douglas Shane Davis & Claude
Date: 2026-01-22
"""

from web3 import Web3
from eth_account import Account
import json

# Configuration
MONAD_TESTNET_RPC = "https://testnet-rpc.monad.xyz"
PRIVATE_KEY = "c411a4d4365560753ef3ceceac1652ec89240704346bf58ad900d65574f541c9"
ADDRESS = "0x24f6b1ce11c57d40b542f91ac85fa9eb61f78771"

def print_header(title):
    """Print section header"""
    print(f"\n{'='*80}")
    print(f"{title:^80}")
    print(f"{'='*80}\n")

def verify_key_ownership():
    """Verify you own the private key"""
    print_header("🔑 PRIVATE KEY OWNERSHIP")

    account = Account.from_key(PRIVATE_KEY)

    print(f"Private Key: {PRIVATE_KEY}")
    print(f"Derived Address: {account.address}")
    print(f"Expected Address: {ADDRESS}")
    print(f"Match: {'✅ YES' if account.address.lower() == ADDRESS.lower() else '❌ NO'}")
    print(f"\n📋 What You Own:")
    print(f"  ✅ Private key (can sign transactions)")
    print(f"  ✅ Address (can receive funds)")
    print(f"  ✅ Control over funds in this address")

def verify_node_ownership():
    """Verify node ownership"""
    print_header("🌐 BLOCKCHAIN NODE OWNERSHIP")

    try:
        w3 = Web3(Web3.HTTPProvider(MONAD_TESTNET_RPC))
        connected = w3.is_connected()

        print(f"RPC Endpoint: {MONAD_TESTNET_RPC}")
        print(f"Connected: {'✅ YES' if connected else '❌ NO'}")

        if connected:
            try:
                chain_id = w3.eth.chain_id
                block_number = w3.eth.block_number
                print(f"Chain ID: {chain_id}")
                print(f"Current Block: {block_number:,}")
            except:
                pass

        print(f"\n📋 What You Own:")
        print(f"  ❌ You DO NOT own this RPC node")
        print(f"  ❌ You DO NOT control the blockchain")
        print(f"  ⚠️  You ARE USING: Monad's public infrastructure")
        print(f"  ⚠️  You ARE TRUSTING: Monad to provide accurate data")

        print(f"\n💡 To Own a Node:")
        print(f"  1. Run your own Ethereum/Monad node")
        print(f"  2. Change RPC to: http://localhost:8545")
        print(f"  3. Maintain your own blockchain copy")
        print(f"  4. Requires: ~2TB disk, 16GB RAM, good internet")

    except Exception as e:
        print(f"❌ Error connecting to RPC: {e}")

def check_balance_and_value():
    """Check balance and explain value"""
    print_header("💰 BALANCE & VALUE ANALYSIS")

    try:
        w3 = Web3(Web3.HTTPProvider(MONAD_TESTNET_RPC))

        if w3.is_connected():
            balance_wei = w3.eth.get_balance(Web3.to_checksum_address(ADDRESS))
            balance_eth = w3.from_wei(balance_wei, 'ether')

            print(f"Address: {ADDRESS}")
            print(f"Balance: {balance_eth:.8f} testnet ETH")
            print(f"Real World Value: $0.00 (TESTNET ONLY)")

            print(f"\n📊 Understanding Testnet Value:")
            print(f"  ⚠️  Testnet ETH = $0.00 (no real value)")
            print(f"  ⚠️  Testnet WBTC = $0.00 (no real value)")
            print(f"  ⚠️  All testnet tokens = FOR TESTING ONLY")

            print(f"\n🔄 What Happens When You Run the Bridge:")
            print(f"  1. You start with: {balance_eth:.8f} testnet ETH")
            print(f"  2. Each session costs: ~0.011 testnet ETH (gas fees)")
            print(f"  3. After 1 session: ~{balance_eth - 0.011:.8f} testnet ETH")
            print(f"  4. You receive: WBTC tokens (testnet, $0 value)")
            print(f"  5. Your ETH balance DECREASES (spent on gas)")

            print(f"\n❌ YOU DO NOT GAIN VALUE")
            print(f"✅ YOU SPEND GAS TO TEST THE SYSTEM")

        else:
            print(f"❌ Cannot connect to check balance")

    except Exception as e:
        print(f"❌ Error: {e}")

def explain_gas_costs():
    """Explain gas costs"""
    print_header("⛽ GAS COST BREAKDOWN")

    print(f"Each Bridge Transaction:")
    print(f"  Gas Limit: 21,000")
    print(f"  Gas Price: ~102 gwei")
    print(f"  Cost per TX: ~0.002142 ETH")
    print(f"  ")
    print(f"Default Session (10 blocks, 5 transactions):")
    print(f"  Total Gas Cost: ~0.011 ETH")
    print(f"  ")
    print(f"If You Fund with 0.1 ETH:")
    print(f"  Session 1: 0.100 - 0.011 = 0.089 ETH remaining")
    print(f"  Session 2: 0.089 - 0.011 = 0.078 ETH remaining")
    print(f"  Session 3: 0.078 - 0.011 = 0.067 ETH remaining")
    print(f"  ...")
    print(f"  You can run ~9 sessions before running out")
    print(f"  ")
    print(f"📉 Your Balance Goes DOWN, Not Up!")

def show_ownership_levels():
    """Show different levels of blockchain ownership"""
    print_header("🏛️ LEVELS OF BLOCKCHAIN OWNERSHIP")

    levels = [
        {
            "level": "1. No Ownership (Wallet Services)",
            "you_control": ["Nothing - service controls your keys"],
            "you_own": ["Account access (can be revoked)"],
            "examples": ["Coinbase, Binance wallets"]
        },
        {
            "level": "2. Key Ownership (YOUR CURRENT LEVEL)",
            "you_control": ["Private key", "Address", "Ability to sign transactions"],
            "you_own": ["Funds in your address"],
            "examples": ["MetaMask, Hardware wallets, Your current setup"]
        },
        {
            "level": "3. Node Ownership",
            "you_control": ["Private key", "Own blockchain copy", "Direct verification"],
            "you_own": ["Complete transaction history", "No RPC dependency"],
            "examples": ["Running Geth/Nethermind/Besu node"]
        },
        {
            "level": "4. Validator Ownership",
            "you_control": ["All of above", "Block proposal rights"],
            "you_own": ["Stake in network security"],
            "examples": ["Ethereum staking with 32 ETH"]
        },
        {
            "level": "5. Network Ownership (Private Chain)",
            "you_control": ["Everything - all nodes, all rules"],
            "you_own": ["Entire blockchain"],
            "examples": ["Private Ethereum chain", "Your Bitcoin regtest"]
        }
    ]

    for i, level in enumerate(levels, 1):
        print(f"{level['level']}")
        print(f"  You Control: {', '.join(level['you_control'])}")
        print(f"  You Own: {', '.join(level['you_own'])}")
        print(f"  Examples: {', '.join(level['examples'])}")
        if i == 2:
            print(f"  👉 YOU ARE HERE")
        print()

def main():
    """Main execution"""
    print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║         🔍 BLOCKCHAIN OWNERSHIP & VALUE VERIFICATION TOOL 🔍                 ║
║                                                                              ║
║  Understand what you own, what you're using, and where value comes from     ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)

    verify_key_ownership()
    verify_node_ownership()
    check_balance_and_value()
    explain_gas_costs()
    show_ownership_levels()

    print_header("🎓 KEY TAKEAWAYS")
    print(f"1. ✅ YOU OWN: Your private key and address")
    print(f"2. ❌ YOU DON'T OWN: The Monad testnet blockchain")
    print(f"3. ⚠️  TESTNET FUNDS: Have $0 real-world value")
    print(f"4. 📉 FUNDING: Decreases your balance (gas costs)")
    print(f"5. 🎯 PURPOSE: Testing and development only")
    print(f"")
    print(f"💡 To Own More:")
    print(f"   - Run your own Ethereum node (see ethereum.org docs)")
    print(f"   - This gives you a complete copy of the blockchain")
    print(f"   - You still won't 'own' the public blockchain")
    print(f"   - But you'll have independent verification")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
