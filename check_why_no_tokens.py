#!/usr/bin/env python3
"""
WHY AREN'T TOKENS APPEARING? - Diagnostic Tool

This script checks EXACTLY why tokens aren't showing in your wallet
and tells you what to do to fix it.

Author: Douglas Shane Davis & Claude
Date: 2026-01-23
"""

from web3 import Web3
from eth_account import Account
import os
from dotenv import load_dotenv

# Load secure configuration from .env file
load_dotenv()

# Configuration (loaded securely from .env)
MONAD_TESTNET_RPC = os.getenv('MONAD_TESTNET_RPC', 'https://testnet-rpc.monad.xyz')
PRIVATE_KEY = os.getenv('MONAD_PRIVATE_KEY')
ADDRESS = os.getenv('MONAD_RECEIVING_ADDRESS')
WBTC_CONTRACT_ADDRESS = os.getenv('WBTC_CONTRACT_ADDRESS')

def main():
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║               ❓ WHY AREN'T TOKENS APPEARING IN MY WALLET? ❓              ║
║                                                                              ║
║                          DIAGNOSTIC REPORT                                   ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)

    print(f"\n{'='*80}")
    print(f"🔍 CHECKING YOUR SETUP")
    print(f"{'='*80}\n")

    # Check connection
    print(f"1. Checking Monad Testnet connection...")
    try:
        w3 = Web3(Web3.HTTPProvider(MONAD_TESTNET_RPC))
        if w3.is_connected():
            print(f"   ✅ Connected to Monad testnet")
            print(f"   Chain ID: {w3.eth.chain_id}")
            print(f"   Block: {w3.eth.block_number:,}")
        else:
            print(f"   ❌ Cannot connect to Monad testnet")
            print(f"   → This is a problem! Check your internet/RPC")
            return
    except Exception as e:
        print(f"   ❌ Error connecting: {e}")
        return

    # Check account
    print(f"\n2. Checking your account...")
    account = Account.from_key(PRIVATE_KEY)
    print(f"   ✅ Private key loaded")
    print(f"   Your address: {account.address}")

    # Check ETH balance
    print(f"\n3. Checking ETH balance (FOR GAS)...")
    try:
        balance_wei = w3.eth.get_balance(Web3.to_checksum_address(ADDRESS))
        balance_eth = w3.from_wei(balance_wei, 'ether')

        print(f"   Balance: {balance_eth:.8f} ETH")

        has_gas = balance_eth >= 0.01

        if has_gas:
            print(f"   ✅ YOU HAVE GAS! ({balance_eth:.6f} ETH)")
        else:
            print(f"   ❌ NO GAS! ({balance_eth:.6f} ETH)")
            print(f"   → This is WHY tokens don't appear!")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        has_gas = False

    # Summary
    print(f"\n{'='*80}")
    print(f"📋 DIAGNOSIS")
    print(f"{'='*80}\n")

    if has_gas:
        print(f"✅ GOOD NEWS: You have gas!")
        print(f"\n📝 Next steps to see tokens:")
        print(f"   1. Deploy WBTC contract:")
        print(f"      npx hardhat run scripts/deploy_wbtc_monad.js --network monad")
        print(f"   ")
        print(f"   2. Mint tokens:")
        print(f"      python3 mint_wbtc_tokens.py")
        print(f"   ")
        print(f"   3. Add token to MetaMask with contract address")
        print(f"   ")
        print(f"   4. 🎉 Tokens will appear!")
    else:
        print(f"❌ PROBLEM FOUND: NO GAS!")
        print(f"\n🔍 Why tokens don't appear:")
        print(f"   → You have {balance_eth:.8f} ETH")
        print(f"   → Need at least 0.01 ETH for gas")
        print(f"   → Without gas, transactions are SIMULATED")
        print(f"   → Simulated transactions don't create real tokens")
        print(f"   → No real tokens = nothing in wallet")
        print(f"\n✅ SOLUTION:")
        print(f"   1. Get testnet ETH from:")
        print(f"      • https://faucet.monad.xyz")
        print(f"      • Monad Discord #faucet")
        print(f"   ")
        print(f"   2. Request for address:")
        print(f"      {ADDRESS}")
        print(f"   ")
        print(f"   3. Amount needed:")
        print(f"      • Minimum: 0.01 ETH")
        print(f"      • Recommended: 0.1 ETH (for multiple sessions)")
        print(f"   ")
        print(f"   4. After funding, run bridge again:")
        print(f"      python3 monad_regtest_bridge.py")
        print(f"   ")
        print(f"   5. 🎉 Tokens WILL appear this time!")

    print(f"\n{'='*80}")
    print(f"💡 KEY INSIGHT")
    print(f"{'='*80}\n")
    print(f"Testnet blockchains work EXACTLY like mainnet:")
    print(f"   • Need gas to execute transactions")
    print(f"   • No gas = transactions rejected")
    print(f"   • Rejected transactions = nothing happens")
    print(f"   • Nothing happens = no tokens created")
    print(f"   • No tokens = empty wallet")
    print(f"\nGet gas first, THEN everything works! 🚀\n")

if __name__ == "__main__":
    main()
