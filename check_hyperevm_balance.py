#!/usr/bin/env python3
"""
Quick script to check your HyperEVM balance
"""

import os
from web3 import Web3

HYPEREVM_RPC = "https://rpc.hyperliquid-testnet.xyz/evm"
WALLET = "0x9FE74D9D6f1Ae0Ce1fb3B51d4a82c05b74e280f3"

try:
    w3 = Web3(Web3.HTTPProvider(HYPEREVM_RPC, request_kwargs={'timeout': 10}))

    if w3.is_connected():
        balance_wei = w3.eth.get_balance(WALLET)
        balance_eth = w3.from_wei(balance_wei, 'ether')

        print(f"\n🌐 HyperEVM Testnet")
        print(f"💼 Wallet: {WALLET}")
        print(f"💰 Balance: {balance_eth} ETH")

        if balance_eth > 0:
            print(f"\n✅ You have ETH! Ready to deploy!")
            print(f"   Run: python3 deploy_hyperevm.py")
        else:
            print(f"\n❌ No ETH yet!")
            print(f"   Get testnet ETH from Hyperliquid Discord")
            print(f"   Join: https://discord.gg/hyperliquid")
    else:
        print("❌ Failed to connect to HyperEVM")

except Exception as e:
    print(f"❌ Error: {e}")
