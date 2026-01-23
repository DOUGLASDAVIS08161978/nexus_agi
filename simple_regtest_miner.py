#!/usr/bin/env python3
"""
Simple Bitcoin Core Regtest Miner

This script demonstrates basic Bitcoin Core regtest mining:
1. Start bitcoind
2. Create wallet
3. Mine blocks
4. Check balance
"""

import subprocess
import time

def run(cmd):
    """Run shell command and return output"""
    return subprocess.check_output(cmd, shell=True).decode().strip()

print("🚀 Starting bitcoind...")
run("bitcoind -regtest -daemon")

time.sleep(5)

print("👛 Creating wallet...")
try:
    run("bitcoin-cli -regtest createwallet miner")
except:
    pass  # Wallet might already exist

addr = run("bitcoin-cli -regtest getnewaddress")
print(f"⛏️  Mining blocks to: {addr}")

# Mine 101 blocks (need 100 confirmations for coinbase)
run(f"bitcoin-cli -regtest generatetoaddress 101 {addr}")

balance = run("bitcoin-cli -regtest getbalance")
print(f"✅ Balance: {balance} BTC")

print(f"\n💡 Your regtest Bitcoin is ready!")
print(f"   Address: {addr}")
print(f"   Balance: {balance} BTC")
print(f"   Network: Regtest (local only)")
print(f"   Value: $0 (not on mainnet)")
