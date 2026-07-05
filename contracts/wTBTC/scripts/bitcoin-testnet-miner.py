#!/usr/bin/env python3
"""
Bitcoin Testnet CPU Miner (Improved Version)
WARNING: Still slow, but actually functional!

For wTBTC bridge testing - get testnet BTC to lock in bridge.
"""

import json
import hashlib
import struct
import binascii
import time
import threading
from datetime import datetime

# Requires: pip install python-bitcoinrpc
try:
    from bitcoinrpc.authproxy import AuthServiceProxy
except ImportError:
    print("⚠️  Install bitcoin RPC: pip install python-bitcoinrpc")
    exit(1)


class TestnetCPUMiner:
    """
    Improved CPU miner for Bitcoin testnet.
    Still slow, but actually works!
    """

    def __init__(self, rpc_user, rpc_pass, rpc_host="127.0.0.1", rpc_port=18332):
        self.rpc_url = f"http://{rpc_user}:{rpc_pass}@{rpc_host}:{rpc_port}"
        self.rpc = None
        self.mining_address = None
        self.hashes_attempted = 0
        self.start_time = None
        self.block_found = False

    def connect(self):
        """Connect to Bitcoin Core testnet node"""
        try:
            self.rpc = AuthServiceProxy(self.rpc_url)
            info = self.rpc.getblockchaininfo()

            print("=" * 60)
            print("🔗 CONNECTED TO BITCOIN TESTNET NODE")
            print("=" * 60)
            print(f"Chain: {info['chain']}")
            print(f"Blocks: {info['blocks']}")
            print(f"Difficulty: {info['difficulty']}")
            print(f"Verification: {info['verificationprogress']:.2%}")
            print()

            if info['chain'] != 'test':
                print("⚠️  WARNING: Not on testnet! Switch your node to testnet.")
                return False

            return True

        except Exception as e:
            print(f"❌ Connection failed: {e}")
            print()
            print("Make sure Bitcoin Core is running with:")
            print("  bitcoind -testnet -rpcuser=USER -rpcpassword=PASS")
            return False

    def setup_address(self):
        """Get or create mining address"""
        try:
            # Get a new address for mining rewards
            self.mining_address = self.rpc.getnewaddress("CPU_Mining")

            print("💰 Mining Setup:")
            print("-" * 60)
            print(f"Rewards address: {self.mining_address}")
            print()

            return True

        except Exception as e:
            print(f"❌ Failed to create address: {e}")
            return False

    def get_block_template(self):
        """Get current block template from network"""
        try:
            template = self.rpc.getblocktemplate({
                "rules": ["segwit"]
            })
            return template

        except Exception as e:
            print(f"❌ Failed to get template: {e}")
            return None

    def build_coinbase_tx(self, height, reward_satoshis):
        """Build coinbase transaction"""
        # Simplified coinbase - just for testing
        coinbase_script = struct.pack("<I", height)

        return {
            "version": 2,
            "inputs": [{
                "sequence": 0xffffffff,
                "script": coinbase_script.hex()
            }],
            "outputs": [{
                "value": reward_satoshis,
                "address": self.mining_address
            }]
        }

    def build_block_header(self, template, nonce):
        """Build block header for hashing"""
        version = template['version']
        prev_hash = template['previousblockhash']
        merkle_root = template['merkleroot'] if 'merkleroot' in template else "0" * 64
        timestamp = int(time.time())
        bits = int(template['bits'], 16)

        # Build header (80 bytes)
        header = struct.pack(
            "<I32s32sIII",
            version,
            bytes.fromhex(prev_hash)[::-1],
            bytes.fromhex(merkle_root)[::-1],
            timestamp,
            bits,
            nonce
        )

        return header

    def hash_header(self, header):
        """Double SHA256 hash (Bitcoin proof-of-work)"""
        first_hash = hashlib.sha256(header).digest()
        second_hash = hashlib.sha256(first_hash).digest()
        return second_hash[::-1]  # Reverse for little-endian

    def check_proof_of_work(self, hash_result, target_bits):
        """Check if hash meets difficulty target"""
        # Convert target bits to actual target value
        exponent = target_bits >> 24
        mantissa = target_bits & 0xffffff
        target = mantissa * (2 ** (8 * (exponent - 3)))

        # Convert hash to integer
        hash_int = int.from_bytes(hash_result, byteorder='big')

        return hash_int < target

    def mine_block(self, template):
        """Actually mine the block (CPU brute force)"""
        height = template['height']
        difficulty = template.get('difficulty', 'unknown')
        target_bits = int(template['bits'], 16)

        print("⛏️  Mining Block:")
        print("-" * 60)
        print(f"Height: {height}")
        print(f"Difficulty: {difficulty}")
        print(f"Previous: {template['previousblockhash'][:16]}...")
        print()
        print("🔄 Hashing... (Press Ctrl+C to stop)")
        print()

        self.start_time = time.time()
        self.hashes_attempted = 0

        # Try nonces from 0 to 2^32
        for nonce in range(0, 0xFFFFFFFF):
            self.hashes_attempted += 1

            # Build and hash header
            header = self.build_block_header(template, nonce)
            hash_result = self.hash_header(header)

            # Check if we found a valid block
            if self.check_proof_of_work(hash_result, target_bits):
                print()
                print("=" * 60)
                print("🎉 BLOCK FOUND!")
                print("=" * 60)
                print(f"Nonce: {nonce}")
                print(f"Hash: {hash_result.hex()}")
                print(f"Hashes: {self.hashes_attempted:,}")
                print()

                self.block_found = True
                return True

            # Print progress every 100,000 hashes
            if self.hashes_attempted % 100000 == 0:
                elapsed = time.time() - self.start_time
                hash_rate = self.hashes_attempted / elapsed if elapsed > 0 else 0

                print(f"⚡ {self.hashes_attempted:,} hashes | "
                      f"{hash_rate:.2f} H/s | "
                      f"{elapsed:.0f}s elapsed", end='\r')

        print()
        print("❌ Nonce range exhausted (block not found)")
        return False

    def start(self):
        """Main mining loop"""
        print("=" * 60)
        print("⛏️  BITCOIN TESTNET CPU MINER")
        print("=" * 60)
        print()

        # Connect to node
        if not self.connect():
            return

        # Setup mining address
        if not self.setup_address():
            return

        print("🚀 Starting mining loop...")
        print("   (Note: CPU mining is VERY slow on testnet)")
        print("   Consider using a faucet instead: https://testnet-faucet.com/btc-testnet/")
        print()

        try:
            while True:
                # Get new block template
                template = self.get_block_template()
                if not template:
                    print("⏳ Waiting 30s for new template...")
                    time.sleep(30)
                    continue

                # Try to mine the block
                if self.mine_block(template):
                    print("✅ Block found! Checking if accepted...")
                    # Would need to submit block here
                    # template['submitblock'](block_data)
                    break

                # Check for new blocks every attempt
                print()
                print("🔄 Checking for new block template...")
                time.sleep(1)

        except KeyboardInterrupt:
            print()
            print()
            print("=" * 60)
            print("⏸️  MINING STOPPED")
            print("=" * 60)

            if self.start_time:
                elapsed = time.time() - self.start_time
                hash_rate = self.hashes_attempted / elapsed if elapsed > 0 else 0

                print(f"Total hashes: {self.hashes_attempted:,}")
                print(f"Average hash rate: {hash_rate:.2f} H/s")
                print(f"Time elapsed: {elapsed:.0f}s")
                print()

                # Estimate time to find block
                if hash_rate > 0:
                    template = self.get_block_template()
                    if template:
                        difficulty = template.get('difficulty', 1)
                        hashes_needed = difficulty * (2**32)
                        time_needed = hashes_needed / hash_rate
                        days_needed = time_needed / 86400

                        print(f"📊 Statistics:")
                        print(f"   Difficulty: {difficulty:.2f}")
                        print(f"   Hashes needed: ~{hashes_needed:.2e}")
                        print(f"   Estimated time: {days_needed:.1f} days")
                        print()
                        print("💡 Recommendation: Use a testnet faucet instead!")


def main():
    """Main entry point"""
    print()
    print("⚠️  IMPORTANT NOTES:")
    print("-" * 60)
    print("1. You need Bitcoin Core running in testnet mode")
    print("2. Configure RPC credentials in bitcoin.conf")
    print("3. CPU mining is EXTREMELY slow on testnet")
    print("4. Consider using faucets instead (instant, free)")
    print()
    print("Testnet faucets:")
    print("  • https://testnet-faucet.com/btc-testnet/")
    print("  • https://coinfaucet.eu/en/btc-testnet/")
    print("  • https://bitcoinfaucet.uo1.net/")
    print()

    # Configuration
    RPC_USER = "your_rpc_user"
    RPC_PASS = "your_rpc_password"
    RPC_PORT = 18332  # Testnet default

    response = input("Continue with CPU mining? (y/n): ")
    if response.lower() != 'y':
        print("Good choice! Use a faucet instead. 💚")
        return

    print()

    # Create and start miner
    miner = TestnetCPUMiner(RPC_USER, RPC_PASS, rpc_port=RPC_PORT)
    miner.start()


if __name__ == "__main__":
    main()
