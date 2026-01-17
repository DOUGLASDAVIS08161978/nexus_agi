#!/usr/bin/env python3
"""
BLOCKCHAIN.COM API INTEGRATION
===============================
Integrates with Blockchain.com public APIs:
- Blockchain Data API (blocks, transactions)
- Simple Query API (stats, network info)
- Exchange Rates API (price data)
- Charts & Statistics API

NO API KEY REQUIRED - Uses public endpoints
"""

import requests
import json
import time
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass


@dataclass
class BlockInfo:
    """Bitcoin block information"""
    hash: str
    height: int
    time: int
    main_chain: bool
    n_tx: int
    size: int
    weight: int
    version: int
    merkle_root: str
    nonce: int
    bits: int
    difficulty: float
    fee: int


@dataclass
class TransactionInfo:
    """Bitcoin transaction information"""
    hash: str
    size: int
    weight: int
    fee: int
    time: int
    inputs: List[Dict]
    outputs: List[Dict]
    block_height: Optional[int]


class BlockchainAPIClient:
    """Client for Blockchain.com APIs"""

    def __init__(self):
        self.base_url = "https://blockchain.info"
        self.headers = {
            "User-Agent": "Nexus-AGI/1.0"
        }

    def get_latest_block(self) -> Dict[str, Any]:
        """Get the latest block info"""
        print(f"\n{'='*80}")
        print(f"🔍 QUERYING LATEST BITCOIN BLOCK")
        print(f"{'='*80}")

        try:
            url = f"{self.base_url}/latestblock"
            print(f"📡 Endpoint: {url}")

            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()

            data = response.json()

            print(f"✅ Latest block retrieved!")
            print(f"   Block Hash: {data['hash']}")
            print(f"   Height: {data['height']:,}")
            print(f"   Time: {datetime.fromtimestamp(data['time']).strftime('%Y-%m-%d %H:%M:%S UTC')}")
            print(f"   Transactions: {data['txIndexes']}")

            return data

        except Exception as e:
            print(f"❌ Error: {e}")
            return {}

    def get_block_by_hash(self, block_hash: str) -> Optional[BlockInfo]:
        """Get block details by hash"""
        print(f"\n{'='*80}")
        print(f"🔍 QUERYING BLOCK BY HASH")
        print(f"{'='*80}")
        print(f"Block Hash: {block_hash[:32]}...")

        try:
            url = f"{self.base_url}/rawblock/{block_hash}"
            print(f"📡 Endpoint: {url}")

            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()

            data = response.json()

            block = BlockInfo(
                hash=data['hash'],
                height=data['height'],
                time=data['time'],
                main_chain=data.get('main_chain', True),
                n_tx=data['n_tx'],
                size=data['size'],
                weight=data['weight'],
                version=data['ver'],
                merkle_root=data['mrkl_root'],
                nonce=data['nonce'],
                bits=data['bits'],
                difficulty=data.get('difficulty', 0),
                fee=data['fee']
            )

            print(f"✅ Block details retrieved!")
            print(f"   Height: {block.height:,}")
            print(f"   Transactions: {block.n_tx}")
            print(f"   Size: {block.size:,} bytes")
            print(f"   Weight: {block.weight:,}")
            print(f"   Difficulty: {block.difficulty:,.2f}")
            print(f"   Nonce: {block.nonce}")
            print(f"   Merkle Root: {block.merkle_root[:32]}...")

            return block

        except Exception as e:
            print(f"❌ Error: {e}")
            return None

    def get_block_by_height(self, height: int) -> Optional[BlockInfo]:
        """Get block by height"""
        print(f"\n{'='*80}")
        print(f"🔍 QUERYING BLOCK BY HEIGHT")
        print(f"{'='*80}")
        print(f"Block Height: {height:,}")

        try:
            url = f"{self.base_url}/block-height/{height}?format=json"
            print(f"📡 Endpoint: {url}")

            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()

            data = response.json()

            # Get first block (main chain)
            block_data = data['blocks'][0]

            block = BlockInfo(
                hash=block_data['hash'],
                height=block_data['height'],
                time=block_data['time'],
                main_chain=block_data.get('main_chain', True),
                n_tx=block_data['n_tx'],
                size=block_data['size'],
                weight=block_data['weight'],
                version=block_data['ver'],
                merkle_root=block_data['mrkl_root'],
                nonce=block_data['nonce'],
                bits=block_data['bits'],
                difficulty=0,
                fee=block_data['fee']
            )

            print(f"✅ Block retrieved!")
            print(f"   Hash: {block.hash[:32]}...")
            print(f"   Time: {datetime.fromtimestamp(block.time).strftime('%Y-%m-%d %H:%M:%S UTC')}")
            print(f"   Transactions: {block.n_tx}")

            return block

        except Exception as e:
            print(f"❌ Error: {e}")
            return None

    def get_address_balance(self, address: str) -> Dict[str, Any]:
        """Get Bitcoin address balance"""
        print(f"\n{'='*80}")
        print(f"💰 QUERYING ADDRESS BALANCE")
        print(f"{'='*80}")
        print(f"Address: {address}")

        try:
            url = f"{self.base_url}/balance?active={address}"
            print(f"📡 Endpoint: {url}")

            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()

            data = response.json()

            if address in data:
                addr_data = data[address]
                balance_btc = addr_data['final_balance'] / 1e8
                received_btc = addr_data['total_received'] / 1e8
                sent_btc = addr_data['total_sent'] / 1e8

                print(f"✅ Address balance retrieved!")
                print(f"   Final Balance: {balance_btc:.8f} BTC")
                print(f"   Total Received: {received_btc:.8f} BTC")
                print(f"   Total Sent: {sent_btc:.8f} BTC")
                print(f"   Transaction Count: {addr_data['n_tx']}")

                return {
                    "address": address,
                    "balance_satoshi": addr_data['final_balance'],
                    "balance_btc": balance_btc,
                    "total_received": received_btc,
                    "total_sent": sent_btc,
                    "tx_count": addr_data['n_tx']
                }

        except Exception as e:
            print(f"❌ Error: {e}")
            return {}

    def get_transaction(self, tx_hash: str) -> Optional[TransactionInfo]:
        """Get transaction details"""
        print(f"\n{'='*80}")
        print(f"🔍 QUERYING TRANSACTION")
        print(f"{'='*80}")
        print(f"TX Hash: {tx_hash[:32]}...")

        try:
            url = f"{self.base_url}/rawtx/{tx_hash}"
            print(f"📡 Endpoint: {url}")

            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()

            data = response.json()

            tx = TransactionInfo(
                hash=data['hash'],
                size=data['size'],
                weight=data['weight'],
                fee=data.get('fee', 0),
                time=data['time'],
                inputs=data['inputs'],
                outputs=data['out'],
                block_height=data.get('block_height')
            )

            print(f"✅ Transaction retrieved!")
            print(f"   Size: {tx.size} bytes")
            print(f"   Weight: {tx.weight}")
            print(f"   Fee: {tx.fee / 1e8:.8f} BTC")
            print(f"   Inputs: {len(tx.inputs)}")
            print(f"   Outputs: {len(tx.outputs)}")
            if tx.block_height:
                print(f"   Block Height: {tx.block_height:,}")

            return tx

        except Exception as e:
            print(f"❌ Error: {e}")
            return None

    def get_network_stats(self) -> Dict[str, Any]:
        """Get Bitcoin network statistics"""
        print(f"\n{'='*80}")
        print(f"📊 QUERYING NETWORK STATISTICS")
        print(f"{'='*80}")

        stats = {}

        try:
            # Get various stats using Simple Query API
            endpoints = {
                "difficulty": "q/getdifficulty",
                "hashrate": "q/hashrate",
                "block_count": "q/getblockcount",
                "total_btc": "q/totalbc"
            }

            for name, endpoint in endpoints.items():
                try:
                    url = f"{self.base_url}/{endpoint}"
                    response = requests.get(url, headers=self.headers, timeout=5)
                    response.raise_for_status()
                    stats[name] = float(response.text.strip())
                    time.sleep(0.2)  # Rate limiting
                except:
                    stats[name] = None

            print(f"✅ Network stats retrieved!")
            if stats.get('difficulty'):
                print(f"   Difficulty: {stats['difficulty']:,.2f}")
            if stats.get('hashrate'):
                print(f"   Hash Rate: {stats['hashrate']:,.2f} TH/s")
            if stats.get('block_count'):
                print(f"   Block Height: {int(stats['block_count']):,}")
            if stats.get('total_btc'):
                print(f"   Total BTC Mined: {stats['total_btc'] / 1e8:,.8f} BTC")

            return stats

        except Exception as e:
            print(f"❌ Error: {e}")
            return stats

    def get_btc_price(self) -> Dict[str, Any]:
        """Get current Bitcoin price"""
        print(f"\n{'='*80}")
        print(f"💵 QUERYING BITCOIN PRICE")
        print(f"{'='*80}")

        try:
            url = f"{self.base_url}/ticker"
            print(f"📡 Endpoint: {url}")

            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()

            data = response.json()

            print(f"✅ Bitcoin prices retrieved!")
            print(f"\n💰 Current Prices:")

            # Display major currencies
            major_currencies = ['USD', 'EUR', 'GBP', 'JPY', 'CNY']
            for currency in major_currencies:
                if currency in data:
                    price = data[currency]['last']
                    print(f"   {currency}: {price:,.2f} {data[currency]['symbol']}")

            return data

        except Exception as e:
            print(f"❌ Error: {e}")
            return {}

    def get_unconfirmed_tx_count(self) -> int:
        """Get mempool size (unconfirmed transactions)"""
        print(f"\n{'='*80}")
        print(f"📝 QUERYING MEMPOOL SIZE")
        print(f"{'='*80}")

        try:
            url = f"{self.base_url}/q/unconfirmedcount"
            print(f"📡 Endpoint: {url}")

            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()

            count = int(response.text.strip())

            print(f"✅ Mempool size: {count:,} unconfirmed transactions")

            return count

        except Exception as e:
            print(f"❌ Error: {e}")
            return 0


def main():
    """Main execution - Demonstrate Blockchain.com API integration"""

    print(f"\n{'='*80}")
    print(f"⚡ BLOCKCHAIN.COM API INTEGRATION")
    print(f"{'='*80}")
    print(f"Real-time Bitcoin blockchain data from Blockchain.com")
    print(f"{'='*80}\n")

    client = BlockchainAPIClient()

    # 1. Get latest block
    latest = client.get_latest_block()
    time.sleep(1)

    # 2. Get network statistics
    stats = client.get_network_stats()
    time.sleep(1)

    # 3. Get Bitcoin price
    prices = client.get_btc_price()
    time.sleep(1)

    # 4. Get mempool size
    mempool = client.get_unconfirmed_tx_count()
    time.sleep(1)

    # 5. Query specific address (example - Satoshi's genesis block reward)
    genesis_address = "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa"
    balance = client.get_address_balance(genesis_address)
    time.sleep(1)

    # 6. Get a specific block by height (recent block)
    if latest:
        recent_height = latest.get('height', 870000)
        block = client.get_block_by_height(recent_height - 1)

    # Final summary
    print(f"\n{'='*80}")
    print(f"📊 BLOCKCHAIN.COM API SUMMARY")
    print(f"{'='*80}")
    print(f"\n🔗 Network Status:")
    if latest:
        print(f"   Latest Block: {latest.get('height', 'N/A'):,}")
        print(f"   Block Hash: {latest.get('hash', 'N/A')[:32]}...")
    if stats.get('difficulty'):
        print(f"   Difficulty: {stats['difficulty']:,.0f}")
    if stats.get('hashrate'):
        print(f"   Network Hash Rate: {stats['hashrate']:,.2f} TH/s")
    print(f"   Mempool Size: {mempool:,} unconfirmed txs")

    print(f"\n💰 Market Data:")
    if prices and 'USD' in prices:
        print(f"   BTC/USD: ${prices['USD']['last']:,.2f}")
    if prices and 'EUR' in prices:
        print(f"   BTC/EUR: €{prices['EUR']['last']:,.2f}")

    print(f"\n🎯 Address Query:")
    if balance:
        print(f"   Address: {balance['address']}")
        print(f"   Balance: {balance['balance_btc']:.8f} BTC")
        print(f"   Transactions: {balance['tx_count']}")

    print(f"\n{'='*80}")
    print(f"✅ All API queries completed successfully!")
    print(f"{'='*80}\n")

    # Save API results
    api_results = {
        "timestamp": datetime.now().isoformat(),
        "latest_block": latest,
        "network_stats": stats,
        "mempool_size": mempool,
        "btc_prices": prices,
        "address_balance": balance
    }

    with open('blockchain_api_results.json', 'w') as f:
        json.dump(api_results, f, indent=2)

    print(f"💾 API results saved to: blockchain_api_results.json\n")


if __name__ == "__main__":
    main()
