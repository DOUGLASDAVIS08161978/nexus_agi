#!/usr/bin/env python3
"""
Esplora API Examples
Demonstrates all features of the Blockstream Esplora API

Based on: https://github.com/Blockstream/esplora/blob/master/API.md

Usage:
    python esplora_api_examples.py
"""

import sys
import json
from esplora_withdrawal_bridge import EsploraAPIClient


def example_transactions():
    """Example: Transaction queries"""
    print("\n" + "="*80)
    print("📝 TRANSACTION EXAMPLES")
    print("="*80)

    client = EsploraAPIClient("testnet")

    # Example testnet transaction
    txid = "96f32efcb5d81fd22383213312ad96c30b48ff336cf135cddc4af90341a204f3"

    print(f"\n1. Get Transaction Info (GET /tx/:txid)")
    tx_info = client.get_transaction(txid)
    print(f"   TXID: {tx_info['txid'][:16]}...")
    print(f"   Size: {tx_info['size']} bytes")
    print(f"   Weight: {tx_info['weight']} WU")
    print(f"   Fee: {tx_info['fee']} sats")
    print(f"   Inputs: {len(tx_info['vin'])}")
    print(f"   Outputs: {len(tx_info['vout'])}")

    print(f"\n2. Get Transaction Status (GET /tx/:txid/status)")
    tx_status = client.get_transaction_status(txid)
    print(f"   Confirmed: {tx_status['confirmed']}")
    if tx_status['confirmed']:
        print(f"   Block Height: {tx_status['block_height']}")
        print(f"   Block Hash: {tx_status['block_hash'][:16]}...")

    print(f"\n3. Get Transaction Hex (GET /tx/:txid/hex)")
    tx_hex = client.get_transaction_hex(txid)
    print(f"   Hex Length: {len(tx_hex)} characters")
    print(f"   Hex Preview: {tx_hex[:64]}...")


def example_addresses():
    """Example: Address queries"""
    print("\n" + "="*80)
    print("📬 ADDRESS EXAMPLES")
    print("="*80)

    client = EsploraAPIClient("testnet")

    # Example testnet address (from previous queries)
    address = "tb1qw508d6qejxtdg4y5r3zarvary0c5xw7kxpjzsx"

    print(f"\n1. Get Address Info (GET /address/:address)")
    addr_info = client.get_address_info(address)
    print(f"   Address: {address}")
    print(f"   Chain Tx Count: {addr_info['chain_stats']['tx_count']}")
    print(f"   Chain Funded: {addr_info['chain_stats']['funded_txo_sum']} sats")
    print(f"   Chain Spent: {addr_info['chain_stats']['spent_txo_sum']} sats")
    print(f"   Mempool Tx Count: {addr_info['mempool_stats']['tx_count']}")

    print(f"\n2. Get Address UTXOs (GET /address/:address/utxo)")
    utxos = client.get_address_utxos(address)
    total_value = sum(utxo['value'] for utxo in utxos)
    print(f"   Total UTXOs: {len(utxos)}")
    print(f"   Total Value: {total_value:,} sats ({total_value/1e8:.8f} BTC)")

    if utxos:
        print(f"\n   First UTXO:")
        utxo = utxos[0]
        print(f"      TXID: {utxo['txid'][:16]}...")
        print(f"      Vout: {utxo['vout']}")
        print(f"      Value: {utxo['value']:,} sats")
        print(f"      Confirmed: {utxo['status']['confirmed']}")

    print(f"\n3. Get Address Transactions (GET /address/:address/txs)")
    txs = client.get_address_txs(address)
    print(f"   Recent Transactions: {len(txs)}")

    if txs:
        print(f"\n   Most Recent Transaction:")
        tx = txs[0]
        print(f"      TXID: {tx['txid'][:16]}...")
        print(f"      Fee: {tx['fee']} sats")
        print(f"      Size: {tx['size']} bytes")


def example_blocks():
    """Example: Block queries"""
    print("\n" + "="*80)
    print("⛏️  BLOCK EXAMPLES")
    print("="*80)

    client = EsploraAPIClient("testnet")

    print(f"\n1. Get Tip Height (GET /blocks/tip/height)")
    tip_height = client.get_tip_height()
    print(f"   Current Height: {tip_height:,}")

    print(f"\n2. Get Tip Hash (GET /blocks/tip/hash)")
    tip_hash = client.get_tip_hash()
    print(f"   Current Hash: {tip_hash}")

    print(f"\n3. Get Block Info (GET /block/:hash)")
    block = client.get_block(tip_hash)
    print(f"   Height: {block['height']:,}")
    print(f"   Timestamp: {block['timestamp']}")
    print(f"   Tx Count: {block['tx_count']}")
    print(f"   Size: {block['size']:,} bytes")
    print(f"   Weight: {block['weight']:,} WU")
    print(f"   Merkle Root: {block['merkle_root'][:16]}...")
    print(f"   Previous Block: {block['previousblockhash'][:16]}...")

    print(f"\n4. Get Block at Height (GET /block-height/:height)")
    block_hash = client.get_block_at_height(tip_height - 1)
    print(f"   Height {tip_height-1:,}: {block_hash[:16]}...")


def example_mempool():
    """Example: Mempool queries"""
    print("\n" + "="*80)
    print("💧 MEMPOOL EXAMPLES")
    print("="*80)

    client = EsploraAPIClient("testnet")

    print(f"\n1. Get Mempool Info (GET /mempool)")
    mempool = client.get_mempool_info()
    print(f"   Transaction Count: {mempool['count']:,}")
    print(f"   Total Size: {mempool['vsize']:,} vB")
    print(f"   Total Fees: {mempool['total_fee']:,} sats")

    print(f"\n   Fee Histogram:")
    for i, (fee_rate, size) in enumerate(mempool['fee_histogram'][:5], 1):
        print(f"      {i}. {fee_rate:8.2f} sat/vB: {size:,} vB")

    print(f"\n2. Get Mempool TXIDs (GET /mempool/txids)")
    txids = client.get_mempool_txids()
    print(f"   Total TXIDs: {len(txids)}")
    if txids:
        print(f"   Sample TXIDs:")
        for txid in txids[:3]:
            print(f"      {txid[:16]}...")


def example_fee_estimates():
    """Example: Fee estimation"""
    print("\n" + "="*80)
    print("💰 FEE ESTIMATION EXAMPLES")
    print("="*80)

    client = EsploraAPIClient("testnet")

    print(f"\n1. Get Fee Estimates (GET /fee-estimates)")
    estimates = client.get_fee_estimates()

    print(f"\n   Confirmation Target → Fee Rate (sat/vB)")
    print(f"   " + "-"*50)

    # Common targets
    targets = [1, 2, 3, 6, 12, 24, 144, 504, 1008]

    for target in targets:
        if str(target) in estimates:
            fee = estimates[str(target)]
            blocks_text = "block" if target == 1 else "blocks"

            if target == 1:
                time_est = "~10 min"
            elif target <= 6:
                time_est = f"~{target * 10} min"
            elif target <= 144:
                time_est = f"~{target // 6} hours"
            else:
                time_est = f"~{target // 144} days"

            print(f"   {target:4d} {blocks_text:6s} ({time_est:>10s}): {fee:8.2f} sat/vB")

    # Calculate costs for typical transaction sizes
    print(f"\n   Estimated Transaction Costs:")
    typical_fee = float(estimates.get('6', 1.0))

    tx_types = [
        ("Simple P2WPKH (1 in, 2 out)", 140),
        ("Complex P2WPKH (2 in, 2 out)", 207),
        ("Multi-sig 2-of-3 (2 in, 2 out)", 340),
    ]

    for tx_name, vbytes in tx_types:
        cost_sats = int(vbytes * typical_fee)
        cost_btc = cost_sats / 1e8
        print(f"      {tx_name:30s}: {cost_sats:6,} sats ({cost_btc:.8f} BTC)")


def example_network_comparison():
    """Example: Compare different networks"""
    print("\n" + "="*80)
    print("🌐 NETWORK COMPARISON")
    print("="*80)

    networks = ["mainnet", "testnet", "signet"]

    print(f"\n{'Network':12s} {'Height':>10s} {'Mempool':>10s} {'Fee (6 blks)':>15s}")
    print(f"{'-'*60}")

    for network in networks:
        try:
            client = EsploraAPIClient(network)

            height = client.get_tip_height()
            mempool = client.get_mempool_info()
            fees = client.get_fee_estimates()
            fee_6 = float(fees.get('6', 0))

            print(f"{network:12s} {height:>10,} {mempool['count']:>10,} {fee_6:>12.2f} sat/vB")

        except Exception as e:
            print(f"{network:12s} {'Error':>10s} {str(e)[:30]:>10s}")


def example_real_world_scenarios():
    """Example: Real-world usage scenarios"""
    print("\n" + "="*80)
    print("🔧 REAL-WORLD SCENARIOS")
    print("="*80)

    client = EsploraAPIClient("testnet")

    # Scenario 1: Check if address has funds
    print(f"\n1. Check if Address Has Funds")
    address = "tb1qw508d6qejxtdg4y5r3zarvary0c5xw7kxpjzsx"
    addr_info = client.get_address_info(address)

    balance = (
        addr_info['chain_stats']['funded_txo_sum'] -
        addr_info['chain_stats']['spent_txo_sum']
    )

    print(f"   Address: {address}")
    print(f"   Balance: {balance:,} sats ({balance/1e8:.8f} BTC)")
    print(f"   Has Funds: {'✓ Yes' if balance > 0 else '✗ No'}")

    # Scenario 2: Calculate optimal fee for urgent transaction
    print(f"\n2. Calculate Fee for Urgent Transaction (Next Block)")
    estimates = client.get_fee_estimates()
    urgent_fee = float(estimates.get('1', estimates.get('2', 1.0)))

    tx_size = 140  # vBytes for simple transaction

    print(f"   Target: Next block (~10 minutes)")
    print(f"   Fee Rate: {urgent_fee:.2f} sat/vB")
    print(f"   Transaction Size: {tx_size} vB")
    print(f"   Total Fee: {int(urgent_fee * tx_size):,} sats")

    # Scenario 3: Monitor transaction confirmation
    print(f"\n3. Monitor Transaction Status")
    txid = "96f32efcb5d81fd22383213312ad96c30b48ff336cf135cddc4af90341a204f3"
    status = client.get_transaction_status(txid)

    if status['confirmed']:
        current_height = client.get_tip_height()
        confirmations = current_height - status['block_height'] + 1

        print(f"   TXID: {txid[:16]}...")
        print(f"   Status: ✓ Confirmed")
        print(f"   Block Height: {status['block_height']:,}")
        print(f"   Confirmations: {confirmations:,}")

        # Security level based on confirmations
        if confirmations >= 6:
            security = "🟢 High (6+ confirmations)"
        elif confirmations >= 3:
            security = "🟡 Medium (3-5 confirmations)"
        else:
            security = "🟠 Low (1-2 confirmations)"

        print(f"   Security Level: {security}")
    else:
        print(f"   TXID: {txid[:16]}...")
        print(f"   Status: ⏳ Pending (in mempool)")

    # Scenario 4: Find spendable UTXOs
    print(f"\n4. Find Spendable UTXOs for Payment")
    utxos = client.get_address_utxos(address)
    confirmed_utxos = [u for u in utxos if u['status']['confirmed']]

    print(f"   Total UTXOs: {len(utxos)}")
    print(f"   Confirmed UTXOs: {len(confirmed_utxos)}")

    total_spendable = sum(u['value'] for u in confirmed_utxos)
    print(f"   Spendable Amount: {total_spendable:,} sats ({total_spendable/1e8:.8f} BTC)")


def main():
    """Run all examples"""

    print("\n" + "="*80)
    print("🌟 BLOCKSTREAM ESPLORA API - COMPREHENSIVE EXAMPLES")
    print("="*80)
    print("\nBased on: https://github.com/Blockstream/esplora/blob/master/API.md")
    print("Networks: Bitcoin Mainnet, Testnet, Signet, Liquid, Liquid Testnet")

    try:
        # Run all examples
        example_transactions()
        example_addresses()
        example_blocks()
        example_mempool()
        example_fee_estimates()
        example_network_comparison()
        example_real_world_scenarios()

        print("\n" + "="*80)
        print("✅ ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("="*80)
        print("\nFor more information:")
        print("   API Docs: https://github.com/Blockstream/esplora/blob/master/API.md")
        print("   Mainnet: https://blockstream.info/api/")
        print("   Testnet: https://blockstream.info/testnet/api/")
        print("="*80 + "\n")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
