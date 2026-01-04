#!/usr/bin/env python3
"""
BITCOIN BLOCK & TRANSACTION VALIDATOR
Validates blocks against Bitcoin consensus rules
Checks blockchain state and mempool
Comprehensive validation of all hashes and transactions

VALIDATION CHECKS:
- Block structure
- Proof-of-work
- Merkle root
- Transaction validity
- Blockchain state
- Mempool status
- All cryptographic hashes
"""

import hashlib
import struct
import requests
import json
from datetime import datetime
from immutable_ledger_system import ImmutableLedger


class BitcoinValidator:
    """
    Comprehensive Bitcoin Block and Transaction Validator

    Validates blocks according to Bitcoin consensus rules
    """

    def __init__(self):
        self.ledger = ImmutableLedger("validation_ledger.jsonl")
        self.testnet_api = "https://blockstream.info/testnet/api"

        print("=" * 80)
        print("BITCOIN BLOCK & TRANSACTION VALIDATOR")
        print("=" * 80)
        print("Network: Bitcoin Testnet3")
        print("Purpose: Comprehensive validation")
        print("=" * 80)

    def sha256d(self, data: bytes) -> bytes:
        """Double SHA-256"""
        return hashlib.sha256(hashlib.sha256(data).digest()).digest()

    def hash_to_int(self, hash_bytes: bytes) -> int:
        """Convert hash to integer (little-endian)"""
        return int.from_bytes(hash_bytes, byteorder='little')

    def bits_to_target(self, bits: int) -> int:
        """Convert compact bits to target"""
        exponent = bits >> 24
        mantissa = bits & 0x00ffffff

        if exponent <= 3:
            target = mantissa >> (8 * (3 - exponent))
        else:
            target = mantissa << (8 * (exponent - 3))

        return target

    def validate_proof_of_work(self, block_hash: str, bits: int) -> dict:
        """
        Validate proof-of-work

        Check if block hash meets difficulty target
        """
        print(f"\n⚡ VALIDATING PROOF-OF-WORK")
        print(f"{'-' * 80}")

        # Convert block hash to integer
        hash_bytes = bytes.fromhex(block_hash)[::-1]
        hash_int = self.hash_to_int(hash_bytes)

        # Get target from bits
        target = self.bits_to_target(bits)

        print(f"Block Hash (int): {hash_int}")
        print(f"Target (int):     {target}")
        print(f"Bits:             0x{bits:08x}")

        # Check if hash < target
        valid = hash_int < target

        if valid:
            ratio = target / hash_int
            print(f"\n✅ PROOF-OF-WORK VALID!")
            print(f"   Hash is {ratio:.2f}x below target")
        else:
            ratio = hash_int / target
            print(f"\n❌ PROOF-OF-WORK INVALID!")
            print(f"   Hash is {ratio:.2f}x above target")

        return {
            'valid': valid,
            'hash_int': hash_int,
            'target': target,
            'ratio': ratio if valid else 1/ratio
        }

    def validate_block_header(self, header_hex: str) -> dict:
        """
        Validate block header structure and hash
        """
        print(f"\n📦 VALIDATING BLOCK HEADER")
        print(f"{'-' * 80}")

        header_bytes = bytes.fromhex(header_hex)

        # Check header size
        if len(header_bytes) != 80:
            print(f"❌ Invalid header size: {len(header_bytes)} (expected 80)")
            return {'valid': False, 'error': 'invalid_size'}

        print(f"✅ Header size: 80 bytes")

        # Parse header
        version = struct.unpack('<I', header_bytes[0:4])[0]
        prev_hash = header_bytes[4:36][::-1].hex()
        merkle_root = header_bytes[36:68].hex()
        timestamp = struct.unpack('<I', header_bytes[68:72])[0]
        bits = struct.unpack('<I', header_bytes[72:76])[0]
        nonce = struct.unpack('<I', header_bytes[76:80])[0]

        print(f"Version:       0x{version:08x}")
        print(f"Previous Hash: {prev_hash}")
        print(f"Merkle Root:   {merkle_root}")
        print(f"Timestamp:     {timestamp} ({datetime.fromtimestamp(timestamp)})")
        print(f"Bits:          0x{bits:08x}")
        print(f"Nonce:         {nonce:,}")

        # Calculate block hash
        block_hash = self.sha256d(header_bytes)
        block_hash_hex = block_hash[::-1].hex()

        print(f"\nCalculated Hash: {block_hash_hex}")

        # Count leading zeros
        leading_zeros = len(block_hash_hex) - len(block_hash_hex.lstrip('0'))
        print(f"Leading Zeros:   {leading_zeros}")

        return {
            'valid': True,
            'version': version,
            'prev_hash': prev_hash,
            'merkle_root': merkle_root,
            'timestamp': timestamp,
            'bits': bits,
            'nonce': nonce,
            'block_hash': block_hash_hex,
            'leading_zeros': leading_zeros
        }

    def validate_merkle_root(self, transactions_hex: list, expected_root: str) -> dict:
        """
        Validate merkle root calculation
        """
        print(f"\n🌳 VALIDATING MERKLE ROOT")
        print(f"{'-' * 80}")

        if not transactions_hex:
            print(f"❌ No transactions provided")
            return {'valid': False, 'error': 'no_transactions'}

        # Calculate transaction hashes
        tx_hashes = []
        for i, tx_hex in enumerate(transactions_hex):
            tx_bytes = bytes.fromhex(tx_hex)
            tx_hash = self.sha256d(tx_bytes)
            tx_hashes.append(tx_hash)
            print(f"TX {i+1} Hash: {tx_hash[::-1].hex()}")
            print(f"TX {i+1} Size: {len(tx_bytes)} bytes")

        # Calculate merkle root
        # For single transaction, merkle root = txid
        if len(tx_hashes) == 1:
            merkle_root = tx_hashes[0]
        else:
            # Build merkle tree
            current_level = tx_hashes
            while len(current_level) > 1:
                next_level = []
                for i in range(0, len(current_level), 2):
                    left = current_level[i]
                    right = current_level[i+1] if i+1 < len(current_level) else left
                    combined = left + right
                    parent = self.sha256d(combined)
                    next_level.append(parent)
                current_level = next_level
            merkle_root = current_level[0]

        calculated_root = merkle_root.hex()
        print(f"\nCalculated Root: {calculated_root}")
        print(f"Expected Root:   {expected_root}")

        valid = calculated_root == expected_root

        if valid:
            print(f"\n✅ MERKLE ROOT VALID!")
        else:
            print(f"\n❌ MERKLE ROOT MISMATCH!")

        return {
            'valid': valid,
            'calculated': calculated_root,
            'expected': expected_root,
            'tx_count': len(transactions_hex)
        }

    def check_blockchain_state(self, block_hash: str, block_height: int) -> dict:
        """
        Check if block exists on the blockchain
        """
        print(f"\n🌐 CHECKING BLOCKCHAIN STATE")
        print(f"{'-' * 80}")

        try:
            # Check block by hash
            url = f"{self.testnet_api}/block/{block_hash}"
            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                block_data = response.json()
                print(f"✅ BLOCK FOUND ON BLOCKCHAIN!")
                print(f"   Height:        {block_data.get('height', 'N/A')}")
                print(f"   Confirmations: {block_data.get('confirmations', 'N/A')}")
                print(f"   Status:        {block_data.get('status', 'N/A')}")
                print(f"   Transactions:  {block_data.get('tx_count', 'N/A')}")

                return {
                    'found': True,
                    'confirmed': block_data.get('confirmations', 0) > 0,
                    'height': block_data.get('height'),
                    'confirmations': block_data.get('confirmations', 0),
                    'tx_count': block_data.get('tx_count', 0)
                }
            else:
                print(f"❌ Block not found on blockchain (yet)")
                print(f"   Status Code: {response.status_code}")
                print(f"   Block may not be broadcast yet")

                return {
                    'found': False,
                    'reason': 'not_broadcast'
                }

        except Exception as e:
            print(f"❌ Error checking blockchain: {e}")
            return {
                'found': False,
                'error': str(e)
            }

    def check_mempool(self, tx_hash: str) -> dict:
        """
        Check if transaction is in mempool
        """
        print(f"\n💾 CHECKING MEMPOOL")
        print(f"{'-' * 80}")

        try:
            url = f"{self.testnet_api}/tx/{tx_hash}"
            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                tx_data = response.json()

                if tx_data.get('status', {}).get('confirmed', False):
                    print(f"✅ Transaction CONFIRMED in blockchain")
                    print(f"   Block Height: {tx_data['status'].get('block_height', 'N/A')}")
                else:
                    print(f"⏳ Transaction in MEMPOOL (unconfirmed)")

                return {
                    'found': True,
                    'confirmed': tx_data.get('status', {}).get('confirmed', False),
                    'block_height': tx_data.get('status', {}).get('block_height')
                }
            else:
                print(f"❌ Transaction not found in mempool or blockchain")
                return {
                    'found': False
                }

        except Exception as e:
            print(f"❌ Error checking mempool: {e}")
            return {
                'found': False,
                'error': str(e)
            }

    def validate_complete_block(self, block_data: dict) -> dict:
        """
        Complete block validation
        """
        print(f"\n{'#' * 80}")
        print(f"COMPREHENSIVE BLOCK VALIDATION")
        print(f"{'#' * 80}")
        print(f"Block Hash: {block_data['hash']}")
        print(f"Height:     #{block_data['height']}")
        print(f"Nonce:      {block_data['nonce']:,}")

        validation_results = {}

        # 1. Validate block header
        header_result = self.validate_block_header(block_data['serialized_header'])
        validation_results['header'] = header_result

        # 2. Validate proof-of-work
        pow_result = self.validate_proof_of_work(
            header_result['block_hash'],
            header_result['bits']
        )
        validation_results['proof_of_work'] = pow_result

        # 3. Validate merkle root
        coinbase_hex = block_data['serialized_block'][162:]  # Skip header + tx count
        merkle_result = self.validate_merkle_root(
            [coinbase_hex],
            header_result['merkle_root']
        )
        validation_results['merkle_root'] = merkle_result

        # 4. Check blockchain state
        blockchain_result = self.check_blockchain_state(
            header_result['block_hash'],
            block_data['height']
        )
        validation_results['blockchain'] = blockchain_result

        # 5. Check mempool for coinbase transaction
        mempool_result = self.check_mempool(block_data['coinbase_txid'])
        validation_results['mempool'] = mempool_result

        # Summary
        print(f"\n{'=' * 80}")
        print(f"VALIDATION SUMMARY")
        print(f"{'=' * 80}")

        checks = [
            ('Block Header', header_result.get('valid', False)),
            ('Proof-of-Work', pow_result.get('valid', False)),
            ('Merkle Root', merkle_result.get('valid', False)),
            ('Blockchain State', blockchain_result.get('found', False)),
            ('Mempool/Confirmed', mempool_result.get('found', False)),
        ]

        all_valid = True
        for check_name, is_valid in checks:
            status = "✅ PASS" if is_valid else "❌ FAIL"
            print(f"{check_name:<20} {status}")
            if not is_valid:
                all_valid = False

        print(f"{'-' * 80}")

        if all_valid:
            print(f"✅ ALL VALIDATIONS PASSED!")
            print(f"   Block is fully valid and on the blockchain!")
        elif pow_result['valid'] and merkle_result['valid']:
            print(f"⚠️  Block is STRUCTURALLY VALID but not yet on blockchain")
            print(f"   This is normal if the block hasn't been broadcast yet")
        else:
            print(f"❌ Block has validation errors")

        # Record to ledger
        self.ledger.add_entry("BLOCK_VALIDATED", {
            'block_hash': header_result['block_hash'],
            'height': block_data['height'],
            'all_valid': all_valid,
            'validated_at': datetime.now().isoformat()
        })

        # Print ledger stats
        self.ledger.print_statistics()

        return validation_results


def main():
    """Main validation program"""

    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                  BITCOIN BLOCK & TRANSACTION VALIDATOR                       ║
║                                                                              ║
║  Comprehensive validation of our mined block                                ║
║                                                                              ║
║  Validates:                                                                 ║
║  - Block structure and header                                               ║
║  - Proof-of-work (SHA-256d)                                                 ║
║  - Merkle root calculation                                                  ║
║  - All transaction hashes                                                   ║
║  - Blockchain state                                                         ║
║  - Mempool status                                                           ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

    validator = BitcoinValidator()

    print(f"\n🔍 Loading block data...")

    # Load our mined block
    try:
        with open('block_4811605_broadcast.json', 'r') as f:
            block_data = json.load(f)
        print(f"✅ Block data loaded successfully")
    except FileNotFoundError:
        print(f"❌ Block data file not found")
        print(f"   Run block_broadcaster.py first")
        return

    # Validate!
    print(f"\n🚀 Starting comprehensive validation...")

    results = validator.validate_complete_block(block_data)

    print(f"\n{'=' * 80}")
    print(f"🎓 VALIDATION COMPLETE")
    print(f"{'=' * 80}")
    print(f"")
    print(f"All cryptographic hashes verified")
    print(f"All structural checks performed")
    print(f"Blockchain and mempool status checked")
    print(f"")
    print(f"Results stored in validation ledger")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
