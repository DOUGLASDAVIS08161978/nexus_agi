#!/usr/bin/env python3
"""
MASSIVE BITCOIN MINING DEPLOYMENT
10 Parallel Instances × 10 Iterations Each = 100 Mining Attempts

FEATURES:
- 10 concurrent mining instances
- 10 iterations per instance
- Real-time output from all instances
- Aggregated statistics
- Multi-core utilization
- Immutable ledger recording
"""

import hashlib
import struct
import time
import requests
from datetime import datetime
from multiprocessing import Pool, Manager, cpu_count
from immutable_ledger_system import ImmutableLedger


def sha256d(data: bytes) -> bytes:
    """Double SHA-256"""
    return hashlib.sha256(hashlib.sha256(data).digest()).digest()


def hash_to_int(hash_bytes: bytes) -> int:
    """Convert hash to integer"""
    return int.from_bytes(hash_bytes, byteorder='little')


def bits_to_target(bits: int) -> int:
    """Convert compact bits to target"""
    exponent = bits >> 24
    mantissa = bits & 0x00ffffff
    if exponent <= 3:
        target = mantissa >> (8 * (3 - exponent))
    else:
        target = mantissa << (8 * (exponent - 3))
    return target


def get_block_info():
    """Get current testnet block info"""
    try:
        testnet_api = "https://blockstream.info/testnet/api"

        url = f"{testnet_api}/blocks/tip/height"
        response = requests.get(url, timeout=10)
        height = int(response.text.strip())

        url = f"{testnet_api}/blocks/tip/hash"
        response = requests.get(url, timeout=10)
        block_hash = response.text.strip()

        url = f"{testnet_api}/block/{block_hash}"
        response = requests.get(url, timeout=10)
        data = response.json()

        return {
            'height': height,
            'hash': block_hash,
            'bits': data.get('bits', 0x1d00ffff),
            'difficulty': data.get('difficulty', 1.0)
        }
    except:
        return None


def mine_chunk(args):
    """Mine a chunk of nonces"""
    instance_id, iteration, start_nonce, chunk_size, header_template, target = args

    best_hash = 2**256
    best_nonce = 0
    hashes = 0

    for nonce in range(start_nonce, start_nonce + chunk_size):
        header = header_template + struct.pack('<I', nonce)
        block_hash = sha256d(header)
        hash_int = hash_to_int(block_hash)
        hashes += 1

        if hash_int < best_hash:
            best_hash = hash_int
            best_nonce = nonce

        if hash_int < target:
            return {
                'found': True,
                'instance_id': instance_id,
                'iteration': iteration,
                'nonce': nonce,
                'hash': block_hash[::-1].hex(),
                'hashes': hashes,
                'best_hash': best_hash
            }

    return {
        'found': False,
        'instance_id': instance_id,
        'iteration': iteration,
        'hashes': hashes,
        'best_hash': best_hash,
        'best_nonce': best_nonce
    }


def run_instance(args):
    """Run a single mining instance with multiple iterations"""
    instance_id, iterations, hashes_per_iteration = args

    print(f"\n{'=' * 80}")
    print(f"🚀 INSTANCE #{instance_id} STARTING - {iterations} iterations")
    print(f"{'=' * 80}")

    results = []
    total_hashes = 0
    instance_start = time.time()

    # Get block info
    block_info = get_block_info()
    if not block_info:
        return {'instance_id': instance_id, 'error': 'no_block_info'}

    # Build header template
    version = 0x20000000
    prev_hash = bytes.fromhex(block_info['hash'])[::-1]

    # Simplified coinbase
    height = block_info['height'] + 1
    tx = struct.pack('<I', 2) + b'\x01' + b'\x00' * 32 + b'\xff\xff\xff\xff'
    height_bytes = struct.pack('<I', height)[:3]
    script = bytes([len(height_bytes)]) + height_bytes + b'\x00' * 4
    tx += bytes([len(script)]) + script + b'\xff\xff\xff\xff'
    tx += b'\x01' + struct.pack('<Q', 625000000)
    tx += b'\x16\x00\x14' + b'\x00' * 20 + b'\x00\x00\x00\x00'

    merkle_root = sha256d(tx)
    timestamp = int(time.time())
    bits = block_info['bits']
    target = bits_to_target(bits)

    header_template = struct.pack('<I', version)
    header_template += prev_hash
    header_template += merkle_root
    header_template += struct.pack('<I', timestamp)
    header_template += struct.pack('<I', bits)

    print(f"Instance #{instance_id}: Block #{height}, Target difficulty: {block_info['difficulty']}")

    # Run iterations
    for iteration in range(iterations):
        iter_start = time.time()

        # Random starting nonce for this iteration
        import random
        start_nonce = random.randint(0, 2**32 - hashes_per_iteration)

        result = mine_chunk((
            instance_id,
            iteration + 1,
            start_nonce,
            hashes_per_iteration,
            header_template,
            target
        ))

        iter_time = time.time() - iter_start
        hashrate = result['hashes'] / iter_time if iter_time > 0 else 0
        total_hashes += result['hashes']

        distance = result['best_hash'] / target

        status = "🎉 BLOCK FOUND!" if result['found'] else f"⚡ {distance:.2f}x away"

        print(f"Instance #{instance_id} Iter {iteration + 1}/{iterations}: "
              f"{result['hashes']:,} hashes | {hashrate:,.0f} H/s | {status}")

        results.append(result)

        if result['found']:
            print(f"\n{'!' * 80}")
            print(f"🏆 INSTANCE #{instance_id} FOUND BLOCK ON ITERATION {iteration + 1}!")
            print(f"{'!' * 80}")
            print(f"Block Hash: {result['hash']}")
            print(f"Nonce: {result['nonce']:,}")
            print(f"{'!' * 80}\n")

    instance_time = time.time() - instance_start
    avg_hashrate = total_hashes / instance_time if instance_time > 0 else 0

    blocks_found = sum(1 for r in results if r['found'])

    print(f"\n{'=' * 80}")
    print(f"✅ INSTANCE #{instance_id} COMPLETE")
    print(f"{'=' * 80}")
    print(f"Iterations: {iterations}")
    print(f"Total Hashes: {total_hashes:,}")
    print(f"Time: {instance_time:.2f}s")
    print(f"Avg Hashrate: {avg_hashrate:,.0f} H/s")
    print(f"Blocks Found: {blocks_found}")
    print(f"{'=' * 80}\n")

    return {
        'instance_id': instance_id,
        'iterations': iterations,
        'total_hashes': total_hashes,
        'time': instance_time,
        'hashrate': avg_hashrate,
        'blocks_found': blocks_found,
        'results': results
    }


class MassiveMiningDeployment:
    """
    Massive Mining Deployment System

    Deploys 10 parallel instances, each running 10 iterations
    """

    def __init__(self):
        self.ledger = ImmutableLedger("massive_deployment_ledger.jsonl")
        self.num_instances = 10
        self.iterations_per_instance = 10
        self.hashes_per_iteration = 10_000_000  # 10M per iteration

        print("=" * 80)
        print("MASSIVE BITCOIN MINING DEPLOYMENT")
        print("=" * 80)
        print(f"Instances:       {self.num_instances}")
        print(f"Iterations/Inst: {self.iterations_per_instance}")
        print(f"Hashes/Iter:     {self.hashes_per_iteration:,}")
        print(f"Total Attempts:  {self.num_instances * self.iterations_per_instance}")
        print(f"Total Hashes:    {self.num_instances * self.iterations_per_instance * self.hashes_per_iteration:,}")
        print(f"CPU Cores:       {cpu_count()}")
        print("=" * 80)

    def deploy(self):
        """Deploy all instances"""
        print(f"\n{'#' * 80}")
        print(f"DEPLOYING {self.num_instances} PARALLEL MINING INSTANCES")
        print(f"{'#' * 80}\n")

        self.ledger.add_entry("DEPLOYMENT_START", {
            'instances': self.num_instances,
            'iterations_per_instance': self.iterations_per_instance,
            'hashes_per_iteration': self.hashes_per_iteration,
            'started_at': datetime.now().isoformat()
        })

        deployment_start = time.time()

        # Prepare work for all instances
        work_items = [
            (i + 1, self.iterations_per_instance, self.hashes_per_iteration)
            for i in range(self.num_instances)
        ]

        # Deploy in parallel
        print(f"🚀 Launching {self.num_instances} instances in parallel...\n")

        with Pool(processes=min(self.num_instances, cpu_count())) as pool:
            instance_results = pool.map(run_instance, work_items)

        deployment_time = time.time() - deployment_start

        # Aggregate results
        print(f"\n{'#' * 80}")
        print(f"DEPLOYMENT COMPLETE - AGGREGATING RESULTS")
        print(f"{'#' * 80}\n")

        total_hashes = sum(r.get('total_hashes', 0) for r in instance_results if 'error' not in r)
        total_blocks = sum(r.get('blocks_found', 0) for r in instance_results if 'error' not in r)
        total_hashrate = total_hashes / deployment_time if deployment_time > 0 else 0

        print(f"{'=' * 80}")
        print(f"FINAL STATISTICS")
        print(f"{'=' * 80}")
        print(f"Instances Deployed:  {self.num_instances}")
        print(f"Total Iterations:    {self.num_instances * self.iterations_per_instance}")
        print(f"Total Hashes:        {total_hashes:,}")
        print(f"Total Time:          {deployment_time:.2f}s")
        print(f"Combined Hashrate:   {total_hashrate:,.0f} H/s")
        print(f"Blocks Found:        {total_blocks}")
        print(f"{'=' * 80}\n")

        # Show per-instance summary
        print(f"PER-INSTANCE SUMMARY:")
        print(f"{'-' * 80}")
        for result in instance_results:
            if 'error' in result:
                print(f"Instance #{result['instance_id']}: ❌ Error")
            else:
                print(f"Instance #{result['instance_id']}: "
                      f"{result['total_hashes']:,} hashes | "
                      f"{result['hashrate']:,.0f} H/s | "
                      f"{result['blocks_found']} blocks")
        print(f"{'-' * 80}\n")

        # Record completion
        self.ledger.add_entry("DEPLOYMENT_COMPLETE", {
            'instances': self.num_instances,
            'total_iterations': self.num_instances * self.iterations_per_instance,
            'total_hashes': total_hashes,
            'deployment_time': deployment_time,
            'combined_hashrate': total_hashrate,
            'blocks_found': total_blocks,
            'completed_at': datetime.now().isoformat()
        })

        # Print ledger
        self.ledger.print_statistics()

        return instance_results


def main():
    """Main deployment program"""

    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                  MASSIVE BITCOIN MINING DEPLOYMENT                           ║
║                                                                              ║
║  10 Parallel Instances × 10 Iterations Each                                 ║
║                                                                              ║
║  Total Mining Attempts:  100                                                ║
║  Total Hashes:           1,000,000,000 (1 BILLION!)                         ║
║                                                                              ║
║  This deployment will utilize all available CPU cores                       ║
║  to maximize mining throughput across 10 simultaneous                       ║
║  mining instances, each running 10 iterations.                              ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

    deployment = MassiveMiningDeployment()

    print(f"\n🚀 Initializing massive deployment...")
    print(f"⏳ This will take several minutes...")
    print(f"💪 Maximum CPU utilization engaged!\n")

    time.sleep(2)

    # DEPLOY!
    results = deployment.deploy()

    # Final summary
    blocks_found = sum(r.get('blocks_found', 0) for r in results if 'error' not in r)

    print(f"\n{'#' * 80}")
    if blocks_found > 0:
        print(f"🎉🎉🎉 SUCCESS! {blocks_found} BLOCK(S) FOUND! 🎉🎉🎉")
        print(f"{'#' * 80}")
        for result in results:
            if 'error' not in result and result.get('blocks_found', 0) > 0:
                for iter_result in result['results']:
                    if iter_result['found']:
                        print(f"Instance #{result['instance_id']} Iteration {iter_result['iteration']}:")
                        print(f"  Hash: {iter_result['hash']}")
                        print(f"  Nonce: {iter_result['nonce']:,}")
    else:
        print(f"DEPLOYMENT COMPLETE - NO BLOCKS FOUND")
        print(f"{'#' * 80}")
        print(f"All 100 mining attempts completed successfully")
        print(f"Continue mining for eventual block discovery")
    print(f"{'#' * 80}\n")


if __name__ == "__main__":
    main()
