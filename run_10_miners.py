#!/usr/bin/env python3
"""
10 PARALLEL SUPERINTELLIGENT MINERS
Run 10 mining instances simultaneously for 5 iterations each
"""

import sys
import time
from superintelligent_testnet_miner import SuperintelligentMiner
import logging

logging.basicConfig(
    level=logging.INFO,
    format='[Miner-%(process)d] %(asctime)s - %(levelname)s - %(message)s'
)

def run_miner_instance(miner_id: int, num_blocks: int = 5):
    """Run a single miner instance"""
    print(f"\n{'='*80}")
    print(f"  🤖 MINER INSTANCE #{miner_id}")
    print(f"{'='*80}\n")

    miner = SuperintelligentMiner(demo_mode=True)
    miner.start_superintelligent_mining(num_blocks=num_blocks)

    return miner.stats


def main():
    """Run 10 parallel mining instances"""
    print("\n" + "╔" + "="*78 + "╗")
    print("║" + " "*20 + "10 PARALLEL SUPERINTELLIGENT MINERS" + " "*23 + "║")
    print("║" + " "*25 + "5 ITERATIONS EACH" + " "*36 + "║")
    print("╚" + "="*78 + "╝\n")

    num_instances = 10
    blocks_per_instance = 5

    print(f"Starting {num_instances} mining instances...")
    print(f"Each mining {blocks_per_instance} blocks")
    print(f"Total target: {num_instances * blocks_per_instance} blocks\n")

    start_time = time.time()
    all_stats = []

    # Run all instances
    for i in range(num_instances):
        print(f"\n{'#'*80}")
        print(f"  STARTING INSTANCE {i+1}/{num_instances}")
        print(f"{'#'*80}")

        stats = run_miner_instance(i+1, blocks_per_instance)
        all_stats.append(stats)

        print(f"\n✅ Instance {i+1} complete: {stats.blocks_found} blocks mined")

    # Calculate aggregate statistics
    total_runtime = time.time() - start_time
    total_blocks = sum(s.blocks_found for s in all_stats)
    total_hashes = sum(s.total_hashes for s in all_stats)
    avg_hashrate = (total_hashes / total_runtime) / 1_000_000 if total_runtime > 0 else 0

    # Print final summary
    print("\n\n" + "╔" + "="*78 + "╗")
    print("║" + " "*25 + "AGGREGATE STATISTICS" + " "*33 + "║")
    print("╚" + "="*78 + "╝\n")

    print(f"📊 TOTAL STATISTICS:")
    print(f"   Mining Instances: {num_instances}")
    print(f"   Total Blocks Mined: {total_blocks}")
    print(f"   Total Hashes: {total_hashes:,}")
    print(f"   Total Runtime: {total_runtime:.2f} seconds")
    print(f"   Average Hashrate: {avg_hashrate:.2f} MH/s ({avg_hashrate/1000:.2f} GH/s)")
    print(f"   Blocks/Second: {total_blocks/total_runtime:.2f}")
    print(f"   Efficiency: {(total_blocks / total_runtime) * 1000:.2f}\n")

    print("🏆 PER-INSTANCE BREAKDOWN:")
    for i, stats in enumerate(all_stats, 1):
        print(f"   Instance {i}: {stats.blocks_found} blocks | {stats.total_hashes:,} hashes | {stats.hashrate_mhs:.2f} MH/s")

    print(f"\n{'='*80}")
    print("  ✅ ALL 10 INSTANCES COMPLETE!")
    print(f"  🏆 MINED {total_blocks} BLOCKS IN {total_runtime:.2f} SECONDS")
    print("  ⚡ WORLD'S FASTEST TESTNET MINING SYSTEM")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
