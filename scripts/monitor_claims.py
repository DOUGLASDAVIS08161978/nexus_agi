#!/usr/bin/env python3
"""
Monitor NexusRewardToken claims in real-time
Watches for new claims and displays notifications
"""
import time
import sys
import os
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.contract_interactor import ContractInteractor, ERC20Reader

# Configuration - UPDATE THESE AFTER DEPLOYMENT
CONTRACT_ADDRESS = "0xYourContractAddressHere"
NETWORK = "ethereum_sepolia"  # or "ethereum_mainnet"
CHECK_INTERVAL = 15  # seconds between checks


def get_total_claimers(interactor):
    """Get total unique claimers"""
    data = "0x0e3a2e15"  # totalClaimers()
    result = interactor.eth_call(NETWORK, CONTRACT_ADDRESS, data)
    if result and result.startswith("0x"):
        return int(result, 16)
    return None


def get_total_rewards_claimed(interactor):
    """Get total rewards claimed"""
    data = "0xd54ad2a1"  # totalRewardsClaimed()
    result = interactor.eth_call(NETWORK, CONTRACT_ADDRESS, data)
    if result and result.startswith("0x"):
        return int(result, 16) / 10**18
    return None


def get_reward_pool_balance(reader):
    """Get available rewards in pool"""
    balance = reader.get_balance(NETWORK, CONTRACT_ADDRESS, CONTRACT_ADDRESS)
    return balance if balance is not None else 0


def main():
    """Monitor contract for new claims"""
    interactor = ContractInteractor()
    reader = ERC20Reader()

    print("=" * 80)
    print("NEXUS REWARD TOKEN - REAL-TIME CLAIM MONITOR")
    print("=" * 80)
    print(f"\n📡 Monitoring Configuration:")
    print(f"   Contract: {CONTRACT_ADDRESS}")
    print(f"   Network: {NETWORK}")
    print(f"   Check Interval: {CHECK_INTERVAL} seconds")
    print(f"\n⏳ Starting monitoring... (Press Ctrl+C to stop)\n")
    print("-" * 80)

    # Get initial state
    last_claimers = get_total_claimers(interactor)
    last_claimed = get_total_rewards_claimed(interactor)
    last_pool_balance = get_reward_pool_balance(reader)

    if last_claimers is None or last_claimed is None:
        print("❌ Error: Could not connect to contract")
        print("   Please check CONTRACT_ADDRESS and NETWORK settings")
        return

    print(f"[{datetime.now().strftime('%H:%M:%S')}] 📊 Initial State:")
    print(f"   Total Claimers: {last_claimers:,}")
    print(f"   Total Claimed: {last_claimed:,.2f} NREW")
    print(f"   Pool Balance: {last_pool_balance:,.2f} NREW")
    print("-" * 80)
    print()

    claim_count = 0

    try:
        while True:
            # Get current state
            total_claimers = get_total_claimers(interactor)
            total_claimed = get_total_rewards_claimed(interactor)
            pool_balance = get_reward_pool_balance(reader)

            if total_claimers is None or total_claimed is None:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️  Connection error, retrying...")
                time.sleep(CHECK_INTERVAL)
                continue

            # Check for new claims
            new_claimers = total_claimers != last_claimers
            new_amount = total_claimed != last_claimed
            pool_changed = pool_balance != last_pool_balance

            if new_claimers or new_amount or pool_changed:
                claim_count += 1
                timestamp = datetime.now().strftime('%H:%M:%S')

                print(f"[{timestamp}] 🎉 Activity Detected! (Event #{claim_count})")

                if new_claimers:
                    claimers_diff = total_claimers - last_claimers
                    print(f"   👤 New Claimers: +{claimers_diff:,} (Total: {total_claimers:,})")

                if new_amount:
                    amount_diff = total_claimed - last_claimed
                    print(f"   💰 Claimed: +{amount_diff:,.2f} NREW (Total: {total_claimed:,.2f} NREW)")

                if pool_changed:
                    pool_diff = pool_balance - last_pool_balance
                    change_type = "increased" if pool_diff > 0 else "decreased"
                    print(f"   🏦 Pool {change_type}: {pool_diff:+,.2f} NREW (Available: {pool_balance:,.2f} NREW)")

                    # Alert if pool is low
                    if pool_balance < 1000:
                        print(f"   ⚠️  WARNING: Low pool balance! ({pool_balance:,.2f} NREW remaining)")

                print("-" * 80)
                print()

                # Update last values
                last_claimers = total_claimers
                last_claimed = total_claimed
                last_pool_balance = pool_balance

            # Sleep until next check
            time.sleep(CHECK_INTERVAL)

    except KeyboardInterrupt:
        print("\n")
        print("=" * 80)
        print("🛑 Monitoring Stopped")
        print("=" * 80)
        print(f"\n📊 Session Summary:")
        print(f"   Events Detected: {claim_count:,}")
        print(f"   Final Claimers: {total_claimers:,}")
        print(f"   Final Total Claimed: {total_claimed:,.2f} NREW")
        print(f"   Final Pool Balance: {pool_balance:,.2f} NREW")
        print("\n✅ Monitor closed successfully\n")


if __name__ == "__main__":
    main()
