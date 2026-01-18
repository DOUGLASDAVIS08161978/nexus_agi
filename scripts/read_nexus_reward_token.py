#!/usr/bin/env python3
"""
Read NexusRewardToken contract state
Uses the ContractInteractor tools to read deployed contract
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.contract_interactor import ERC20Reader

# Configuration - UPDATE THESE AFTER DEPLOYMENT
CONTRACT_ADDRESS = "0xYourContractAddressHere"  # Replace with your deployed address
NETWORK = "ethereum_sepolia"  # or "ethereum_mainnet"
USER_ADDRESS = "0x24f6b1ce11c57d40b542f91ac85fa9eb61f78771"  # Default user to check


def main():
    """Read and display NexusRewardToken contract state"""
    reader = ERC20Reader()

    print("=" * 80)
    print("NEXUS REWARD TOKEN - CONTRACT STATE READER")
    print("=" * 80)
    print(f"\nContract: {CONTRACT_ADDRESS}")
    print(f"Network: {NETWORK}")
    print(f"User: {USER_ADDRESS}")

    # Get token info
    print("\n📊 Token Information:")
    print("-" * 80)
    info = reader.get_token_info(NETWORK, CONTRACT_ADDRESS)
    if info:
        print(f"   Name: {info['name']}")
        print(f"   Symbol: {info['symbol']}")
        print(f"   Decimals: {info['decimals']}")
        print(f"   Total Supply: {info['totalSupply']:,} {info['symbol']}")
        print(f"   ✅ Token info retrieved successfully")
    else:
        print("   ❌ Error: Could not retrieve token info")
        print("   Make sure CONTRACT_ADDRESS and NETWORK are set correctly")
        return

    # Get contract balance (reward pool)
    print("\n💰 Reward Pool Status:")
    print("-" * 80)
    pool_balance = reader.get_balance(NETWORK, CONTRACT_ADDRESS, CONTRACT_ADDRESS)
    if pool_balance is not None:
        print(f"   Available Rewards: {pool_balance:,.4f} NREW")
        if pool_balance < 100:
            print("   ⚠️  Warning: Low reward pool! Contract needs more tokens")
        else:
            claims_available = int(pool_balance / 100)
            print(f"   ℹ️  Approximately {claims_available:,} claims available")
    else:
        print("   ❌ Error: Could not retrieve reward pool balance")

    # Check user status
    print(f"\n👤 User Status ({USER_ADDRESS[:10]}...{USER_ADDRESS[-6:]}):")
    print("-" * 80)

    # Get user balance
    user_balance = reader.get_balance(NETWORK, CONTRACT_ADDRESS, USER_ADDRESS)
    if user_balance is not None:
        print(f"   Current Balance: {user_balance:,.4f} NREW")
        if user_balance == 0:
            print("   ℹ️  User has not claimed any rewards yet")
        else:
            claims_made = int(user_balance / 100)
            print(f"   ℹ️  Estimated claims made: {claims_made:,}")
    else:
        print("   ❌ Error: Could not retrieve user balance")

    print("\n" + "=" * 80)
    print("✅ Contract state read complete")
    print("=" * 80)
    print("\nNext steps:")
    print("1. If this is a new contract, fund it with depositRewards()")
    print("2. Users can claim rewards every 1 hour")
    print("3. Use check_reward_stats.py for detailed analytics")
    print("4. Use monitor_claims.py to watch claims in real-time")
    print("\n")


if __name__ == "__main__":
    main()
