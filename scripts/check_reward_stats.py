#!/usr/bin/env python3
"""
Check NexusRewardToken advanced statistics
Reads contract state variables and configuration
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.contract_interactor import ContractInteractor

# Configuration - UPDATE THESE AFTER DEPLOYMENT
CONTRACT_ADDRESS = "0xYourContractAddressHere"
NETWORK = "ethereum_sepolia"  # or "ethereum_mainnet"
USER_ADDRESS = "0x24f6b1ce11c57d40b542f91ac85fa9eb61f78771"


def get_reward_amount(interactor):
    """Get the reward amount per claim"""
    # rewardAmount() - function selector
    data = "0x228cb733"
    result = interactor.eth_call(NETWORK, CONTRACT_ADDRESS, data)
    if result and result.startswith("0x"):
        return int(result, 16) / 10**18
    return None


def get_cooldown_period(interactor):
    """Get the cooldown period in seconds"""
    # cooldownPeriod() - function selector
    data = "0x5348cbca"
    result = interactor.eth_call(NETWORK, CONTRACT_ADDRESS, data)
    if result and result.startswith("0x"):
        return int(result, 16)
    return None


def get_total_claimers(interactor):
    """Get total unique claimers"""
    # totalClaimers() - function selector
    data = "0x0e3a2e15"
    result = interactor.eth_call(NETWORK, CONTRACT_ADDRESS, data)
    if result and result.startswith("0x"):
        return int(result, 16)
    return None


def get_total_rewards_claimed(interactor):
    """Get total rewards claimed across all users"""
    # totalRewardsClaimed() - function selector
    data = "0xd54ad2a1"
    result = interactor.eth_call(NETWORK, CONTRACT_ADDRESS, data)
    if result and result.startswith("0x"):
        return int(result, 16) / 10**18
    return None


def get_last_claim_time(interactor, address):
    """Get last claim time for an address"""
    # lastClaimTime(address) - function selector + padded address
    # Remove 0x prefix from address and pad to 32 bytes
    addr_param = address[2:].lower().zfill(64)
    data = "0x4c0f38c2" + addr_param
    result = interactor.eth_call(NETWORK, CONTRACT_ADDRESS, data)
    if result and result.startswith("0x"):
        return int(result, 16)
    return None


def get_has_claimed(interactor, address):
    """Check if address has ever claimed"""
    # hasClaimed(address) - function selector + padded address
    addr_param = address[2:].lower().zfill(64)
    data = "0xe0886f90" + addr_param
    result = interactor.eth_call(NETWORK, CONTRACT_ADDRESS, data)
    if result and result.startswith("0x"):
        return int(result, 16) == 1
    return None


def can_claim_now(interactor, address):
    """Check if address can claim right now"""
    # canClaim(address) - function selector + padded address
    addr_param = address[2:].lower().zfill(64)
    data = "0x402914f5" + addr_param
    result = interactor.eth_call(NETWORK, CONTRACT_ADDRESS, data)
    if result and result.startswith("0x"):
        return int(result, 16) == 1
    return None


def time_until_next_claim(interactor, address):
    """Get time until next claim in seconds"""
    # timeUntilNextClaim(address) - function selector + padded address
    addr_param = address[2:].lower().zfill(64)
    data = "0x0a5f8f18" + addr_param
    result = interactor.eth_call(NETWORK, CONTRACT_ADDRESS, data)
    if result and result.startswith("0x"):
        return int(result, 16)
    return None


def main():
    """Read and display advanced contract statistics"""
    interactor = ContractInteractor()

    print("=" * 80)
    print("NEXUS REWARD TOKEN - ADVANCED STATISTICS")
    print("=" * 80)
    print(f"\nContract: {CONTRACT_ADDRESS}")
    print(f"Network: {NETWORK}")

    # Get reward configuration
    print("\n🎁 Reward Configuration:")
    print("-" * 80)

    reward_amount = get_reward_amount(interactor)
    if reward_amount is not None:
        print(f"   Reward per Claim: {reward_amount:,.2f} NREW")
    else:
        print("   ❌ Could not fetch reward amount")

    cooldown_seconds = get_cooldown_period(interactor)
    if cooldown_seconds is not None:
        cooldown_hours = cooldown_seconds / 3600
        cooldown_minutes = cooldown_seconds / 60
        print(f"   Cooldown Period: {cooldown_hours:.1f} hours ({cooldown_minutes:.0f} minutes)")
    else:
        print("   ❌ Could not fetch cooldown period")

    # Get community statistics
    print("\n👥 Community Statistics:")
    print("-" * 80)

    total_claimers = get_total_claimers(interactor)
    if total_claimers is not None:
        print(f"   Total Unique Claimers: {total_claimers:,}")
    else:
        print("   ❌ Could not fetch total claimers")

    total_claimed = get_total_rewards_claimed(interactor)
    if total_claimed is not None:
        print(f"   Total Rewards Claimed: {total_claimed:,.2f} NREW")

        if total_claimers and total_claimers > 0:
            avg_per_user = total_claimed / total_claimers
            print(f"   Average per User: {avg_per_user:,.2f} NREW")

        if reward_amount:
            total_claims = int(total_claimed / reward_amount)
            print(f"   Total Claims Made: {total_claims:,}")
    else:
        print("   ❌ Could not fetch total rewards claimed")

    # Get user-specific info
    print(f"\n🔍 User Analysis ({USER_ADDRESS[:10]}...{USER_ADDRESS[-6:]}):")
    print("-" * 80)

    has_claimed = get_has_claimed(interactor, USER_ADDRESS)
    if has_claimed is not None:
        if has_claimed:
            print("   ✅ User has claimed before")

            last_claim = get_last_claim_time(interactor, USER_ADDRESS)
            if last_claim and last_claim > 0:
                from datetime import datetime
                last_claim_dt = datetime.fromtimestamp(last_claim)
                print(f"   Last Claim: {last_claim_dt.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            print("   ℹ️  User has never claimed")
    else:
        print("   ❌ Could not fetch claim history")

    can_claim = can_claim_now(interactor, USER_ADDRESS)
    if can_claim is not None:
        if can_claim:
            print("   ✅ User CAN claim right now!")
        else:
            print("   ⏳ User cannot claim yet")

            time_remaining = time_until_next_claim(interactor, USER_ADDRESS)
            if time_remaining is not None and time_remaining > 0:
                hours = time_remaining // 3600
                minutes = (time_remaining % 3600) // 60
                seconds = time_remaining % 60
                print(f"   Time Until Next Claim: {hours}h {minutes}m {seconds}s")
            elif time_remaining == 0:
                print("   ℹ️  Should be ready to claim (check reward pool balance)")
    else:
        print("   ❌ Could not check claim eligibility")

    # Get contract balance (reward pool)
    print("\n💰 Reward Pool Analysis:")
    print("-" * 80)

    block_number = interactor.get_block_number(NETWORK)
    if block_number:
        print(f"   Current Block: {block_number:,}")

    balance = interactor.get_balance(NETWORK, CONTRACT_ADDRESS)
    if balance is not None:
        print(f"   Contract ETH Balance: {balance:.6f} ETH")

    # For ERC20 balance, we'd need to call balanceOf(contract_address)
    # This is shown in read_nexus_reward_token.py

    print("\n" + "=" * 80)
    print("✅ Statistics collection complete")
    print("=" * 80)

    # Provide recommendations
    print("\n💡 Recommendations:")
    if can_claim:
        print("   • User is eligible to claim - visit Remix or Etherscan to claim")
    if total_claimers == 0:
        print("   • No claims yet - be the first to claim!")
    if has_claimed and not can_claim:
        print("   • Wait for cooldown period to expire before next claim")

    print("\n")


if __name__ == "__main__":
    main()
