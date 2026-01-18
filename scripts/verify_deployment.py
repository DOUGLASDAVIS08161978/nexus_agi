#!/usr/bin/env python3
"""
Comprehensive NexusRewardToken deployment verification
Runs all checks to verify contract was deployed correctly
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.contract_interactor import ContractInteractor, ERC20Reader

# Configuration - UPDATE THESE AFTER DEPLOYMENT
CONTRACT_ADDRESS = "0xYourContractAddressHere"
NETWORK = "ethereum_sepolia"  # or "ethereum_mainnet"
EXPECTED_OWNER = "0xYourOwnerAddressHere"  # The address that deployed the contract


def check_contract_exists(interactor):
    """Verify contract exists at address"""
    is_contract = interactor.is_contract(NETWORK, CONTRACT_ADDRESS)
    return is_contract


def check_token_details(reader):
    """Verify token name, symbol, decimals"""
    info = reader.get_token_info(NETWORK, CONTRACT_ADDRESS)
    if not info:
        return None

    expected = {
        "name": "Nexus Reward Token",
        "symbol": "NREW",
        "decimals": 18
    }

    matches = (
        info["name"] == expected["name"] and
        info["symbol"] == expected["symbol"] and
        info["decimals"] == expected["decimals"]
    )

    return {
        "info": info,
        "expected": expected,
        "matches": matches
    }


def check_total_supply(reader):
    """Verify initial supply was minted"""
    info = reader.get_token_info(NETWORK, CONTRACT_ADDRESS)
    if not info:
        return None

    # Expected: 1,000,000 tokens
    expected_supply = 1_000_000
    actual_supply = info["totalSupply"]

    return {
        "expected": expected_supply,
        "actual": actual_supply,
        "matches": actual_supply == expected_supply
    }


def check_owner_balance(reader):
    """Check owner received initial supply"""
    if EXPECTED_OWNER == "0xYourOwnerAddressHere":
        return None

    balance = reader.get_balance(NETWORK, CONTRACT_ADDRESS, EXPECTED_OWNER)
    return balance


def check_contract_configuration(interactor):
    """Check reward amount and cooldown settings"""
    # Get reward amount
    data = "0x228cb733"  # rewardAmount()
    result = interactor.eth_call(NETWORK, CONTRACT_ADDRESS, data)
    reward_amount = int(result, 16) / 10**18 if result and result.startswith("0x") else None

    # Get cooldown period
    data = "0x5348cbca"  # cooldownPeriod()
    result = interactor.eth_call(NETWORK, CONTRACT_ADDRESS, data)
    cooldown = int(result, 16) if result and result.startswith("0x") else None

    return {
        "reward_amount": reward_amount,
        "cooldown_seconds": cooldown,
        "cooldown_hours": cooldown / 3600 if cooldown else None
    }


def check_contract_balance(reader):
    """Check if contract has been funded with rewards"""
    balance = reader.get_balance(NETWORK, CONTRACT_ADDRESS, CONTRACT_ADDRESS)
    return balance


def main():
    """Run comprehensive deployment verification"""
    interactor = ContractInteractor()
    reader = ERC20Reader()

    print("=" * 80)
    print("NEXUS REWARD TOKEN - DEPLOYMENT VERIFICATION")
    print("=" * 80)
    print(f"\nContract: {CONTRACT_ADDRESS}")
    print(f"Network: {NETWORK}")
    print(f"Expected Owner: {EXPECTED_OWNER}")
    print("\n" + "=" * 80)

    all_checks_passed = True

    # Check 1: Contract exists
    print("\n✓ Check 1: Contract Existence")
    print("-" * 80)
    exists = check_contract_exists(interactor)
    if exists:
        print("   ✅ PASS: Contract found at address")
    else:
        print("   ❌ FAIL: No contract at this address")
        all_checks_passed = False

    # Check 2: Token details
    print("\n✓ Check 2: Token Details")
    print("-" * 80)
    token_details = check_token_details(reader)
    if token_details:
        info = token_details["info"]
        print(f"   Name: {info['name']}")
        print(f"   Symbol: {info['symbol']}")
        print(f"   Decimals: {info['decimals']}")

        if token_details["matches"]:
            print("   ✅ PASS: All token details match expected values")
        else:
            print("   ❌ FAIL: Token details don't match")
            all_checks_passed = False
    else:
        print("   ❌ FAIL: Could not retrieve token details")
        all_checks_passed = False

    # Check 3: Total supply
    print("\n✓ Check 3: Total Supply")
    print("-" * 80)
    supply_check = check_total_supply(reader)
    if supply_check:
        print(f"   Expected: {supply_check['expected']:,} NREW")
        print(f"   Actual: {supply_check['actual']:,} NREW")

        if supply_check["matches"]:
            print("   ✅ PASS: Total supply matches expected")
        else:
            print("   ❌ FAIL: Total supply mismatch")
            all_checks_passed = False
    else:
        print("   ❌ FAIL: Could not retrieve total supply")
        all_checks_passed = False

    # Check 4: Owner balance
    print("\n✓ Check 4: Owner Initial Balance")
    print("-" * 80)
    if EXPECTED_OWNER != "0xYourOwnerAddressHere":
        owner_balance = check_owner_balance(reader)
        if owner_balance is not None:
            print(f"   Owner Balance: {owner_balance:,} NREW")

            if owner_balance > 0:
                print("   ✅ PASS: Owner received tokens")
            else:
                print("   ⚠️  WARNING: Owner has 0 balance (tokens may have been transferred)")
        else:
            print("   ❌ FAIL: Could not retrieve owner balance")
            all_checks_passed = False
    else:
        print("   ⏭️  SKIPPED: Update EXPECTED_OWNER in script to run this check")

    # Check 5: Configuration
    print("\n✓ Check 5: Contract Configuration")
    print("-" * 80)
    config = check_contract_configuration(interactor)
    if config["reward_amount"] is not None:
        print(f"   Reward Amount: {config['reward_amount']:,.2f} NREW per claim")

        if config["reward_amount"] == 100:
            print("   ✅ PASS: Reward amount is set to default (100 NREW)")
        else:
            print(f"   ℹ️  INFO: Reward amount has been customized")

        print(f"   Cooldown Period: {config['cooldown_hours']:.1f} hours ({config['cooldown_seconds']} seconds)")

        if config["cooldown_seconds"] == 3600:
            print("   ✅ PASS: Cooldown is set to default (1 hour)")
        else:
            print(f"   ℹ️  INFO: Cooldown has been customized")
    else:
        print("   ❌ FAIL: Could not retrieve configuration")
        all_checks_passed = False

    # Check 6: Contract funding
    print("\n✓ Check 6: Reward Pool Funding")
    print("-" * 80)
    contract_balance = check_contract_balance(reader)
    if contract_balance is not None:
        print(f"   Pool Balance: {contract_balance:,} NREW")

        if contract_balance >= 100:
            claims_available = int(contract_balance / (config["reward_amount"] or 100))
            print(f"   ✅ PASS: Contract is funded ({claims_available:,} claims available)")
        elif contract_balance > 0:
            print(f"   ⚠️  WARNING: Low balance (less than 1 claim)")
        else:
            print("   ⚠️  WARNING: Contract not funded yet")
            print("   ℹ️  Use depositRewards() to fund the reward pool")
    else:
        print("   ❌ FAIL: Could not retrieve contract balance")
        all_checks_passed = False

    # Check 7: Network connectivity
    print("\n✓ Check 7: Network Connectivity")
    print("-" * 80)
    block_number = interactor.get_block_number(NETWORK)
    if block_number:
        print(f"   Current Block: {block_number:,}")
        print("   ✅ PASS: RPC connection working")
    else:
        print("   ❌ FAIL: Could not connect to network")
        all_checks_passed = False

    # Final summary
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)

    if all_checks_passed:
        print("\n🎉 ALL CHECKS PASSED!")
        print("\nYour contract is deployed and configured correctly.")
        print("\nNext steps:")
        print("1. Fund the contract with depositRewards() if not done yet")
        print("2. Test claiming with claimReward()")
        print("3. Use monitor_claims.py to watch for claims")
        print("4. Use check_reward_stats.py for analytics")
    else:
        print("\n⚠️  SOME CHECKS FAILED")
        print("\nPlease review the failures above and:")
        print("1. Verify CONTRACT_ADDRESS is correct")
        print("2. Verify NETWORK is correct")
        print("3. Check that contract was deployed successfully")
        print("4. Ensure RPC endpoint is accessible")

    print("\n" + "=" * 80)
    print()


if __name__ == "__main__":
    main()
