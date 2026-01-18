# NexusRewardToken - Remix IDE Deployment Guide

## Overview

This guide walks you through deploying the NexusRewardToken smart contract using Remix IDE, verifying it on Etherscan, and interacting with it using our custom tools.

**Contract:** `hashproof-token/contracts/NexusRewardToken.sol`
**Target Network:** Ethereum Sepolia Testnet (free) or Ethereum Mainnet
**Final Destination:** Bridge tokens to Bitcoin address `bc1qyhkq7usdefhhhynkjksdqfx32u3rmv94y0htsal`

---

## Part 1: Deploying with Remix IDE

### Step 1: Access Remix IDE

1. Open your browser and navigate to: https://remix.ethereum.org
2. You'll see the Remix IDE interface with file explorer on the left

### Step 2: Create the Contract File

1. In the file explorer (left sidebar), click the "+" icon to create a new file
2. Name it: `NexusRewardToken.sol`
3. Copy the entire contract code from `/home/user/nexus_agi/hashproof-token/contracts/NexusRewardToken.sol`
4. Paste it into the Remix editor

### Step 3: Install Dependencies

The contract uses OpenZeppelin contracts. Remix will auto-import them, but you can verify:

1. Click on the "Solidity Compiler" tab (left sidebar)
2. Select compiler version: `0.8.20` or higher
3. Enable "Auto compile" (optional but helpful)
4. Click "Compile NexusRewardToken.sol"

You should see a green checkmark if compilation succeeds.

### Step 4: Connect MetaMask

1. Install MetaMask browser extension if you haven't: https://metamask.io
2. Create/import your wallet
3. **For Sepolia Testnet:**
   - Switch MetaMask network to "Sepolia Test Network"
   - Get free Sepolia ETH from faucets:
     - https://sepoliafaucet.com
     - https://www.alchemy.com/faucets/ethereum-sepolia
     - https://faucet.quicknode.com/ethereum/sepolia

4. **For Ethereum Mainnet:**
   - Ensure you have sufficient ETH for gas (~$50-100 USD)
   - **WARNING:** Mainnet transactions cost real money!

### Step 5: Deploy the Contract

1. Click on "Deploy & Run Transactions" tab (left sidebar)
2. In the "ENVIRONMENT" dropdown, select **"Injected Provider - MetaMask"**
3. MetaMask will pop up asking to connect - click "Connect"
4. Verify the correct account is selected
5. In the "CONTRACT" dropdown, select **"NexusRewardToken"**
6. Under the constructor parameters, enter:
   - `initialOwner`: Your wallet address (e.g., `0xYourAddressHere`)
7. Click the orange **"Deploy"** button
8. MetaMask will pop up with a transaction confirmation
9. Review the gas fee and click **"Confirm"**
10. Wait for transaction confirmation (15-30 seconds on Sepolia, 1-5 minutes on mainnet)

### Step 6: Verify Deployment

Once deployed, you'll see the contract instance at the bottom of the "Deploy & Run" panel.

**Save these important details:**
- **Contract Address:** (e.g., `0x1234...abcd`)
- **Deployment Transaction Hash:** (e.g., `0xabcd...1234`)
- **Network:** Sepolia or Mainnet
- **Block Number:** When it was deployed

---

## Part 2: Verify Contract on Etherscan

### Why Verify?

Contract verification makes your source code public on Etherscan, allowing:
- Users to read the contract code
- Trust and transparency
- Direct interaction via Etherscan UI

### Verification Steps

1. Go to Etherscan:
   - Sepolia: https://sepolia.etherscan.io
   - Mainnet: https://etherscan.io

2. Search for your contract address in the search bar

3. Click the "Contract" tab, then "Verify and Publish"

4. Fill in the verification form:
   - **Compiler Type:** Solidity (Single file)
   - **Compiler Version:** v0.8.20+commit.a1b79de6
   - **Open Source License Type:** MIT
   - Click "Continue"

5. On the next page:
   - Paste your **flattened contract code** (see below)
   - **Constructor Arguments ABI-encoded:** Leave blank or use Remix to get encoded args
   - **Optimization:** Enabled
   - **Runs:** 200
   - Click "Verify and Publish"

### Getting Flattened Contract Code

In Remix:
1. Right-click on `NexusRewardToken.sol`
2. Select "Flatten"
3. Copy all the flattened code
4. Paste into Etherscan verification form

**OR** manually combine all imports:
- Copy OpenZeppelin contracts from node_modules
- Combine into single file

---

## Part 3: Fund the Contract with Rewards

After deployment, the contract needs tokens in its balance to distribute rewards.

### Using Remix

1. In the deployed contract instance, find the `depositRewards` function
2. Enter amount in wei (e.g., `100000000000000000000` = 100 tokens)
3. Click "transact"
4. Confirm in MetaMask

### Using Etherscan (after verification)

1. Go to your contract on Etherscan
2. Click "Contract" → "Write Contract"
3. Click "Connect to Web3" and connect MetaMask
4. Find `depositRewards` function
5. Enter amount: `100000000000000000000` (100 tokens with 18 decimals)
6. Click "Write" and confirm transaction

**Recommended initial funding:** 10,000 tokens = `10000000000000000000000` wei

---

## Part 4: Interact with Contract Using Our Tools

### Setup Python Environment

```bash
cd /home/user/nexus_agi
python3 -m pip install web3 requests
```

### Read Contract State

Create `scripts/read_nexus_reward_token.py`:

```python
#!/usr/bin/env python3
"""
Read NexusRewardToken contract state
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.contract_interactor import ERC20Reader

# Configuration
CONTRACT_ADDRESS = "0xYourContractAddressHere"  # Replace with your deployed address
NETWORK = "ethereum_sepolia"  # or "ethereum_mainnet"

def main():
    reader = ERC20Reader()

    print("=" * 80)
    print("NEXUS REWARD TOKEN - CONTRACT STATE")
    print("=" * 80)

    # Get token info
    print("\n📊 Token Information:")
    info = reader.get_token_info(NETWORK, CONTRACT_ADDRESS)
    if info:
        print(f"   Name: {info['name']}")
        print(f"   Symbol: {info['symbol']}")
        print(f"   Decimals: {info['decimals']}")
        print(f"   Total Supply: {info['totalSupply']:,} {info['symbol']}")

    # Get contract balance (reward pool)
    print("\n💰 Reward Pool:")
    pool_balance = reader.get_balance(NETWORK, CONTRACT_ADDRESS, CONTRACT_ADDRESS)
    if pool_balance is not None:
        print(f"   Available Rewards: {pool_balance:,.4f} NREW")

    # Check if user can claim
    user_address = "0x24f6b1ce11c57d40b542f91ac85fa9eb61f78771"
    print(f"\n👤 User Status ({user_address}):")

    # Get user balance
    user_balance = reader.get_balance(NETWORK, CONTRACT_ADDRESS, user_address)
    if user_balance is not None:
        print(f"   Current Balance: {user_balance:,.4f} NREW")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
```

Run it:
```bash
python3 scripts/read_nexus_reward_token.py
```

### Check Advanced Stats

Create `scripts/check_reward_stats.py`:

```python
#!/usr/bin/env python3
"""
Check NexusRewardToken advanced statistics
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.contract_interactor import ContractInteractor

CONTRACT_ADDRESS = "0xYourContractAddressHere"
NETWORK = "ethereum_sepolia"

def main():
    interactor = ContractInteractor()

    print("=" * 80)
    print("NEXUS REWARD TOKEN - ADVANCED STATISTICS")
    print("=" * 80)

    # Get stats using custom function
    # getStats() returns (contractBalance, rewardsClaimedTotal, uniqueClaimers, currentRewardAmount, currentCooldown)

    # Function selector for getStats()
    data = "0x" + "c59d4847"  # keccak256("getStats()")[:8]

    result = interactor.eth_call(NETWORK, CONTRACT_ADDRESS, data)

    if result:
        print(f"\n📈 Contract Statistics:")
        print(f"   Raw Response: {result}")
        # Decode the response (5 uint256 values)
        # In production, use web3.py's decode_abi for proper parsing

    # Check reward amount
    data = "0x" + "228cb733"  # keccak256("rewardAmount()")[:8]
    result = interactor.eth_call(NETWORK, CONTRACT_ADDRESS, data)
    if result:
        # Convert hex to decimal
        reward_amount = int(result, 16) / 10**18
        print(f"\n🎁 Reward Configuration:")
        print(f"   Reward per Claim: {reward_amount:,.2f} NREW")

    # Check cooldown period
    data = "0x" + "5348cbca"  # keccak256("cooldownPeriod()")[:8]
    result = interactor.eth_call(NETWORK, CONTRACT_ADDRESS, data)
    if result:
        cooldown_seconds = int(result, 16)
        cooldown_hours = cooldown_seconds / 3600
        print(f"   Cooldown Period: {cooldown_hours:.1f} hours ({cooldown_seconds} seconds)")

    # Check total claimers
    data = "0x" + "0e3a2e15"  # keccak256("totalClaimers()")[:8]
    result = interactor.eth_call(NETWORK, CONTRACT_ADDRESS, data)
    if result:
        total_claimers = int(result, 16)
        print(f"\n👥 Community Statistics:")
        print(f"   Total Unique Claimers: {total_claimers:,}")

    # Check total rewards claimed
    data = "0x" + "d54ad2a1"  # keccak256("totalRewardsClaimed()")[:8]
    result = interactor.eth_call(NETWORK, CONTRACT_ADDRESS, data)
    if result:
        total_claimed = int(result, 16) / 10**18
        print(f"   Total Rewards Claimed: {total_claimed:,.2f} NREW")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
```

Run it:
```bash
python3 scripts/check_reward_stats.py
```

---

## Part 5: Claim Rewards (User Guide)

### Using Remix

1. Go to "Deploy & Run Transactions" in Remix
2. Load your deployed contract instance
3. Find the `claimReward` function
4. Click "transact" (no parameters needed)
5. Confirm in MetaMask
6. Wait for confirmation
7. Check your balance increased by 100 NREW

### Using Etherscan

1. Go to contract on Etherscan
2. Click "Contract" → "Write Contract"
3. Connect MetaMask
4. Find `claimReward` function
5. Click "Write"
6. Confirm transaction
7. Wait for confirmation

### Checking Claim Eligibility

Before claiming, check if you can claim:

**Using `canClaim` function:**
1. In Remix or Etherscan, find `canClaim` (read function)
2. Enter your address
3. Returns `true` if you can claim, `false` otherwise

**Using `timeUntilNextClaim` function:**
1. Find `timeUntilNextClaim` (read function)
2. Enter your address
3. Returns seconds until next claim (0 if ready)

---

## Part 6: Bridge Tokens to Bitcoin Address

**Target Bitcoin Address:** `bc1qyhkq7usdefhhhynkjksdqfx32u3rmv94y0htsal`

### Important Notes

⚠️ **Direct ERC-20 to Bitcoin bridging is complex and requires:**
1. A bridge service (like RenBridge, tBTC, or WBTC)
2. Wrapping/unwrapping mechanisms
3. Custodial services or decentralized protocols

### Option 1: Use WBTC Protocol (Recommended)

1. **Convert NREW to ETH:**
   - Use Uniswap or another DEX
   - Swap NREW → ETH

2. **Convert ETH to WBTC:**
   - Use a DEX like Uniswap
   - Swap ETH → WBTC
   - WBTC address: `0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599`

3. **Unwrap WBTC to BTC:**
   - Go to https://wbtc.network
   - Click "Burn" (convert WBTC → BTC)
   - Enter Bitcoin address: `bc1qyhkq7usdefhhhynkjksdqfx32u3rmv94y0htsal`
   - Follow merchant instructions
   - Wait for confirmation (usually 6 BTC blocks)

### Option 2: Use Centralized Exchange

1. **Send NREW to Exchange:**
   - Deposit NREW to exchange (if listed)
   - Or swap NREW → ETH first, then send ETH

2. **Sell for BTC:**
   - Trade NREW or ETH for BTC on exchange

3. **Withdraw BTC:**
   - Enter address: `bc1qyhkq7usdefhhhynkjksdqfx32u3rmv94y0htsal`
   - Confirm withdrawal
   - Pay network fee
   - Wait for confirmations

### Option 3: Use Thorchain or RenBridge

**Thorchain:**
- Cross-chain swaps
- Native BTC support
- Visit: https://app.thorswap.finance

**RenBridge (if still operational):**
- Decentralized bridging
- Visit: https://bridge.renproject.io

---

## Part 7: Security Best Practices

### Private Key Safety

1. **NEVER share your private keys**
2. **NEVER share your seed phrase**
3. Use hardware wallets for large amounts (Ledger, Trezor)
4. Enable 2FA on exchanges
5. Verify all addresses before sending

### Smart Contract Interactions

1. **Always verify contract on Etherscan** before interacting
2. **Check function parameters** carefully
3. **Start with small amounts** when testing
4. **Use testnet first** (Sepolia) before mainnet
5. **Review transaction details** in MetaMask before confirming

### Gas Optimization

1. **Check gas prices** before transacting: https://etherscan.io/gastracker
2. **Adjust gas limit** if transactions fail
3. **Use low priority** for non-urgent transactions
4. **Batch operations** when possible

---

## Part 8: Troubleshooting

### Contract Won't Deploy

**Problem:** "Out of gas" error
**Solution:** Increase gas limit in MetaMask (try 5,000,000)

**Problem:** "Execution reverted"
**Solution:** Check constructor parameters are correct

### Can't Claim Rewards

**Problem:** "Cooldown period not elapsed"
**Solution:** Wait 1 hour between claims

**Problem:** "Insufficient contract balance"
**Solution:** Contract needs more tokens - use `depositRewards`

### Verification Failed

**Problem:** "Compiled bytecode doesn't match"
**Solution:**
- Ensure compiler version matches exactly (0.8.20)
- Check optimization is enabled (200 runs)
- Use flattened code from Remix

### Bridge Issues

**Problem:** "Transaction pending for hours"
**Solution:**
- Check gas price - may need to speed up
- Use MetaMask "Speed Up" feature
- Or cancel and retry with higher gas

---

## Part 9: Monitoring and Analytics

### Track Contract Activity

**Using Etherscan:**
1. Go to your contract address
2. Click "Events" tab to see all claims
3. Click "Token Transfers" to see distribution
4. Click "Holders" to see token distribution

**Using our tools:**

Create `scripts/monitor_claims.py`:

```python
#!/usr/bin/env python3
"""
Monitor NexusRewardToken claims in real-time
"""
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.contract_interactor import ContractInteractor

CONTRACT_ADDRESS = "0xYourContractAddressHere"
NETWORK = "ethereum_sepolia"

def main():
    interactor = ContractInteractor()

    print("📡 Monitoring NexusRewardToken claims...")
    print(f"   Contract: {CONTRACT_ADDRESS}")
    print(f"   Network: {NETWORK}")
    print("\nPress Ctrl+C to stop\n")

    last_claimers = 0
    last_claimed = 0

    try:
        while True:
            # Get total claimers
            data = "0x" + "0e3a2e15"  # totalClaimers()
            result = interactor.eth_call(NETWORK, CONTRACT_ADDRESS, data)

            if result:
                total_claimers = int(result, 16)

                # Get total claimed
                data = "0x" + "d54ad2a1"  # totalRewardsClaimed()
                result2 = interactor.eth_call(NETWORK, CONTRACT_ADDRESS, data)

                if result2:
                    total_claimed = int(result2, 16) / 10**18

                    # Check for changes
                    if total_claimers != last_claimers or total_claimed != last_claimed:
                        print(f"[{time.strftime('%H:%M:%S')}] 🎉 New claim detected!")
                        print(f"   Total Claimers: {total_claimers:,}")
                        print(f"   Total Claimed: {total_claimed:,.2f} NREW\n")

                        last_claimers = total_claimers
                        last_claimed = total_claimed

            time.sleep(15)  # Check every 15 seconds

    except KeyboardInterrupt:
        print("\n\n✅ Monitoring stopped")

if __name__ == "__main__":
    main()
```

---

## Part 10: Summary Checklist

### Deployment Checklist

- [ ] Access Remix IDE (https://remix.ethereum.org)
- [ ] Create and compile NexusRewardToken.sol
- [ ] Connect MetaMask to correct network
- [ ] Get testnet ETH or have mainnet ETH ready
- [ ] Deploy contract with your address as initialOwner
- [ ] Save contract address and deployment details
- [ ] Verify contract on Etherscan
- [ ] Fund contract with reward tokens
- [ ] Test claim functionality
- [ ] Set up monitoring scripts

### Interaction Checklist

- [ ] Install Python dependencies (web3, requests)
- [ ] Update contract address in scripts
- [ ] Run read_nexus_reward_token.py
- [ ] Run check_reward_stats.py
- [ ] Test claim via Remix or Etherscan
- [ ] Verify claim worked (check balance)
- [ ] Set up monitoring script (optional)

### Bridge to Bitcoin Checklist

- [ ] Choose bridge method (WBTC, exchange, Thorchain)
- [ ] Swap NREW → ETH (if using DEX route)
- [ ] Swap ETH → WBTC (if using WBTC route)
- [ ] Unwrap WBTC → BTC
- [ ] Send to bc1qyhkq7usdefhhhynkjksdqfx32u3rmv94y0htsal
- [ ] Wait for Bitcoin confirmations (6 blocks)
- [ ] Verify receipt on Bitcoin blockchain

---

## Support and Resources

### Documentation
- OpenZeppelin Contracts: https://docs.openzeppelin.com
- Remix IDE Docs: https://remix-ide.readthedocs.io
- MetaMask Guide: https://metamask.io/faqs/
- Etherscan API: https://docs.etherscan.io

### Testnet Faucets
- Sepolia: https://sepoliafaucet.com
- Alchemy Sepolia: https://www.alchemy.com/faucets/ethereum-sepolia
- QuickNode: https://faucet.quicknode.com/ethereum/sepolia

### Bridge Services
- WBTC Network: https://wbtc.network
- Thorchain: https://app.thorswap.finance
- RenBridge: https://bridge.renproject.io (check status)

### Block Explorers
- Ethereum Mainnet: https://etherscan.io
- Sepolia Testnet: https://sepolia.etherscan.io
- Bitcoin: https://mempool.space

---

## Contact Information

**Project:** Nexus AGI
**Repository:** https://github.com/DOUGLASDAVIS08161978/nexus_agi
**Bitcoin Destination:** bc1qyhkq7usdefhhhynkjksdqfx32u3rmv94y0htsal

---

*Created as part of the Nexus AGI integrated blockchain and AI system*
*Version: 1.0*
*Date: 2026-01-18*
