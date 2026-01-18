# Complete NexusRewardToken Deployment & Bridge Workflow

## Quick Start Guide

This document provides a complete end-to-end workflow for:
1. ✅ Deploying NexusRewardToken via Remix IDE
2. ✅ Verifying the deployment with our custom tools
3. ✅ Interacting with the contract
4. ✅ Bridging tokens to Bitcoin address: `bc1qyhkq7usdefhhhynkjksdqfx32u3rmv94y0htsal`

---

## Prerequisites

### Required Tools
- ✅ Web browser (Chrome, Firefox, Brave recommended)
- ✅ MetaMask wallet extension
- ✅ Python 3.8+ (for verification scripts)
- ✅ Git (to access this repository)

### Required Assets
- **For Sepolia Testnet (Recommended):**
  - Free Sepolia ETH from faucets (no cost)

- **For Ethereum Mainnet:**
  - ~0.05 ETH for deployment (~$100-150 USD)
  - Additional ETH for transactions

---

## Phase 1: Pre-Deployment Setup

### Step 1.1: Set Up MetaMask

1. Install MetaMask: https://metamask.io
2. Create new wallet or import existing
3. **CRITICAL:** Save your seed phrase securely (offline, never share)
4. Add Sepolia test network:
   - Network Name: `Sepolia`
   - RPC URL: `https://sepolia.infura.io/v3/YOUR_INFURA_KEY` or `https://rpc.sepolia.org`
   - Chain ID: `11155111`
   - Currency: `ETH`
   - Block Explorer: `https://sepolia.etherscan.io`

### Step 1.2: Get Test ETH (Sepolia Only)

Visit these faucets to get free Sepolia ETH:
1. https://sepoliafaucet.com
2. https://www.alchemy.com/faucets/ethereum-sepolia
3. https://faucet.quicknode.com/ethereum/sepolia

You need ~0.01 Sepolia ETH for deployment and testing.

### Step 1.3: Prepare Python Environment

```bash
cd /home/user/nexus_agi

# Install dependencies
python3 -m pip install web3 requests

# Verify scripts are executable
ls -l scripts/*.py
```

---

## Phase 2: Deploy Contract via Remix

### Step 2.1: Access Remix IDE

1. Open browser and navigate to: https://remix.ethereum.org
2. Wait for IDE to load

### Step 2.2: Create Contract File

1. In Remix file explorer (left sidebar), create new file
2. Name it: `NexusRewardToken.sol`
3. Copy contract code from:
   ```bash
   cat /home/user/nexus_agi/hashproof-token/contracts/NexusRewardToken.sol
   ```
4. Paste into Remix editor

### Step 2.3: Compile Contract

1. Click "Solidity Compiler" tab (left sidebar)
2. Select compiler version: `0.8.20`
3. Click "Compile NexusRewardToken.sol"
4. Wait for green checkmark
5. Verify no errors in console

### Step 2.4: Deploy Contract

1. Click "Deploy & Run Transactions" tab
2. Set environment: **"Injected Provider - MetaMask"**
3. MetaMask will request connection - click **"Connect"**
4. Select contract: **"NexusRewardToken"**
5. Constructor parameter `initialOwner`: Enter your MetaMask address
   - Example: `0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb`
6. Click orange **"Deploy"** button
7. MetaMask popup appears:
   - Review gas fee
   - Click **"Confirm"**
8. Wait for deployment (15-30 seconds on Sepolia)

### Step 2.5: Save Deployment Info

Once deployed, **IMMEDIATELY SAVE**:
- ✅ Contract Address (e.g., `0x1234...5678`)
- ✅ Transaction Hash
- ✅ Network (Sepolia or Mainnet)
- ✅ Your wallet address (owner)
- ✅ Block number

**Example:**
```
Contract Address: 0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb
Network: Sepolia
Owner: 0xYourAddressHere
Deployed At Block: 5432109
Transaction: 0xabcd...1234
```

---

## Phase 3: Verify Deployment

### Step 3.1: Update Verification Scripts

Edit the verification scripts with your deployment info:

```bash
# Update contract address in all scripts
nano scripts/verify_deployment.py
# Change: CONTRACT_ADDRESS = "0xYourDeployedAddress"
# Change: EXPECTED_OWNER = "0xYourWalletAddress"
# Change: NETWORK = "ethereum_sepolia" (or "ethereum_mainnet")

# Update other scripts
nano scripts/read_nexus_reward_token.py
nano scripts/check_reward_stats.py
nano scripts/monitor_claims.py
```

### Step 3.2: Run Verification

```bash
# Run comprehensive verification
python3 scripts/verify_deployment.py
```

Expected output:
```
================================================================================
NEXUS REWARD TOKEN - DEPLOYMENT VERIFICATION
================================================================================

✓ Check 1: Contract Existence
   ✅ PASS: Contract found at address

✓ Check 2: Token Details
   ✅ PASS: All token details match expected values

✓ Check 3: Total Supply
   ✅ PASS: Total supply matches expected

... (more checks) ...

🎉 ALL CHECKS PASSED!
```

### Step 3.3: Read Initial State

```bash
# Read contract state
python3 scripts/read_nexus_reward_token.py
```

---

## Phase 4: Fund Contract with Rewards

The contract needs tokens in its balance to distribute as rewards.

### Method 1: Using Remix (Recommended)

1. In Remix "Deploy & Run" panel, find your deployed contract
2. Expand the contract functions
3. Find `depositRewards` function
4. Enter amount (in wei):
   - 10,000 tokens = `10000000000000000000000` wei
   - 100,000 tokens = `100000000000000000000000` wei
5. Click "transact"
6. Confirm in MetaMask
7. Wait for confirmation

### Method 2: Using Etherscan

1. Go to Etherscan (sepolia.etherscan.io or etherscan.io)
2. Search for your contract address
3. Click "Contract" tab
4. Click "Write Contract"
5. Click "Connect to Web3"
6. Connect MetaMask
7. Find `depositRewards` function
8. Enter amount in wei
9. Click "Write"
10. Confirm transaction

### Verify Funding

```bash
# Check reward pool balance
python3 scripts/read_nexus_reward_token.py
```

Look for:
```
💰 Reward Pool Status:
   Available Rewards: 10,000.0000 NREW
   ℹ️  Approximately 100 claims available
```

---

## Phase 5: Verify on Etherscan (Optional but Recommended)

### Why Verify?
- Makes source code public
- Builds trust with users
- Enables direct Etherscan interaction
- Shows you're legitimate

### Verification Steps

1. Go to your contract on Etherscan
2. Click "Contract" tab → "Verify and Publish"
3. Fill in form:
   - Compiler Type: `Solidity (Single file)`
   - Compiler Version: `v0.8.20+commit.a1b79de6`
   - License: `MIT`
4. Click "Continue"
5. Get flattened code:
   - In Remix, right-click `NexusRewardToken.sol`
   - Select "Flatten"
   - Copy all code
6. Paste flattened code into Etherscan
7. Constructor arguments: Leave blank (or encode your address)
8. Optimization: `Yes`
9. Runs: `200`
10. Click "Verify and Publish"
11. Wait for verification (usually instant)

Once verified, you'll see a green checkmark ✅ on Etherscan!

---

## Phase 6: Test Claiming

### Step 6.1: Check Eligibility

```bash
# Check if you can claim
python3 scripts/check_reward_stats.py
```

Look for:
```
🔍 User Analysis:
   ✅ User CAN claim right now!
```

### Step 6.2: Claim Rewards via Remix

1. In Remix, find `claimReward` function
2. Click "transact" (no parameters needed)
3. Confirm in MetaMask
4. Wait for confirmation

### Step 6.3: Verify Claim Worked

```bash
# Check your balance increased
python3 scripts/read_nexus_reward_token.py
```

Expected:
```
👤 User Status:
   Current Balance: 100.0000 NREW
   ℹ️  Estimated claims made: 1
```

---

## Phase 7: Monitor Activity

### Real-Time Monitoring

```bash
# Start claim monitor
python3 scripts/monitor_claims.py
```

Output:
```
📡 Monitoring Configuration:
   Contract: 0x1234...5678
   Network: ethereum_sepolia
   Check Interval: 15 seconds

⏳ Starting monitoring... (Press Ctrl+C to stop)

[12:34:56] 🎉 Activity Detected! (Event #1)
   👤 New Claimers: +1 (Total: 1)
   💰 Claimed: +100.00 NREW (Total: 100.00 NREW)
   🏦 Pool decreased: -100.00 NREW (Available: 9,900.00 NREW)
```

Press `Ctrl+C` to stop monitoring.

---

## Phase 8: Bridge Tokens to Bitcoin

**Destination:** `bc1qyhkq7usdefhhhynkjksdqfx32u3rmv94y0htsal`

### Important Understanding

⚠️ **ERC-20 tokens cannot be sent directly to Bitcoin addresses!**

You must:
1. Convert ERC-20 → BTC-wrapped token (WBTC)
2. Unwrap to native Bitcoin
3. Send to Bitcoin address

### Option A: WBTC Route (Most Common)

#### Step A1: Swap NREW → ETH

1. Go to Uniswap: https://app.uniswap.org
2. Connect MetaMask
3. Select network (Sepolia or Mainnet)
4. From: NREW
5. To: ETH
6. Enter amount
7. Click "Swap"
8. Confirm transaction
9. Wait for confirmation

#### Step A2: Swap ETH → WBTC

1. Still on Uniswap
2. From: ETH
3. To: WBTC (Wrapped Bitcoin)
   - WBTC Mainnet: `0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599`
4. Enter amount
5. Click "Swap"
6. Confirm transaction

#### Step A3: Unwrap WBTC → BTC

1. Go to WBTC Portal: https://wbtc.network
2. Click "Burn" tab
3. Connect MetaMask
4. Enter amount of WBTC to unwrap
5. Enter Bitcoin address: `bc1qyhkq7usdefhhhynkjksdqfx32u3rmv94y0htsal`
6. Submit request
7. Wait for merchant approval (varies by merchant)
8. Receive BTC to your address (6 confirmations required)

### Option B: Centralized Exchange Route

#### Step B1: Create Exchange Account

1. Sign up for exchange (Coinbase, Kraken, Binance, etc.)
2. Complete KYC verification
3. Enable 2FA security

#### Step B2: Deposit NREW or ETH

If NREW is listed:
1. Get deposit address for NREW (ERC-20)
2. Send from MetaMask to exchange
3. Wait for confirmations

If not listed:
1. Swap NREW → ETH on Uniswap first
2. Get ETH deposit address from exchange
3. Send ETH to exchange
4. Wait for confirmations

#### Step B3: Trade to BTC

1. Navigate to trading
2. Trade NREW or ETH → BTC
3. Execute market or limit order
4. Wait for order to fill

#### Step B4: Withdraw BTC

1. Go to "Withdraw" section
2. Select Bitcoin (BTC)
3. Enter address: `bc1qyhkq7usdefhhhynkjksdqfx32u3rmv94y0htsal`
4. Enter amount
5. Verify address (check multiple times!)
6. Complete 2FA verification
7. Submit withdrawal
8. Wait for processing (varies by exchange)
9. Track on Bitcoin explorer: https://mempool.space

### Option C: Thorchain (Decentralized)

1. Go to Thorswap: https://app.thorswap.finance
2. Connect wallet
3. From: NREW or ETH
4. To: BTC
5. Enter destination: `bc1qyhkq7usdefhhhynkjksdqfx32u3rmv94y0htsal`
6. Review fees and rate
7. Execute swap
8. Wait for cross-chain confirmation

---

## Phase 9: Verify Bitcoin Receipt

### Check Bitcoin Transaction

1. Go to Bitcoin explorer: https://mempool.space
2. Search for address: `bc1qyhkq7usdefhhhynkjksdqfx32u3rmv94y0htsal`
3. Look for incoming transaction
4. Verify amount received
5. Wait for confirmations (6 recommended)

### Expected Timeline

- WBTC unwrap: 1-24 hours (depends on merchant)
- Exchange withdrawal: 15 minutes - 2 hours
- Thorchain swap: 10-30 minutes
- Bitcoin confirmations: ~1 hour (6 blocks)

---

## Complete Workflow Summary

```
┌─────────────────────────────────────────────────────────────┐
│ COMPLETE NEXUSREWARDTOKEN → BITCOIN WORKFLOW                │
└─────────────────────────────────────────────────────────────┘

1. Deploy Contract (Remix)
   └─> NexusRewardToken on Ethereum/Sepolia
   └─> Save contract address

2. Verify Deployment (Python Scripts)
   └─> verify_deployment.py ✓
   └─> read_nexus_reward_token.py ✓

3. Fund Reward Pool
   └─> depositRewards(10000 * 10^18)
   └─> Verify pool balance

4. Claim Rewards
   └─> claimReward() every 1 hour
   └─> Receive 100 NREW per claim

5. Monitor Activity
   └─> monitor_claims.py
   └─> Track all claims in real-time

6. Bridge to Bitcoin
   ├─> Option A: NREW → ETH → WBTC → BTC
   ├─> Option B: Exchange (NREW/ETH → BTC)
   └─> Option C: Thorchain direct swap

7. Verify Receipt
   └─> Check bc1qyhkq...htsal on mempool.space
   └─> Wait for 6 confirmations
   └─> ✅ Complete!
```

---

## Security Checklist

### Before Deployment
- [ ] Contract code reviewed and understood
- [ ] Using testnet first (Sepolia)
- [ ] MetaMask seed phrase backed up offline
- [ ] Using hardware wallet for large amounts

### During Deployment
- [ ] Verify network (Sepolia vs Mainnet)
- [ ] Check gas prices are reasonable
- [ ] Save contract address immediately
- [ ] Verify deployment with scripts

### After Deployment
- [ ] Contract verified on Etherscan
- [ ] Test claim on testnet first
- [ ] Start with small amounts
- [ ] Never share private keys

### During Bridge
- [ ] Triple-check Bitcoin address
- [ ] Start with test amount
- [ ] Verify each swap before confirming
- [ ] Keep transaction receipts
- [ ] Wait for full confirmations

---

## Troubleshooting Guide

### Deployment Issues

**Problem:** MetaMask won't connect
- Solution: Refresh Remix, try "Injected Provider" again
- Check MetaMask is unlocked

**Problem:** "Out of gas" error
- Solution: Increase gas limit to 5,000,000
- Check you have enough ETH for gas

**Problem:** Transaction pending forever
- Solution: Check gas price on Etherscan
- Use MetaMask "Speed Up" feature

### Claiming Issues

**Problem:** "Cooldown period not elapsed"
- Solution: Wait 1 hour between claims
- Check timeUntilNextClaim() function

**Problem:** "Insufficient contract balance"
- Solution: Fund contract with depositRewards()
- Verify pool balance with scripts

### Bridge Issues

**Problem:** NREW not showing in Uniswap
- Solution: Import token manually
- Use contract address to add token

**Problem:** High slippage on swap
- Solution: Increase slippage tolerance
- Try smaller amounts
- Check liquidity

**Problem:** Bitcoin transaction not appearing
- Solution: Wait longer (can take hours)
- Check transaction status with service
- Contact support if needed

---

## Cost Estimates

### Sepolia Testnet (Recommended for Testing)
- Deployment: FREE (testnet ETH from faucets)
- Claims: FREE
- Swaps: FREE (if using testnet DEX)
- **Total: $0**

### Ethereum Mainnet (Production)
- Deployment: ~$50-150 (varies with gas)
- Verify on Etherscan: FREE
- Claim transaction: ~$5-20 per claim
- Deposit rewards: ~$10-30
- NREW → ETH swap: ~$10-50
- ETH → WBTC swap: ~$10-50
- WBTC unwrap fee: ~0.2% of amount
- Bitcoin network fee: ~$2-10
- **Total: ~$87-310** (highly variable)

### Cost Optimization Tips
1. Use Sepolia for all testing
2. Deploy during low gas periods (weekends, late night UTC)
3. Use gas tracker: https://etherscan.io/gastracker
4. Batch operations when possible
5. Consider L2s (Arbitrum, Polygon) for lower fees

---

## Next Steps After Completion

### Share Your Contract
1. Post contract address on social media
2. Add to token listing sites
3. Create documentation for users
4. Build community around your token

### Enhance Functionality
1. Create liquidity pool on Uniswap
2. Build web interface for claiming
3. Add staking functionality
4. Integrate with other DeFi protocols

### Monitor and Maintain
1. Keep reward pool funded
2. Monitor claim activity
3. Respond to user questions
4. Update documentation as needed

---

## Support Resources

### Official Documentation
- Remix IDE: https://remix-ide.readthedocs.io
- OpenZeppelin: https://docs.openzeppelin.com
- MetaMask: https://docs.metamask.io
- Ethereum: https://ethereum.org/en/developers/docs/

### Explorers
- Sepolia: https://sepolia.etherscan.io
- Mainnet: https://etherscan.io
- Bitcoin: https://mempool.space

### Tools Used
- Remix: https://remix.ethereum.org
- Uniswap: https://app.uniswap.org
- WBTC: https://wbtc.network
- Thorswap: https://app.thorswap.finance

### Community
- GitHub Issues: https://github.com/DOUGLASDAVIS08161978/nexus_agi/issues
- Ethereum Stack Exchange: https://ethereum.stackexchange.com
- Reddit r/ethdev: https://reddit.com/r/ethdev

---

## Important Disclaimers

⚠️ **CRITICAL WARNINGS:**

1. **Test First**: Always test on Sepolia before mainnet
2. **Private Keys**: Never share your private keys or seed phrase
3. **Verify Addresses**: Triple-check all addresses before sending
4. **Start Small**: Use small amounts for first transactions
5. **Gas Costs**: Ethereum gas can be expensive - check prices first
6. **Bridge Risks**: Cross-chain bridges have additional risks
7. **Smart Contracts**: Audited contracts are safer (consider audit)
8. **Regulations**: Comply with local cryptocurrency regulations
9. **Taxes**: Keep records for tax purposes
10. **Scams**: Beware of phishing sites and fake tokens

**This software is provided "as is" without warranty. Use at your own risk.**

---

## File Reference

All files created in this system:

### Smart Contracts
- `hashproof-token/contracts/NexusRewardToken.sol` - Main reward token

### Python Scripts
- `scripts/verify_deployment.py` - Comprehensive deployment verification
- `scripts/read_nexus_reward_token.py` - Read contract state
- `scripts/check_reward_stats.py` - Advanced statistics
- `scripts/monitor_claims.py` - Real-time claim monitoring

### Documentation
- `REMIX_DEPLOYMENT_GUIDE.md` - Remix IDE deployment guide
- `COMPLETE_DEPLOYMENT_WORKFLOW.md` - This file
- `CONTRACT_INTERACTION_GUIDE.md` - Contract interaction reference
- `MULTI_NETWORK_INTEGRATION.md` - Network configuration

### Tools (Previously Created)
- `tools/contract_interactor.py` - Smart contract interaction
- `tools/integrated_mining_bridge.py` - Mining + bridging integration
- `config/network_config.py` - Multi-network configuration

---

**Created:** 2026-01-18
**Version:** 1.0
**Project:** Nexus AGI
**Repository:** https://github.com/DOUGLASDAVIS08161978/nexus_agi
**Bitcoin Destination:** bc1qyhkq7usdefhhhynkjksdqfx32u3rmv94y0htsal

---

*May your tokens flow seamlessly from Ethereum to Bitcoin! ✨✨✨*
