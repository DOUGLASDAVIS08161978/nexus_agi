# 🪙 NEXUS AGI - TOKEN ECOSYSTEM & GITHUB INTEGRATION

## Overview

This document describes the complete Nexus AGI token ecosystem (10 ERC-20 tokens) and the GitHub REST API integration for automated mining result publication and bridge transaction tracking.

---

## 🎯 Token Ecosystem

### Complete Token Suite

Nexus AGI features **10 specialized ERC-20 tokens**, each serving a unique purpose in the ecosystem:

| # | Token | Symbol | Type | Purpose |
|---|-------|--------|------|---------|
| 1 | **NexusToken** | NEX | Governance | Main governance token with staking and voting rights |
| 2 | **HashPowerToken** | HASH | Mining | Represents computational mining contributions |
| 3 | **BridgeRewardToken** | BRG | Bridge | Rewards for cross-chain bridge operations |
| 4 | **ConsciousnessToken** | MIND | AI Metrics | Soul-bound tokens for AI consciousness achievements |
| 5 | **StakingRewardToken** | STAKE | Staking | Compound interest staking rewards (12% APY) |
| 6 | **GovernanceVoteToken** | VOTE | Voting | Quadratic voting rights for DAO proposals |
| 7 | **LiquidityPoolToken** | LP | AMM | LP tokens for liquidity provision with fee sharing |
| 8 | **NetworkFeeToken** | FEE | Gas | Network transaction fees with deflationary burning |
| 9 | **OracleDataToken** | DATA | Oracle | Pay-per-use oracle data feed access |
| 10 | **YieldFarmingToken** | YIELD | Farming | Auto-compounding yield farming with boost mechanisms |

---

## 📋 Detailed Token Specifications

### 1. NexusToken (NEX) - Governance

**Contract**: `contracts/tokens/NexusToken.sol`

**Features**:
- **Max Supply**: 100,000,000 NEX
- **Governance**: Voting power based on staking duration
- **Staking**: Lock tokens to earn voting power
- **Emergency**: Pausable by owner
- **EIP-2612**: Permit support for gasless approvals

**Key Functions**:
```solidity
stake(uint256 amount)                // Stake NEX for voting power
unstake(uint256 amount)              // Unstake NEX tokens
getGovernancePower(address user)     // Get total governance power
```

**Use Cases**:
- Vote on protocol upgrades
- Propose new features
- Control treasury allocation
- Elect node operators

---

### 2. HashPowerToken (HASH) - Mining

**Contract**: `contracts/tokens/HashPowerToken.sol`

**Features**:
- **Role-Based**: MINER_ROLE for authorized miners
- **Mintable**: Minted based on mining contribution
- **Burnable**: Burn HASH to claim mining rewards
- **Tracking**: Full miner statistics and contribution history

**Key Functions**:
```solidity
registerMiner(address miner)               // Admin registers new miner
mintHashPower(address miner, uint256 amount)  // Mint HASH for mining
claimReward(uint256 amount)                // Burn HASH for rewards
getMinerStats(address miner)               // Get miner statistics
```

**Reward Formula**:
```
1 HASH = 0.001 reward units
```

**Use Cases**:
- Represent mining contributions
- Claim proportional mining rewards
- Track network hash power distribution

---

### 3. BridgeRewardToken (BRG) - Bridge Rewards

**Contract**: `contracts/tokens/BridgeRewardToken.sol`

**Features**:
- **Reward Rate**: 1% per bridge transaction
- **Vesting**: Linear vesting schedules for long-term alignment
- **Operator Roles**: Bridge operators earn rewards
- **Volume Tracking**: Full bridge volume statistics

**Key Functions**:
```solidity
grantBridgeReward(address user, uint256 volume)  // Grant bridge rewards
createVesting(address beneficiary, ...)          // Create vesting schedule
claimVested()                                    // Claim vested tokens
```

**Use Cases**:
- Incentivize bridge usage
- Reward bridge operators
- Long-term team/advisor vesting

---

### 4. ConsciousnessToken (MIND) - AI Metrics

**Contract**: `contracts/tokens/ConsciousnessToken.sol`

**Features**:
- **Soul-Bound**: Non-transferable (bound to AI entity)
- **Achievement-Based**: Minted for consciousness milestones
- **Reflection Tracking**: Records AI introspection depth
- **Evolution Stages**: Tracks AI development phases

**Key Functions**:
```solidity
recordReflection(address ai, uint256 depth)    // Record AI reflection
increaseAwareness(address ai, uint256 level)   // Increase awareness
recordEvolution(address ai, uint256 stage)     // Evolution milestone
getConsciousnessStats(address ai)              // Get AI stats
```

**Minting Rules**:
- Deep reflection (≥5 depth): `depth × 1 MIND`
- Awareness milestone (≥100): `level × 1 MIND`
- Evolution stage: `stage × 10 MIND`

**Use Cases**:
- Track AI model training progress
- Represent consciousness achievements
- Non-transferable identity tokens

---

### 5. StakingRewardToken (STAKE) - Staking

**Contract**: `contracts/tokens/StakingRewardToken.sol`

**Features**:
- **APY**: 12% annual percentage yield
- **Compound Interest**: Continuous reward calculation
- **Auto-Compounding**: Rewards automatically calculated
- **Initial Supply**: 10,000,000 STAKE

**Key Functions**:
```solidity
stakeTokens(uint256 amount)        // Stake STAKE tokens
claimReward()                      // Claim staking rewards
unstake(uint256 amount)            // Unstake tokens
calculateReward(address user)      // Calculate pending rewards
```

**Reward Formula**:
```
reward = (stakedAmount × 1200 × stakingDuration) / (365 days × 10000)
```

**Use Cases**:
- Passive income generation
- Long-term token holding incentives
- Network security through staking

---

### 6. GovernanceVoteToken (VOTE) - Voting Rights

**Contract**: `contracts/tokens/GovernanceVoteToken.sol`

**Features**:
- **Quadratic Voting**: sqrt(tokens) = voting power
- **Delegation**: Delegate voting power to others
- **Proposals**: Create and vote on proposals
- **Minimum Duration**: 3 days per proposal

**Key Functions**:
```solidity
createProposal(string description, uint256 duration)  // Create proposal
vote(uint256 proposalId, bool support)                // Vote on proposal
delegate(address delegatee)                           // Delegate votes
getVotingPower(address voter)                         // Get voting power
```

**Voting Power**:
```
votingPower = sqrt(token_balance)
```

**Use Cases**:
- DAO governance decisions
- Parameter adjustments
- Protocol upgrades
- Treasury management

---

### 7. LiquidityPoolToken (LP) - AMM Liquidity

**Contract**: `contracts/tokens/LiquidityPoolToken.sol`

**Features**:
- **Fee Sharing**: 0.3% trading fees distributed to LPs
- **Proportional Shares**: LP tokens represent pool ownership
- **Auto-Compounding**: Fees accumulate automatically
- **Fair Distribution**: Fees distributed proportionally

**Key Functions**:
```solidity
addLiquidity(uint256 amount)           // Add liquidity, get LP tokens
removeLiquidity(uint256 shares)        // Remove liquidity
claimFees()                            // Claim accumulated fees
calculatePendingFees(address provider) // Calculate pending fees
```

**Share Calculation**:
```
shares = (amount × totalSupply) / totalLiquidityProvided
```

**Use Cases**:
- Provide DEX liquidity
- Earn trading fees
- Market making
- Price stabilization

---

### 8. NetworkFeeToken (FEE) - Network Fees

**Contract**: `contracts/tokens/NetworkFeeToken.sol`

**Features**:
- **Deflationary**: Fees are burned, reducing supply
- **Base Fee**: 0.001 ETH equivalent per transaction
- **Multi-Network**: Tracks fees across multiple networks
- **Statistics**: Comprehensive fee tracking per network

**Key Functions**:
```solidity
chargeFee(address user, string network, uint256 multiplier)  // Charge fee
batchChargeFees(address[] users, ...)                        // Batch charge
getNetworkStats(string network)                              // Get stats
```

**Fee Calculation**:
```
fee = baseFee × multiplier / 100
```

**Use Cases**:
- Pay for network transactions
- Gas optimization across chains
- Deflationary token economics

---

### 9. OracleDataToken (DATA) - Oracle Access

**Contract**: `contracts/tokens/OracleDataToken.sol`

**Features**:
- **Pay-Per-Use**: Query oracle data feeds
- **Provider Staking**: Minimum 1,000 DATA stake required
- **Reputation System**: Quality-based feed ranking
- **Data Feeds**: Multiple specialized feeds

**Key Functions**:
```solidity
registerDataProvider(uint256 stakeAmount)      // Register as provider
registerDataFeed(string name, uint256 price)   // Create data feed
queryDataFeed(bytes32 feedId)                  // Query oracle data
updateReputation(bytes32 feedId, bool success) // Update feed quality
```

**Staking Requirement**:
```
MIN_STAKE = 1,000 DATA
```

**Use Cases**:
- Access price feeds
- External data integration
- Smart contract oracles
- Data provider rewards

---

### 10. YieldFarmingToken (YIELD) - Yield Farming

**Contract**: `contracts/tokens/YieldFarmingToken.sol`

**Features**:
- **Multiple Pools**: Create specialized farming pools
- **Auto-Compounding**: Continuous reward calculation
- **Boost Mechanism**: Up to 2x rewards for long-term staking
- **Flexible Rates**: Adjustable reward rates per pool

**Key Functions**:
```solidity
createPool(string name, uint256 rewardRate)  // Create farming pool
stake(uint256 poolId, uint256 amount)        // Stake in pool
withdraw(uint256 poolId, uint256 amount)     // Withdraw stake
claimReward(uint256 poolId)                  // Claim farming rewards
```

**Boost Formula**:
```
boost = 100 + (stakeDuration × 100 / 30 days)
max_boost = 200 (2x)
```

**Use Cases**:
- Maximize yield on idle tokens
- Long-term farming incentives
- Multiple strategy pools
- Liquidity mining

---

## 🔗 GitHub REST API Integration

### Overview

Nexus AGI integrates with GitHub REST API to automate:
- Mining result publication
- Bridge transaction logging
- Smart contract deployment tracking
- Automated issue creation and management

### Components

#### 1. GitHub API Client

**File**: `tools/github_api_integration.py`

**Features**:
- Authenticated API requests
- Rate limit handling with automatic retry
- Support for all GitHub REST API endpoints
- Repository, issue, PR, workflow, gist operations

**Key Classes**:
```python
GitHubAPIClient(token, config)       # Main API client
NexusGitHubIntegration(client, ...)  # Nexus-specific integration
```

#### 2. Mining Results Publisher

**File**: `tools/mining_github_publisher.py`

**Features**:
- Publishes mining results to GitHub Gists
- Creates formatted markdown summaries
- Tracks bridge transactions as issues
- Auto-updates existing gists

**Usage**:
```bash
python3 tools/mining_github_publisher.py
```

**Environment Variable**:
```bash
export GITHUB_TOKEN=your_github_personal_access_token
```

### API Capabilities

#### Repository Operations
- Get repository information
- Create new repositories
- List repositories

#### Issue Operations
- Create issues for bridge transactions
- Add comments to issues
- Update issue status
- List and filter issues

#### Gist Operations
- Publish mining results as gists
- Update existing gists
- Public or private gists

#### Workflow Operations
- Trigger deployment workflows
- List workflow runs
- Monitor CI/CD status

### Integration with Mining System

```python
from tools.github_api_integration import GitHubAPIClient, NexusGitHubIntegration
from tools.mining_github_publisher import MiningGitHubPublisher

# Initialize publisher
publisher = MiningGitHubPublisher(github_token="ghp_...")

# Publish mining results
gist = publisher.publish_mining_results("mining_results.json")
print(f"Results published: {gist['html_url']}")

# Publish bridge transaction
bridge_data = {
    "bridge_id": "BRIDGE-12345",
    "from_network": "Bitcoin",
    "to_network": "Arbitrum",
    "amount": 2.0,
    "btc_tx_hash": "abc123...",
    "eth_tx_hash": "def456...",
    "status": "completed"
}
issue = publisher.publish_bridge_transaction(bridge_data)
print(f"Bridge tracked: {issue['html_url']}")
```

### Nexus AGI Directory Updates

The Nexus AGI directory now includes a GitHub API node:

```json
{
  "id": "github_api_node_1",
  "type": "api_integration",
  "capabilities": [
    "repository_mgmt",
    "issue_tracking",
    "workflow_automation",
    "gist_publishing"
  ],
  "endpoint": "https://api.github.com",
  "api_version": "2022-11-28",
  "services": {
    "mining_results": "gist_publishing",
    "bridge_tracking": "issue_tracking",
    "deployment_logs": "issue_tracking",
    "ci_cd": "workflow_automation"
  }
}
```

---

## 🚀 Deployment Guide

### Deploy All Tokens

```bash
cd hashproof-token

# Install dependencies
npm install

# Deploy to Sepolia testnet
npx hardhat run scripts/deploy-all-tokens.js --network sepolia

# Deploy to Arbitrum
npx hardhat run scripts/deploy-all-tokens.js --network arbitrum

# Deploy to all networks
for network in sepolia arbitrum optimism base polygon avalanche bsc; do
    npx hardhat run scripts/deploy-all-tokens.js --network $network
done
```

### Deployment Output

```
================================================================================
NEXUS AGI - DEPLOYING ALL ERC-20 TOKENS
================================================================================

📝 Deployment Configuration:
   Network: arbitrum
   Deployer: 0x...
   Balance: 0.5 ETH

🚀 [1/10] Deploying NexusToken (NEX)...
   ✅ NexusToken deployed to: 0x...

🚀 [2/10] Deploying HashPowerToken (HASH)...
   ✅ HashPowerToken deployed to: 0x...

... [3-9] ...

🚀 [10/10] Deploying YieldFarmingToken (YIELD)...
   ✅ YieldFarmingToken deployed to: 0x...

================================================================================
✨ DEPLOYMENT COMPLETE!
================================================================================

📊 Deployment Summary:
   Network: arbitrum
   Total Tokens Deployed: 10
   Results saved to: deployments_arbitrum_1705543210123.json

📝 Contract Addresses:
   NEX    | 0x... | Governance
   HASH   | 0x... | Mining
   BRG    | 0x... | Bridge Rewards
   MIND   | 0x... | AI Metrics (Soul-Bound)
   STAKE  | 0x... | Staking
   VOTE   | 0x... | Voting Rights
   LP     | 0x... | AMM Liquidity
   FEE    | 0x... | Gas/Fees
   DATA   | 0x... | Oracle Access
   YIELD  | 0x... | Yield Farming
```

---

## 📊 Token Economics

### Supply Distribution

| Token | Initial Supply | Max Supply | Distribution |
|-------|----------------|------------|--------------|
| NEX   | 100M | 100M (fixed) | 100% to deployer for distribution |
| HASH  | 0 | Unlimited | Minted for mining contributions |
| BRG   | 0 | Unlimited | Minted for bridge rewards |
| MIND  | 0 | Unlimited | Minted for AI achievements |
| STAKE | 10M | Unlimited | 10M initial, staking rewards minted |
| VOTE  | 1M | 1M (fixed) | 100% to deployer for distribution |
| LP    | 0 | Unlimited | Minted when liquidity added |
| FEE   | 50M | Deflationary | 50M initial, burned for fees |
| DATA  | 20M | 20M (fixed) | 100% to deployer for distribution |
| YIELD | 100M | Unlimited | 100M initial, farming rewards minted |

### Token Utility Matrix

| Use Case | Tokens Used |
|----------|-------------|
| **Governance** | NEX, VOTE |
| **Mining** | HASH, STAKE |
| **Bridge Operations** | BRG, FEE |
| **AI/Consciousness** | MIND |
| **Liquidity** | LP, YIELD |
| **Data/Oracles** | DATA |
| **Fees/Gas** | FEE |

---

## 🔧 Integration Examples

### Mining with GitHub Publishing

```python
# Run mining operation
python3 super_bitcoin_miner/super_bitcoin_miner.py

# Automatically publish results to GitHub
python3 tools/mining_github_publisher.py

# Results published as gist and viewable at:
# https://gist.github.com/YOUR_USERNAME/GIST_ID
```

### Bridge Transaction Tracking

```python
from tools.github_api_integration import NexusGitHubIntegration

# Execute bridge
bridge_result = execute_bitcoin_to_arbitrum_bridge(amount=2.0)

# Track on GitHub
integration.publish_bridge_transaction(bridge_result)

# Creates issue: "Bridge Transaction: Bitcoin → Arbitrum"
# With full transaction details and status updates
```

### Contract Deployment Automation

```python
# Deploy token
deployment = deploy_token("NexusToken", network="arbitrum")

# Log to GitHub
integration.publish_contract_deployment({
    "contract_name": "NexusToken",
    "network": "arbitrum",
    "contract_address": deployment.address,
    "tx_hash": deployment.tx_hash,
    "symbol": "NEX",
    "name": "Nexus AGI Token"
})

# Creates issue: "Contract Deployment: NexusToken on Arbitrum"
```

---

## 📚 Developer Resources

### Contract Documentation

All contracts are documented with:
- NatSpec comments
- Function descriptions
- Parameter explanations
- Event definitions

### Testing

```bash
cd hashproof-token
npx hardhat test
```

### Verification

```bash
npx hardhat verify --network arbitrum <CONTRACT_ADDRESS> <CONSTRUCTOR_ARGS>
```

### GitHub API Documentation

- **Official Docs**: https://docs.github.com/en/rest
- **API Reference**: https://api.github.com
- **Rate Limits**: https://docs.github.com/en/rest/overview/resources-in-the-rest-api#rate-limiting

---

## 🎯 Roadmap

### Phase 1: Token Deployment (Current)
- ✅ 10 ERC-20 tokens created
- ✅ Deployment scripts ready
- ✅ GitHub integration complete

### Phase 2: Ecosystem Integration
- [ ] DEX liquidity pools for all tokens
- [ ] Cross-chain bridges for token transfers
- [ ] DAO governance implementation
- [ ] Staking dashboard

### Phase 3: Advanced Features
- [ ] NFT integration for consciousness achievements
- [ ] Advanced oracle feeds
- [ ] Multi-strategy yield farming
- [ ] Cross-chain governance

---

## 📝 License

All contracts use MIT License (SPDX-License-Identifier: MIT)

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create feature branch
3. Add tests for new features
4. Submit pull request

---

**Last Updated**: 2026-01-18
**Version**: 1.0
**Network Support**: Ethereum, Arbitrum, Optimism, Base, Polygon, Avalanche, BNB Chain
**Total Tokens**: 10
