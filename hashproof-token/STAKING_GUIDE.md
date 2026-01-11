# 🏦 HASHPROOF STAKING GUIDE

**Earn Passive Income with HPROOF Staking - Up to 30% APY!**

---

## 📋 Table of Contents

1. [What is Staking?](#what-is-staking)
2. [Available Staking Pools](#available-staking-pools)
3. [How to Stake](#how-to-stake)
4. [Checking Your Rewards](#checking-your-rewards)
5. [Claiming Rewards](#claiming-rewards)
6. [Unstaking](#unstaking)
7. [Auto-Compounding](#auto-compounding)
8. [Strategies](#strategies)
9. [FAQ](#faq)

---

## 🎯 What is Staking?

**Staking** is the process of locking your HPROOF tokens in a smart contract to earn rewards over time. Think of it like a crypto savings account with high interest rates!

### How It Works:

1. **Lock tokens** in a staking pool for a specific period
2. **Earn rewards** automatically based on the pool's APY (Annual Percentage Yield)
3. **Claim rewards** anytime or enable auto-compounding
4. **Unstake** when the lock period ends (or pay a 10% penalty for early withdrawal)

### Benefits:

- ✅ **Passive Income** - Earn while you hold
- ✅ **No Active Trading** - Set it and forget it
- ✅ **High APY** - Up to 30% annual returns
- ✅ **Auto-Compounding** - Maximize gains with compound interest
- ✅ **Secure** - Audited smart contracts

---

## 🏊 Available Staking Pools

| Pool | Lock Period | APY | Best For | Flexibility |
|------|-------------|-----|----------|------------|
| **Flexible** | No lock | **5%** | Testing, short-term | ⭐⭐⭐⭐⭐ |
| **30-Day** | 30 days | **10%** | Conservative staking | ⭐⭐⭐⭐ |
| **90-Day** | 90 days | **15%** | Balanced approach | ⭐⭐⭐ |
| **180-Day** | 6 months | **20%** | Committed holders | ⭐⭐ |
| **365-Day** | 1 year | **30%** | Maximum returns | ⭐ |

### Pool Details:

#### Pool 0: Flexible (5% APY)
- **No lock period** - Withdraw anytime without penalty
- Best for: Testing staking, maintaining liquidity
- Lowest APY but maximum flexibility

#### Pool 1: 30-Day (10% APY)
- **30-day lock** - Short commitment
- Best for: Conservative investors, testing longer stakes
- Decent APY with low commitment

#### Pool 2: 90-Day (15% APY)
- **90-day lock** - Quarterly commitment
- Best for: Balanced risk/reward
- Good APY for medium-term holders

#### Pool 3: 180-Day (20% APY)
- **180-day lock** - Half-year commitment
- Best for: Serious investors
- High APY for committed holders

#### Pool 4: 365-Day (30% APY)
- **365-day lock** - Full year commitment
- Best for: Long-term believers, maximalists
- **Highest APY** for maximum commitment

---

## 💰 Earnings Calculator

### Example: Staking 10,000 HPROOF

| Pool | Daily Earnings | Monthly Earnings | Yearly Earnings | After 1 Year |
|------|----------------|------------------|-----------------|--------------|
| Flexible (5%) | 1.37 HPROOF | 41.67 HPROOF | 500 HPROOF | 10,500 HPROOF |
| 30-Day (10%) | 2.74 HPROOF | 83.33 HPROOF | 1,000 HPROOF | 11,000 HPROOF |
| 90-Day (15%) | 4.11 HPROOF | 125.00 HPROOF | 1,500 HPROOF | 11,500 HPROOF |
| 180-Day (20%) | 5.48 HPROOF | 166.67 HPROOF | 2,000 HPROOF | 12,000 HPROOF |
| **365-Day (30%)** | **8.22 HPROOF** | **250.00 HPROOF** | **3,000 HPROOF** | **13,000 HPROOF** |

### Value at Different Token Prices:

**Staking 10,000 HPROOF in 365-Day Pool (30% APY):**

| Token Price | Your Investment | Yearly Rewards | Total After 1 Year |
|-------------|-----------------|----------------|-------------------|
| $0.10 | $1,000 | $300 | $1,300 |
| $0.50 | $5,000 | $1,500 | $6,500 |
| $1.00 | $10,000 | $3,000 | $13,000 |
| $5.00 | $50,000 | $15,000 | $65,000 |
| $10.00 | $100,000 | $30,000 | $130,000 |

---

## 📝 How to Stake

### Prerequisites:

1. **Have HPROOF tokens** in your wallet
2. **Minimum stake**: 100 HPROOF per pool
3. **Network**: Deployed on Polygon (low fees) or your chosen network
4. **Wallet**: MetaMask or compatible wallet

### Step-by-Step Instructions:

#### Method 1: Using Scripts (Hardhat)

1. **Navigate to project directory:**
   ```bash
   cd hashproof-token
   ```

2. **Edit the staking script** (`scripts/stake.js`):
   ```javascript
   // Update these parameters:
   const poolId = 4;           // Choose pool: 0-4
   const stakeAmount = hre.ethers.parseEther("1000"); // Amount to stake
   const autoCompound = true;  // Enable auto-compounding?
   ```

3. **Run the staking script:**
   ```bash
   npx hardhat run scripts/stake.js --network polygon
   ```

4. **Follow the prompts** and confirm transactions

#### Method 2: Direct Contract Interaction (Web3)

1. **Connect to contract** using Web3 or ethers.js
2. **Approve tokens:**
   ```javascript
   await hashProofToken.approve(stakingAddress, amount);
   ```

3. **Stake tokens:**
   ```javascript
   await stakingContract.stake(amount, poolId, autoCompound);
   ```

### Configuration Options:

**Pool Selection (`poolId`):**
- `0` = Flexible (5% APY)
- `1` = 30-Day (10% APY)
- `2` = 90-Day (15% APY)
- `3` = 180-Day (20% APY)
- `4` = 365-Day (30% APY)

**Auto-Compound (`autoCompound`):**
- `true` = Rewards automatically re-staked
- `false` = Rewards can be claimed manually

---

## 📊 Checking Your Rewards

### Using the Check Rewards Script:

```bash
npx hardhat run scripts/check-rewards.js --network polygon
```

### What You'll See:

- 📌 All your active stakes
- 💰 Pending rewards for each stake
- 🔒 Lock status and unlock dates
- 📈 Expected reward rates (daily, weekly, monthly, yearly)
- 💵 Reward values at different token prices

### Example Output:

```
📌 STAKE #0:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Pool:              365-Day (30% APY)
  Staked Amount:     10,000 HPROOF
  Pending Rewards:   246.58 HPROOF
  Auto-Compound:     Yes ✅
  Stake Age:         30 days, 5 hours
  Status:            🔒 LOCKED
  Unlocks in:        335 days

  💵 Reward Rates:
     Daily:          8.2192 HPROOF
     Weekly:         57.5342 HPROOF
     Monthly:        246.58 HPROOF
     Yearly (est):   3,000.00 HPROOF
```

---

## 💸 Claiming Rewards

### When to Claim:

- ✅ Anytime you want to realize profits
- ✅ When you need liquidity
- ✅ To sell rewards at high prices
- ❌ Not needed if auto-compounding is enabled

### Using the Claim Script:

```bash
npx hardhat run scripts/claim-rewards.js --network polygon
```

### What Happens:

1. **Script checks** all your stakes for pending rewards
2. **Claims rewards** from each stake
3. **Transfers HPROOF** to your wallet
4. **Shows summary** of total claimed

### Important Notes:

- 🔄 **Auto-compound stakes**: Rewards are added to stake amount (increases future rewards)
- 💰 **Manual stakes**: Rewards are sent to your wallet
- ⛽ **Gas fees**: Small network fee applies
- 📈 **Reward calculation**: Based on time elapsed and APY

### Example Output:

```
🎉 CLAIMING COMPLETE!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Stakes Claimed:     2 / 2
Total Claimed:      350.25 HPROOF
Balance Before:     5,000 HPROOF
Balance After:      5,350.25 HPROOF
Balance Increase:   350.25 HPROOF

💵 VALUE GAINED:
  At $0.10/token:  $35.03
  At $1.00/token:  $350.25
  At $10.0/token:  $3,502.50
```

---

## 🔓 Unstaking

### When Can You Unstake?

- ✅ **After lock period** - No penalty, get full amount
- ⚠️ **During lock period** - 10% penalty on staked amount (rewards not penalized)
- ✅ **Flexible pool** - Anytime, no penalty

### Using the Unstake Script:

1. **Edit the script** (`scripts/unstake.js`):
   ```javascript
   const stakeIndexToUnstake = 0; // Change this to your stake index
   ```

2. **Run the script:**
   ```bash
   npx hardhat run scripts/unstake.js --network polygon
   ```

### Early Withdrawal Penalty:

If you unstake before the lock period ends:

- **Penalty**: 10% of staked amount
- **Rewards**: Not penalized, claimed in full
- **Example**:
  - Staked: 10,000 HPROOF
  - Rewards: 500 HPROOF
  - Penalty: 1,000 HPROOF (10% of stake)
  - **You receive**: 9,000 + 500 = 9,500 HPROOF

### Recommendation:

**Wait until unlock date** to avoid penalty and maximize returns!

### Example Output:

```
⚠️  WARNING: EARLY WITHDRAWAL!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔒 This stake is still locked!
⚠️  You will lose 10% of your staked amount as penalty!

  Penalty Amount:    1,000.00 HPROOF
  You will receive:  9,000.00 HPROOF
  + Rewards:         246.58 HPROOF
  ─────────────────────────────────────
  Total Received:    9,246.58 HPROOF

💡 RECOMMENDATION: Wait until 2026-12-15
   to avoid penalty and get full amount!
```

---

## 🔄 Auto-Compounding

### What is Auto-Compounding?

Instead of claiming rewards to your wallet, **rewards are automatically re-staked** to your original stake, increasing your total staked amount and future reward rate.

### Benefits:

- ✅ **Compound interest** - Earn rewards on rewards
- ✅ **Exponential growth** - Accelerates gains over time
- ✅ **Hands-off** - No manual claiming needed
- ✅ **Maximizes APY** - Highest total returns

### Manual vs Auto-Compound Comparison:

**Staking 10,000 HPROOF in 365-Day Pool (30% APY):**

| Period | Manual Staking | Auto-Compound | Difference |
|--------|----------------|---------------|------------|
| 6 months | 11,500 | 11,553 | +53 HPROOF |
| 1 year | 13,000 | 13,000 | 0 (same) |
| 2 years | 16,000 | 16,900 | +900 HPROOF |
| 5 years | 25,000 | 37,129 | +12,129 HPROOF |

*Auto-compound assumes rewards are re-staked monthly*

### How to Enable:

When staking, set `autoCompound = true`:

```javascript
const autoCompound = true; // Enable auto-compounding
await staking.stake(amount, poolId, autoCompound);
```

### How It Works:

1. You stake with auto-compound enabled
2. Rewards accumulate in the contract
3. When you claim or check rewards, they're added to your stake
4. Your staked amount increases
5. Future rewards are calculated on the new higher amount
6. Process repeats = **compound growth**

---

## 📈 Staking Strategies

### Strategy 1: The Ladder 🪜

**Objective**: Balance liquidity and returns

**Method**:
- 25% in Flexible pool (immediate liquidity)
- 25% in 30-Day pool
- 25% in 90-Day pool
- 25% in 365-Day pool

**Pros**: Diversified, regular unlock dates, good average APY
**Cons**: Lower overall APY than all-in 365-day

### Strategy 2: The Maximalist 💎

**Objective**: Maximum returns

**Method**:
- 100% in 365-Day pool with auto-compound

**Pros**: Highest possible APY (30%), compound growth
**Cons**: No liquidity for 1 year, early withdrawal penalty

### Strategy 3: The Conservative 🛡️

**Objective**: Low risk, high flexibility

**Method**:
- 50% in Flexible pool
- 50% in 30-Day pool

**Pros**: Can exit anytime, low commitment
**Cons**: Lower APY (7.5% average)

### Strategy 4: The Dollar-Cost Averaging 📊

**Objective**: Gradual entry, risk mitigation

**Method**:
- Stake a portion each month
- Start with flexible, gradually move to longer pools
- Increase allocation as you gain confidence

**Pros**: Reduces timing risk, learn as you go
**Cons**: Miss out on early gains

### Strategy 5: The Profit-Taker 💰

**Objective**: Regular income

**Method**:
- Stake with auto-compound OFF
- Claim rewards weekly/monthly
- Sell rewards or re-stake manually

**Pros**: Regular cash flow, realize profits
**Cons**: Miss compound growth, more gas fees

---

## ❓ FAQ

### General Questions:

**Q: What is the minimum stake amount?**
A: 100 HPROOF per stake.

**Q: Can I have multiple stakes?**
A: Yes! Stake as many times as you want across different pools.

**Q: Are rewards paid in HPROOF?**
A: Yes, all rewards are paid in HPROOF tokens.

**Q: Where do staking rewards come from?**
A: From the 30,000,000 HPROOF staking rewards pool allocated in the tokenomics.

### Rewards Questions:

**Q: How often are rewards calculated?**
A: Every block! Your rewards accumulate continuously.

**Q: When can I claim rewards?**
A: Anytime, even during the lock period.

**Q: Do unclaimed rewards count toward my stake?**
A: Only if auto-compound is enabled.

**Q: What happens to rewards if I unstake early?**
A: Rewards are claimed automatically when you unstake (no penalty on rewards).

### Penalty Questions:

**Q: What is the early withdrawal penalty?**
A: 10% of your staked amount (not rewards).

**Q: Can I avoid the penalty?**
A: Yes, wait until the lock period ends.

**Q: Does the penalty apply to Flexible pool?**
A: No, Flexible pool has no lock period or penalty.

**Q: Where does the penalty go?**
A: Burned or redistributed (check smart contract implementation).

### Technical Questions:

**Q: Are the staking contracts audited?**
A: Built with OpenZeppelin's audited libraries.

**Q: Can I lose my staked tokens?**
A: No, unless there's a smart contract exploit (use at your own risk).

**Q: What network is staking on?**
A: Depends on your deployment (Polygon, Ethereum, Arbitrum, etc.).

**Q: How much gas do staking operations cost?**
A: Varies by network:
- Polygon: ~$0.01-0.05
- Ethereum: $5-50 (depending on gas prices)
- Arbitrum: ~$0.10-1.00

### Strategy Questions:

**Q: Should I use auto-compound?**
A: Yes, if you want to maximize long-term gains. No, if you want regular income.

**Q: Which pool should I choose?**
A: Depends on your goals:
- Need liquidity? → Flexible
- Maximum returns? → 365-Day
- Balanced? → 90-Day or 180-Day

**Q: Can I move between pools?**
A: Not directly. You must unstake from one pool and re-stake in another.

**Q: What if the token price goes down?**
A: Your APY rewards still accumulate, but the USD value changes. Staking reduces selling pressure though!

---

## 🎯 Quick Start Checklist

Ready to start earning? Follow this checklist:

- [ ] Buy HPROOF tokens (DEX or exchange)
- [ ] Add HPROOF to MetaMask (see METAMASK_GUIDE.md)
- [ ] Decide which pool(s) to use
- [ ] Calculate expected returns
- [ ] Choose auto-compound or manual
- [ ] Edit and run `scripts/stake.js`
- [ ] Confirm transaction in wallet
- [ ] Verify stake with `scripts/check-rewards.js`
- [ ] Set reminder for unlock date
- [ ] Watch your rewards grow! 📈

---

## 📞 Support & Resources

- **Check Rewards**: `npx hardhat run scripts/check-rewards.js`
- **Claim Rewards**: `npx hardhat run scripts/claim-rewards.js`
- **Unstake**: `npx hardhat run scripts/unstake.js`
- **Contract Addresses**: See `deployment-info.json`
- **Whitepaper**: `HASHPROOF_WHITEPAPER.md`
- **Community**: Discord/Telegram (TBD)

---

## ⚠️ Risk Disclaimer

**Staking carries risks:**

- Smart contract vulnerabilities (though audited libraries used)
- Token price volatility affects USD value of rewards
- Early withdrawal penalties
- Network risks (chain reorgs, attacks)
- Opportunity cost (could earn more elsewhere)

**Not financial advice. Do your own research. Only stake what you can afford to lose.**

---

## 🎊 Start Earning Today!

**The best time to start staking was yesterday. The second best time is NOW!**

Choose your pool, stake your HPROOF, and start earning passive income! 💰

---

*Built with ❤️ by the NexusAGI Team*

*Powered by Claude AI & Human Creativity*

**Let's make that passive income! 🚀✨**
