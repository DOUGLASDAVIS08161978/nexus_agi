# 💧 How to Add Liquidity to $WTBTC

After deploying your WTBTC token, here's how to add liquidity on decentralized exchanges (DEXs).

---

## 📋 Prerequisites

Before adding liquidity, you need:

1. ✅ **Deployed WTBTC contract** (run deployment script first)
2. ✅ **WTBTC tokens in your wallet** (mint some tokens)
3. ✅ **MATIC in your wallet** (for pairing and gas fees)
4. ✅ **Wallet connected** (MetaMask on Polygon network)

---

## 🚀 Quick Start - 3 Commands

### **Step 1: Deploy Contracts**
```bash
./deploy_quick.sh
```

### **Step 2: Mint Tokens to Your Wallet**
```bash
node scripts/mint_tokens.js
```

### **Step 3: Add Liquidity**
```bash
node scripts/add_liquidity.js
```

**That's it!** Your WTBTC token will have liquidity on QuickSwap (Polygon's main DEX).

---

## 🌐 Manual Method - Using DEX UI

### **Option 1: QuickSwap (Recommended for Polygon)**

1. **Go to QuickSwap:**
   - Visit: https://quickswap.exchange/#/pool

2. **Connect Wallet:**
   - Click "Connect Wallet"
   - Select MetaMask
   - Make sure you're on Polygon network

3. **Add Liquidity:**
   - Click "New Position" or "Add Liquidity"
   - Click "Select Token"
   - Paste your WTBTC contract address (from `deployment_info.json`)
   - Import the token

4. **Select Pair:**
   - Token 1: Your WTBTC token
   - Token 2: MATIC (or WMATIC, USDC, USDT)

5. **Enter Amounts:**
   - Enter amount of WTBTC (e.g., 100 WTBTC)
   - Enter amount of MATIC (e.g., 10 MATIC)
   - The ratio sets the initial price

6. **Review and Confirm:**
   - Review the price ratio
   - Click "Supply"
   - Approve token spending (if first time)
   - Confirm transaction in MetaMask

7. **Done!**
   - You'll receive LP tokens
   - Your liquidity is live!

---

### **Option 2: Uniswap V3 (Polygon)**

1. **Go to Uniswap:**
   - Visit: https://app.uniswap.org/
   - Switch to Polygon network

2. **Add Liquidity:**
   - Click "Pool" → "New Position"
   - Select fee tier (0.3% recommended)
   - Import your WTBTC token
   - Pair with WMATIC or USDC

3. **Set Price Range:**
   - Choose full range for maximum liquidity
   - Or set a custom range

4. **Deposit Amounts:**
   - Enter token amounts
   - Review and confirm

---

## 💻 Automated Method - Using Scripts

### **Complete Workflow:**

```bash
# 1. Deploy contracts
./deploy_quick.sh

# 2. Mint 1000 WTBTC to your wallet
node scripts/mint_tokens.js

# 3. Add liquidity (10 WTBTC + 0.01 MATIC)
node scripts/add_liquidity.js
```

---

## ⚙️ Customizing Liquidity Amounts

Edit `scripts/add_liquidity.js` to change amounts:

```javascript
// Line ~90
const tokenAmountToAdd = ethers.parseUnits('10', decimals); // Change '10' to your amount
const maticAmountToAdd = ethers.parseEther('0.01'); // Change '0.01' to your amount
```

---

## 📊 Understanding Liquidity Pools

### **What is Liquidity?**

When you add liquidity, you're:
1. **Depositing two tokens** (e.g., WTBTC + MATIC)
2. **Getting LP tokens** (proof of your share)
3. **Earning fees** (from traders who use the pool)

### **Price Calculation**

The initial price is set by the ratio:

```
Price of WTBTC = (Amount of MATIC) / (Amount of WTBTC)
```

Example:
- You add: 100 WTBTC + 10 MATIC
- Initial price: 1 WTBTC = 0.1 MATIC

### **Impermanent Loss**

If token prices diverge significantly, you might have "impermanent loss."
- **Solution:** Only provide liquidity if you believe in both tokens long-term
- **Benefit:** You earn trading fees (usually 0.3% per trade)

---

## 🎯 Recommended Liquidity Ratios

### **For New Token Launch:**

**Conservative (Low Liquidity):**
- 100 WTBTC + 1 MATIC
- Initial price: 1 WTBTC = 0.01 MATIC ($0.01 if MATIC = $1)

**Moderate (Medium Liquidity):**
- 1,000 WTBTC + 10 MATIC
- Initial price: 1 WTBTC = 0.01 MATIC

**Aggressive (High Liquidity):**
- 10,000 WTBTC + 100 MATIC
- Initial price: 1 WTBTC = 0.01 MATIC

---

## 🔍 After Adding Liquidity

### **View Your Position:**

1. **QuickSwap:**
   - Go to https://quickswap.exchange/#/pool
   - Your LP position will show

2. **PolygonScan:**
   - View transaction: `https://polygonscan.com/tx/YOUR_TX_HASH`
   - View LP token balance

### **Check Pool Info:**

```bash
# View pool on QuickSwap
https://info.quickswap.exchange/pair/YOUR_PAIR_ADDRESS

# Or use script
node scripts/check_pool_info.js
```

---

## 🛡️ Security Tips

1. **Start Small:**
   - Test with small amounts first
   - Scale up after confirming everything works

2. **Verify Contract:**
   - Make sure you're using YOUR deployed contract
   - Check address in `deployment_info.json`

3. **Double-Check Network:**
   - Always confirm you're on Polygon Mainnet
   - Chain ID: 137

4. **Save LP Tokens:**
   - Keep track of your LP tokens
   - They represent your liquidity position

---

## 🚨 Troubleshooting

### **"Insufficient liquidity" error**
- You're the first liquidity provider!
- This is normal for new tokens
- Just set your desired initial price

### **"Approve" transaction first**
- First transaction: Approve token spending
- Second transaction: Add liquidity
- Both are needed

### **High gas fees**
- Polygon fees are usually very low ($0.01-0.10)
- If high, try again later
- Check gas price: https://polygonscan.com/gastracker

### **Transaction fails**
- Check you have enough MATIC for gas
- Check you have tokens to add
- Try increasing slippage to 1-2%

---

## 📈 Managing Your Liquidity

### **Add More Liquidity:**
```bash
# Run the script again
node scripts/add_liquidity.js
```

### **Remove Liquidity:**
1. Go to QuickSwap Pool page
2. Click your position
3. Click "Remove"
4. Choose amount to remove (0-100%)
5. Confirm transaction

### **Claim Fees:**
- Fees auto-compound in your position
- When you remove liquidity, you get back tokens + fees

---

## 🎉 Success Checklist

After adding liquidity, verify:

- ✅ Transaction confirmed on PolygonScan
- ✅ LP tokens in your wallet
- ✅ Pool visible on QuickSwap
- ✅ Others can trade your token
- ✅ You're earning fees from trades

---

## 📚 Additional Resources

### **DEX Platforms (Polygon):**
- QuickSwap: https://quickswap.exchange
- Uniswap V3: https://app.uniswap.org
- SushiSwap: https://www.sushi.com
- Balancer: https://balancer.fi

### **Analytics:**
- QuickSwap Info: https://info.quickswap.exchange
- DexScreener: https://dexscreener.com/polygon
- GeckoTerminal: https://www.geckoterminal.com/polygon

### **Helpful Commands:**

```bash
# Check token balance
cast balance WTBTC_ADDRESS --rpc-url $POLYGON_RPC_URL

# Check MATIC balance
cast balance YOUR_ADDRESS --rpc-url $POLYGON_RPC_URL

# View deployment info
cat deployment_info.json
```

---

## 🔗 Contract Addresses

**Your Contracts** (from deployment_info.json):
- TestnetWBTC: `[Will be filled after deployment]`
- EthereumBridgeToken: `[Will be filled after deployment]`

**Polygon DEX Contracts:**
- QuickSwap Router: `0xa5E0829CaCEd8fFDD4De3c43696c57F7D7A678ff`
- WMATIC: `0x0d500B1d8E8eF31E21C99d1Db9A6444d3ADf1270`

---

## 💡 Pro Tips

1. **Initial Price Strategy:**
   - Set a low initial price (e.g., $0.01)
   - Let the market discover fair value
   - As demand grows, price will increase

2. **Liquidity Incentives:**
   - Consider offering rewards for LPs
   - Use farming platforms
   - Attract more liquidity providers

3. **Marketing:**
   - Announce your pool on Twitter
   - Share contract address
   - Show PolygonScan verification

4. **Monitoring:**
   - Watch your pool daily
   - Track volume and fees
   - Adjust liquidity as needed

---

**Created:** 2026-02-01
**Network:** Polygon Mainnet
**Status:** Ready to Add Liquidity 💧
