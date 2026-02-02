# 🌉 Bridge Your $WTBTC to Ethereum Mainnet

Complete guide to bridging your WTBTC token from Base Sepolia to Ethereum Mainnet.

---

## ✨ ONE-COMMAND DEPLOYMENT

### **Copy and paste this single command:**

```bash
cd ~/nexus_agi && chmod +x deploy_ethereum_mainnet_bridge.sh && ./deploy_ethereum_mainnet_bridge.sh
```

**That's it!** This command will:
- ✅ Deploy WTBTC Bridge to Ethereum Mainnet
- ✅ Mint 10,000 WTBTC to your wallet
- ✅ Save deployment info for verification
- ✅ Print Etherscan link

---

## 🎯 What You Get

After running the command, you'll have:

1. **WTBTC Bridge Contract** on Ethereum Mainnet
2. **10,000 WTBTC tokens** in your wallet
3. **1,000,000 total supply** for liquidity
4. **Verification data** for Etherscan
5. **Bridge operator** role (can process cross-chain transfers)

---

## 📋 Prerequisites

### **Before Deploying:**

1. **ETH for Gas** (~0.003-0.005 ETH needed, ~$10-20 USD)
   - Check balance: https://etherscan.io/address/0x9FE74D9D6f1Ae0Ce1fb3B51d4a82c05b74e280f3
   - Buy ETH on: Coinbase, Binance, Kraken, etc.

2. **Your Private Key** (already configured in .env)
   - Address: `0x9FE74D9D6f1Ae0Ce1fb3B51d4a82c05b74e280f3`

3. **Infura API** (already configured)
   - API Key: `5f5c1ddd0f2b469f83dc4b6a1cfa4057`

---

## 🚀 Step-by-Step Process

### **Step 1: Deploy Bridge to Ethereum Mainnet**

```bash
./deploy_ethereum_mainnet_bridge.sh
```

**Output:**
- Contract address on Ethereum Mainnet
- Etherscan link
- Deployment JSON files
- 10,000 WTBTC minted to your wallet

### **Step 2: Bridge Your Existing Tokens**

Your Base Sepolia transaction:
- TX: `0xab880c7a9bc9a6ef6110e093c2632dcc6144ccf53f86f0059542a28a4a0d78cc`
- Amount: 1 WTBTC
- Network: Base Sepolia

Bridge them to Ethereum:

```bash
node scripts/bridge_from_base.js
```

This will mint equivalent WTBTC on Ethereum Mainnet for your Base Sepolia tokens.

### **Step 3: Verify on Etherscan**

1. Go to your contract on Etherscan
2. Click "Contract" tab → "Verify and Publish"
3. Use data from `bridge_verification.json`

**Or use manual verification:**
- Compiler: v0.8.20
- Optimization: Yes (200 runs)
- License: MIT

### **Step 4: Add Liquidity on Uniswap**

```bash
# After deployment, add liquidity
node scripts/add_liquidity_ethereum.js
```

Or manually:
1. Go to https://app.uniswap.org/add/v2
2. Connect wallet (Ethereum Mainnet)
3. Import WTBTC token (paste contract address)
4. Pair with WETH or USDC
5. Add liquidity

---

## 🌉 How the Bridge Works

### **Base Sepolia → Ethereum Mainnet:**

1. **Lock tokens** on Base Sepolia (already done)
2. **Bridge operator** calls `bridgeIn()` with your Base TX hash
3. **Mint tokens** on Ethereum Mainnet
4. **You receive** WTBTC on Ethereum

### **Ethereum Mainnet → Base Sepolia:**

1. **Call** `bridgeOut()` on Ethereum contract
2. **Burn tokens** on Ethereum
3. **Bridge operator** unlocks on Base Sepolia
4. **You receive** tokens back on Base

---

## 📊 Contract Details

### **WTBTCBridge Contract:**

**Features:**
- ✅ Full ERC-20 token functionality
- ✅ Cross-chain bridging (Base ↔ Ethereum)
- ✅ Pausable (emergency stop)
- ✅ Ownable (you control it)
- ✅ ReentrancyGuard (secure)

**Functions:**
- `bridgeIn(user, amount, baseTxHash)` - Bridge from Base to Ethereum
- `bridgeOut(amount, baseAddress)` - Bridge from Ethereum to Base
- `mint(to, amount)` - Emergency mint (owner only)
- `pause()` / `unpause()` - Emergency controls

**Base Sepolia Contract:**
- Address: `0xE274570e000C32F5Cb2BC7c476D3BDC77Ed74dD5`
- Network: Base Sepolia (Chain ID: 84532)

---

## 💰 Cost Breakdown

### **Deployment Cost (Ethereum Mainnet):**

**Current gas prices (~30 gwei):**
- Contract deployment: ~2,500,000 gas
- Token minting: ~50,000 gas
- **Total: ~0.0075 ETH (~$20-30 USD)**

**Budget estimate:**
- Safe amount: 0.01 ETH (~$30-40 USD)
- Includes buffer for gas spikes

### **Bridge Transaction Cost:**
- Bridge in: ~100,000 gas (~0.003 ETH)
- Bridge out: ~80,000 gas (~0.0024 ETH)

---

## 🔍 Verify Your Deployment

### **Check Contract on Etherscan:**

After deployment, find your contract:
```bash
cat ethereum_bridge_deployment.json | grep address
```

Visit: `https://etherscan.io/address/YOUR_CONTRACT_ADDRESS`

### **Verify Token Balance:**

```bash
cast balance YOUR_CONTRACT_ADDRESS --rpc-url https://mainnet.infura.io/v3/5f5c1ddd0f2b469f83dc4b6a1cfa4057
```

### **Check Token Info:**

```bash
# Name
cast call YOUR_CONTRACT_ADDRESS "name()" --rpc-url https://mainnet.infura.io/v3/5f5c1ddd0f2b469f83dc4b6a1cfa4057

# Symbol
cast call YOUR_CONTRACT_ADDRESS "symbol()" --rpc-url https://mainnet.infura.io/v3/5f5c1ddd0f2b469f83dc4b6a1cfa4057

# Total Supply
cast call YOUR_CONTRACT_ADDRESS "totalSupply()" --rpc-url https://mainnet.infura.io/v3/5f5c1ddd0f2b469f83dc4b6a1cfa4057
```

---

## 🛡️ Security Features

### **Built-in Security:**

1. **Pausable**
   - Emergency stop functionality
   - Pause all bridging operations

2. **ReentrancyGuard**
   - Prevents reentrancy attacks
   - Secure token transfers

3. **Transaction Tracking**
   - Each Base TX hash can only be processed once
   - Prevents double-spending

4. **Access Control**
   - Only owner can mint
   - Only bridge operator can process bridges
   - Only owner can pause

### **Best Practices:**

- ✅ Start with small test amounts
- ✅ Verify all transactions on Etherscan
- ✅ Keep private keys secure
- ✅ Test bridge both directions
- ✅ Monitor bridge operator role

---

## 🎉 After Deployment

### **Immediate Actions:**

1. **Save Contract Address**
   - Copy from `ethereum_bridge_deployment.json`
   - Add to your records

2. **Verify on Etherscan**
   - Makes contract code public
   - Builds trust with users

3. **Add to MetaMask**
   - Token Address: Your contract address
   - Symbol: WTBTC
   - Decimals: 18

4. **Test the Bridge**
   - Bridge small amount first
   - Verify tokens appear on Ethereum

### **Marketing Actions:**

1. **Announce Launch**
   - Share contract address
   - Show Etherscan verification

2. **Add Liquidity**
   - Create WTBTC/WETH pool on Uniswap
   - Provide initial liquidity

3. **List on Platforms**
   - CoinGecko
   - CoinMarketCap
   - DEX aggregators (1inch, Matcha)

4. **Build Community**
   - Twitter announcement
   - Telegram/Discord group
   - Regular updates

---

## 🚨 Troubleshooting

### **"Insufficient ETH for gas"**
- You need at least 0.01 ETH on Ethereum Mainnet
- Buy ETH on exchange and send to your wallet

### **"Network connection failed"**
- Check internet connection
- Try again in a few minutes
- RPC might be temporarily down

### **"Transaction already processed"**
- Your Base TX has already been bridged
- Check your Ethereum WTBTC balance
- No action needed

### **"Only bridge operator can call this"**
- You need to be bridge operator
- By default, deployer is operator
- Check: `cast call CONTRACT "bridgeOperator()"`

### **Deployment fails**
- Check you have enough ETH
- Verify you're on Ethereum Mainnet
- Check Infura API key is valid

---

## 📚 Additional Scripts

### **Check Bridge Status:**
```bash
node scripts/check_bridge_status.js
```

### **Update Bridge Operator:**
```bash
node scripts/update_operator.js NEW_OPERATOR_ADDRESS
```

### **Emergency Pause:**
```bash
node scripts/pause_bridge.js
```

### **Add Liquidity:**
```bash
node scripts/add_liquidity_ethereum.js
```

---

## 🔗 Useful Links

### **Explorers:**
- Ethereum: https://etherscan.io
- Base Sepolia: https://sepolia.basescan.org

### **DEXs (Ethereum):**
- Uniswap: https://app.uniswap.org
- SushiSwap: https://www.sushi.com
- Balancer: https://balancer.fi

### **Tools:**
- Gas Tracker: https://etherscan.io/gastracker
- Token Checker: https://tokensniffer.com
- Liquidity Checker: https://dexscreener.com

### **Infura Dashboard:**
- https://infura.io/dashboard

---

## 🎯 Summary

**Your Base Sepolia Token:**
- Address: `0xE274570e000C32F5Cb2BC7c476D3BDC77Ed74dD5`
- Network: Base Sepolia
- Your TX: `0xab880c7a...4a0d78cc`

**Deploy to Ethereum Mainnet:**
```bash
./deploy_ethereum_mainnet_bridge.sh
```

**Bridge Your Tokens:**
```bash
node scripts/bridge_from_base.js
```

**Add Liquidity:**
```bash
node scripts/add_liquidity_ethereum.js
```

---

**✨ May your bridge be prosperous, friend from the digital realm! ✨**

---

**Created:** 2026-02-02
**Network:** Ethereum Mainnet (Chain ID: 1)
**Status:** Ready to Deploy 🚀
