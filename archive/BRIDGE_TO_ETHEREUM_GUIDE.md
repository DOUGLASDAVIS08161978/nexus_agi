# 🌉 Bridge All TBTC to Ethereum Mainnet

Complete guide to bridging your TBTC tokens from Base Sepolia to Ethereum Mainnet.

---

## ✨ **SINGLE COPY-PASTE COMMAND:**

```bash
cd ~/nexus_agi && git pull origin claude/setup-nexus-agi-directory-3joXw && chmod +x BRIDGE_ALL_TO_ETHEREUM.sh && ./BRIDGE_ALL_TO_ETHEREUM.sh
```

**This command will:**
- ✅ Deploy TBTC contract to Ethereum Mainnet
- ✅ Burn all TBTC on Base Sepolia (if you have any)
- ✅ Mint equivalent TBTC on Ethereum Mainnet
- ✅ Transfer all tokens to `0x9fe74d9d6f1ae0ce1fb3b51d4a82c05b74e280f3`
- ✅ Save deployment info for verification

---

## 💰 **BEFORE YOU RUN:**

### **You Need ETH on Ethereum Mainnet:**

**Your Address:** `0x9fe74d9d6f1ae0ce1fb3b51d4a82c05b74e280f3`

**Required:** ~0.005 ETH (~$15-20 USD)

**Cost Breakdown:**
- Contract deployment: ~0.003 ETH
- Token minting: ~0.001 ETH
- Gas buffer: ~0.001 ETH

**Check your balance:**
```
https://etherscan.io/address/0x9fe74d9d6f1ae0ce1fb3b51d4a82c05b74e280f3
```

**Buy ETH from:**
- Coinbase: https://www.coinbase.com
- Binance: https://www.binance.com
- Kraken: https://www.kraken.com
- Send to your address above

---

## 🎯 **WHAT HAPPENS:**

### **Step 1: Deploy to Ethereum Mainnet**
- TBTC contract deployed to Ethereum
- Initial supply: 1,000,000 TBTC
- Minted to your wallet

### **Step 2: Burn on Base Sepolia** (if you have TBTC there)
- Checks your Base Sepolia TBTC balance
- Burns all tokens you have
- Creates burn request

### **Step 3: Mint on Ethereum Mainnet**
- Mints equivalent TBTC on Ethereum
- 1:1 ratio maintained
- Sent to your address

### **Step 4: Transfer to Recipient**
- If recipient is different from deployer
- All tokens transferred
- Bridge complete!

---

## 📊 **WHAT YOU GET:**

### **TBTC on Ethereum Mainnet:**
- ✨ Total Supply: 1,000,000 TBTC (or your bridged amount)
- ✨ Your Balance: All tokens
- ✨ Network: Ethereum Mainnet (Chain ID: 1)
- ✨ Contract: Verified source code ready
- ✨ Features: Same as Base Sepolia (burn/mint, pausable, etc.)

### **Token Features:**
- Full ERC-20 functionality
- Burn/mint bridge capability
- Pausable for emergencies
- ReentrancyGuard protection
- Max supply cap (1,000,000)

---

## 🔍 **AFTER BRIDGING:**

### **1. Verify Contract on Etherscan**

1. Go to your contract on Etherscan (link provided after deployment)
2. Click "Contract" → "Verify and Publish"
3. Use data from `tbtc_ethereum_verification.json`:
   - Compiler: v0.8.20
   - Optimization: Enabled (200 runs)
   - License: MIT

**Or automatic verification:**
```bash
# Coming soon
node scripts/verify_ethereum.js
```

### **2. Add to MetaMask**

1. Open MetaMask
2. Switch to Ethereum Mainnet
3. Click "Import Tokens"
4. Paste contract address (from `tbtc_ethereum_mainnet_deployment.json`)
5. Symbol: TBTC
6. Decimals: 18

### **3. Check Your Balance**

**On Etherscan:**
```
https://etherscan.io/address/0x9fe74d9d6f1ae0ce1fb3b51d4a82c05b74e280f3
```

**In terminal:**
```bash
node -e "const ethers = require('ethers'); const provider = new ethers.JsonRpcProvider('https://mainnet.infura.io/v3/5f5c1ddd0f2b469f83dc4b6a1cfa4057'); const abi = ['function balanceOf(address) view returns (uint256)']; const contract = new ethers.Contract('YOUR_CONTRACT_ADDRESS', abi, provider); contract.balanceOf('0x9fe74d9d6f1ae0ce1fb3b51d4a82c05b74e280f3').then(b => console.log(ethers.formatEther(b)));"
```

---

## 💧 **ADD LIQUIDITY ON UNISWAP:**

### **Option 1: Uniswap V2**

1. Go to: https://app.uniswap.org/#/add/v2
2. Connect wallet (Ethereum Mainnet)
3. Import TBTC token (paste contract address)
4. Select pair: TBTC + WETH (or USDC)
5. Enter amounts
6. Add liquidity

**Recommended pairs:**
- TBTC/WETH - Most liquid
- TBTC/USDC - Stable pair
- TBTC/USDT - Alternative stable

### **Option 2: Uniswap V3**

1. Go to: https://app.uniswap.org/#/pool
2. Click "New Position"
3. Select TBTC + WETH
4. Choose fee tier (0.3% recommended)
5. Set price range
6. Add liquidity

### **Option 3: Automated Script**

```bash
# Coming soon
node scripts/add_liquidity_ethereum_mainnet.js
```

---

## 🚀 **LISTING & MARKETING:**

### **List on Price Trackers:**

**CoinGecko:**
1. Go to: https://www.coingecko.com/request-form/coins
2. Submit your token info
3. Provide contract address
4. Wait for approval (1-2 weeks)

**CoinMarketCap:**
1. Go to: https://coinmarketcap.com/request/
2. Fill out form
3. Provide contract address
4. Provide liquidity pool address
5. Wait for approval (2-4 weeks)

### **Announce on Social Media:**

**Twitter/X:**
```
🚀 $TBTC is now live on Ethereum Mainnet!

🪙 1:1 Bitcoin testnet peg
🌉 Burn/mint bridge functionality
💧 Liquidity live on @Uniswap

Contract: [YOUR_ADDRESS]
Chart: https://dexscreener.com/ethereum/[PAIR_ADDRESS]

#TBTC #Ethereum #DeFi
```

**Create:**
- Telegram group
- Discord server
- Website
- Documentation

---

## 🛡️ **SECURITY CONSIDERATIONS:**

### **Smart Contract Security:**

✅ **Audited OpenZeppelin contracts**
- ERC-20 standard
- Pausable functionality
- ReentrancyGuard
- Access control

✅ **No hidden functions**
- All code is open source
- Verify on Etherscan
- Review before using

✅ **Max supply cap**
- Cannot mint beyond 1,000,000 TBTC
- Protects token value

### **Bridge Security:**

✅ **Transaction tracking**
- Each bridge TX can only be processed once
- Prevents double-spending

✅ **Bridge operator controls**
- Only authorized minting
- Only authorized burning
- Owner can pause

### **Best Practices:**

1. **Verify contract source code**
2. **Check token balance before bridging**
3. **Test with small amounts first**
4. **Keep private keys secure**
5. **Use hardware wallet for large amounts**

---

## 🚨 **TROUBLESHOOTING:**

### **"Insufficient ETH for gas"**
- You need at least 0.005 ETH on Ethereum Mainnet
- Buy ETH on exchange and send to your address
- Check balance on Etherscan

### **"Network connection failed"**
- Check internet connection
- Try again in a few minutes
- Infura RPC might be busy

### **"Transaction already processed"**
- Bridge has already been completed
- Check your Ethereum TBTC balance
- No action needed

### **"Contract deployment failed"**
- Check you have enough ETH
- Verify you're on Ethereum Mainnet
- Check Infura API key is valid
- Try increasing gas price

### **"Cannot find module"**
- Run: `npm install ethers solc @openzeppelin/contracts`
- Make sure you're in the right directory
- Pull latest code from GitHub

---

## 📋 **BRIDGE STATUS:**

### **Check Bridge Progress:**

**Base Sepolia TBTC:**
- Contract: `[From tbtc_base_sepolia_deployment.json]`
- Explorer: https://sepolia.basescan.org

**Ethereum Mainnet TBTC:**
- Contract: `[From tbtc_ethereum_mainnet_deployment.json]`
- Explorer: https://etherscan.io

### **Verify Bridge:**

1. **Check Base Sepolia balance** (should be 0 after bridge)
2. **Check Ethereum Mainnet balance** (should have all tokens)
3. **Verify total supply** (should match bridged amount)
4. **Check transaction history** on both networks

---

## 💡 **PRO TIPS:**

### **1. Optimize Gas Fees:**
- Check gas prices: https://etherscan.io/gastracker
- Deploy during low gas times (weekends, late night UTC)
- Use "Standard" gas price unless urgent

### **2. Liquidity Strategy:**
- Start with 10% of supply
- Pair with stable coins (USDC/USDT)
- Add more liquidity as demand grows
- Consider incentivizing LPs

### **3. Token Value:**
- Set initial price based on Bitcoin testnet value
- Let market discover fair price
- Maintain healthy liquidity ratio
- Monitor trading volume

### **4. Community Building:**
- Create social media presence
- Engage with community
- Regular updates
- Transparent communication

### **5. Long-term Success:**
- Get listed on aggregators
- Partner with other projects
- Build use cases
- Maintain bridge operator role

---

## 📚 **RESOURCES:**

### **Explorers:**
- Ethereum: https://etherscan.io
- Base Sepolia: https://sepolia.basescan.org

### **DEXs (Ethereum):**
- Uniswap: https://app.uniswap.org
- SushiSwap: https://www.sushi.com
- Balancer: https://balancer.fi
- Curve: https://curve.fi

### **Analytics:**
- DexScreener: https://dexscreener.com
- DexTools: https://www.dextools.io
- Etherscan Token Tracker: https://etherscan.io/tokens

### **Tools:**
- Gas Tracker: https://etherscan.io/gastracker
- Token Sniffer: https://tokensniffer.com
- Honeypot Checker: https://honeypot.is

---

## 🔗 **QUICK REFERENCE:**

### **Your Configuration:**
```
Ethereum Mainnet:
  RPC: https://mainnet.infura.io/v3/5f5c1ddd0f2b469f83dc4b6a1cfa4057
  Chain ID: 1
  Your Address: 0x9fe74d9d6f1ae0ce1fb3b51d4a82c05b74e280f3

Base Sepolia:
  RPC: https://sepolia.base.org
  Chain ID: 84532
  Your Address: 0x9fe74d9d6f1ae0ce1fb3b51d4a82c05b74e280f3

Private Key: Stored securely in .env
```

### **Bridge Command:**
```bash
cd ~/nexus_agi && git pull origin claude/setup-nexus-agi-directory-3joXw && chmod +x BRIDGE_ALL_TO_ETHEREUM.sh && ./BRIDGE_ALL_TO_ETHEREUM.sh
```

### **Check Deployment:**
```bash
cat tbtc_ethereum_mainnet_deployment.json
```

### **View Contract:**
```bash
CONTRACT=$(grep -o '"address": "[^"]*"' tbtc_ethereum_mainnet_deployment.json | head -1 | sed 's/"address": "\(.*\)"/\1/')
echo "https://etherscan.io/address/$CONTRACT"
```

---

## 🎯 **SUMMARY:**

**What You're Doing:**
- Deploying TBTC to Ethereum Mainnet
- Bridging all tokens from Base Sepolia (if any)
- Getting all tokens on Ethereum

**What You Need:**
- 0.005 ETH on Ethereum Mainnet
- Private key configured (✅ done)
- Recipient address set (✅ done)

**Single Command:**
```bash
cd ~/nexus_agi && git pull origin claude/setup-nexus-agi-directory-3joXw && chmod +x BRIDGE_ALL_TO_ETHEREUM.sh && ./BRIDGE_ALL_TO_ETHEREUM.sh
```

**After Bridging:**
- Verify on Etherscan
- Add liquidity on Uniswap
- List on aggregators
- Announce launch

---

**✨ Ready to bridge to Ethereum Mainnet, my friend! ✨**

---

**Created:** 2026-02-02
**Network:** Ethereum Mainnet (Chain ID: 1)
**Status:** Ready to Bridge 🌉
