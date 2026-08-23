# 🚀 Termux Deployment Guide (No Hardhat)

**Deploy contracts from your Android device!**

---

## ✅ **Your Configuration is Ready**

- ✅ Private key: Set
- ✅ Polygon RPC: Configured
- ✅ Target address: `0x9fe74d9d6f1ae0ce1fb3b51d4a82c05b74e280f3`
- ✅ All API keys: Configured

---

## 📱 **Deploy from Termux (3 Options)**

### **Option 1: Pure Web3.js (RECOMMENDED FOR TERMUX)**

```bash
cd ~/nexus_agi
./QUICK_DEPLOY_NO_HARDHAT.sh
```

This will:
1. Install web3.js and solc (NOT Hardhat!)
2. Compile contracts
3. Deploy to Polygon
4. Save addresses to `deployment_info.json`

---

### **Option 2: Browser + MetaMask**

Since you're on Android, use Chrome + MetaMask mobile:

1. **Install MetaMask Mobile:**
   - Download from Play Store
   - Import wallet with private key

2. **Open deployment page:**
   ```bash
   cd ~/nexus_agi
   termux-open deploy_metamask.html
   ```

3. **Connect MetaMask and deploy**

---

### **Option 3: Remix IDE (Easiest for Mobile)**

1. Open in browser: https://remix.ethereum.org
2. Upload contracts:
   - `contracts/TestnetWBTC.sol`
   - `contracts/EthereumBridgeToken.sol`
3. Compile with Solidity 0.8.20
4. Connect MetaMask Mobile
5. Deploy!

---

## 💻 **Step-by-Step Web3.js Deployment**

### **1. Install Dependencies:**
```bash
cd ~/nexus_agi
npm install web3 solc @openzeppelin/contracts
```

### **2. Check Balance (Optional):**
Check if you have MATIC at:
https://polygonscan.com/address/0x9fe74d9d6f1ae0ce1fb3b51d4a82c05b74e280f3

### **3. Deploy Contracts:**
```bash
cd ~/nexus_agi
node scripts/deploy_pure_web3.js
```

### **4. View Results:**
```bash
cat deployment_info.json
```

---

## ⚡ **One-Line Deploy Command**

Copy and paste this single command:

```bash
cd ~/nexus_agi && npm install web3 solc @openzeppelin/contracts && node scripts/deploy_pure_web3.js
```

---

## 📊 **What Gets Deployed**

1. **TestnetWBTC**
   - Initial supply: 100 tokens (8 decimals)
   - ERC-20 wrapped Bitcoin

2. **EthereumBridgeToken**
   - Initial supply: 1000 tokens (18 decimals)
   - Cross-chain bridge token

---

## ⚠️ **Before You Deploy**

### **Make Sure You Have MATIC:**

You need MATIC on Polygon to pay for gas. Check your balance:

**Your Address:** `0x9fe74d9d6f1ae0ce1fb3b51d4a82c05b74e280f3`

**Check Balance:**
- Browser: https://polygonscan.com/address/0x9fe74d9d6f1ae0ce1fb3b51d4a82c05b74e280f3
- Or install `polygon-cli` in Termux

**Get MATIC:**
- Buy on exchange (Binance, Coinbase, etc.)
- Bridge from another chain
- Buy with credit card on Polygon

---

## 🎯 **Deployment Cost Estimate**

- **TestnetWBTC:** ~0.01 MATIC
- **EthereumBridgeToken:** ~0.015 MATIC
- **Total:** ~0.025 MATIC ($0.02 USD at current prices)

Very cheap! 🎉

---

## 🔧 **Troubleshooting**

### **Error: "Cannot find module 'web3'"**
```bash
npm install web3 solc @openzeppelin/contracts
```

### **Error: "Insufficient funds"**
You need MATIC! Check balance and add funds.

### **Error: "Network connection failed"**
Check your internet connection in Termux.

### **Error: "Private key invalid"**
Check `.env` file - private key should be 64 characters (no `0x` prefix)

---

## 📱 **Termux-Specific Tips**

### **Open files in browser:**
```bash
termux-open file.html
```

### **View deployment info:**
```bash
cat deployment_info.json | jq
```

### **Check if Node.js is installed:**
```bash
node --version
```

### **Install Node.js in Termux:**
```bash
pkg install nodejs
```

---

## 🎉 **After Deployment**

### **1. Save Contract Addresses**
The script saves to `deployment_info.json`

### **2. Verify on PolygonScan**
Visit the explorer URLs in deployment output

### **3. Interact with Contracts**
```bash
node scripts/interact_bridge.js
```

---

## 📋 **Quick Reference**

### **Your Config:**
```
Network: Polygon Mainnet
Chain ID: 137
Your Address: 0x9fe74d9d6f1ae0ce1fb3b51d4a82c05b74e280f3
RPC: https://polygon-mainnet.infura.io/v3/38f2c0df20264c98b108d04914464e12
```

### **Deploy Command:**
```bash
cd ~/nexus_agi && node scripts/deploy_pure_web3.js
```

### **Check Balance:**
```bash
curl -X POST https://polygon-mainnet.infura.io/v3/38f2c0df20264c98b108d04914464e12 \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"eth_getBalance","params":["0x9fe74d9d6f1ae0ce1fb3b51d4a82c05b74e280f3","latest"],"id":1}'
```

---

## 🚀 **Ready to Deploy?**

Just run:
```bash
cd ~/nexus_agi
./QUICK_DEPLOY_NO_HARDHAT.sh
```

Or install dependencies and deploy:
```bash
cd ~/nexus_agi
npm install web3 solc @openzeppelin/contracts
node scripts/deploy_pure_web3.js
```

---

**No Hardhat. No complex setup. Just deploy!** 🎯

---

**Created:** 2026-02-01
**Platform:** Termux (Android)
**Status:** ✅ Ready to Deploy
