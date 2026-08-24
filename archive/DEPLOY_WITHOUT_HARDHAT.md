# 🎯 Deploy Contracts WITHOUT Hardhat

**Your Node.js 25.3.0 is incompatible with Hardhat. Here are 3 alternative methods.**

---

## ✅ **METHOD 1: MetaMask Browser Deployment (RECOMMENDED)**

You already have `deploy_metamask.html` which bypasses Hardhat entirely!

### **Steps:**

1. **Open the file in a browser:**
   ```bash
   # On Linux/Mac:
   xdg-open /home/user/nexus_agi/deploy_metamask.html

   # Or manually navigate to:
   file:///home/user/nexus_agi/deploy_metamask.html
   ```

2. **Make sure MetaMask is installed:**
   - Install from: https://metamask.io
   - Import your wallet using private key: `0eee6f45b0af8f5a6a24744a1a978346d5bd66b41c64dc30bd18a32e246515cd`

3. **Switch MetaMask to Polygon:**
   - Network Name: Polygon Mainnet
   - RPC URL: `https://polygon-mainnet.infura.io/v3/38f2c0df20264c98b108d04914464e12`
   - Chain ID: 137
   - Currency: MATIC

4. **Click "Deploy All (Recommended)"**

5. **Approve each transaction in MetaMask**

6. **Done!** Contract addresses will be displayed.

---

## 🌐 **METHOD 2: Remix IDE (Pure Solidity, No Installation)**

Remix is a browser-based Solidity IDE - perfect for deploying without CLI tools.

### **Steps:**

1. **Open Remix:** https://remix.ethereum.org

2. **Upload your contracts:**
   - Click "File Explorer" (left sidebar)
   - Upload `contracts/TestnetWBTC.sol`
   - Upload `contracts/EthereumBridgeToken.sol`

3. **Install OpenZeppelin:**
   - Click "Plugin Manager" → Install "OpenZeppelin"
   - Or use GitHub import in each contract

4. **Compile:**
   - Click "Solidity Compiler" tab
   - Select compiler version: `0.8.20`
   - Click "Compile TestnetWBTC.sol"
   - Click "Compile EthereumBridgeToken.sol"

5. **Deploy:**
   - Click "Deploy & Run Transactions" tab
   - Environment: Select "Injected Provider - MetaMask"
   - MetaMask will connect
   - Switch MetaMask to Polygon network
   - Select contract: `TestnetWBTC`
   - Constructor args: `100` (initial supply)
   - Click "Deploy"
   - Approve transaction in MetaMask
   - Repeat for `EthereumBridgeToken` with args: `1000`

6. **Done!** Contracts deployed to Polygon.

---

## 💻 **METHOD 3: Pure JavaScript Deployment Script**

Uses web3.js directly without Hardhat.

### **Installation:**

```bash
cd /home/user/nexus_agi
npm install web3 @openzeppelin/contracts solc
```

### **Create deployment script:**

See `scripts/deploy_pure_web3.js` (created for you below)

### **Run:**

```bash
node scripts/deploy_pure_web3.js
```

---

## 📋 **Which Method Should You Use?**

| Method | Difficulty | Requirements | Best For |
|--------|-----------|--------------|----------|
| **MetaMask HTML** | ⭐ Easiest | Browser + MetaMask | Quick deployment |
| **Remix IDE** | ⭐⭐ Easy | Browser only | Testing/learning |
| **Pure Web3.js** | ⭐⭐⭐ Medium | Node.js + npm | Automation |

---

## 🎯 **Recommended: Use MetaMask Browser Deployment**

Your `deploy_metamask.html` file is already configured with:
- ✅ All contract bytecode embedded
- ✅ Polygon RPC configuration
- ✅ Automatic deployment
- ✅ ZetaLink Bitcoin integration
- ✅ No CLI required

**Just open it in a browser and click deploy!**

---

## ⚠️ **Before Deploying:**

Make sure you have **MATIC** for gas fees on Polygon:
- Check balance: https://polygonscan.com/address/0x9fe74d9d6f1ae0ce1fb3b51d4a82c05b74e280f3
- Buy MATIC if needed or bridge from another chain

---

## 🚀 **Quick Start Command (MetaMask Method):**

```bash
cd /home/user/nexus_agi
./launch_bridge.sh
# Then select: "1) 🌐 Open MetaMask Deployment (Browser - RECOMMENDED)"
```

This will open your browser and you can deploy with one click!

---

**Created:** 2026-02-01
**No Hardhat Required!** ✅
