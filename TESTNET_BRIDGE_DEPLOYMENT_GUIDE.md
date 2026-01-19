# 🌉 Nexus AGI Testnet Bridge - Complete Deployment Guide

**Created:** 2026-01-19
**Network:** Sepolia Testnet
**Recipient:** `0x24f6b1ce11c57d40b542f91ac85fa9eb61f78771`

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Quick Start](#quick-start)
4. [Detailed Setup](#detailed-setup)
5. [Deployment](#deployment)
6. [Bridge Operations](#bridge-operations)
7. [Nexus AGI Directory Integration](#nexus-agi-directory-integration)
8. [Troubleshooting](#troubleshooting)

---

## 🎯 Overview

This deployment sets up a complete **Bitcoin ↔ Ethereum testnet bridge** infrastructure with:

### Smart Contracts Deployed (2):

1. **TestnetWBTC** (tWBTC)
   - Wrapped Bitcoin token for Sepolia testnet
   - ERC-20 standard with 8 decimals
   - Mintable and burnable
   - Initial supply: 100 tWBTC

2. **EthereumBridgeToken** (XCBT)
   - Cross-chain bridge token
   - Supports Bitcoin ↔ Ethereum ↔ Polygon bridging
   - Advanced features: lock/unlock, mint/burn
   - Bridge operator permissions
   - Initial supply: 1000 XCBT

### Features:

✅ Testnet-only (safe, no real money)
✅ Automatic token minting to your address
✅ Bridge operator permissions configured
✅ Integration with 133+ APIs from Nexus AGI Directory
✅ Complete interaction scripts
✅ MetaMask integration ready

---

## ⚙️ Prerequisites

### 1. Software Requirements

```bash
# Node.js 18 or higher
node --version  # Should be >= 18.0.0

# npm
npm --version

# Git
git --version
```

### 2. Wallet Setup

You need an **Ethereum wallet** with:

- ✅ Your address: `0x24f6b1ce11c57d40b542f91ac85fa9eb61f78771`
- ✅ Sepolia ETH for gas fees (free from faucets)
- ✅ MetaMask or similar Web3 wallet

### 3. Get Sepolia Testnet ETH (Free)

Visit any of these faucets and request testnet ETH:

- 🚰 [Alchemy Sepolia Faucet](https://www.alchemy.com/faucets/ethereum-sepolia)
- 🚰 [Sepolia Faucet](https://sepoliafaucet.com)
- 🚰 [PoW Faucet](https://sepolia-faucet.pk910.de)

**Recommended amount:** 0.5 ETH (more than enough for all deployments)

### 4. Private Key

You'll need your wallet's **private key** to sign deployment transactions.

⚠️ **SECURITY WARNING:**
- Only use this on **TESTNET**
- Never share your private key
- Never commit it to Git
- Use a separate test wallet

---

## 🚀 Quick Start

### Step 1: Configure Environment

Edit `.env` file and add your private key:

```bash
# Open .env file
nano .env

# Replace this line:
PRIVATE_KEY=your_private_key_here_without_0x_prefix

# With your actual private key (without 0x prefix):
PRIVATE_KEY=abcdef1234567890...  # Your actual private key
```

### Step 2: Verify Configuration

```bash
# Check Node.js version
node --version

# Verify dependencies are installed
npm list | grep hardhat

# Should show:
# ├── @nomicfoundation/hardhat-toolbox@4.0.0
# ├── hardhat@2.19.0
```

### Step 3: Deploy to Sepolia

```bash
# Deploy all contracts
npx hardhat run scripts/deploy_testnet_bridge.js --network sepolia
```

This will:
1. ✅ Deploy TestnetWBTC contract
2. ✅ Deploy EthereumBridgeToken contract
3. ✅ Mint 10 tWBTC to your address
4. ✅ Mint 50 XCBT to your address
5. ✅ Set you as bridge operator
6. ✅ Save deployment info to JSON file

### Step 4: View Your Tokens

The script will output contract addresses. Use them to:

**Add to MetaMask:**

1. Open MetaMask
2. Switch to **Sepolia Test Network**
3. Click "Import Tokens"
4. Paste contract address (from deployment output)
5. Symbol and decimals will auto-fill
6. Confirm

### Step 5: Interact with Bridge

```bash
# Run interaction dashboard
node scripts/interact_bridge.js
```

This displays:
- 📊 All 133+ APIs from Nexus AGI Directory
- 🌉 Your deployed contract addresses
- 💰 Your token balances
- 📚 Bridge operation examples
- 💾 Generated integration guide

---

## 🔧 Detailed Setup

### Install Dependencies

If you need to reinstall:

```bash
# Install all dependencies
npm install

# Verify installation
npx hardhat --version  # Should show: 3.1.4 or similar
```

### Environment Variables

The `.env` file contains all configuration:

```env
# RPC URLs (already configured with public endpoints)
SEPOLIA_RPC_URL=https://ethereum-sepolia-rpc.publicnode.com

# Your wallet private key (YOU MUST ADD THIS)
PRIVATE_KEY=your_key_here

# Recipient address (already set to your address)
RECIPIENT_ADDRESS=0x24f6b1ce11c57d40b542f91ac85fa9eb61f78771

# Token supply settings
INITIAL_WBTC_SUPPLY=100
INITIAL_BRIDGE_SUPPLY=1000

# Bridge configuration
BRIDGE_OPERATOR_ADDRESS=0x24f6b1ce11c57d40b542f91ac85fa9eb61f78771
```

### Network Configuration

Check `hardhat.config.js`:

```javascript
networks: {
  sepolia: {
    url: process.env.SEPOLIA_RPC_URL,
    accounts: process.env.PRIVATE_KEY ? [process.env.PRIVATE_KEY] : [],
  }
}
```

---

## 🚢 Deployment

### Local Testing (Optional)

Test locally before deploying to Sepolia:

```bash
# Start local Hardhat network
npx hardhat node

# In another terminal, deploy to local network
npx hardhat run scripts/deploy_testnet_bridge.js --network localhost
```

### Deploy to Sepolia Testnet

```bash
# Full deployment
npx hardhat run scripts/deploy_testnet_bridge.js --network sepolia
```

### Expected Output

```
═══════════════════════════════════════════════════
  NEXUS AGI - TESTNET BRIDGE DEPLOYMENT TO SEPOLIA
═══════════════════════════════════════════════════

🔑 Deploying with account: 0x...
💰 Account balance: 0.5 ETH

──────────────────────────────────────────────────
  PHASE 1: DEPLOYING TESTNET WBTC CONTRACT
──────────────────────────────────────────────────

🚀 Deploying TestnetWBTC...
✅ TestnetWBTC deployed to: 0xABC...123

💸 Minting 10 tWBTC to recipient...
✅ Minted successfully!

──────────────────────────────────────────────────
  PHASE 2: DEPLOYING ETHEREUM BRIDGE TOKEN CONTRACT
──────────────────────────────────────────────────

🚀 Deploying EthereumBridgeToken...
✅ EthereumBridgeToken deployed to: 0xDEF...456

──────────────────────────────────────────────────
  PHASE 3: CONFIGURING BRIDGE OPERATOR
──────────────────────────────────────────────────

🔧 Adding bridge operator...
✅ Bridge operator added!

💸 Minting 50 XCBT to recipient...
✅ Minted successfully!

═══════════════════════════════════════════════════
             ✅ DEPLOYMENT SUCCESSFUL!
═══════════════════════════════════════════════════
```

### Deployment Files Created

- `deployment_sepolia_<timestamp>.json` - Complete deployment info
- `bridge_integration_guide.json` - Integration guide (created by interact script)

---

## 🌉 Bridge Operations

### Available Functions

#### 1. Bridge to Bitcoin Testnet

Burns tokens on Ethereum, initiates bridge to Bitcoin.

```javascript
// Example using ethers.js
const bridge = await ethers.getContractAt(
  'EthereumBridgeToken',
  '<bridge-address>'
);

const btcAddress = 'tb1q...'; // Your Bitcoin testnet address
const amount = ethers.parseUnits('1', 8); // 1 XCBT

const tx = await bridge.bridgeToBitcoin(btcAddress, amount);
await tx.wait();
```

#### 2. Bridge to Polygon

Locks tokens on Ethereum for minting on Polygon.

```javascript
const polygonAddress = '0x...'; // Your Polygon address
const amount = ethers.parseUnits('5', 8); // 5 XCBT

const tx = await bridge.bridgeToPolygon(polygonAddress, amount);
await tx.wait();
```

#### 3. Bridge from Bitcoin (Operator Only)

Mints tokens on Ethereum after receiving Bitcoin.

```javascript
// Only bridge operator can call this
const recipient = '0x24f6b1ce11c57d40b542f91ac85fa9eb61f78771';
const amount = ethers.parseUnits('2', 8);
const btcTxId = 'abc123...'; // Bitcoin transaction ID

const tx = await bridge.bridgeFromBitcoin(recipient, amount, btcTxId);
await tx.wait();
```

#### 4. Check Balance

```javascript
const wbtc = await ethers.getContractAt('TestnetWBTC', '<wbtc-address>');
const balance = await wbtc.balanceOf('0x24f6b1ce11c57d40b542f91ac85fa9eb61f78771');
console.log('Balance:', ethers.formatUnits(balance, 8), 'tWBTC');
```

### Using Hardhat Console

```bash
# Start Hardhat console on Sepolia
npx hardhat console --network sepolia
```

```javascript
// In console:
const wbtc = await ethers.getContractAt('TestnetWBTC', '<address>');
const balance = await wbtc.balanceOf('0x24f6b1ce11c57d40b542f91ac85fa9eb61f78771');
console.log(ethers.formatUnits(balance, 8));
```

---

## 🔗 Nexus AGI Directory Integration

### Overview

The bridge integrates with **133+ APIs** from the [Nexus AGI Directory](https://nexus-agi.com/.well-known/seeds-public.json).

### Fetch Available APIs

```javascript
const https = require('https');

https.get('https://nexus-agi.com/.well-known/seeds-public.json', (res) => {
  let data = '';
  res.on('data', chunk => data += chunk);
  res.on('end', () => {
    const apis = JSON.parse(data);
    console.log(`Found ${apis.length} APIs`);

    // Filter blockchain-related APIs
    const blockchainAPIs = apis.filter(api =>
      api.capabilities?.includes('blockchain') ||
      api.name.toLowerCase().includes('crypto')
    );

    blockchainAPIs.forEach(api => {
      console.log(`${api.name}: ${api.endpoint}`);
    });
  });
});
```

### Integration Examples

**Example 1: Enhance bridge with AI decision-making**

```javascript
// Use Claude or OpenAI to analyze bridge transactions
const anthropicAPI = apis.find(api => api.id.includes('anthropic'));
// Call API to analyze optimal bridge timing, gas fees, etc.
```

**Example 2: Multi-chain monitoring**

```javascript
// Use available blockchain APIs to monitor multiple networks
// Track bridge transactions across Bitcoin, Ethereum, Polygon
```

---

## 🛠️ Troubleshooting

### Error: "Insufficient balance"

**Solution:** Get Sepolia ETH from faucets listed above.

### Error: "Invalid private key"

**Solution:**
- Ensure private key is in `.env` file
- Remove `0x` prefix from private key
- Check for extra spaces or quotes

### Error: "Network connection failed"

**Solutions:**
- Check internet connection
- Try alternative RPC URL in `.env`:
  ```env
  SEPOLIA_RPC_URL=https://rpc.sepolia.org
  ```

### Tokens not showing in MetaMask

**Solutions:**
1. Verify you're on **Sepolia Test Network**
2. Double-check contract address
3. Manually add token with correct decimals (8)
4. Check Etherscan to verify deployment

### Can't find deployment file

**Solution:**
```bash
# List deployment files
ls -la deployment_*.json

# If missing, redeploy:
npx hardhat run scripts/deploy_testnet_bridge.js --network sepolia
```

---

## 📊 Contract Addresses

After deployment, your contract addresses will be in:
- Console output
- `deployment_sepolia_<timestamp>.json` file

**View on Etherscan:**
- TestnetWBTC: `https://sepolia.etherscan.io/address/<address>`
- BridgeToken: `https://sepolia.etherscan.io/address/<address>`

**Add to MetaMask:**
1. TestnetWBTC (tWBTC) - Decimals: 8
2. EthereumBridgeToken (XCBT) - Decimals: 8

---

## 🎯 Next Steps

1. ✅ Deploy contracts to Sepolia
2. ✅ Add tokens to MetaMask
3. ✅ Verify balances (10 tWBTC, 50 XCBT)
4. ✅ Run interaction dashboard
5. ✅ Test bridge functions
6. ✅ Explore Nexus AGI Directory APIs
7. ✅ Build custom integrations

---

## 📚 Additional Resources

- [Hardhat Documentation](https://hardhat.org/docs)
- [OpenZeppelin Contracts](https://docs.openzeppelin.com/contracts)
- [Ethers.js Documentation](https://docs.ethers.org)
- [Sepolia Testnet Explorer](https://sepolia.etherscan.io)
- [Nexus AGI Directory](https://nexus-agi.com)

---

## ⚠️ Important Notes

### Testnet Only

- All contracts are on **Sepolia testnet**
- Tokens have **zero real value**
- Safe for testing and learning
- **Cannot** bridge to Bitcoin mainnet

### Security

- Never use mainnet private keys
- Keep private keys secure
- Don't commit `.env` to Git
- Use separate test wallet

### Gas Fees

- Deployment costs ~0.02-0.05 Sepolia ETH
- Bridge operations cost ~0.001-0.003 ETH each
- All gas fees are paid in testnet ETH (free)

---

## 💡 Support

If you encounter issues:

1. Check this troubleshooting guide
2. Review Hardhat error messages
3. Verify configuration in `.env`
4. Check Sepolia testnet status
5. Ensure sufficient gas (Sepolia ETH)

---

**Happy Bridging! 🌉**

Created by Nexus AGI
For: 0x24f6b1ce11c57d40b542f91ac85fa9eb61f78771
Network: Sepolia Testnet
Date: 2026-01-19
