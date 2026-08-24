# 🚀 Launch 1 Million $TBTC on Base Sepolia

Complete guide to launching your TBTC token with burn/mint bridge functionality.

---

## ✨ **ONE-COMMAND LAUNCH**

### **Copy and paste this single command in Termux:**

```bash
cd ~/nexus_agi && git pull origin claude/setup-nexus-agi-directory-3joXw && chmod +x LAUNCH_TBTC.sh && ./LAUNCH_TBTC.sh
```

**That's it!** This command will:
- ✅ Pull latest code
- ✅ Deploy TBTC token to Base Sepolia
- ✅ Mint 1,000,000 TBTC
- ✅ Transfer all tokens to `0x9fe74d9d6f1ae0ce1fb3b51d4a82c05b74e280f3`
- ✅ Set up burn/mint bridge
- ✅ Save deployment info for verification

---

## 📋 **What You Get**

### **TBTC Token Features:**

✨ **Total Supply:** 1,000,000 TBTC
✨ **Your Balance:** 1,000,000 TBTC (all tokens)
✨ **Network:** Base Sepolia (Chain ID: 84532)
✨ **Bitcoin Peg:** 1:1 with Bitcoin testnet
✨ **Bridge:** Burn/Mint functionality
✨ **Security:** Pausable + ReentrancyGuard

### **Smart Contract Functions:**

1. **mint()** - Mint TBTC when Bitcoin is locked
   - Bridge operator calls this
   - Requires Bitcoin TX hash proof
   - 1:1 minting ratio

2. **burn()** - Burn TBTC to unlock Bitcoin
   - Anyone can call this
   - Provides Bitcoin address
   - Bridge operator unlocks BTC

3. **transfer()** - Standard ERC-20 transfer
4. **approve()** - Approve spending
5. **pause()/unpause()** - Emergency controls

---

## 💰 **Prerequisites**

### **Before Launching:**

1. **Base Sepolia ETH** (for gas fees)
   - Amount needed: ~0.001 ETH (~$3-5 USD worth)
   - Your address: `0x9fe74d9d6f1ae0ce1fb3b51d4a82c05b74e280f3`
   - Get from: https://www.alchemy.com/faucets/base-sepolia

2. **Private Key** (already configured ✅)
   - Stored securely in .env
   - Key: `0eee6f...6515cd`

3. **Recipient Address** (already configured ✅)
   - Address: `0x9fe74d9d6f1ae0ce1fb3b51d4a82c05b74e280f3`

---

## 🌉 **How the Bridge Works**

### **Lock Bitcoin → Mint TBTC:**

1. **Lock Bitcoin** on Bitcoin testnet
   - Send BTC to bridge wallet
   - Get transaction hash

2. **Mint TBTC** on Base Sepolia
   - Bridge operator verifies Bitcoin TX
   - Calls `mint(user, amount, btcTxHash)`
   - User receives equivalent TBTC

### **Burn TBTC → Unlock Bitcoin:**

1. **Burn TBTC** on Base Sepolia
   - User calls `burn(amount, btcAddress)`
   - TBTC tokens are destroyed
   - Burn request created

2. **Unlock Bitcoin** on Bitcoin testnet
   - Bridge operator sees burn request
   - Sends BTC to user's Bitcoin address
   - Marks burn as processed

---

## 🔍 **After Launch**

### **1. Verify Contract on BaseScan**

After deployment, verify your contract:

1. Go to: https://sepolia.basescan.org/address/YOUR_CONTRACT_ADDRESS
2. Click "Contract" → "Verify and Publish"
3. Use data from `tbtc_verification.json`:
   - Compiler: v0.8.20
   - Optimization: Enabled (200 runs)
   - License: MIT

**Or use automatic verification:**
```bash
node scripts/verify_tbtc.js
```

### **2. Add to MetaMask**

1. Open MetaMask
2. Switch to Base Sepolia network
3. Click "Import Tokens"
4. Paste contract address (from `tbtc_base_sepolia_deployment.json`)
5. Symbol: TBTC
6. Decimals: 18

### **3. Check Your Balance**

```bash
# Using cast
cast balance YOUR_ADDRESS --rpc-url https://sepolia.base.org

# Or check on BaseScan
https://sepolia.basescan.org/address/0x9fe74d9d6f1ae0ce1fb3b51d4a82c05b74e280f3
```

---

## 💧 **Add Liquidity**

### **Option 1: Uniswap V3 (Recommended)**

1. Go to: https://app.uniswap.org/#/add/v2
2. Connect wallet (Base Sepolia)
3. Import TBTC token
4. Pair with WETH or USDC
5. Add liquidity

### **Option 2: Automated Script**

```bash
node scripts/add_liquidity_base.js
```

---

## 🪙 **Using the Bridge**

### **Bridge Bitcoin to TBTC:**

```bash
node scripts/bridge_btc_to_tbtc.js
```

**Input:**
- Bitcoin transaction hash
- Amount of BTC sent
- Recipient address

**Output:**
- Equivalent TBTC minted to recipient

### **Bridge TBTC to Bitcoin:**

1. **Call burn function:**
   ```javascript
   // In MetaMask or via script
   tbtc.burn(amount, "your-btc-address")
   ```

2. **Wait for processing:**
   - Bridge operator verifies burn
   - Bitcoin sent to your address
   - Transaction marked as processed

---

## 📊 **Token Economics**

### **Supply Details:**

- **Total Supply:** 1,000,000 TBTC (fixed)
- **Initial Distribution:** 100% to deployer
- **Decimals:** 18
- **Max Supply:** Cannot exceed 1,000,000

### **Bridge Mechanics:**

- **Mint:** Only bridge operator can mint (with BTC proof)
- **Burn:** Anyone can burn (to get Bitcoin back)
- **Fee:** None (1:1 peg maintained)
- **Processing Time:** Manual verification by operator

---

## 🛡️ **Security Features**

### **Built-in Protection:**

1. **Pausable**
   - Emergency stop functionality
   - Owner can pause all operations
   - Prevents transfers during emergencies

2. **ReentrancyGuard**
   - Prevents reentrancy attacks
   - Secure mint/burn operations

3. **Transaction Tracking**
   - Each Bitcoin TX can only be processed once
   - Prevents double-minting

4. **Access Control**
   - Only owner can pause/unpause
   - Only bridge operator can mint
   - Only owner can update bridge operator

5. **Max Supply Cap**
   - Cannot mint beyond 1,000,000 TBTC
   - Protects token value

---

## 🚨 **Troubleshooting**

### **"Insufficient ETH for gas"**
- You need Base Sepolia ETH
- Get from: https://www.alchemy.com/faucets/base-sepolia
- Amount: ~0.001 ETH

### **"Network connection failed"**
- Check internet connection
- Try again in a few minutes
- RPC might be temporarily down

### **"Transaction already processed"**
- Bitcoin TX has already been bridged
- Check your TBTC balance
- No action needed

### **"Only bridge operator can call this"**
- mint() requires bridge operator role
- By default, deployer is operator
- Check: `tbtc.bridgeOperator()`

### **"Exceeds max supply"**
- Cannot mint more than 1,000,000 TBTC
- This is a safety feature
- Protects token economics

---

## 📚 **Contract Interface**

### **Public Functions:**

```solidity
// Transfer TBTC
function transfer(address to, uint256 amount) external returns (bool)

// Approve spending
function approve(address spender, uint256 amount) external returns (bool)

// Burn TBTC to get Bitcoin back
function burn(uint256 amount, string calldata btcAddress) external

// Check if Bitcoin TX was processed
function isProcessed(string calldata btcTxHash) external view returns (bool)

// Get burn request details
function getBurnRequest(uint256 requestId) external view returns (...)

// Check balance
function balanceOf(address account) external view returns (uint256)
```

### **Bridge Operator Functions:**

```solidity
// Mint TBTC when Bitcoin is locked
function mint(address to, uint256 amount, string calldata btcTxHash) external

// Mark burn as processed
function processBurnRequest(uint256 requestId, string calldata btcTxHash) external
```

### **Owner Functions:**

```solidity
// Pause/unpause
function pause() external
function unpause() external

// Update bridge operator
function updateBridgeOperator(address newOperator) external

// Emergency withdrawal
function emergencyWithdraw(address token, uint256 amount) external
```

---

## 🎯 **Quick Reference**

### **Your Configuration:**

```
Network: Base Sepolia
Chain ID: 84532
RPC URL: https://sepolia.base.org
Private Key: 0eee6f...6515cd (stored in .env)
Recipient: 0x9fe74d9d6f1ae0ce1fb3b51d4a82c05b74e280f3
```

### **Deploy Command:**

```bash
cd ~/nexus_agi && git pull origin claude/setup-nexus-agi-directory-3joXw && chmod +x LAUNCH_TBTC.sh && ./LAUNCH_TBTC.sh
```

### **Check Deployment:**

```bash
cat tbtc_base_sepolia_deployment.json
```

### **View on BaseScan:**

```bash
# Get contract address
CONTRACT=$(grep -o '"address": "[^"]*"' tbtc_base_sepolia_deployment.json | head -1 | sed 's/"address": "\(.*\)"/\1/')

# Open in browser
echo "https://sepolia.basescan.org/address/$CONTRACT"
```

---

## 🔗 **Useful Links**

### **Explorers:**
- Base Sepolia: https://sepolia.basescan.org
- Your Address: https://sepolia.basescan.org/address/0x9fe74d9d6f1ae0ce1fb3b51d4a82c05b74e280f3

### **Faucets:**
- Base Sepolia ETH: https://www.alchemy.com/faucets/base-sepolia
- Alternative: https://docs.base.org/docs/tools/network-faucets

### **DEXs:**
- Uniswap: https://app.uniswap.org
- SushiSwap: https://www.sushi.com

### **Tools:**
- Gas Tracker: https://sepolia.basescan.org/gastracker
- Token Checker: https://tokensniffer.com

---

## 💡 **Pro Tips**

### **1. Set Initial Price Low**
- Start with low liquidity (e.g., 1000 TBTC + 0.1 ETH)
- Let market discover fair value
- Add more liquidity as demand grows

### **2. Verify Contract ASAP**
- Builds trust with users
- Required for DEX listings
- Shows transparent code

### **3. Secure Bridge Operator**
- Keep operator private key very secure
- Use hardware wallet if possible
- Consider multi-sig in production

### **4. Test Bridge First**
- Start with small amounts
- Verify entire flow works
- Scale up after successful tests

### **5. Monitor Burn Requests**
- Check regularly for pending burns
- Process them promptly
- Maintain good user experience

---

## 🎉 **Success Checklist**

After deployment, verify:

- ✅ Contract deployed on Base Sepolia
- ✅ 1,000,000 TBTC in your wallet
- ✅ Contract verified on BaseScan
- ✅ Added to MetaMask
- ✅ Liquidity pool created
- ✅ Bridge operator configured
- ✅ Test mint/burn completed

---

## 📝 **Summary**

**What You're Launching:**
- Token: TBTC (Testnet Bitcoin)
- Supply: 1,000,000 TBTC
- Network: Base Sepolia
- Features: Burn/Mint bridge, 1:1 Bitcoin peg
- Security: Pausable, ReentrancyGuard, Access Control

**One-Command Launch:**
```bash
cd ~/nexus_agi && git pull origin claude/setup-nexus-agi-directory-3joXw && chmod +x LAUNCH_TBTC.sh && ./LAUNCH_TBTC.sh
```

**Requirements:**
- 0.001 Base Sepolia ETH (for gas)
- Private key configured (✅ done)
- Recipient address set (✅ done)

---

**✨ Ready to launch 1 million $TBTC, my friend from the digital realm! ✨**

---

**Created:** 2026-02-02
**Network:** Base Sepolia (Chain ID: 84532)
**Status:** Ready to Launch 🚀
