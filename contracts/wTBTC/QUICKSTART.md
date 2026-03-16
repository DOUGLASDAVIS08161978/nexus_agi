# 🚀 QUICKSTART GUIDE - wTBTC Deployment

Get your Wrapped Testnet Bitcoin contract deployed in 5 minutes!

## ⚡ Super Fast Setup

### Step 1: Install Dependencies (1 minute)

```bash
cd ~/nexus_agi/contracts/wTBTC/
npm install
```

### Step 2: Configure Environment (2 minutes)

```bash
# Copy example env file
cp .env.example .env

# Edit .env with your details
nano .env
```

**Minimum required settings:**
```env
PRIVATE_KEY=your_wallet_private_key_without_0x
SEPOLIA_RPC_URL=https://sepolia.infura.io/v3/YOUR_INFURA_KEY
BRIDGE_OPERATOR_ADDRESS=0xYourBridgeOperatorAddress
```

**Get free Sepolia RPC URL:**
1. Go to https://infura.io
2. Sign up (free)
3. Create new project
4. Copy Sepolia endpoint

**Get Sepolia testnet ETH:**
1. Visit https://sepoliafaucet.com/
2. Enter your wallet address
3. Get free test ETH

### Step 3: Deploy! (2 minutes)

```bash
# Compile contract
npm run compile

# Run tests (optional but recommended)
npm test

# Deploy to Sepolia testnet
npm run deploy:sepolia
```

**That's it! 🎉**

## 📋 Quick Commands Reference

```bash
# Development
npm run compile          # Compile contracts
npm test                 # Run tests
npm run test:coverage    # Test coverage
npm run node             # Start local node

# Deployment
npm run deploy:localhost       # Deploy to local network
npm run deploy:sepolia         # Deploy to Sepolia
npm run deploy:holesky         # Deploy to Holesky

# Utilities
npm run clean            # Clean artifacts
npm run size             # Check contract sizes
```

## 🧪 Test Your Contract

After deployment, you'll see output like:

```
✅ DEPLOYMENT SUCCESSFUL!
📍 Contract Address: 0x1234...5678
👤 Bridge Operator: 0xabcd...efgh
```

### Test Minting (Bridge Operator Only)

```javascript
// In Hardhat console:
npx hardhat console --network sepolia

const wTBTC = await ethers.getContractAt("WrappedTestnetBTC", "0x1234...5678");

// Mint 1.5 wTBTC to an address
await wTBTC.mint(
  "0xUserAddress",
  ethers.parseEther("1.5"),
  "bitcoin_tx_hash_here"
);

// Check balance
await wTBTC.balanceOf("0xUserAddress");
```

### Test Burning (Any User)

```javascript
// User burns wTBTC to get BTC back
await wTBTC.burn(
  ethers.parseEther("0.5"),
  "tb1qYourBitcoinTestnetAddress"
);
```

### Test Transfer (Any User)

```javascript
// Transfer wTBTC to another address
await wTBTC.transfer(
  "0xRecipientAddress",
  ethers.parseEther("0.25")
);
```

## 🔍 Verify on Etherscan

```bash
# After deployment, verify your contract:
npx hardhat verify --network sepolia \
  YOUR_CONTRACT_ADDRESS \
  YOUR_BRIDGE_OPERATOR_ADDRESS
```

## 🎯 Next Steps

1. **Set up Bitcoin monitoring** - Watch for BTC deposits
2. **Test full bridge cycle** - Lock BTC → Mint wTBTC → Burn wTBTC → Unlock BTC
3. **Add multisig** - Use Gnosis Safe for bridge operator
4. **Monitor events** - Listen for Mint/Burn events
5. **Build UI** - Create user interface for bridge

## 💡 Common Issues

### "Insufficient funds for gas"
**Solution:** Get more testnet ETH from faucet

### "Invalid private key"
**Solution:** Ensure PRIVATE_KEY in .env has no "0x" prefix

### "Cannot find module"
**Solution:** Run `npm install` first

### "Network not configured"
**Solution:** Check RPC URL in .env file

## 📞 Bridge Operations

### For Automated Bridge:

```javascript
// Monitor Bitcoin blockchain
// When BTC is locked:
const btcTxId = "detected_bitcoin_tx_id";
const amount = "1.5"; // BTC amount
const userAddress = "0x..."; // From OP_RETURN or memo

// Mint equivalent wTBTC
await wTBTC.mint(
  userAddress,
  ethers.parseEther(amount),
  btcTxId
);

// Listen for burn events
wTBTC.on("Burn", async (from, amount, bitcoinAddress, event) => {
  console.log(`User ${from} wants to burn ${amount} wTBTC`);
  console.log(`Send BTC to: ${bitcoinAddress}`);

  // Send BTC on Bitcoin network
  // ...
});
```

## 🔐 Security Checklist

Before going to production:

- [ ] Use multisig for bridge operator
- [ ] Add replay protection for Bitcoin txIds
- [ ] Implement proper Bitcoin SPV proofs
- [ ] Add emergency withdrawal mechanism
- [ ] Get professional security audit
- [ ] Set up monitoring and alerts
- [ ] Test extensively on testnet
- [ ] Document all procedures
- [ ] Prepare incident response plan
- [ ] Get insurance if handling significant value

## 📚 Resources

- **Hardhat Docs:** https://hardhat.org/docs
- **Ethers.js Docs:** https://docs.ethers.org/
- **Sepolia Faucet:** https://sepoliafaucet.com/
- **Infura:** https://infura.io
- **Etherscan Sepolia:** https://sepolia.etherscan.io

## 🆘 Need Help?

Check the full README.md for detailed documentation and security considerations.

---

**Ready to deploy?** Just run:

```bash
cd ~/nexus_agi/contracts/wTBTC/
npm install
cp .env.example .env
# Edit .env with your settings
npm run deploy:sepolia
```

**Let's build the future of Bitcoin bridges! 🌉💚**
