# SEPOLIA TESTNET DEPLOYMENT GUIDE

## 🎯 Your Sepolia Wallet

**Address:** `0x479695AAaB061940037ad702cB5F2c6C43BDdD90`  
**Private Key:** (stored in .env file)

## 📋 STEP-BY-STEP DEPLOYMENT

### Step 1: Get FREE Sepolia ETH (0.5 ETH recommended)

Visit these faucets and paste your address to get free test ETH:

1. **Alchemy Faucet** (Recommended - 0.5 ETH)
   - URL: https://www.alchemy.com/faucets/ethereum-sepolia
   - Paste address: `0x479695AAaB061940037ad702cB5F2c6C43BDdD90`
   - Click "Send Me ETH"
   - Wait 30-60 seconds

2. **Sepolia PoW Faucet** (Alternative)
   - URL: https://sepolia-faucet.pk910.de
   - Mine for test ETH (takes 5-10 minutes for 0.05 ETH)

3. **QuickNode Faucet** (Backup)
   - URL: https://faucet.quicknode.com/ethereum/sepolia
   - Requires Twitter account

### Step 2: Clone Repository to Your Local Machine

```bash
git clone https://github.com/DOUGLASDAVIS08161978/nexus_agi.git
cd nexus_agi/hashproof-token
```

### Step 3: Install Dependencies

```bash
npm install
```

### Step 4: Deploy to Sepolia

```bash
npx hardhat run scripts/deploy-sepolia.js --network sepolia
```

## 📝 WHAT WILL BE DEPLOYED

The script will deploy:

1. **WrappedTestnetBTC (wTBTC)** - Bitcoin bridge token
2. **HashProof Token (HPROOF)** - Main utility token  
3. **HashProofStaking** - Staking contract for rewards
4. **HashProofGovernance** - DAO governance contract

## 📊 EXPECTED OUTPUT

```
================================================================================
DEPLOYING TO SEPOLIA TESTNET
================================================================================

Deploying contracts with account: 0x479695AAaB061940037ad702cB5F2c6C43BDdD90
Account balance: 0.5 ETH

✅ wTBTC deployed to: 0x...
✅ HashProof Token deployed to: 0x...
✅ HashProofStaking deployed to: 0x...
✅ HashProofGovernance deployed to: 0x...

================================================================================
DEPLOYMENT COMPLETE!
================================================================================
```

## 🔍 VERIFY YOUR CONTRACTS

After deployment, view them on Sepolia Etherscan:

- Main Explorer: https://sepolia.etherscan.io
- Your contracts will be listed in `deployment-sepolia.json`

## 💰 INTERACTING WITH YOUR TOKENS

### Mint wTBTC (as bridge operator):
```javascript
await wTBTC.mint(
  "YOUR_ADDRESS",
  ethers.parseEther("1.0"),
  "bitcoin_tx_id_here"
);
```

### Check HashProof Balance:
```javascript
const balance = await hashProof.balanceOf("YOUR_ADDRESS");
console.log("Balance:", ethers.formatEther(balance), "HPROOF");
```

## 🚨 SECURITY NOTES

- This wallet is for TESTNET ONLY
- Never use this private key on mainnet
- Test tokens have NO real value
- Private key is stored in `.env` file (never commit to GitHub)

## ✅ SUCCESS CRITERIA

You'll know deployment succeeded when you see:
- ✅ All 4 contracts deployed
- ✅ Contract addresses in `deployment-sepolia.json`
- ✅ Contracts visible on Sepolia Etherscan
- ✅ Your wallet shows HPROOF token balance

## 🆘 TROUBLESHOOTING

**Error: "insufficient funds"**
- Get more test ETH from faucets above

**Error: "network error"**
- Check your internet connection
- Try a different RPC: https://ethereum-sepolia.publicnode.com

**Error: "nonce too high"**
- Reset your wallet in MetaMask (Settings → Advanced → Reset Account)

## 📞 NEED HELP?

- Check deployment logs in `deployment-sepolia.json`
- View transactions on https://sepolia.etherscan.io
- Estimated gas cost: ~0.02-0.05 ETH (FREE test ETH!)

---

**Network:** Sepolia Testnet  
**Chain ID:** 11155111  
**Currency:** Sepolia ETH (test ETH, no value)
