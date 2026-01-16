# 🚀 QUICK START: Deploy to Sepolia in 5 Minutes

## ✅ EVERYTHING IS READY!

All files have been committed to your repository. Here's what to do:

---

## 📋 YOUR WALLET INFO

**Address:** `0x479695AAaB061940037ad702cB5F2c6C43BDdD90`

**⚠️ IMPORTANT:** The private key is in the `.env` file (DO NOT commit to GitHub!)

---

## 🎯 3 SIMPLE STEPS

### 1️⃣ GET FREE TEST ETH (2 minutes)

Visit: **https://www.alchemy.com/faucets/ethereum-sepolia**

- Paste your address: `0x479695AAaB061940037ad702cB5F2c6C43BDdD90`
- Click "Send Me ETH"
- You'll get **0.5 Sepolia ETH** (FREE!)
- Wait 30-60 seconds

### 2️⃣ PULL THE CODE (on your local computer)

```bash
cd nexus_agi
git pull origin claude/expand-network-domains-0hJ5n
cd hashproof-token
npm install
```

### 3️⃣ DEPLOY!

```bash
npx hardhat run scripts/deploy-sepolia.js --network sepolia
```

**That's it!** 🎉

---

## 📊 WHAT YOU'LL GET

After deployment, you'll receive:

✅ **4 Smart Contract Addresses:**
- wTBTC (Wrapped Testnet Bitcoin)
- HashProof Token
- HashProofStaking
- HashProofGovernance

✅ **Tokens in your wallet:**
- Initial HPROOF tokens
- Ability to mint wTBTC

✅ **Deployment JSON file:**
- All contract addresses
- Etherscan links
- Deployment timestamp

---

## 🔍 VIEW YOUR CONTRACTS

After deployment, check them here:

**Sepolia Etherscan:**
https://sepolia.etherscan.io/address/0x479695AAaB061940037ad702cB5F2c6C43BDdD90

You'll see:
- Your ETH balance
- Your token balances
- Your deployed contracts
- All transactions

---

## 💡 COST BREAKDOWN

| Item | Cost |
|------|------|
| Test ETH from faucet | **FREE** |
| wTBTC deployment | **FREE** |
| HashProof deployment | **FREE** |
| Staking deployment | **FREE** |
| Governance deployment | **FREE** |
| **TOTAL** | **$0.00** |

*No credit card required. No mainnet ETH needed.*

---

## 🆘 IF YOU GET STUCK

**Problem:** "Insufficient funds"
**Solution:** Get more test ETH from faucet (you need ~0.1 ETH total)

**Problem:** "Network error"
**Solution:** Check your internet connection

**Problem:** "Can't find .env file"
**Solution:** The .env file is gitignored. Create it manually:
```bash
echo 'PRIVATE_KEY=0x43ffdfa3db136240610b932e0cd4f62c1bdcd0137c6af748f971317923f3a5ac' > .env
```

---

## 🎊 AFTER DEPLOYMENT

You can:

1. **Mint wTBTC tokens** (you're the bridge operator)
2. **Transfer HPROOF tokens** to other addresses
3. **Stake tokens** for rewards
4. **Create governance proposals**
5. **Test all contract functions**

---

## 🔐 SECURITY REMINDER

- ✅ This wallet is for TESTNET ONLY
- ✅ Never send real ETH to this address
- ✅ Test tokens have ZERO value
- ✅ Feel free to experiment!

---

**Ready to deploy?** Just run the 3 commands above! 🚀
