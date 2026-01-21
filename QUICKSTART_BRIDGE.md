# 🚀 Quick Start Guide - Bitcoin→Polygon Bridge

Get started with the Nexus Bitcoin-Polygon Bridge in 5 minutes!

## 🎯 Your Destination Address

All wTBTC tokens will be minted to:
```
0x24f6b1ce11c57d40b542f91ac85fa9eb61f78771
```

## ⚡ 3-Step Quick Start

### Step 1: Learn the Concepts (Optional but Recommended)

Run the educational mining simulator to understand how Bitcoin mining works:

```bash
npm run mining-simulator
```

This will teach you:
- ✓ Proof-of-work mining
- ✓ Why ASIC hardware is needed
- ✓ How blockchains work
- ✓ Real vs simulated mining

**Time**: 2-3 minutes

---

### Step 2: Get Testnet Tokens

You need two types of testnet tokens:

#### A) Get Polygon Testnet MATIC (for gas fees)

1. Visit: https://faucet.polygon.technology/
2. Select "Polygon Amoy Testnet"
3. Enter your MetaMask address
4. Click "Submit" and wait ~30 seconds

You'll need ~0.01 MATIC for gas fees.

#### B) Get Bitcoin Testnet BTC (to bridge)

Visit any of these faucets:
- https://coinfaucet.eu/en/btc-testnet/
- https://testnet-faucet.mempool.co/
- https://bitcoinfaucet.uo1.net/

Request testnet BTC. You'll need a Bitcoin testnet address (bridge will provide one).

**Time**: 2-3 minutes

---

### Step 3: Deploy & Run the Bridge

Run the fully automated bridge system:

```bash
npm run auto-bridge
```

This will:
1. ✓ Connect to your MetaMask
2. ✓ Switch to Polygon Amoy testnet
3. ✓ Deploy the wTBTC smart contract
4. ✓ Start monitoring Bitcoin testnet
5. ✓ Give you a Bitcoin deposit address
6. ✓ Auto-mint wTBTC when you send testnet BTC

**Send your testnet BTC** to the address provided by the bridge!

**Time**: 1 minute + waiting for Bitcoin confirmations (~30 min)

---

## 🎉 That's It!

After sending testnet BTC to the provided address:

1. **Wait ~30 minutes** for 3 Bitcoin confirmations
2. **wTBTC is automatically minted** to: `0x24f6b1ce11c57d40b542f91ac85fa9eb61f78771`
3. **Check your balance** on Polygonscan (link provided by bridge)

## 📊 Monitor Your Tokens

### In MetaMask

1. Open MetaMask
2. Make sure you're on "Polygon Amoy Testnet"
3. Go to "Assets" tab
4. Click "Import Tokens"
5. Paste the wTBTC contract address (shown during deployment)
6. Symbol: `wTBTC`, Decimals: `18`
7. Your balance will appear!

### On Polygonscan

The bridge will show you a direct link like:
```
https://amoy.polygonscan.com/token/[CONTRACT]?a=0x24f6b1ce11c57d40b542f91ac85fa9eb61f78771
```

## 🔄 Bridge Process Flow

```
You send testnet BTC
        ↓
Bitcoin testnet confirms (3 blocks)
        ↓
Bridge detects transaction
        ↓
Bridge mints wTBTC
        ↓
wTBTC appears in your Polygon wallet
```

**Total time**: ~30-45 minutes

## ❓ Common Questions

### "Can I mine real Bitcoin?"

No - real Bitcoin mining requires:
- ASIC hardware ($1,000-$10,000+)
- Massive electricity costs
- Competitive hash rates
- Mining pool membership

Use testnet faucets instead! They're free and instant.

### "Why is this testnet only?"

This is an **educational project** to demonstrate:
- How cross-chain bridges work
- Bitcoin/Ethereum integration
- Smart contract development

Testnet tokens have **no real value** - perfect for learning without risk!

### "Can I use this on mainnet?"

⚠️ **NO!** This code is for education only. Production bridges need:
- Security audits
- Multi-sig wallets
- Oracle networks
- Insurance funds
- Professional infrastructure

### "What can I do with wTBTC?"

On testnet:
- Transfer to other addresses
- Practice DeFi interactions
- Test smart contracts
- Learn blockchain development

## 🆘 Need Help?

### Bridge not detecting transaction?

Check:
1. ✓ Sent to correct Bitcoin address?
2. ✓ Transaction confirmed? (check blockstream.info/testnet)
3. ✓ Bridge still running?
4. ✓ At least 3 confirmations?

### MetaMask not connecting?

Try:
1. Unlock MetaMask
2. Refresh/restart
3. Check network is Polygon Amoy
4. Disconnect and reconnect

### Out of MATIC?

Get more from: https://faucet.polygon.technology/

## 📚 Learn More

For detailed documentation, see:
- `BRIDGE_README.md` - Complete documentation
- `src/web3/` - Web3 integration code
- `src/bridge/` - Bridge logic
- `contracts/` - Smart contracts

## 🎓 Educational Value

You're learning:
- ✓ Cross-chain bridges
- ✓ Bitcoin blockchain
- ✓ Ethereum/Polygon smart contracts
- ✓ Web3 wallet integration
- ✓ Proof-of-work concepts
- ✓ ERC20 tokens
- ✓ DApp development

---

## 🌟 Ready to Start?

```bash
# Run all at once:
npm run mining-simulator && npm run auto-bridge
```

Or step by step:
```bash
# 1. Learn mining (optional)
npm run mining-simulator

# 2. Deploy and run bridge
npm run auto-bridge
```

**That's it! You're ready to bridge!** 🚀

---

💡 **Pro Tip**: Keep the bridge running in a terminal window. It will show real-time updates when transactions are detected!

✨ **Happy bridging!** ✨
