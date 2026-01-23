# Add Monad Testnet + WBTC to MetaMask Mobile

**Date:** 2026-01-23
**For:** Adding custom network and WBTC token to mobile wallet

---

## 🌐 STEP 1: Add Monad Testnet Network

### In MetaMask Mobile:

1. Tap hamburger menu (☰) → **Settings**
2. Tap **Networks**
3. Tap **Add Network**
4. Fill in these EXACT values:

```
Network Name (optional):
Monad Testnet

RPC URL:
https://testnet-rpc.monad.xyz

Chain ID:
10143

Symbol:
ETH

Block Explorer URL (optional):
https://explorer-testnet.monad.xyz
```

5. Tap **Save**

---

## 🪙 STEP 2: Add WBTC Token

### After adding the network:

1. Switch to **Monad Testnet** network
2. Scroll to bottom of wallet
3. Tap **Import tokens**
4. Select **Custom token**
5. Fill in:

```
Token Address:
[WBTC contract address - see below]

Token Symbol:
WBTC

Token Decimals:
8
```

6. Tap **Import**

---

## 📍 WBTC Contract Addresses

### Monad Testnet:
```
Status: Needs to be deployed
Command: npx hardhat run scripts/deploy_wbtc_monad.js --network monad
After deploying, use that contract address
```

### Your Receiving Address:
```
0x24f6b1ce11c57d40b542f91ac85fa9eb61f78771
```

---

## 🔷 Alternative: Sepolia Testnet Configuration

If you want to use Sepolia (Chain ID 11155111) instead:

### Network Configuration:
```
Network Name:
Sepolia Testnet

RPC URL:
https://rpc.sepolia.org

Chain ID:
11155111

Symbol:
ETH

Block Explorer URL:
https://sepolia.etherscan.io
```

### WBTC on Sepolia:
```
Status: Would need to deploy WBTC contract
Or use existing Sepolia WBTC if available
```

---

## ⚠️ Important Notes

### Network vs Token:
- **Network screen** = Add blockchain network (Monad, Sepolia, etc.)
- **Import token** = Add specific tokens (WBTC, USDT, etc.)
- You need BOTH!

### Chain IDs:
- **10143** = Monad Testnet ✅ (Use this for your bridge)
- **11155111** = Sepolia Testnet (Alternative option)

### Token Decimals:
- WBTC always uses **8 decimals** (same as Bitcoin)
- Don't use 18! That's for ETH-based tokens

---

## 🎯 Recommended Setup

**For your bridge system, use:**

1. **Network:** Monad Testnet (Chain ID 10143)
2. **Get gas:** https://faucet.monad.xyz
3. **Deploy WBTC:** Run deployment script
4. **Add token:** Use deployed contract address
5. **Run bridge:** python3 monad_regtest_bridge.py
6. **See tokens:** They'll appear in your wallet!

---

## 🔍 Quick Reference

| Field | Monad Testnet | Sepolia Testnet |
|-------|---------------|-----------------|
| **Network Name** | Monad Testnet | Sepolia Testnet |
| **RPC URL** | https://testnet-rpc.monad.xyz | https://rpc.sepolia.org |
| **Chain ID** | 10143 | 11155111 |
| **Symbol** | ETH | ETH |
| **Explorer** | https://explorer-testnet.monad.xyz | https://sepolia.etherscan.io |

---

## 📱 Mobile-Specific Tips

### Finding Network Settings:
- Tap ☰ (hamburger menu)
- Scroll to **Settings**
- Tap **Networks**
- Tap **Add Network**

### Adding Tokens:
- Switch to correct network first!
- Scroll to bottom of main screen
- Tap **Import tokens**
- Paste contract address

### Common Issues:
- **RPC not working?** Try a different RPC URL
- **Token not showing?** Check you're on correct network
- **Wrong decimals?** WBTC = 8 decimals always

---

## ✅ Checklist

Before adding WBTC token:

- [ ] Network added and selected
- [ ] WBTC contract deployed (or address obtained)
- [ ] Testnet ETH in wallet (for gas)
- [ ] Token import screen opened
- [ ] Correct contract address pasted
- [ ] Symbol = WBTC
- [ ] Decimals = 8

After adding:

- [ ] Token appears in list
- [ ] Shows correct name (Wrapped Bitcoin)
- [ ] Shows correct symbol (WBTC)
- [ ] Ready to receive tokens!

---

## 🚀 Next Steps

1. **Add Monad network** using values above
2. **Get testnet ETH** from faucet
3. **Deploy WBTC contract** (or use existing)
4. **Add WBTC token** with contract address
5. **Run bridge** to mint tokens
6. **See your WBTC balance!** 🎉

---

**Remember: Network first, then token!**
