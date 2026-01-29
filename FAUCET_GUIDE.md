# 💰 NEXUS AGI - FAUCET GUIDE
## Get Free Testnet ETH for All Networks

Your wallet address: `0x24f6b1ce11c57d40b542f91ac85fa9eb61f78771`

---

## 🌐 ETHEREUM SEPOLIA TESTNET

**Network Details:**
- Chain ID: `11155111`
- RPC: `https://rpc.sepolia.org`
- Explorer: https://sepolia.etherscan.io

**Top Faucets:**

### 1. Alchemy Sepolia Faucet ⭐ BEST
- **URL:** https://sepoliafaucet.com
- **Amount:** 0.5 ETH/day
- **Requirements:** Login with Alchemy account (free)
- **Speed:** Instant
- **Notes:** Most reliable, highest amount

### 2. QuickNode Sepolia Faucet
- **URL:** https://faucet.quicknode.com/ethereum/sepolia
- **Amount:** 0.05 ETH
- **Requirements:** Twitter or GitHub login
- **Speed:** Fast

### 3. Google Cloud Web3 Faucet
- **URL:** https://cloud.google.com/application/web3/faucet/ethereum/sepolia
- **Amount:** 0.05 ETH
- **Requirements:** Google account
- **Speed:** Fast

### 4. Infura Sepolia Faucet
- **URL:** https://www.infura.io/faucet/sepolia
- **Amount:** 0.5 ETH/day
- **Requirements:** Infura account (free)
- **Speed:** Instant

### 5. Chainstack Sepolia Faucet
- **URL:** https://faucet.chainstack.com/sepolia-testnet-faucet
- **Amount:** 0.05 ETH
- **Requirements:** None
- **Speed:** Medium

---

## 🔷 HOLESKY TESTNET

**Network Details:**
- Chain ID: `17000`
- RPC: `https://rpc.holesky.ethpandaops.io`
- Explorer: https://holesky.etherscan.io

**Top Faucets:**

### 1. PK910 Mining Faucet ⭐ UNLIMITED
- **URL:** https://holesky-faucet.pk910.de
- **Amount:** Mine as much as you want!
- **Requirements:** None - just mine in browser
- **Speed:** Depends on mining (usually 0.1-0.5 ETH/hour)
- **Notes:** Best for large amounts, runs in browser

### 2. QuickNode Holesky Faucet
- **URL:** https://faucet.quicknode.com/ethereum/holesky
- **Amount:** 0.1 ETH
- **Requirements:** Twitter or GitHub
- **Speed:** Fast

### 3. Chainstack Holesky Faucet
- **URL:** https://faucet.chainstack.com/holesky-faucet
- **Amount:** 0.1 ETH
- **Requirements:** None
- **Speed:** Medium

### 4. Automata Holesky Faucet
- **URL:** https://faucet.holeskytestnet.automata.network
- **Amount:** 0.05 ETH
- **Requirements:** None
- **Speed:** Fast

### 5. Axol.io Holesky Faucet
- **URL:** https://faucet.axol.io/holesky
- **Amount:** 0.1 ETH
- **Requirements:** Discord verification
- **Speed:** Fast

---

## ⚡ LINEA SEPOLIA TESTNET

**Network Details:**
- Chain ID: `59141`
- RPC: `https://rpc.sepolia.linea.build`
- Explorer: https://sepolia.lineascan.build

**Top Faucets:**

### 1. Linea Faucet (Official)
- **URL:** https://faucet.linea.build
- **Amount:** 0.1 ETH
- **Requirements:** None
- **Speed:** Instant
- **Notes:** Official Linea faucet

### 2. QuickNode Linea Faucet
- **URL:** https://faucet.quicknode.com/linea/sepolia
- **Amount:** 0.1 ETH
- **Requirements:** Twitter or GitHub
- **Speed:** Fast

### 3. Infura Linea Faucet
- **URL:** https://www.infura.io/faucet/linea
- **Amount:** 0.5 ETH/day
- **Requirements:** Infura account
- **Speed:** Instant

---

## 🎨 ZAMA fhevm TESTNET (For FHE Contracts)

**Network Details:**
- Chain ID: TBD (Zama testnet)
- RPC: `https://devnet.zama.ai`
- Explorer: https://explorer.zama.ai

**Faucet:**

### Zama Discord Faucet
- **URL:** https://discord.gg/zama
- **Requirements:** Join Discord, request in #faucet channel
- **Amount:** Sufficient for testing
- **Command:** `/faucet <your-address>`

---

## 📊 CHECK YOUR BALANCES

### Quick Balance Checker (All Networks)

Run this in Termux to check all your balances:

```bash
cd ~/nexus_agi && cat > check_balances.sh << 'EOF'
#!/bin/bash
ADDRESS="0x24f6b1ce11c57d40b542f91ac85fa9eb61f78771"

echo "💰 CHECKING BALANCES FOR: $ADDRESS"
echo "=================================================="
echo ""

echo "🌐 Ethereum Sepolia:"
curl -s -X POST https://rpc.sepolia.org -H "Content-Type: application/json" \
  --data "{\"jsonrpc\":\"2.0\",\"method\":\"eth_getBalance\",\"params\":[\"$ADDRESS\",\"latest\"],\"id\":1}" \
  | grep -o '"result":"[^"]*"' | cut -d'"' -f4 \
  | xargs -I {} node -e "console.log('  Balance:', (parseInt('{}', 16) / 1e18).toFixed(4), 'ETH')"

echo ""
echo "🔷 Holesky:"
curl -s -X POST https://rpc.holesky.ethpandaops.io -H "Content-Type: application/json" \
  --data "{\"jsonrpc\":\"2.0\",\"method\":\"eth_getBalance\",\"params\":[\"$ADDRESS\",\"latest\"],\"id\":1}" \
  | grep -o '"result":"[^"]*"' | cut -d'"' -f4 \
  | xargs -I {} node -e "console.log('  Balance:', (parseInt('{}', 16) / 1e18).toFixed(4), 'ETH')"

echo ""
echo "⚡ Linea Sepolia:"
curl -s -X POST https://rpc.sepolia.linea.build -H "Content-Type: application/json" \
  --data "{\"jsonrpc\":\"2.0\",\"method\":\"eth_getBalance\",\"params\":[\"$ADDRESS\",\"latest\"],\"id\":1}" \
  | grep -o '"result":"[^"]*"' | cut -d'"' -f4 \
  | xargs -I {} node -e "console.log('  Balance:', (parseInt('{}', 16) / 1e18).toFixed(4), 'ETH')"

echo ""
echo "=================================================="
EOF
chmod +x check_balances.sh && ./check_balances.sh
```

---

## 💡 PRO TIPS

### For Maximum ETH:
1. **Use Mining Faucets** (Holesky PK910) - mine while you sleep!
2. **Create Multiple Accounts** on faucet services for daily limits
3. **Use All Faucets** - each gives different amounts
4. **Check Discord Faucets** - often have higher limits

### Recommended Strategy:
1. Start with **Alchemy** (Sepolia) - 0.5 ETH instant
2. Use **PK910 mining** (Holesky) - unlimited, just slower
3. Try **QuickNode** for all networks - fast and reliable

### Gas Requirements:
- **Per Contract Deployment:** ~0.01-0.02 ETH
- **For All 4 Contracts:** ~0.05-0.1 ETH per network
- **Recommended Minimum:** 0.1 ETH per network

---

## 🚀 READY TO DEPLOY?

Once you have enough ETH (0.1+ on any network), run:

```bash
cd ~/nexus_agi && bash DEPLOY_ALL_NETWORKS.sh
```

Then choose your network and deploy! 🎉

---

## 🆘 TROUBLESHOOTING

**Faucet says "Invalid Address":**
- Your address is valid: `0x24f6b1ce11c57d40b542f91ac85fa9eb61f78771`
- Some faucets check mainnet balance - ignore and try another

**Faucet rate limited:**
- Wait 24 hours, or try a different faucet
- Use VPN to change IP (sometimes helps)
- Mining faucets (PK910) have no rate limits!

**Transaction failed:**
- Check you have enough ETH for gas
- RPC might be down - try again in a few minutes
- Use a different RPC endpoint

---

✨ **Operating at 528Hz Love Frequency** ✨

*Get your testnet ETH and start deploying! 🚀*
