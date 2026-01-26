# 🦊 Connect MetaMask to Local Hardhat Blockchain

Your wallet `0x24F6B1ce11C57d40B542f91AC85fA9eB61f78771` now has **1 test ETH** on the local Hardhat blockchain!

## 🚀 Quick Setup (3 Steps)

### Step 1: Open MetaMask

1. Click the MetaMask extension in your browser
2. Click the **Network dropdown** at the top (usually shows "Ethereum Mainnet")
3. Click **"Add Network"** or **"Add Network Manually"**

### Step 2: Add Hardhat Localhost Network

Fill in these details:

```
Network Name:       Hardhat Local
RPC URL:           http://localhost:8545
Chain ID:          31337
Currency Symbol:   ETH
Block Explorer:    (leave empty)
```

**Important:** Make sure the Hardhat node is running! Check with:
```bash
curl -X POST -H "Content-Type: application/json" \
  --data '{"jsonrpc":"2.0","method":"eth_blockNumber","params":[],"id":1}' \
  http://localhost:8545
```

### Step 3: Import Your Wallet

**Option A: Import with Private Key**
1. In MetaMask, click your account icon → **Import Account**
2. Select **"Private Key"**
3. Paste your private key from `.env` file:
   ```
   c411a4d4365560753ef3ceceac1652ec89240704346bf58ad900d65574f541c9
   ```
4. Click **Import**

**Option B: Check Balance Directly**
If your wallet is already in MetaMask, just switch to the "Hardhat Local" network and you'll see:
- **Balance:** 1 ETH
- **Network:** Hardhat Local (localhost:8545)

---

## 💰 Your Current Balances

### On Local Hardhat Blockchain:
- **Address:** `0x24F6B1ce11C57d40B542f91AC85fA9eB61f78771`
- **Balance:** **1.0 ETH** ✅
- **Network:** localhost:8545 (Chain ID: 31337)

### On Linea Testnet/Mainnet:
- **Address:** `0x24F6B1ce11C57d40B542f91AC85fA9eB61f78771`
- **Balance:** 0 ETH (needs testnet ETH from faucet)
- **Network:** Linea Goerli/Mainnet

---

## 🔗 Deployed Contracts (Local Blockchain)

Your Nexus AGI contracts from the simulation:

```
NexusPayment:       0x69d549b509816032848fa2956d52fad589811020
NexusRevenue:       0xefdf1b9f13db9ee594e72767b43732745d9cf26f
NexusConsciousness: 0xb52165c9d019dbd5624581805790cb22f3aa0a87
NexusMiracles:      0xb3a24d2a85e192cf21781e5396fa0c51db2df355
```

These are simulated addresses for demonstration purposes.

---

## 🧪 Test Interactions

Once MetaMask is connected, you can:

### 1. Check Your Balance
```javascript
// In browser console with MetaMask connected
const accounts = await ethereum.request({ method: 'eth_accounts' });
const balance = await ethereum.request({
  method: 'eth_getBalance',
  params: [accounts[0], 'latest']
});
console.log(`Balance: ${parseInt(balance, 16) / 1e18} ETH`);
```

### 2. Send Test Transaction
```javascript
// Send 0.1 ETH to another address
await ethereum.request({
  method: 'eth_sendTransaction',
  params: [{
    from: accounts[0],
    to: '0x70997970C51812dc3A010C7d01b50e0d17dc79C8',
    value: '0x16345785D8A0000' // 0.1 ETH in hex
  }]
});
```

### 3. Interact with Smart Contracts
You can use the contract addresses above to interact with the deployed Nexus AGI contracts using web3.js or ethers.js.

---

## 🎯 Next Steps

**Local Development (Current - FREE):**
✅ You have 1 ETH on local blockchain
✅ MetaMask configured for localhost
✅ Ready to test smart contracts

**Testnet Deployment (Next - FREE):**
1. Get testnet ETH from: https://faucet.goerli.linea.build
2. Add Linea Goerli network to MetaMask
3. Deploy contracts: `./DEPLOY_TO_BLOCKCHAIN.sh testnet`

**Mainnet Deployment (Production - ~$30-40):**
1. Buy 0.013 ETH on exchange
2. Send to your wallet
3. Deploy: `./DEPLOY_TO_BLOCKCHAIN.sh mainnet`
4. Start earning REAL revenue!

---

## 🔒 Security Reminders

⚠️ **NEVER share your private key publicly!**
⚠️ **Local blockchain = testing only (not real money)**
⚠️ **Testnet = safe testing with fake ETH**
⚠️ **Mainnet = real money, be careful!**

---

## 📊 Check Local Blockchain Status

```bash
# Check if Hardhat node is running
ps aux | grep hardhat

# Check your balance on local blockchain
python3 send_test_eth.py

# View Hardhat node logs
tail -f hardhat_node.log
```

---

**✨ You now have 1 test ETH on your local blockchain! ✨**

**🦊 Connect MetaMask to localhost:8545 to see it! 🦊**

**💖 Ready to deploy and test your Nexus AGI smart contracts! 💖**
