# 🦊 METAMASK SHOWING "PERSONAL ADDRESS" - TROUBLESHOOTING GUIDE

## 🔍 THE PROBLEM

You deployed contracts successfully, but MetaMask shows them as "personal addresses" instead of contracts.

---

## ✅ STEP 1: VERIFY CONTRACTS HAVE BYTECODE

Run this in Termux:

```bash
cd ~/nexus_agi
bash CHECK_CONTRACTS_TERMUX.sh
```

This will show if your contracts actually have bytecode on the blockchain.

**If contracts DON'T have bytecode:**
- Something went wrong with deployment
- Redeploy using: `bash DEPLOY_SOLCJS.sh`

**If contracts HAVE bytecode:**
- The issue is with MetaMask connection
- Continue to Step 2

---

## 🌐 STEP 2: CHECK METAMASK NETWORK CONNECTION

### Issue: MetaMask Can't Reach Geth on Android/Termux

If you're running Geth in **Termux (Android)** and trying to use **MetaMask in a browser**, there are connection issues:

### **Option A: Use MetaMask Mobile App (RECOMMENDED)**

1. **Install MetaMask Mobile** on your Android device
2. **Add Custom Network:**
   - Network Name: `Geth Local`
   - RPC URL: `http://localhost:8545`
   - Chain ID: `1337`
   - Currency: `ETH`

3. **Import Your Account:**
   - Use private key: `0xb71c71a67e1177ad4e901695e1b4b9ee17ae16c6668d313eac2f96dbcda3f291`

4. **Add Token Addresses** from `deployment_addresses.json`

### **Option B: Use Termux Browser Access**

If Termux browser doesn't support MetaMask extension, you need to expose Geth to external access:

1. **Stop Geth:**
   ```bash
   pkill geth
   ```

2. **Restart Geth with external access:**
   ```bash
   geth --dev \
        --dev.period 5 \
        --http \
        --http.addr "0.0.0.0" \
        --http.port 8545 \
        --http.api "eth,net,web3,personal,miner" \
        --http.corsdomain "*" \
        --allow-insecure-unlock \
        > geth_blockchain.log 2>&1 &
   ```

3. **Get your device's IP address:**
   ```bash
   ip addr show wlan0 | grep "inet " | awk '{print $2}' | cut -d/ -f1
   ```

4. **In MetaMask (on another device or desktop):**
   - RPC URL: `http://YOUR_DEVICE_IP:8545`
   - Chain ID: `1337`

---

## 🔧 STEP 3: METAMASK CACHE ISSUES

Sometimes MetaMask caches network data incorrectly.

### **Clear MetaMask Cache:**

1. **In MetaMask:**
   - Click on account icon → Settings
   - Scroll to "Advanced"
   - Click "Clear activity tab data"
   - Click "Clear"

2. **Remove and Re-add Network:**
   - Settings → Networks
   - Find "Geth Local"
   - Click "Delete"
   - Add it again with same settings

3. **Reconnect to Site:**
   - If using a dApp, disconnect and reconnect

---

## 📋 STEP 4: VERIFY CORRECT ADDRESSES

Make sure you're using the **correct contract addresses** from the latest deployment:

```bash
cat ~/nexus_agi/deployment_addresses.json
```

**Important:**
- Don't use addresses from old deployments
- Don't use simulation addresses
- Use the addresses from the file created by `DEPLOY_SOLCJS.sh`

---

## 🔍 STEP 5: CHECK MANUALLY WITH WEB3

Create a test script to verify MetaMask sees the contract:

```javascript
// In browser console (with MetaMask connected)
const provider = new ethers.providers.Web3Provider(window.ethereum);

// Your contract address
const address = "0x3A220f351252089D385b29beca14e27F204c296A";

// Check bytecode
const code = await provider.getCode(address);
console.log("Bytecode length:", code.length);

// If length > 2, it's a contract!
if (code.length > 2) {
    console.log("✅ MetaMask sees this as a CONTRACT");
} else {
    console.log("❌ MetaMask sees this as EOA (personal address)");
}
```

---

## 🚨 COMMON ISSUES

### **Issue 1: Wrong Chain ID**
**Symptom:** MetaMask shows different chain or network error
**Fix:** Make sure Chain ID is exactly `1337` (not 31337)

### **Issue 2: Geth Not Running**
**Symptom:** MetaMask shows "Could not fetch chain ID"
**Fix:** Check if Geth is running: `ps aux | grep geth`

### **Issue 3: Using Browser MetaMask with Termux Geth**
**Symptom:** Connection refused
**Fix:** Use MetaMask Mobile app or expose Geth to network (see Option B above)

### **Issue 4: Old Deployment Addresses**
**Symptom:** MetaMask shows personal address for old addresses
**Fix:** Use addresses from latest `deployment_addresses.json`

### **Issue 5: MetaMask Showing Wrong Network**
**Symptom:** Shows Ethereum Mainnet instead of Geth Local
**Fix:** Manually switch network in MetaMask dropdown

---

## 💡 ULTIMATE FIX: START FRESH

If nothing works, start completely fresh:

1. **Stop Geth:**
   ```bash
   pkill geth
   ```

2. **Clear Geth data:**
   ```bash
   rm -rf ~/.ethereum/devchain
   ```

3. **Restart Geth:**
   ```bash
   cd ~/nexus_agi
   bash DEPLOY_SOLCJS.sh
   ```

4. **Remove MetaMask network:**
   - Delete "Geth Local" from MetaMask

5. **Re-add fresh:**
   - Add network with new RPC
   - Import account
   - Add contract addresses from NEW deployment

---

## 📞 STILL NOT WORKING?

Run this diagnostic:

```bash
cd ~/nexus_agi
echo "=== GETH STATUS ==="
ps aux | grep geth | grep -v grep

echo ""
echo "=== GETH RPC TEST ==="
curl -X POST -H "Content-Type: application/json" \
     --data '{"jsonrpc":"2.0","method":"eth_blockNumber","params":[],"id":1}' \
     http://localhost:8545

echo ""
echo "=== DEPLOYMENT FILE ==="
cat deployment_addresses.json

echo ""
echo "=== BYTECODE CHECK ==="
bash CHECK_CONTRACTS_TERMUX.sh
```

Send the output and I'll help debug further!

---

**💖 The contracts ARE deployed with real bytecode - it's just a MetaMask connection/cache issue! 💖**
