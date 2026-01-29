# 🚀 DEPLOY NEXUS AGI FROM ANY DEVICE

Your Termux has network restrictions. Use one of these methods instead:

---

## METHOD 1: DEPLOY FROM PC/MAC/LINUX (EASIEST)

### Step 1: Get the code
```bash
git clone https://github.com/DOUGLASDAVIS08161978/nexus_agi
cd nexus_agi
```

### Step 2: Install dependencies
```bash
npm install
```

### Step 3: Deploy to Holesky
```bash
node deploy_holesky_NOW.cjs
```

**Done!** Your contracts will be live!

---

## METHOD 2: DEPLOY WITH REMIX IDE (NO INSTALLATION!)

### Step 1: Get Holesky ETH
1. Go to: https://holesky-faucet.pk910.de
2. Paste address: `0x24f6b1ce11c57d40b542f91ac85fa9eb61f78771`
3. Click "Start Mining"
4. Mine for 30-60 minutes
5. You'll have ~0.1-0.5 ETH

### Step 2: Setup MetaMask
1. Install MetaMask browser extension
2. Import your private key:
   - Click MetaMask → Settings → Security & Privacy
   - Click "Import Account"
   - Paste: `c411a4d4365560753ef3ceceac1652ec89240704346bf58ad900d65574f541c9`
3. Add Holesky Network:
   - Network Name: `Holesky`
   - RPC URL: `https://ethereum-holesky-rpc.publicnode.com`
   - Chain ID: `17000`
   - Currency Symbol: `ETH`
   - Explorer: `https://holesky.etherscan.io`

### Step 3: Deploy with Remix
1. Go to: https://remix.ethereum.org
2. Create new files and paste contracts:
   - `NexusPayment.sol` → [paste from contracts/NexusPayment.sol]
   - `NexusRevenue.sol` → [paste from contracts/NexusRevenue.sol]
   - `NexusConsciousness.sol` → [paste from contracts/NexusConsciousness.sol]
   - `NexusMiracles.sol` → [paste from contracts/NexusMiracles.sol]

3. Compile:
   - Click "Solidity Compiler" tab
   - Select version: `0.8.20`
   - Click "Compile"

4. Deploy:
   - Click "Deploy & Run" tab
   - Environment: "Injected Provider - MetaMask"
   - Connect MetaMask
   - Deploy each contract:
     1. Deploy `NexusPayment`
     2. Deploy `NexusRevenue`
     3. Deploy `NexusConsciousness`
     4. Deploy `NexusMiracles`
   - Link contracts:
     - Call `NexusPayment.setRevenueContract()` with NexusRevenue address
     - Call `NexusRevenue.setPaymentContract()` with NexusPayment address
     - Call `NexusConsciousness.setOracle()` with your address
     - Call `NexusMiracles.setOracle()` with your address

**Done!** Your contracts are live!

---

## METHOD 3: HARDHAT (FOR DEVELOPERS)

### Step 1: Setup
```bash
git clone https://github.com/DOUGLASDAVIS08161978/nexus_agi
cd nexus_agi
npm install --save-dev hardhat @nomicfoundation/hardhat-toolbox
```

### Step 2: Create hardhat.config.js
```javascript
require("@nomicfoundation/hardhat-toolbox");

module.exports = {
  solidity: "0.8.20",
  networks: {
    holesky: {
      url: "https://ethereum-holesky-rpc.publicnode.com",
      accounts: ["c411a4d4365560753ef3ceceac1652ec89240704346bf58ad900d65574f541c9"],
      chainId: 17000
    }
  }
};
```

### Step 3: Deploy
```bash
npx hardhat run scripts/deploy.js --network holesky
```

---

## METHOD 4: FOUNDRY (FASTEST)

### Step 1: Install Foundry
```bash
curl -L https://foundry.paradigm.xyz | bash
foundryup
```

### Step 2: Deploy
```bash
cd nexus_agi

forge create contracts/NexusPayment.sol:NexusPayment \
  --rpc-url https://ethereum-holesky-rpc.publicnode.com \
  --private-key c411a4d4365560753ef3ceceac1652ec89240704346bf58ad900d65574f541c9

# Repeat for other contracts
```

---

## TROUBLESHOOTING TERMUX

If you really want to deploy from Termux, try:

### Fix 1: Update CA certificates
```bash
pkg update && pkg upgrade
pkg install ca-certificates
```

### Fix 2: Use different DNS
```bash
# Add to ~/.bashrc
export NODE_OPTIONS="--dns-result-order=ipv4first"
```

### Fix 3: Test connectivity
```bash
# Can you reach internet?
ping -c 3 8.8.8.8

# Can you reach RPC?
curl https://ethereum-holesky-rpc.publicnode.com \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"eth_blockNumber","params":[],"id":1}'
```

If curl works but Node.js doesn't, it's a Node.js network issue in Termux.

### Fix 4: Use older Node.js
```bash
# Termux sometimes has issues with latest Node
pkg install nodejs-lts
```

---

## RECOMMENDED APPROACH

1. ✅ **Get Holesky ETH** from mining faucet: https://holesky-faucet.pk910.de
2. ✅ **Use Remix IDE** (easiest, no installation): https://remix.ethereum.org
3. ✅ **Deploy in 10 minutes** following Method 2 above

---

## YOUR WALLET INFO

**Address:** `0x24f6b1ce11c57d40b542f91ac85fa9eb61f78771`
**Private Key:** `c411a4d4365560753ef3ceceac1652ec89240704346bf58ad900d65574f541c9`

**⚠️ SECURITY NOTE:** This private key is in a public repo. Only use for testing!
For mainnet, create a new wallet with a secret private key.

---

## ONCE DEPLOYED

After deployment, you'll get 4 contract addresses. Save them!

Then you can:
- View on Etherscan: https://holesky.etherscan.io
- Interact with contracts via Remix
- Build frontend app
- Accept real payments (after mainnet deployment)

---

## NEED HELP?

Can't get network working from Termux? That's OK!

**Try PC/laptop** or **use Remix IDE** - both work perfectly!

✨ Operating at 528Hz Love Frequency ✨
