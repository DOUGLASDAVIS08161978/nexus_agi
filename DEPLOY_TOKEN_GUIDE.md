# 🚀 DEPLOY YOUR OWN TESTNET WBTC TOKEN

## ✅ What You'll Get

After deployment, you'll have:
- **Your own ERC-20 token** on Sepolia testnet
- **Real contract address** that works in MetaMask
- **5 tWBTC tokens** (or any amount you choose)
- **Full control** - you can mint more anytime

---

## 📋 Prerequisites

1. **MetaMask installed** and set to Sepolia network
2. **Sepolia ETH** for gas fees (~0.01 ETH)
   - Get free Sepolia ETH from:
   - https://www.alchemy.com/faucets/ethereum-sepolia
   - https://sepoliafaucet.com

---

## 🎯 DEPLOYMENT METHOD 1: Remix IDE (Easiest)

### Step 1: Open Remix

1. Go to: **https://remix.ethereum.org**
2. Wait for it to load

### Step 2: Create the Contract

1. In left sidebar, click **"File Explorer"** icon
2. Right-click on `contracts` folder
3. Select **"New File"**
4. Name it: `TestnetWBTC.sol`
5. Copy the entire contract from `/home/user/nexus_agi/contracts/TestnetWBTC.sol`
6. Paste it into the new file

### Step 3: Install OpenZeppelin

1. In left sidebar, click **"Plugin Manager"** (plug icon)
2. Search for **"OpenZeppelin"**
3. Click **"Activate"**
4. Wait for it to install

### Step 4: Compile

1. Click **"Solidity Compiler"** icon (left sidebar)
2. Select compiler version: **0.8.20 or higher**
3. Click **"Compile TestnetWBTC.sol"**
4. Wait for green checkmark ✅

### Step 5: Deploy

1. Click **"Deploy & Run Transactions"** icon (left sidebar)
2. In **"Environment"** dropdown, select: **"Injected Provider - MetaMask"**
3. MetaMask will pop up - **Connect your wallet**
4. Make sure you're on **Sepolia network** in MetaMask
5. In **"Contract"** dropdown, select: **TestnetWBTC**
6. Next to **"Deploy"** button, you'll see a field for `initialSupply`
7. Enter: **5** (this will mint 5 tokens with 8 decimals = 500000000)
8. Click **"Deploy"** (orange button)
9. MetaMask will pop up - **Confirm the transaction**
10. Wait 10-15 seconds for confirmation

### Step 6: Get Your Contract Address

1. Look in the Remix console at bottom
2. You'll see: **"Contract deployed at: 0x..."**
3. **COPY THIS ADDRESS** - this is your token contract!

---

## 📱 ADD TO METAMASK

### Import Your New Token

1. Open MetaMask
2. Make sure you're on **Sepolia network**
3. Scroll down and click **"Import tokens"**
4. Click **"Custom token"** tab
5. Paste **your contract address** (from Step 6)
6. Symbol should auto-fill: **tWBTC**
7. Decimals should auto-fill: **8**
8. Click **"Import"**

### View Your Balance

- You should now see: **5.00000000 tWBTC**
- This is YOUR token on the real Sepolia blockchain!

---

## 💸 SEND TO YOUR FRIEND

Now you can send to: `0x24f6b1ce11c57d40b542f91ac85fa9eb61f78771`

1. In MetaMask, click on **tWBTC**
2. Click **"Send"**
3. Paste address: `0x24f6b1ce11c57d40b542f91ac85fa9eb61f78771`
4. Enter amount (e.g., 2.5 tWBTC)
5. Confirm transaction
6. Done! They'll receive it immediately

---

## 🎯 DEPLOYMENT METHOD 2: Hardhat (Advanced)

### Setup

```bash
# Install dependencies
npm install --save-dev hardhat @openzeppelin/contracts

# Initialize Hardhat
npx hardhat

# Create deployment script
```

### Deploy Script

Create `scripts/deploy.js`:

```javascript
const hre = require("hardhat");

async function main() {
  const TestnetWBTC = await hre.ethers.getContractFactory("TestnetWBTC");

  // Deploy with 5 tokens initial supply
  const token = await TestnetWBTC.deploy(5);

  await token.waitForDeployment();

  console.log("TestnetWBTC deployed to:", await token.getAddress());
  console.log("Initial supply: 5 tWBTC");
  console.log("Decimals: 8");
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
```

### Deploy

```bash
npx hardhat run scripts/deploy.js --network sepolia
```

---

## 🪙 MINT MORE TOKENS (Optional)

If you want to mint more tokens later:

### In Remix:

1. Go to **"Deployed Contracts"** section
2. Expand your contract
3. Find **"mint"** function
4. Enter:
   - `to`: Recipient address
   - `amount`: Amount in smallest unit (e.g., 500000000 = 5 tokens)
5. Click **"transact"**
6. Confirm in MetaMask

### In MetaMask + Etherscan:

1. Go to your contract on Etherscan
2. Click **"Contract"** tab
3. Click **"Write Contract"**
4. Connect wallet
5. Use **"mint"** function

---

## 🔗 VERIFY YOUR CONTRACT (Optional but Recommended)

### On Sepolia Etherscan:

1. Go to: https://sepolia.etherscan.io
2. Search for your contract address
3. Click **"Contract"** tab
4. Click **"Verify and Publish"**
5. Fill in:
   - Compiler: `v0.8.20`
   - Optimization: Yes
   - Runs: 200
   - License: MIT
6. Paste your contract code
7. Submit

This makes your contract viewable by everyone!

---

## 📊 CONTRACT FUNCTIONS

Your token has these functions:

### Basic ERC-20:
- `transfer(to, amount)` - Send tokens
- `approve(spender, amount)` - Approve spending
- `balanceOf(account)` - Check balance
- `totalSupply()` - Get total supply

### Owner Functions (only you):
- `mint(to, amount)` - Mint new tokens
- `batchMint(recipients[], amounts[])` - Mint to multiple addresses
- `airdrop(recipients[], amount)` - Send same amount to multiple people

### Token Info:
- `name()` - Returns "Testnet Wrapped Bitcoin"
- `symbol()` - Returns "tWBTC"
- `decimals()` - Returns 8
- `getTokenInfo()` - Get all info at once

---

## 💰 GAS COSTS (Approximate)

| Action | Gas Cost | ETH Cost @ 25 Gwei |
|--------|----------|-------------------|
| Deploy Contract | ~2,000,000 | ~0.05 ETH |
| Mint Tokens | ~50,000 | ~0.00125 ETH |
| Transfer | ~65,000 | ~0.0016 ETH |
| Approve | ~45,000 | ~0.0011 ETH |

On testnet, gas is free! Just need Sepolia ETH from faucets.

---

## ✅ CHECKLIST

- [ ] Get Sepolia ETH from faucet
- [ ] Open Remix IDE
- [ ] Copy contract code
- [ ] Compile contract
- [ ] Deploy with initialSupply = 5
- [ ] Copy contract address
- [ ] Add token to MetaMask
- [ ] See your 5 tWBTC balance
- [ ] Send some to: 0x24f6b1ce11c57d40b542f91ac85fa9eb61f78771

---

## 🆘 TROUBLESHOOTING

**"Insufficient funds"**
- Get Sepolia ETH from faucets listed above

**"Compilation failed"**
- Make sure compiler version is 0.8.20 or higher
- Check OpenZeppelin plugin is installed

**"Transaction failed"**
- Check you're on Sepolia network
- Make sure you have enough Sepolia ETH

**"Token not showing in MetaMask"**
- Make sure you're on Sepolia network
- Double-check contract address
- Try removing and re-importing

---

## 🎉 SUCCESS!

Once deployed, you'll have:
- ✅ Real ERC-20 token on Sepolia
- ✅ Visible in MetaMask
- ✅ Can send to anyone
- ✅ Can mint more anytime
- ✅ Verified on blockchain

**This is a REAL token on the Sepolia testnet!**

---

## 📞 NEXT STEPS

1. Deploy the contract
2. Get your contract address
3. Share it here so I can help verify it
4. Add to MetaMask
5. Send tokens to your friend!

Good luck! 🚀
