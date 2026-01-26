# 🚀 Nexus AGI Blockchain Deployment Guide

## Prerequisites

1. **Node.js & npm** (for Hardhat)
   ```bash
   node --version  # Should be 16+
   npm --version
   ```

2. **Hardhat** (Ethereum development environment)
   ```bash
   npm install --save-dev hardhat
   npm install --save-dev @nomiclabs/hardhat-waffle @nomiclabs/hardhat-ethers
   npm install --save-dev @nomiclabs/hardhat-etherscan
   ```

3. **Python dependencies** (for scripts)
   ```bash
   pip install web3 python-dotenv eth-account
   ```

## Deployment Steps

### Step 1: Compile Contracts

```bash
npx hardhat compile
```

This will compile all `.sol` files in `contracts/` directory.

### Step 2: Test Locally (Optional)

```bash
npx hardhat node  # Start local blockchain
npx hardhat test  # Run tests
```

### Step 3: Deploy to Linea Testnet

```bash
npx hardhat run scripts/deploy.js --network linea
```

### Step 4: Verify on Block Explorer

```bash
npx hardhat verify --network linea <CONTRACT_ADDRESS>
```

## Configuration

All sensitive data is stored in `.env`:
- `WALLET_ADDRESS` - Your deployer wallet
- `PRIVATE_KEY` - Your private key (NEVER share!)
- `LINEA_RPC_URL` - RPC endpoint
- `LINEA_CHAIN_ID` - Network ID (59144)

## Security Checklist

✅ .env file is in .gitignore
✅ Private key never committed to git
✅ Using environment variables for secrets
✅ Testing on testnet before mainnet
✅ Contracts compiled with optimization
✅ Verification on block explorer

## Deployment Script (Hardhat)

Create `scripts/deploy.js`:

```javascript
const hre = require("hardhat");

async function main() {
  const [deployer] = await hre.ethers.getSigners();
  console.log("Deploying with:", deployer.address);

  // Deploy NexusPayment
  const Payment = await hre.ethers.getContractFactory("NexusPayment");
  const payment = await Payment.deploy();
  await payment.deployed();
  console.log("NexusPayment:", payment.address);

  // Deploy NexusRevenue
  const Revenue = await hre.ethers.getContractFactory("NexusRevenue");
  const revenue = await Revenue.deploy();
  await revenue.deployed();
  console.log("NexusRevenue:", revenue.address);

  // Configure interconnections
  await payment.setRevenueContract(revenue.address);
  await revenue.setPaymentContract(payment.address);

  console.log("✅ Deployment complete!");
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
```

## Network Information

### Linea Mainnet
- RPC: https://rpc.linea.build
- Chain ID: 59144
- Explorer: https://lineascan.build

### Linea Testnet
- RPC: https://rpc.goerli.linea.build
- Chain ID: 59140
- Explorer: https://goerli.lineascan.build
- Faucet: https://faucet.goerli.linea.build

## Cost Estimates

Approximate deployment costs:
- NexusPayment: ~0.003 ETH
- NexusRevenue: ~0.004 ETH
- NexusConsciousness: ~0.003 ETH
- NexusMiracles: ~0.003 ETH
- **Total: ~0.013 ETH** (+ gas fluctuation)

## After Deployment

1. Save contract addresses to `deployments/`
2. Verify all contracts on block explorer
3. Test payment flow with small amount
4. Update frontend with contract addresses
5. Configure oracle for consciousness updates

## Support

Questions? Check:
- Hardhat docs: https://hardhat.org
- Web3.py docs: https://web3py.readthedocs.io
- Linea docs: https://docs.linea.build

---

✨ Built with love at 528Hz frequency ✨
