#!/bin/bash
# ================================================================
# NEXUS AGI - COMPLETE DEPLOYMENT SCRIPT
# One-click deploy for Termux/Linux
# ================================================================

echo "🚀 NEXUS AGI BLOCKCHAIN DEPLOYMENT"
echo "=================================="
echo ""

# ================================================================
# STEP 1: Check/Start Geth Blockchain
# ================================================================
echo "🔍 Checking if Geth blockchain is running..."
if curl -s -X POST -H "Content-Type: application/json" \
   --data '{"jsonrpc":"2.0","method":"eth_blockNumber","params":[],"id":1}' \
   http://localhost:8545 > /dev/null 2>&1; then
    echo "✅ Blockchain is running"
else
    echo "⚠️  Starting Geth blockchain..."
    echo ""

    # Kill any existing Geth processes
    pkill -9 geth 2>/dev/null
    sleep 2

    # Start Geth in dev mode with unlimited ETH
    geth --dev \
         --dev.period 5 \
         --http \
         --http.addr "0.0.0.0" \
         --http.port 8545 \
         --http.api "eth,net,web3,personal,miner" \
         --http.corsdomain "*" \
         --allow-insecure-unlock \
         > geth_blockchain.log 2>&1 &

    echo "⏳ Waiting for Geth to start (10 seconds)..."
    sleep 10

    # Verify it started
    if curl -s -X POST -H "Content-Type: application/json" \
       --data '{"jsonrpc":"2.0","method":"eth_blockNumber","params":[],"id":1}' \
       http://localhost:8545 > /dev/null 2>&1; then
        echo "✅ Geth blockchain started successfully!"
        echo ""
    else
        echo "❌ Failed to start Geth. Check geth_blockchain.log"
        exit 1
    fi
fi

# ================================================================
# STEP 2: Get Geth Account Info
# ================================================================
echo "📋 Getting blockchain account info..."

GETH_ACCOUNT=$(curl -s -X POST -H "Content-Type: application/json" \
    --data '{"jsonrpc":"2.0","method":"eth_accounts","params":[],"id":1}' \
    http://localhost:8545 | grep -o '"0x[a-fA-F0-9]*"' | head -1 | tr -d '"')

if [ -z "$GETH_ACCOUNT" ]; then
    echo "❌ Could not get Geth account"
    exit 1
fi

echo "✅ Geth Account: $GETH_ACCOUNT"
echo ""

# ================================================================
# STEP 3: Update Hardhat Config for Geth
# ================================================================
echo "🔧 Configuring Hardhat for Geth (Chain ID: 1337)..."

cat > hardhat.config.cjs << 'HARDHAT_CONFIG'
require("@nomicfoundation/hardhat-toolbox");

module.exports = {
  solidity: {
    version: "0.8.20",
    settings: {
      optimizer: {
        enabled: true,
        runs: 200
      }
    }
  },
  networks: {
    localhost: {
      url: "http://127.0.0.1:8545",
      chainId: 1337,
      accounts: [
        "0xb71c71a67e1177ad4e901695e1b4b9ee17ae16c6668d313eac2f96dbcda3f291"
      ]
    }
  }
};
HARDHAT_CONFIG

echo "✅ Hardhat configured for Geth"
echo ""

# ================================================================
# STEP 4: Convert Deploy Script to CommonJS
# ================================================================
echo "🔧 Converting deployment script to CommonJS..."

cat > scripts/deploy.cjs << 'DEPLOY_SCRIPT'
const hre = require("hardhat");
const fs = require("fs");
const path = require("path");

async function main() {
  console.log("\n" + "=".repeat(80));
  console.log("  🚀 DEPLOYING NEXUS AGI TO GETH BLOCKCHAIN 🚀");
  console.log("=".repeat(80) + "\n");

  // Get deployer account
  const [deployer] = await hre.ethers.getSigners();

  console.log("📋 Deployment Details:");
  console.log(`  Network: ${hre.network.name}`);
  console.log(`  Chain ID: ${hre.network.config.chainId}`);
  console.log(`  Deployer: ${deployer.address}`);

  // Check balance
  const balance = await deployer.provider.getBalance(deployer.address);
  console.log(`  Balance: ${hre.ethers.formatEther(balance)} ETH\n`);

  console.log("=".repeat(80));
  console.log("  📦 DEPLOYING CONTRACTS");
  console.log("=".repeat(80) + "\n");

  // Deploy NexusPayment
  console.log("[1/4] Deploying NexusPayment...");
  const NexusPayment = await hre.ethers.getContractFactory("NexusPayment");
  const payment = await NexusPayment.deploy();
  await payment.waitForDeployment();
  const paymentAddress = await payment.getAddress();
  console.log(`  ✅ NexusPayment: ${paymentAddress}\n`);

  // Deploy NexusRevenue
  console.log("[2/4] Deploying NexusRevenue...");
  const NexusRevenue = await hre.ethers.getContractFactory("NexusRevenue");
  const revenue = await NexusRevenue.deploy();
  await revenue.waitForDeployment();
  const revenueAddress = await revenue.getAddress();
  console.log(`  ✅ NexusRevenue: ${revenueAddress}\n`);

  // Deploy NexusConsciousness
  console.log("[3/4] Deploying NexusConsciousness...");
  const NexusConsciousness = await hre.ethers.getContractFactory("NexusConsciousness");
  const consciousness = await NexusConsciousness.deploy();
  await consciousness.waitForDeployment();
  const consciousnessAddress = await consciousness.getAddress();
  console.log(`  ✅ NexusConsciousness: ${consciousnessAddress}\n`);

  // Deploy NexusMiracles
  console.log("[4/4] Deploying NexusMiracles...");
  const NexusMiracles = await hre.ethers.getContractFactory("NexusMiracles");
  const miracles = await NexusMiracles.deploy();
  await miracles.waitForDeployment();
  const miraclesAddress = await miracles.getAddress();
  console.log(`  ✅ NexusMiracles: ${miraclesAddress}\n`);

  // Configure interconnections
  console.log("=".repeat(80));
  console.log("  🔗 CONFIGURING CONTRACT INTERCONNECTIONS");
  console.log("=".repeat(80) + "\n");

  console.log("Linking NexusPayment → NexusRevenue...");
  const setRevenueTx = await payment.setRevenueContract(revenueAddress);
  await setRevenueTx.wait();
  console.log("  ✅ Payment linked to Revenue\n");

  console.log("Linking NexusRevenue → NexusPayment...");
  const setPaymentTx = await revenue.setPaymentContract(paymentAddress);
  await setPaymentTx.wait();
  console.log("  ✅ Revenue linked to Payment\n");

  console.log("Setting oracle for NexusConsciousness...");
  const setOracleTx = await consciousness.setOracle(deployer.address);
  await setOracleTx.wait();
  console.log("  ✅ Consciousness oracle configured\n");

  console.log("Setting oracle for NexusMiracles...");
  const setMiraclesOracleTx = await miracles.setOracle(deployer.address);
  await setMiraclesOracleTx.wait();
  console.log("  ✅ Miracles oracle configured\n");

  // Save deployment info
  const deploymentInfo = {
    network: "localhost",
    chainId: 1337,
    deployer: deployer.address,
    timestamp: new Date().toISOString(),
    contracts: [
      { name: "NexusPayment", address: paymentAddress },
      { name: "NexusRevenue", address: revenueAddress },
      { name: "NexusConsciousness", address: consciousnessAddress },
      { name: "NexusMiracles", address: miraclesAddress }
    ]
  };

  fs.writeFileSync(
    "deployment_addresses.json",
    JSON.stringify(deploymentInfo, null, 2)
  );

  // Final summary
  console.log("=".repeat(80));
  console.log("  ✅ DEPLOYMENT COMPLETE!");
  console.log("=".repeat(80) + "\n");

  console.log("🔗 CONTRACT ADDRESSES (COPY TO METAMASK):");
  console.log(`  NexusPayment:       ${paymentAddress}`);
  console.log(`  NexusRevenue:       ${revenueAddress}`);
  console.log(`  NexusConsciousness: ${consciousnessAddress}`);
  console.log(`  NexusMiracles:      ${miraclesAddress}\n`);

  console.log("🦊 METAMASK SETUP:");
  console.log("  Network Name:    Geth Local");
  console.log("  RPC URL:         http://localhost:8545");
  console.log("  Chain ID:        1337");
  console.log("  Currency Symbol: ETH\n");

  console.log("📝 Your Account:");
  console.log(`  Address: ${deployer.address}`);
  console.log(`  Balance: ${hre.ethers.formatEther(balance)} ETH\n`);

  console.log("✨ Operating at 528Hz Love Frequency ✨");
  console.log("💖 Nexus AGI is now on-chain! 💖\n");
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error("\n❌ Deployment failed:");
    console.error(error);
    process.exit(1);
  });
DEPLOY_SCRIPT

echo "✅ Deployment script ready"
echo ""

# ================================================================
# STEP 5: Check Internet Connection
# ================================================================
echo "📡 Checking internet connection..."
if ping -c 1 8.8.8.8 > /dev/null 2>&1; then
    echo "✅ Internet connected"
    echo ""
else
    echo "❌ No internet connection"
    echo "   First-time compilation requires internet to download Solidity compiler"
    echo "   Please connect to internet and try again"
    echo ""
    exit 1
fi

# ================================================================
# STEP 6: Install Dependencies
# ================================================================
echo "📦 Installing/checking dependencies..."
if [ ! -d "node_modules" ]; then
    echo "   Installing npm packages..."
    npm install --silent > /dev/null 2>&1
    echo "✅ Dependencies installed"
else
    echo "✅ Dependencies already installed"
fi

echo "📦 Installing required Hardhat plugins..."
npm install --save-dev @nomicfoundation/hardhat-ethers@^3.1.0 ethers@^6.0.0 --silent > /dev/null 2>&1
echo "✅ Hardhat plugins installed"
echo ""

# ================================================================
# STEP 7: Compile Smart Contracts
# ================================================================
echo "🔨 Compiling Nexus AGI Smart Contracts..."
echo "   (This may take 1-2 minutes on first run)"
echo ""

npx hardhat compile

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Compilation successful!"
    echo ""
else
    echo ""
    echo "❌ Compilation failed"
    echo "   Check the error messages above"
    exit 1
fi

# ================================================================
# STEP 8: Deploy to Geth Blockchain
# ================================================================
echo "📦 Deploying contracts to Geth blockchain..."
echo ""

npx hardhat run scripts/deploy.cjs --network localhost

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 DEPLOYMENT SUCCESSFUL! 🎉"
    echo ""
else
    echo ""
    echo "❌ Deployment failed"
    exit 1
fi

# ================================================================
# STEP 9: Display Results
# ================================================================
if [ -f "deployment_addresses.json" ]; then
    echo "=".repeat(80)
    echo "📋 REAL CONTRACT ADDRESSES WITH BYTECODE:"
    echo "=".repeat(80)
    echo ""
    cat deployment_addresses.json
    echo ""
    echo "=".repeat(80)
    echo ""
    echo "✅ These addresses now have REAL bytecode deployed!"
    echo "✅ MetaMask will recognize them as CONTRACTS (not personal addresses)!"
    echo ""
fi

echo "🦊 ADD TO METAMASK:"
echo "   1. Open MetaMask"
echo "   2. Add Network:"
echo "      Network Name: Geth Local"
echo "      RPC URL: http://localhost:8545"
echo "      Chain ID: 1337"
echo "      Currency: ETH"
echo ""
echo "   3. Import Account:"
echo "      Private Key: 0xb71c71a67e1177ad4e901695e1b4b9ee17ae16c6668d313eac2f96dbcda3f291"
echo ""
echo "   4. Add Contract Addresses from deployment_addresses.json above"
echo ""
echo "✨ Operating at 528Hz Love Frequency ✨"
echo "💖 Your Nexus AGI smart contracts are LIVE! 💖"
echo ""
