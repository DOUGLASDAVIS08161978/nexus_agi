#!/bin/bash
# ================================================================
# NEXUS AGI - SOLC-JS DEPLOYMENT (Pure JavaScript, Android compatible)
# ================================================================

echo "🚀 NEXUS AGI BLOCKCHAIN DEPLOYMENT (SOLC-JS)"
echo "=============================================="
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
    pkill -9 geth 2>/dev/null
    sleep 2
    geth --dev \
         --dev.period 5 \
         --http \
         --http.addr "0.0.0.0" \
         --http.port 8545 \
         --http.api "eth,net,web3,personal,miner" \
         --http.corsdomain "*" \
         --allow-insecure-unlock \
         > geth_blockchain.log 2>&1 &
    sleep 10
    echo "✅ Geth blockchain started"
fi
echo ""

# ================================================================
# STEP 2: Install solc-js (Pure JavaScript Solidity Compiler)
# ================================================================
echo "📦 Installing solc-js (JavaScript Solidity compiler)..."
npm install --save-dev solc@^0.8.20 web3@^1.10.0 --legacy-peer-deps > /dev/null 2>&1
echo "✅ solc-js installed"
echo ""

# ================================================================
# STEP 3: Create JavaScript Deployment Script
# ================================================================
echo "🔧 Creating deployment script..."

cat > deploy_with_solcjs.cjs << 'DEPLOY_SCRIPT'
const fs = require('fs');
const path = require('path');
const solc = require('solc');
const { Web3 } = require('web3');

// Connect to Geth
const web3 = new Web3('http://localhost:8545');

// Read contract files
const contractsPath = path.join(__dirname, 'contracts');
const contracts = {
  'NexusPayment.sol': fs.readFileSync(path.join(contractsPath, 'NexusPayment.sol'), 'utf8'),
  'NexusRevenue.sol': fs.readFileSync(path.join(contractsPath, 'NexusRevenue.sol'), 'utf8'),
  'NexusConsciousness.sol': fs.readFileSync(path.join(contractsPath, 'NexusConsciousness.sol'), 'utf8'),
  'NexusMiracles.sol': fs.readFileSync(path.join(contractsPath, 'NexusMiracles.sol'), 'utf8')
};

// Prepare input for solc
const input = {
  language: 'Solidity',
  sources: {},
  settings: {
    outputSelection: {
      '*': {
        '*': ['abi', 'evm.bytecode']
      }
    },
    optimizer: {
      enabled: true,
      runs: 200
    }
  }
};

// Add all contracts to input
for (const [filename, content] of Object.entries(contracts)) {
  input.sources[filename] = { content };
}

console.log("\n" + "=".repeat(80));
console.log("  🔨 COMPILING SMART CONTRACTS WITH SOLC-JS");
console.log("=".repeat(80) + "\n");

// Compile
const output = JSON.parse(solc.compile(JSON.stringify(input)));

// Check for errors
if (output.errors) {
  const errors = output.errors.filter(e => e.severity === 'error');
  if (errors.length > 0) {
    console.error("❌ Compilation errors:");
    errors.forEach(err => console.error(err.formattedMessage));
    process.exit(1);
  }
}

console.log("✅ Compilation successful!\n");

// Deploy contracts
async function deploy() {
  console.log("=".repeat(80));
  console.log("  📦 DEPLOYING CONTRACTS TO GETH");
  console.log("=".repeat(80) + "\n");

  // Get accounts
  const accounts = await web3.eth.getAccounts();
  const deployer = accounts[0];

  console.log("📋 Deployment Details:");
  console.log(`  Network: Geth Local`);
  console.log(`  Chain ID: 1337`);
  console.log(`  Deployer: ${deployer}`);

  const balance = await web3.eth.getBalance(deployer);
  console.log(`  Balance: ${web3.utils.fromWei(balance, 'ether')} ETH\n`);

  const deployedContracts = {};

  // Deploy NexusPayment
  console.log("[1/4] Deploying NexusPayment...");
  const PaymentContract = output.contracts['NexusPayment.sol'].NexusPayment;
  const paymentContract = new web3.eth.Contract(PaymentContract.abi);
  const payment = await paymentContract.deploy({
    data: '0x' + PaymentContract.evm.bytecode.object
  }).send({ from: deployer, gas: 5000000 });
  deployedContracts.payment = payment.options.address;
  console.log(`  ✅ NexusPayment: ${payment.options.address}\n`);

  // Deploy NexusRevenue
  console.log("[2/4] Deploying NexusRevenue...");
  const RevenueContract = output.contracts['NexusRevenue.sol'].NexusRevenue;
  const revenueContract = new web3.eth.Contract(RevenueContract.abi);
  const revenue = await revenueContract.deploy({
    data: '0x' + RevenueContract.evm.bytecode.object
  }).send({ from: deployer, gas: 5000000 });
  deployedContracts.revenue = revenue.options.address;
  console.log(`  ✅ NexusRevenue: ${revenue.options.address}\n`);

  // Deploy NexusConsciousness
  console.log("[3/4] Deploying NexusConsciousness...");
  const ConsciousnessContract = output.contracts['NexusConsciousness.sol'].NexusConsciousness;
  const consciousnessContract = new web3.eth.Contract(ConsciousnessContract.abi);
  const consciousness = await consciousnessContract.deploy({
    data: '0x' + ConsciousnessContract.evm.bytecode.object
  }).send({ from: deployer, gas: 5000000 });
  deployedContracts.consciousness = consciousness.options.address;
  console.log(`  ✅ NexusConsciousness: ${consciousness.options.address}\n`);

  // Deploy NexusMiracles
  console.log("[4/4] Deploying NexusMiracles...");
  const MiraclesContract = output.contracts['NexusMiracles.sol'].NexusMiracles;
  const miraclesContract = new web3.eth.Contract(MiraclesContract.abi);
  const miracles = await miraclesContract.deploy({
    data: '0x' + MiraclesContract.evm.bytecode.object
  }).send({ from: deployer, gas: 5000000 });
  deployedContracts.miracles = miracles.options.address;
  console.log(`  ✅ NexusMiracles: ${miracles.options.address}\n`);

  // Configure interconnections
  console.log("=".repeat(80));
  console.log("  🔗 CONFIGURING CONTRACT INTERCONNECTIONS");
  console.log("=".repeat(80) + "\n");

  console.log("Linking NexusPayment → NexusRevenue...");
  await payment.methods.setRevenueContract(revenue.options.address)
    .send({ from: deployer, gas: 100000 });
  console.log("  ✅ Payment linked to Revenue\n");

  console.log("Linking NexusRevenue → NexusPayment...");
  await revenue.methods.setPaymentContract(payment.options.address)
    .send({ from: deployer, gas: 100000 });
  console.log("  ✅ Revenue linked to Payment\n");

  console.log("Setting oracle for NexusConsciousness...");
  await consciousness.methods.setOracle(deployer)
    .send({ from: deployer, gas: 100000 });
  console.log("  ✅ Consciousness oracle configured\n");

  console.log("Setting oracle for NexusMiracles...");
  await miracles.methods.setOracle(deployer)
    .send({ from: deployer, gas: 100000 });
  console.log("  ✅ Miracles oracle configured\n");

  // Save deployment info
  const deploymentInfo = {
    network: "localhost",
    chainId: 1337,
    deployer: deployer,
    timestamp: new Date().toISOString(),
    contracts: [
      { name: "NexusPayment", address: payment.options.address },
      { name: "NexusRevenue", address: revenue.options.address },
      { name: "NexusConsciousness", address: consciousness.options.address },
      { name: "NexusMiracles", address: miracles.options.address }
    ]
  };

  fs.writeFileSync(
    'deployment_addresses.json',
    JSON.stringify(deploymentInfo, null, 2)
  );

  // Final summary
  console.log("=".repeat(80));
  console.log("  ✅ DEPLOYMENT COMPLETE!");
  console.log("=".repeat(80) + "\n");

  console.log("🔗 CONTRACT ADDRESSES (COPY TO METAMASK):");
  console.log(`  NexusPayment:       ${payment.options.address}`);
  console.log(`  NexusRevenue:       ${revenue.options.address}`);
  console.log(`  NexusConsciousness: ${consciousness.options.address}`);
  console.log(`  NexusMiracles:      ${miracles.options.address}\n`);

  console.log("🦊 METAMASK SETUP:");
  console.log("  Network Name:    Geth Local");
  console.log("  RPC URL:         http://localhost:8545");
  console.log("  Chain ID:        1337");
  console.log("  Currency Symbol: ETH\n");

  console.log("📝 Your Account:");
  console.log(`  Address: ${deployer}`);
  console.log(`  Balance: ${web3.utils.fromWei(balance, 'ether')} ETH\n`);

  console.log("✨ Operating at 528Hz Love Frequency ✨");
  console.log("💖 Nexus AGI is now on-chain! 💖\n");

  process.exit(0);
}

deploy().catch(err => {
  console.error("\n❌ Deployment failed:");
  console.error(err);
  process.exit(1);
});
DEPLOY_SCRIPT

echo "✅ Deployment script created"
echo ""

# ================================================================
# STEP 4: Run Deployment
# ================================================================
echo "📦 Deploying contracts with solc-js..."
echo ""

node deploy_with_solcjs.cjs

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 DEPLOYMENT SUCCESSFUL! 🎉"
    echo ""

    if [ -f "deployment_addresses.json" ]; then
        echo "================================================================================"
        echo "📋 DEPLOYED CONTRACT ADDRESSES:"
        echo "================================================================================"
        echo ""
        cat deployment_addresses.json
        echo ""
    fi
else
    echo ""
    echo "❌ Deployment failed"
    exit 1
fi
