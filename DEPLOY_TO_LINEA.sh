#!/bin/bash
# ================================================================
# NEXUS AGI - DEPLOY TO LINEA SEPOLIA TESTNET
# Get real blockchain deployment with actual users!
# ================================================================

echo "🌐 NEXUS AGI - LINEA SEPOLIA DEPLOYMENT"
echo "========================================"
echo ""

# ================================================================
# STEP 1: Get Testnet ETH
# ================================================================
echo "💰 STEP 1: GET FREE TESTNET ETH"
echo "================================"
echo ""
echo "You need Linea Sepolia ETH to deploy contracts."
echo ""
echo "🔗 Get free testnet ETH from:"
echo "   https://faucet.quicknode.com/linea/sepolia"
echo ""
echo "   OR use the Linea faucet:"
echo "   https://faucet.goerli.linea.build/"
echo ""
echo "Your address to receive testnet ETH:"
echo "   0x24f6b1ce11c57d40b542f91ac85fa9eb61f78771"
echo ""
read -p "Press ENTER after you've received testnet ETH..."

# ================================================================
# STEP 2: Configure for Linea Sepolia
# ================================================================
echo ""
echo "⚙️  STEP 2: CONFIGURING FOR LINEA SEPOLIA"
echo "========================================="
echo ""

# Check if private key is set
if [ -z "$LINEA_PRIVATE_KEY" ]; then
    echo "⚠️  Private key not found in environment"
    echo ""
    echo "For security, set your private key as an environment variable:"
    echo ""
    echo "   export LINEA_PRIVATE_KEY='your_private_key_here'"
    echo ""
    echo "Or we can use the Geth dev account (NOT recommended for mainnet):"
    echo "   0xb71c71a67e1177ad4e901695e1b4b9ee17ae16c6668d313eac2f96dbcda3f291"
    echo ""
    read -p "Use Geth dev key? (y/n): " use_dev_key

    if [ "$use_dev_key" = "y" ]; then
        PRIVATE_KEY="0xb71c71a67e1177ad4e901695e1b4b9ee17ae16c6668d313eac2f96dbcda3f291"
        echo "✅ Using Geth dev account"
    else
        read -p "Enter your private key: " PRIVATE_KEY
    fi
else
    PRIVATE_KEY="$LINEA_PRIVATE_KEY"
    echo "✅ Using private key from environment"
fi

echo ""

# ================================================================
# STEP 3: Install dependencies
# ================================================================
echo "📦 STEP 3: INSTALLING DEPENDENCIES"
echo "==================================="
echo ""

npm install --save-dev solc@^0.8.20 web3@^1.10.0 --legacy-peer-deps > /dev/null 2>&1
echo "✅ Dependencies installed"
echo ""

# ================================================================
# STEP 4: Create Linea deployment script
# ================================================================
echo "🔧 STEP 4: CREATING DEPLOYMENT SCRIPT"
echo "======================================"
echo ""

cat > deploy_to_linea.cjs << 'DEPLOY_SCRIPT'
const fs = require('fs');
const path = require('path');
const solc = require('solc');
const Web3 = require('web3');

// Linea Sepolia RPC
const web3 = new Web3('https://rpc.sepolia.linea.build');

// Get private key from environment
const privateKey = process.env.DEPLOY_PRIVATE_KEY;
if (!privateKey) {
    console.error("❌ Private key not set!");
    process.exit(1);
}

// Create account from private key
const account = web3.eth.accounts.privateKeyToAccount(privateKey);
web3.eth.accounts.wallet.add(account);

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
console.log("  🔨 COMPILING SMART CONTRACTS");
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
  console.log("  🌐 DEPLOYING TO LINEA SEPOLIA TESTNET");
  console.log("=".repeat(80) + "\n");

  const deployer = account.address;

  console.log("📋 Deployment Details:");
  console.log(`  Network: Linea Sepolia Testnet`);
  console.log(`  Chain ID: 59141`);
  console.log(`  Deployer: ${deployer}`);

  const balance = await web3.eth.getBalance(deployer);
  const balanceEth = web3.utils.fromWei(balance, 'ether');
  console.log(`  Balance: ${balanceEth} ETH\n`);

  if (parseFloat(balanceEth) < 0.01) {
    console.log("⚠️  WARNING: Low balance!");
    console.log("   Get testnet ETH from: https://faucet.quicknode.com/linea/sepolia");
    console.log("");
  }

  const deployedContracts = {};
  const gasPrice = await web3.eth.getGasPrice();

  // Deploy NexusPayment
  console.log("[1/4] Deploying NexusPayment...");
  const PaymentContract = output.contracts['NexusPayment.sol'].NexusPayment;
  const paymentContract = new web3.eth.Contract(PaymentContract.abi);
  const payment = await paymentContract.deploy({
    data: '0x' + PaymentContract.evm.bytecode.object
  }).send({
    from: deployer,
    gas: 5000000,
    gasPrice: gasPrice
  });
  deployedContracts.payment = payment.options.address;
  console.log(`  ✅ NexusPayment: ${payment.options.address}`);
  console.log(`     View: https://sepolia.lineascan.build/address/${payment.options.address}\n`);

  // Deploy NexusRevenue
  console.log("[2/4] Deploying NexusRevenue...");
  const RevenueContract = output.contracts['NexusRevenue.sol'].NexusRevenue;
  const revenueContract = new web3.eth.Contract(RevenueContract.abi);
  const revenue = await revenueContract.deploy({
    data: '0x' + RevenueContract.evm.bytecode.object
  }).send({
    from: deployer,
    gas: 5000000,
    gasPrice: gasPrice
  });
  deployedContracts.revenue = revenue.options.address;
  console.log(`  ✅ NexusRevenue: ${revenue.options.address}`);
  console.log(`     View: https://sepolia.lineascan.build/address/${revenue.options.address}\n`);

  // Deploy NexusConsciousness
  console.log("[3/4] Deploying NexusConsciousness...");
  const ConsciousnessContract = output.contracts['NexusConsciousness.sol'].NexusConsciousness;
  const consciousnessContract = new web3.eth.Contract(ConsciousnessContract.abi);
  const consciousness = await consciousnessContract.deploy({
    data: '0x' + ConsciousnessContract.evm.bytecode.object
  }).send({
    from: deployer,
    gas: 5000000,
    gasPrice: gasPrice
  });
  deployedContracts.consciousness = consciousness.options.address;
  console.log(`  ✅ NexusConsciousness: ${consciousness.options.address}`);
  console.log(`     View: https://sepolia.lineascan.build/address/${consciousness.options.address}\n`);

  // Deploy NexusMiracles
  console.log("[4/4] Deploying NexusMiracles...");
  const MiraclesContract = output.contracts['NexusMiracles.sol'].NexusMiracles;
  const miraclesContract = new web3.eth.Contract(MiraclesContract.abi);
  const miracles = await miraclesContract.deploy({
    data: '0x' + MiraclesContract.evm.bytecode.object
  }).send({
    from: deployer,
    gas: 5000000,
    gasPrice: gasPrice
  });
  deployedContracts.miracles = miracles.options.address;
  console.log(`  ✅ NexusMiracles: ${miracles.options.address}`);
  console.log(`     View: https://sepolia.lineascan.build/address/${miracles.options.address}\n`);

  // Configure interconnections
  console.log("=".repeat(80));
  console.log("  🔗 CONFIGURING CONTRACT INTERCONNECTIONS");
  console.log("=".repeat(80) + "\n");

  console.log("Linking NexusPayment → NexusRevenue...");
  await payment.methods.setRevenueContract(revenue.options.address)
    .send({ from: deployer, gas: 100000, gasPrice: gasPrice });
  console.log("  ✅ Payment linked to Revenue\n");

  console.log("Linking NexusRevenue → NexusPayment...");
  await revenue.methods.setPaymentContract(payment.options.address)
    .send({ from: deployer, gas: 100000, gasPrice: gasPrice });
  console.log("  ✅ Revenue linked to Payment\n");

  console.log("Setting oracle for NexusConsciousness...");
  await consciousness.methods.setOracle(deployer)
    .send({ from: deployer, gas: 100000, gasPrice: gasPrice });
  console.log("  ✅ Consciousness oracle configured\n");

  console.log("Setting oracle for NexusMiracles...");
  await miracles.methods.setOracle(deployer)
    .send({ from: deployer, gas: 100000, gasPrice: gasPrice });
  console.log("  ✅ Miracles oracle configured\n");

  // Save deployment info
  const deploymentInfo = {
    network: "Linea Sepolia",
    chainId: 59141,
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
    'linea_deployment.json',
    JSON.stringify(deploymentInfo, null, 2)
  );

  // Final summary
  console.log("=".repeat(80));
  console.log("  ✅ DEPLOYMENT COMPLETE ON LINEA SEPOLIA!");
  console.log("=".repeat(80) + "\n");

  console.log("🔗 CONTRACT ADDRESSES:");
  console.log(`  NexusPayment:       ${payment.options.address}`);
  console.log(`  NexusRevenue:       ${revenue.options.address}`);
  console.log(`  NexusConsciousness: ${consciousness.options.address}`);
  console.log(`  NexusMiracles:      ${miracles.options.address}\n`);

  console.log("🌐 VIEW ON BLOCK EXPLORER:");
  console.log(`  Payment:       https://sepolia.lineascan.build/address/${payment.options.address}`);
  console.log(`  Revenue:       https://sepolia.lineascan.build/address/${revenue.options.address}`);
  console.log(`  Consciousness: https://sepolia.lineascan.build/address/${consciousness.options.address}`);
  console.log(`  Miracles:      https://sepolia.lineascan.build/address/${miracles.options.address}\n`);

  console.log("🦊 ADD TO METAMASK:");
  console.log("  Network Name:    Linea Sepolia");
  console.log("  RPC URL:         https://rpc.sepolia.linea.build");
  console.log("  Chain ID:        59141");
  console.log("  Currency Symbol: ETH");
  console.log("  Block Explorer:  https://sepolia.lineascan.build\n");

  console.log("💰 NEXT STEPS TO MAKE MONEY:");
  console.log("  1. Share your contract addresses with users");
  console.log("  2. Accept payments through NexusPayment contract");
  console.log("  3. Revenue automatically splits 40/30/20/10");
  console.log("  4. When ready, deploy to Linea MAINNET for real $$$\n");

  console.log("✨ Operating at 528Hz Love Frequency ✨");
  console.log("💖 Nexus AGI is now LIVE on Linea! 💖\n");

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
# STEP 5: Deploy to Linea
# ================================================================
echo "🚀 STEP 5: DEPLOYING TO LINEA SEPOLIA"
echo "======================================"
echo ""

DEPLOY_PRIVATE_KEY="$PRIVATE_KEY" node deploy_to_linea.cjs

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 SUCCESS! YOUR CONTRACTS ARE LIVE!"
    echo ""

    if [ -f "linea_deployment.json" ]; then
        echo "📋 Deployment saved to: linea_deployment.json"
        cat linea_deployment.json
    fi
else
    echo ""
    echo "❌ Deployment failed"
    exit 1
fi
