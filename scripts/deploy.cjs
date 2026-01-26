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
