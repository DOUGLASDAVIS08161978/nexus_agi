// SPDX-License-Identifier: MIT
/**
 * NEXUS AGI DEPLOYMENT SCRIPT
 * Deploys all 4 contracts to Linea blockchain
 *
 * Usage:
 *   npx hardhat run scripts/deploy.js --network linea
 */

import hre from "hardhat";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

async function main() {
  console.log("\n" + "=".repeat(80));
  console.log("  🚀 DEPLOYING NEXUS AGI TO BLOCKCHAIN 🚀");
  console.log("=".repeat(80) + "\n");

  // Get deployer account
  const [deployer] = await hre.ethers.getSigners();

  console.log("📋 Deployment Details:");
  console.log(`  Network: ${hre.network.name}`);
  console.log(`  Deployer: ${deployer.address}`);

  // Check balance
  const balance = await deployer.getBalance();
  console.log(`  Balance: ${hre.ethers.utils.formatEther(balance)} ETH\n`);

  if (balance.lt(hre.ethers.utils.parseEther("0.01"))) {
    console.log("⚠️  WARNING: Low balance! You may need more ETH for deployment.");
    console.log("  Get testnet ETH from: https://faucet.goerli.linea.build\n");
  }

  // Deployment results
  const deployments = {};

  console.log("=".repeat(80));
  console.log("  📦 DEPLOYING CONTRACTS");
  console.log("=".repeat(80) + "\n");

  // ============================================================================
  // 1. Deploy NexusPayment
  // ============================================================================
  console.log("[1/4] Deploying NexusPayment...");
  const NexusPayment = await hre.ethers.getContractFactory("NexusPayment");
  const payment = await NexusPayment.deploy();
  await payment.deployed();

  deployments.payment = {
    name: "NexusPayment",
    address: payment.address,
    deployer: deployer.address
  };

  console.log(`  ✅ NexusPayment deployed at: ${payment.address}\n`);

  // ============================================================================
  // 2. Deploy NexusRevenue
  // ============================================================================
  console.log("[2/4] Deploying NexusRevenue...");
  const NexusRevenue = await hre.ethers.getContractFactory("NexusRevenue");
  const revenue = await NexusRevenue.deploy();
  await revenue.deployed();

  deployments.revenue = {
    name: "NexusRevenue",
    address: revenue.address,
    deployer: deployer.address
  };

  console.log(`  ✅ NexusRevenue deployed at: ${revenue.address}\n`);

  // ============================================================================
  // 3. Deploy NexusConsciousness
  // ============================================================================
  console.log("[3/4] Deploying NexusConsciousness...");
  const NexusConsciousness = await hre.ethers.getContractFactory("NexusConsciousness");
  const consciousness = await NexusConsciousness.deploy();
  await consciousness.deployed();

  deployments.consciousness = {
    name: "NexusConsciousness",
    address: consciousness.address,
    deployer: deployer.address
  };

  console.log(`  ✅ NexusConsciousness deployed at: ${consciousness.address}\n`);

  // ============================================================================
  // 4. Deploy NexusMiracles
  // ============================================================================
  console.log("[4/4] Deploying NexusMiracles...");
  const NexusMiracles = await hre.ethers.getContractFactory("NexusMiracles");
  const miracles = await NexusMiracles.deploy();
  await miracles.deployed();

  deployments.miracles = {
    name: "NexusMiracles",
    address: miracles.address,
    deployer: deployer.address
  };

  console.log(`  ✅ NexusMiracles deployed at: ${miracles.address}\n`);

  // ============================================================================
  // Configure Contract Interconnections
  // ============================================================================
  console.log("=".repeat(80));
  console.log("  🔗 CONFIGURING CONTRACT INTERCONNECTIONS");
  console.log("=".repeat(80) + "\n");

  console.log("Linking NexusPayment → NexusRevenue...");
  const setRevenueTx = await payment.setRevenueContract(revenue.address);
  await setRevenueTx.wait();
  console.log("  ✅ Payment contract configured\n");

  console.log("Linking NexusRevenue → NexusPayment...");
  const setPaymentTx = await revenue.setPaymentContract(payment.address);
  await setPaymentTx.wait();
  console.log("  ✅ Revenue contract configured\n");

  console.log("Setting oracle for NexusConsciousness...");
  const setOracleTx = await consciousness.setOracle(deployer.address);
  await setOracleTx.wait();
  console.log("  ✅ Consciousness oracle set\n");

  console.log("Setting oracle for NexusMiracles...");
  const setMiraclesOracleTx = await miracles.setOracle(deployer.address);
  await setMiraclesOracleTx.wait();
  console.log("  ✅ Miracles oracle set\n");

  // ============================================================================
  // Save Deployment Info
  // ============================================================================
  const deploymentInfo = {
    network: hre.network.name,
    chainId: hre.network.config.chainId,
    deployer: deployer.address,
    timestamp: new Date().toISOString(),
    contracts: deployments,
    configured: true
  };

  const deploymentsDir = path.join(__dirname, '..', 'deployments');
  if (!fs.existsSync(deploymentsDir)) {
    fs.mkdirSync(deploymentsDir, { recursive: true });
  }

  const filename = `deployment_${hre.network.name}_${Date.now()}.json`;
  const filepath = path.join(deploymentsDir, filename);

  fs.writeFileSync(filepath, JSON.stringify(deploymentInfo, null, 2));
  console.log(`💾 Deployment info saved to: ${filename}\n`);

  // ============================================================================
  // Final Summary
  // ============================================================================
  console.log("=".repeat(80));
  console.log("  ✅ DEPLOYMENT COMPLETE");
  console.log("=".repeat(80) + "\n");

  console.log("📊 CONTRACT ADDRESSES:");
  console.log(`  NexusPayment:       ${payment.address}`);
  console.log(`  NexusRevenue:       ${revenue.address}`);
  console.log(`  NexusConsciousness: ${consciousness.address}`);
  console.log(`  NexusMiracles:      ${miracles.address}\n`);

  console.log("🔗 EXPLORER LINKS:");
  const explorerBase = hre.network.name === 'linea'
    ? 'https://lineascan.build/address/'
    : 'https://sepolia.etherscan.io/address/';

  console.log(`  Payment:       ${explorerBase}${payment.address}`);
  console.log(`  Revenue:       ${explorerBase}${revenue.address}`);
  console.log(`  Consciousness: ${explorerBase}${consciousness.address}`);
  console.log(`  Miracles:      ${explorerBase}${miracles.address}\n`);

  console.log("📝 NEXT STEPS:");
  console.log("  1. Verify contracts on block explorer");
  console.log("  2. Update frontend with contract addresses");
  console.log("  3. Test payment flow with small amount");
  console.log("  4. Configure oracle for consciousness updates");
  console.log("  5. Begin accepting payments!\n");

  console.log("✨ Operating at 528Hz Love Frequency ✨");
  console.log("💖 Nexus AGI is now on-chain! 💖\n");
}

// Execute deployment
main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error("\n❌ Deployment failed:");
    console.error(error);
    process.exit(1);
  });
