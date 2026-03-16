const { ethers } = require("hardhat");
require("dotenv").config();

async function main() {
  console.log("=".repeat(60));
  console.log("🚀 DEPLOYING WRAPPED TESTNET BITCOIN (wTBTC) 🚀");
  console.log("=".repeat(60));
  console.log();

  // Get deployer account
  const [deployer] = await ethers.getSigners();
  const bridgeOperator = process.env.BRIDGE_OPERATOR_ADDRESS || deployer.address;

  console.log("📋 Deployment Information:");
  console.log("-".repeat(60));
  console.log("Deployer address:", deployer.address);
  console.log("Bridge operator:", bridgeOperator);

  // Check deployer balance
  const balance = await ethers.provider.getBalance(deployer.address);
  console.log("Deployer balance:", ethers.formatEther(balance), "ETH");
  console.log();

  if (balance < ethers.parseEther("0.01")) {
    console.log("⚠️  WARNING: Low balance. You may need more ETH for deployment.");
    console.log();
  }

  // Deploy contract
  console.log("📦 Deploying WrappedTestnetBTC contract...");
  console.log();

  const WrappedTestnetBTC = await ethers.getContractFactory("WrappedTestnetBTC");
  const wTBTC = await WrappedTestnetBTC.deploy(bridgeOperator);

  console.log("⏳ Waiting for deployment transaction to be mined...");
  await wTBTC.waitForDeployment();

  const contractAddress = await wTBTC.getAddress();

  console.log();
  console.log("=".repeat(60));
  console.log("✅ DEPLOYMENT SUCCESSFUL!");
  console.log("=".repeat(60));
  console.log();
  console.log("📍 Contract Address:", contractAddress);
  console.log("👤 Bridge Operator:", bridgeOperator);
  console.log();

  // Verify contract details
  console.log("📊 Contract Details:");
  console.log("-".repeat(60));
  const name = await wTBTC.name();
  const symbol = await wTBTC.symbol();
  const decimals = await wTBTC.decimals();
  const totalSupply = await wTBTC.totalSupply();
  const paused = await wTBTC.paused();

  console.log("Name:", name);
  console.log("Symbol:", symbol);
  console.log("Decimals:", decimals);
  console.log("Total Supply:", ethers.formatEther(totalSupply), symbol);
  console.log("Paused:", paused);
  console.log();

  // Verification command
  console.log("=".repeat(60));
  console.log("🔍 ETHERSCAN VERIFICATION");
  console.log("=".repeat(60));
  console.log();
  console.log("Run this command to verify on Etherscan:");
  console.log();
  console.log(`npx hardhat verify --network ${network.name} ${contractAddress} ${bridgeOperator}`);
  console.log();

  // Save deployment info
  const deploymentInfo = {
    network: network.name,
    contractAddress: contractAddress,
    bridgeOperator: bridgeOperator,
    deployer: deployer.address,
    deploymentTime: new Date().toISOString(),
    contractDetails: {
      name: name,
      symbol: symbol,
      decimals: decimals.toString(),
    }
  };

  const fs = require("fs");
  const deploymentFile = `./deployments/${network.name}-deployment.json`;

  // Create deployments directory if it doesn't exist
  if (!fs.existsSync("./deployments")) {
    fs.mkdirSync("./deployments");
  }

  fs.writeFileSync(
    deploymentFile,
    JSON.stringify(deploymentInfo, null, 2)
  );

  console.log("=".repeat(60));
  console.log("💾 DEPLOYMENT INFO SAVED");
  console.log("=".repeat(60));
  console.log();
  console.log("Saved to:", deploymentFile);
  console.log();

  // Next steps
  console.log("=".repeat(60));
  console.log("📋 NEXT STEPS");
  console.log("=".repeat(60));
  console.log();
  console.log("1. Verify contract on Etherscan (command above)");
  console.log("2. Test minting: await wTBTC.mint(address, amount, 'btc_tx_id')");
  console.log("3. Set up Bitcoin monitoring for bridge operations");
  console.log("4. Configure multisig for bridge operator (recommended)");
  console.log("5. Test burning and full bridge workflow");
  console.log();
  console.log("=".repeat(60));
  console.log("🎉 DEPLOYMENT COMPLETE!");
  console.log("=".repeat(60));
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error();
    console.error("❌ DEPLOYMENT FAILED");
    console.error("=".repeat(60));
    console.error(error);
    process.exit(1);
  });
