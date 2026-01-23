// Deploy WBTC Token to Monad Testnet
const hre = require("hardhat");

async function main() {
  console.log("🚀 Deploying Wrapped Bitcoin (WBTC) to Monad Testnet...\n");

  // Get deployer
  const [deployer] = await hre.ethers.getSigners();
  console.log("📝 Deploying with account:", deployer.address);

  const balance = await hre.ethers.provider.getBalance(deployer.address);
  console.log("💰 Account balance:", hre.ethers.formatEther(balance), "ETH\n");

  // Deploy WBTC
  console.log("🔨 Deploying WrappedBitcoin contract...");
  const WrappedBitcoin = await hre.ethers.getContractFactory("WrappedBitcoin");
  const wbtc = await WrappedBitcoin.deploy();

  await wbtc.waitForDeployment();
  const wbtcAddress = await wbtc.getAddress();

  console.log("✅ WBTC deployed to:", wbtcAddress);
  console.log("\n" + "=".repeat(80));
  console.log("📋 DEPLOYMENT SUMMARY");
  console.log("=".repeat(80));
  console.log("Contract: Wrapped Bitcoin (WBTC)");
  console.log("Address:", wbtcAddress);
  console.log("Network: Monad Testnet");
  console.log("Owner:", deployer.address);
  console.log("Decimals: 8 (same as Bitcoin)");
  console.log("Symbol: WBTC");
  console.log("=".repeat(80));

  console.log("\n💡 NEXT STEPS:");
  console.log("1. Save this contract address for your bridge");
  console.log("2. Add WBTC token to MetaMask:");
  console.log("   - Token Address:", wbtcAddress);
  console.log("   - Token Symbol: WBTC");
  console.log("   - Decimals: 8");
  console.log("3. Use bridge to mint WBTC tokens");

  // Save deployment info
  const fs = require('fs');
  const deploymentInfo = {
    network: "monad-testnet",
    contractName: "WrappedBitcoin",
    address: wbtcAddress,
    deployer: deployer.address,
    timestamp: new Date().toISOString(),
    abi: "contracts/WrappedBitcoin.sol:WrappedBitcoin"
  };

  fs.writeFileSync(
    'wbtc_deployment.json',
    JSON.stringify(deploymentInfo, null, 2)
  );
  console.log("\n💾 Deployment info saved to: wbtc_deployment.json\n");
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });
