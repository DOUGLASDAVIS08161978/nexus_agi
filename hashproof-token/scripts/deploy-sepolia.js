const hre = require("hardhat");

async function main() {
  console.log("=" .repeat(80));
  console.log("DEPLOYING TO SEPOLIA TESTNET");
  console.log("=" .repeat(80));

  const [deployer] = await hre.ethers.getSigners();
  console.log("\nDeploying contracts with account:", deployer.address);

  const balance = await hre.ethers.provider.getBalance(deployer.address);
  console.log("Account balance:", hre.ethers.formatEther(balance), "ETH\n");

  if (balance === 0n) {
    console.log("⚠️  WARNING: Account has 0 ETH!");
    console.log("Get free Sepolia ETH from faucets:");
    console.log("  - https://sepoliafaucet.com");
    console.log("  - https://www.alchemy.com/faucets/ethereum-sepolia");
    console.log("  - https://faucet.quicknode.com/ethereum/sepolia\n");
    return;
  }

  const deployedContracts = {};

  // 1. Deploy Wrapped Testnet BTC
  console.log("📝 Deploying Wrapped Testnet BTC (wTBTC)...");
  try {
    const WrappedTestnetBTC = await hre.ethers.getContractFactory("WrappedTestnetBTC");
    const wTBTC = await WrappedTestnetBTC.deploy(deployer.address);
    await wTBTC.waitForDeployment();
    const wTBTCAddress = await wTBTC.getAddress();

    console.log("✅ wTBTC deployed to:", wTBTCAddress);
    console.log("   Bridge Operator:", deployer.address);
    console.log("   Total Supply:", hre.ethers.formatEther(await wTBTC.totalSupply()), "wTBTC\n");

    deployedContracts.wTBTC = wTBTCAddress;
  } catch (error) {
    console.log("❌ wTBTC deployment failed:", error.message, "\n");
  }

  // 2. Deploy HashProof Token
  console.log("📝 Deploying HashProof Token...");
  try {
    const HashProof = await hre.ethers.getContractFactory("HashProof");
    const hashProof = await HashProof.deploy(deployer.address);
    await hashProof.waitForDeployment();
    const hashProofAddress = await hashProof.getAddress();

    console.log("✅ HashProof Token deployed to:", hashProofAddress);
    console.log("   Initial Supply:", hre.ethers.formatEther(await hashProof.totalSupply()), "HPROOF");
    console.log("   Max Supply:", hre.ethers.formatEther(await hashProof.MAX_SUPPLY()), "HPROOF\n");

    deployedContracts.HashProof = hashProofAddress;

    // 3. Deploy Staking Contract
    console.log("📝 Deploying HashProofStaking...");
    const HashProofStaking = await hre.ethers.getContractFactory("HashProofStaking");
    const staking = await HashProofStaking.deploy(hashProofAddress, deployer.address);
    await staking.waitForDeployment();
    const stakingAddress = await staking.getAddress();

    console.log("✅ HashProofStaking deployed to:", stakingAddress);
    console.log("   Pool Count:", (await staking.poolCount()).toString(), "\n");

    deployedContracts.HashProofStaking = stakingAddress;

    // 4. Deploy Governance Contract
    console.log("📝 Deploying HashProofGovernance...");
    const HashProofGovernance = await hre.ethers.getContractFactory("HashProofGovernance");
    const governance = await HashProofGovernance.deploy(hashProofAddress, deployer.address);
    await governance.waitForDeployment();
    const governanceAddress = await governance.getAddress();

    console.log("✅ HashProofGovernance deployed to:", governanceAddress, "\n");

    deployedContracts.HashProofGovernance = governanceAddress;

    // 5. Authorize staking contract to mint rewards
    console.log("🔧 Authorizing Staking contract to mint rewards...");
    const tx = await hashProof.addAuthorizedMinter(stakingAddress);
    await tx.wait();
    console.log("✅ Staking contract authorized\n");

  } catch (error) {
    console.log("❌ HashProof ecosystem deployment failed:", error.message, "\n");
  }

  // Print summary
  console.log("=" .repeat(80));
  console.log("DEPLOYMENT COMPLETE!");
  console.log("=" .repeat(80));
  console.log("\n🌐 NETWORK: Sepolia Testnet (Chain ID: 11155111)");
  console.log("👤 DEPLOYER:", deployer.address);
  console.log("\n📋 CONTRACT ADDRESSES:");

  for (const [name, address] of Object.entries(deployedContracts)) {
    console.log(`   ${name.padEnd(25)} ${address}`);
  }

  console.log("\n🔍 VERIFY ON ETHERSCAN:");
  console.log("   https://sepolia.etherscan.io");

  for (const [name, address] of Object.entries(deployedContracts)) {
    console.log(`   ${name}: https://sepolia.etherscan.io/address/${address}`);
  }

  console.log("\n💰 GET TEST TOKENS:");
  if (deployedContracts.wTBTC) {
    console.log(`   You can mint wTBTC by calling the mint() function as bridge operator`);
  }
  if (deployedContracts.HashProof) {
    console.log(`   You received HashProof tokens at deployment`);
  }

  console.log("\n" + "=" .repeat(80));
  console.log("🚀 READY ON SEPOLIA TESTNET!");
  console.log("=" .repeat(80) + "\n");

  // Save deployment info
  const deploymentInfo = {
    network: "sepolia",
    chainId: 11155111,
    deployer: deployer.address,
    contracts: deployedContracts,
    timestamp: new Date().toISOString(),
    explorerUrl: "https://sepolia.etherscan.io"
  };

  const fs = require("fs");
  fs.writeFileSync(
    "deployment-sepolia.json",
    JSON.stringify(deploymentInfo, null, 2)
  );
  console.log("📄 Deployment info saved to deployment-sepolia.json\n");
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });
