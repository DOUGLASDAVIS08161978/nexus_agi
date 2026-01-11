const hre = require("hardhat");

async function main() {
  console.log("=" .repeat(80));
  console.log("DEPLOYING HASHPROOF TOKEN ECOSYSTEM");
  console.log("=" .repeat(80));

  const [deployer] = await hre.ethers.getSigners();
  console.log("\nDeploying contracts with account:", deployer.address);

  const balance = await hre.ethers.provider.getBalance(deployer.address);
  console.log("Account balance:", hre.ethers.formatEther(balance), "ETH\n");

  // Deploy HashProof Token
  console.log("📝 Deploying HashProof Token...");
  const HashProof = await hre.ethers.getContractFactory("HashProof");
  const hashProof = await HashProof.deploy(deployer.address);
  await hashProof.waitForDeployment();
  const hashProofAddress = await hashProof.getAddress();

  console.log("✅ HashProof Token deployed to:", hashProofAddress);
  console.log("   Initial Supply:", hre.ethers.formatEther(await hashProof.totalSupply()), "HPROOF");
  console.log("   Max Supply:", hre.ethers.formatEther(await hashProof.MAX_SUPPLY()), "HPROOF\n");

  // Deploy Staking Contract
  console.log("📝 Deploying HashProofStaking...");
  const HashProofStaking = await hre.ethers.getContractFactory("HashProofStaking");
  const staking = await HashProofStaking.deploy(hashProofAddress, deployer.address);
  await staking.waitForDeployment();
  const stakingAddress = await staking.getAddress();

  console.log("✅ HashProofStaking deployed to:", stakingAddress);
  console.log("   Pool Count:", (await staking.poolCount()).toString());
  console.log("   Min Stake:", hre.ethers.formatEther(await staking.minStakeAmount()), "HPROOF\n");

  // Deploy Governance Contract
  console.log("📝 Deploying HashProofGovernance...");
  const HashProofGovernance = await hre.ethers.getContractFactory("HashProofGovernance");
  const governance = await HashProofGovernance.deploy(hashProofAddress, deployer.address);
  await governance.waitForDeployment();
  const governanceAddress = await governance.getAddress();

  console.log("✅ HashProofGovernance deployed to:", governanceAddress);
  console.log("   Voting Period:", (await governance.votingPeriod()).toString(), "seconds");
  console.log("   Proposal Threshold:", hre.ethers.formatEther(await governance.proposalThreshold()), "HPROOF\n");

  // Authorize staking contract to mint rewards
  console.log("🔧 Authorizing Staking contract to mint rewards...");
  const tx = await hashProof.addAuthorizedMinter(stakingAddress);
  await tx.wait();
  console.log("✅ Staking contract authorized\n");

  // Print summary
  console.log("=" .repeat(80));
  console.log("DEPLOYMENT COMPLETE!");
  console.log("=" .repeat(80));
  console.log("\n📋 CONTRACT ADDRESSES:");
  console.log("   HashProof Token:     ", hashProofAddress);
  console.log("   HashProofStaking:    ", stakingAddress);
  console.log("   HashProofGovernance: ", governanceAddress);

  console.log("\n💰 TOKENOMICS:");
  console.log("   Total Supply:        ", hre.ethers.formatEther(await hashProof.totalSupply()), "HPROOF");
  console.log("   Max Supply:          ", hre.ethers.formatEther(await hashProof.MAX_SUPPLY()), "HPROOF");
  console.log("   Mining Rewards Pool: ", hre.ethers.formatEther(await hashProof.miningRewardsPool()), "HPROOF");
  console.log("   Staking Rewards Pool:", hre.ethers.formatEther(await hashProof.stakingRewardsPool()), "HPROOF");
  console.log("   Team Pool:           ", hre.ethers.formatEther(await hashProof.teamPool()), "HPROOF");

  console.log("\n📊 STAKING POOLS:");
  for (let i = 0; i < 5; i++) {
    const poolInfo = await staking.getPoolInfo(i);
    const durationDays = Number(poolInfo[0]) / 86400;
    const apy = Number(poolInfo[1]) / 100;
    console.log(`   Pool ${i}: ${durationDays === 0 ? "Flexible" : durationDays + " days"} - ${apy}% APY`);
  }

  console.log("\n🔥 BURN MECHANISM:");
  console.log("   Burn Rate: 1% on every transfer");
  console.log("   Total Burned:", hre.ethers.formatEther(await hashProof.totalBurned()), "HPROOF");

  console.log("\n🎯 YOUR ALLOCATION:");
  console.log("   Your Address:", deployer.address);
  console.log("   Your Balance:", hre.ethers.formatEther(await hashProof.balanceOf(deployer.address)), "HPROOF");

  console.log("\n" + "=" .repeat(80));
  console.log("🚀 READY TO LAUNCH!");
  console.log("=" .repeat(80) + "\n");

  // Save deployment info
  const deploymentInfo = {
    network: hre.network.name,
    deployer: deployer.address,
    contracts: {
      HashProof: hashProofAddress,
      HashProofStaking: stakingAddress,
      HashProofGovernance: governanceAddress
    },
    timestamp: new Date().toISOString()
  };

  const fs = require("fs");
  fs.writeFileSync(
    "deployment-info.json",
    JSON.stringify(deploymentInfo, null, 2)
  );
  console.log("📄 Deployment info saved to deployment-info.json\n");
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });
