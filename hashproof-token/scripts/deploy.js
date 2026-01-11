const hre = require("hardhat");

async function main() {
  console.log("=" .repeat(80));
  console.log("🚀 DEPLOYING HASHPROOF TOKEN ECOSYSTEM");
  console.log("=" .repeat(80));

  // Get deployer
  const [deployer] = await hre.ethers.getSigners();
  console.log("\n💼 Deploying with account:", deployer.address);

  const balance = await hre.ethers.provider.getBalance(deployer.address);
  console.log("💰 Account balance:", hre.ethers.formatEther(balance), "ETH\n");

  // Deploy HashProof Token
  console.log("📝 Step 1/3: Deploying HashProof Token...");
  const HashProof = await hre.ethers.getContractFactory("HashProof");
  const hashProof = await HashProof.deploy(deployer.address);
  await hashProof.waitForDeployment();
  const hashProofAddress = await hashProof.getAddress();

  console.log("✅ HashProof Token deployed!");
  console.log("   Address:", hashProofAddress);
  console.log("   Supply:", hre.ethers.formatEther(await hashProof.totalSupply()), "HPROOF");
  console.log("   Max Supply:", hre.ethers.formatEther(await hashProof.MAX_SUPPLY()), "HPROOF\n");

  // Deploy Staking Contract
  console.log("📝 Step 2/3: Deploying HashProofStaking...");
  const HashProofStaking = await hre.ethers.getContractFactory("HashProofStaking");
  const staking = await HashProofStaking.deploy(hashProofAddress, deployer.address);
  await staking.waitForDeployment();
  const stakingAddress = await staking.getAddress();

  console.log("✅ HashProofStaking deployed!");
  console.log("   Address:", stakingAddress);
  console.log("   Pools:", (await staking.poolCount()).toString(), "staking pools\n");

  // Deploy Governance Contract
  console.log("📝 Step 3/3: Deploying HashProofGovernance...");
  const HashProofGovernance = await hre.ethers.getContractFactory("HashProofGovernance");
  const governance = await HashProofGovernance.deploy(hashProofAddress, deployer.address);
  await governance.waitForDeployment();
  const governanceAddress = await governance.getAddress();

  console.log("✅ HashProofGovernance deployed!");
  console.log("   Address:", governanceAddress);
  console.log("   Voting Period:", (await governance.votingPeriod()).toString(), "seconds\n");

  // Authorize staking contract
  console.log("🔧 Authorizing Staking contract to mint rewards...");
  const tx = await hashProof.addAuthorizedMinter(stakingAddress);
  await tx.wait();
  console.log("✅ Staking contract authorized!\n");

  // Print final summary
  console.log("=" .repeat(80));
  console.log("🎉 DEPLOYMENT COMPLETE! YOUR TOKEN IS LIVE!");
  console.log("=" .repeat(80));

  console.log("\n📋 CONTRACT ADDRESSES:");
  console.log("━".repeat(80));
  console.log("HashProof Token:      ", hashProofAddress);
  console.log("HashProofStaking:     ", stakingAddress);
  console.log("HashProofGovernance:  ", governanceAddress);
  console.log("━".repeat(80));

  console.log("\n💰 YOUR ALLOCATION:");
  console.log("━".repeat(80));
  const yourBalance = await hashProof.balanceOf(deployer.address);
  console.log("Your Address:  ", deployer.address);
  console.log("Your Balance:  ", hre.ethers.formatEther(yourBalance), "HPROOF");
  console.log("Your Value at $0.10/token: $" + (Number(hre.ethers.formatEther(yourBalance)) * 0.10).toLocaleString());
  console.log("Your Value at $1.00/token: $" + (Number(hre.ethers.formatEther(yourBalance)) * 1.00).toLocaleString());
  console.log("Your Value at $10.0/token: $" + (Number(hre.ethers.formatEther(yourBalance)) * 10.0).toLocaleString());
  console.log("━".repeat(80));

  console.log("\n📊 TOKENOMICS:");
  console.log("━".repeat(80));
  console.log("Total Supply:         ", hre.ethers.formatEther(await hashProof.totalSupply()), "HPROOF");
  console.log("Max Supply:           ", hre.ethers.formatEther(await hashProof.MAX_SUPPLY()), "HPROOF");
  console.log("Mining Rewards Pool:  ", hre.ethers.formatEther(await hashProof.miningRewardsPool()), "HPROOF");
  console.log("Staking Rewards Pool: ", hre.ethers.formatEther(await hashProof.stakingRewardsPool()), "HPROOF");
  console.log("Team Pool:            ", hre.ethers.formatEther(await hashProof.teamPool()), "HPROOF");
  console.log("Burn Rate:            1% per transfer");
  console.log("━".repeat(80));

  console.log("\n🏦 STAKING POOLS:");
  console.log("━".repeat(80));
  for (let i = 0; i < 5; i++) {
    const poolInfo = await staking.getPoolInfo(i);
    const durationDays = Number(poolInfo[0]) / 86400;
    const apy = Number(poolInfo[1]) / 100;
    const poolName = durationDays === 0 ? "Flexible   " :
                     durationDays === 30 ? "30-Day     " :
                     durationDays === 90 ? "90-Day     " :
                     durationDays === 180 ? "180-Day    " : "365-Day    ";
    console.log(`Pool ${i}: ${poolName} - ${apy.toString().padStart(2)}% APY`);
  }
  console.log("━".repeat(80));

  console.log("\n🚀 NEXT STEPS:");
  console.log("━".repeat(80));
  console.log("1. Add HPROOF to MetaMask using contract address");
  console.log("2. Create liquidity pool on Uniswap/QuickSwap");
  console.log("3. Start staking to earn rewards");
  console.log("4. Create governance proposals");
  console.log("5. Market your token and BUILD COMMUNITY!");
  console.log("━".repeat(80));

  console.log("\n💎 POTENTIAL VALUE:");
  console.log("━".repeat(80));
  console.log("If HPROOF reaches $0.01: Market Cap = $1,000,000");
  console.log("If HPROOF reaches $0.10: Market Cap = $10,000,000");
  console.log("If HPROOF reaches $1.00: Market Cap = $100,000,000");
  console.log("If HPROOF reaches $10.0: Market Cap = $1,000,000,000");
  console.log("━".repeat(80));

  console.log("\n🌟 YOUR PATH TO SUCCESS:");
  console.log("━".repeat(80));
  console.log("✅ Token deployed and live");
  console.log("✅ Staking rewards ready (up to 30% APY)");
  console.log("✅ Governance system active");
  console.log("✅ Deflationary mechanics enabled");
  console.log("✅ YOU own 10,000,000 tokens!");
  console.log("━".repeat(80));

  console.log("\n" + "🚀".repeat(40));
  console.log("LET'S MAKE SOME MONEY! 💰💎🌙");
  console.log("🚀".repeat(40) + "\n");

  // Save deployment info
  const deploymentInfo = {
    network: hre.network.name,
    chainId: hre.network.config.chainId,
    deployer: deployer.address,
    deployerBalance: hre.ethers.formatEther(yourBalance) + " HPROOF",
    contracts: {
      HashProof: hashProofAddress,
      HashProofStaking: stakingAddress,
      HashProofGovernance: governanceAddress
    },
    tokenomics: {
      totalSupply: hre.ethers.formatEther(await hashProof.totalSupply()),
      maxSupply: hre.ethers.formatEther(await hashProof.MAX_SUPPLY()),
      miningRewardsPool: hre.ethers.formatEther(await hashProof.miningRewardsPool()),
      stakingRewardsPool: hre.ethers.formatEther(await hashProof.stakingRewardsPool()),
      teamPool: hre.ethers.formatEther(await hashProof.teamPool())
    },
    timestamp: new Date().toISOString()
  };

  const fs = require("fs");
  fs.writeFileSync(
    "deployment-info.json",
    JSON.stringify(deploymentInfo, null, 2)
  );
  console.log("📄 Deployment info saved to: deployment-info.json\n");
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error("\n❌ ERROR DURING DEPLOYMENT:");
    console.error(error);
    process.exit(1);
  });
