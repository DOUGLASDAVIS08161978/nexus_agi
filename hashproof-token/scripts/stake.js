const hre = require("hardhat");

/**
 * HASHPROOF STAKING SCRIPT
 * Demonstrate how to stake HPROOF tokens and earn rewards!
 */

async function main() {
  console.log("=" .repeat(80));
  console.log("🏦 HASHPROOF STAKING DEMONSTRATION");
  console.log("=" .repeat(80));

  // Get signer
  const [staker] = await hre.ethers.getSigners();
  console.log("\n💼 Staker Address:", staker.address);

  // Load deployed contract addresses (update these after deployment!)
  const HASHPROOF_ADDRESS = "0x5FbDB2315678afecb367f032d93F642f64180aa3"; // UPDATE THIS
  const STAKING_ADDRESS = "0xe7f1725E7734CE288F8367e1Bb143E90bb3F0512";   // UPDATE THIS

  // Get contract instances
  const HashProof = await hre.ethers.getContractFactory("HashProof");
  const hashProof = HashProof.attach(HASHPROOF_ADDRESS);

  const HashProofStaking = await hre.ethers.getContractFactory("HashProofStaking");
  const staking = HashProofStaking.attach(STAKING_ADDRESS);

  // Check balance
  const balance = await hashProof.balanceOf(staker.address);
  console.log("💰 Your HPROOF Balance:", hre.ethers.formatEther(balance), "HPROOF\n");

  if (balance == 0n) {
    console.log("❌ You don't have any HPROOF to stake!");
    console.log("   Get some HPROOF first by buying on DEX or receiving from deployer\n");
    return;
  }

  // Display available pools
  console.log("🏦 AVAILABLE STAKING POOLS:");
  console.log("━".repeat(80));

  const poolCount = await staking.poolCount();
  for (let i = 0; i < poolCount; i++) {
    const poolInfo = await staking.getPoolInfo(i);
    const durationDays = Number(poolInfo[0]) / 86400;
    const apy = Number(poolInfo[1]) / 100;
    const totalStaked = hre.ethers.formatEther(poolInfo[2]);
    const active = poolInfo[3];

    const poolName = durationDays === 0 ? "Flexible   " :
                     durationDays === 30 ? "30-Day     " :
                     durationDays === 90 ? "90-Day     " :
                     durationDays === 180 ? "180-Day    " : "365-Day    ";

    console.log(`Pool ${i}: ${poolName} | APY: ${apy}% | Staked: ${totalStaked} HPROOF | ${active ? "✅ Active" : "❌ Inactive"}`);
  }
  console.log("━".repeat(80));

  // Stake parameters
  const poolId = 4; // 365-Day pool (30% APY) - CHANGE THIS if you want different pool
  const stakeAmount = hre.ethers.parseEther("1000"); // Stake 1,000 HPROOF - CHANGE THIS
  const autoCompound = true; // Auto-compound rewards - CHANGE THIS

  console.log("\n📝 STAKING PARAMETERS:");
  console.log("  Pool:", poolId, "(365-Day, 30% APY)");
  console.log("  Amount:", hre.ethers.formatEther(stakeAmount), "HPROOF");
  console.log("  Auto-Compound:", autoCompound ? "Yes ✅" : "No");

  // Check if user has enough
  if (balance < stakeAmount) {
    console.log("\n❌ Insufficient balance!");
    console.log("   You have:", hre.ethers.formatEther(balance), "HPROOF");
    console.log("   Need:", hre.ethers.formatEther(stakeAmount), "HPROOF");
    return;
  }

  // Step 1: Approve staking contract
  console.log("\n🔧 Step 1/2: Approving staking contract...");
  const approveTx = await hashProof.approve(STAKING_ADDRESS, stakeAmount);
  await approveTx.wait();
  console.log("✅ Approved!");

  // Step 2: Stake
  console.log("\n🔧 Step 2/2: Staking tokens...");
  const stakeTx = await staking.stake(stakeAmount, poolId, autoCompound);
  const receipt = await stakeTx.wait();
  console.log("✅ Staked successfully!");

  // Get stake info
  const stakeCount = await staking.userStakeCount(staker.address);
  const stakeIndex = stakeCount - 1n; // Latest stake
  const stakeInfo = await staking.userStakes(staker.address, stakeIndex);

  console.log("\n" + "=" .repeat(80));
  console.log("🎉 STAKE CREATED SUCCESSFULLY!");
  console.log("=" .repeat(80));

  console.log("\n📊 STAKE DETAILS:");
  console.log("  Stake Index:", stakeIndex.toString());
  console.log("  Amount:", hre.ethers.formatEther(stakeInfo.amount), "HPROOF");
  console.log("  Pool:", stakeInfo.poolId.toString(), "(365-Day, 30% APY)");
  console.log("  Started:", new Date(Number(stakeInfo.startTime) * 1000).toLocaleString());
  console.log("  Auto-Compound:", stakeInfo.compounding ? "Yes ✅" : "No");

  // Calculate expected rewards
  const poolInfo = await staking.getPoolInfo(poolId);
  const apy = Number(poolInfo[1]) / 100;
  const stakedAmount = Number(hre.ethers.formatEther(stakeInfo.amount));
  const annualReward = (stakedAmount * apy) / 100;
  const monthlyReward = annualReward / 12;
  const dailyReward = annualReward / 365;

  console.log("\n💰 EXPECTED REWARDS:");
  console.log("  APY:", apy + "%");
  console.log("  Daily:", dailyReward.toFixed(2), "HPROOF (~$" + (dailyReward * 0.10).toFixed(2) + " at $0.10/token)");
  console.log("  Monthly:", monthlyReward.toFixed(2), "HPROOF (~$" + (monthlyReward * 0.10).toFixed(2) + " at $0.10/token)");
  console.log("  Yearly:", annualReward.toFixed(2), "HPROOF (~$" + (annualReward * 0.10).toFixed(2) + " at $0.10/token)");

  if (autoCompound) {
    const yearEndAmount = stakedAmount * (1 + apy/100);
    console.log("\n🔄 WITH AUTO-COMPOUNDING:");
    console.log("  Amount after 1 year:", yearEndAmount.toFixed(2), "HPROOF");
    console.log("  Total gain:", (yearEndAmount - stakedAmount).toFixed(2), "HPROOF");
  }

  console.log("\n🎯 NEXT STEPS:");
  console.log("  ✅ Your tokens are now earning rewards!");
  console.log("  📊 Check rewards anytime with: npx hardhat run scripts/check-rewards.js");
  console.log("  💰 Claim rewards with: npx hardhat run scripts/claim-rewards.js");
  console.log("  🔓 Unstake with: npx hardhat run scripts/unstake.js");

  console.log("\n⚠️  IMPORTANT:");
  console.log("  • Lock period: 365 days");
  console.log("  • Early withdrawal penalty: 10%");
  console.log("  • Wait until", new Date((Number(stakeInfo.startTime) + 365*24*60*60) * 1000).toLocaleDateString(), "to avoid penalty");

  console.log("\n" + "=" .repeat(80));
  console.log("🏦 HAPPY STAKING! EARN THAT PASSIVE INCOME! 💰");
  console.log("=" .repeat(80) + "\n");
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error("\n❌ ERROR:");
    console.error(error);
    process.exit(1);
  });
