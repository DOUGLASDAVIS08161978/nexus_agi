const hre = require("hardhat");

/**
 * HASHPROOF REWARDS CLAIMER
 * Claim your staking rewards and get paid!
 */

async function main() {
  console.log("=" .repeat(80));
  console.log("💰 HASHPROOF REWARDS CLAIMER");
  console.log("=" .repeat(80));

  // Get signer
  const [user] = await hre.ethers.getSigners();
  console.log("\n💼 Claiming rewards for:", user.address);

  // Load deployed contract addresses (update these after deployment!)
  const HASHPROOF_ADDRESS = "0x5FbDB2315678afecb367f032d93F642f64180aa3"; // UPDATE THIS
  const STAKING_ADDRESS = "0xe7f1725E7734CE288F8367e1Bb143E90bb3F0512";   // UPDATE THIS

  // Get contract instances
  const HashProof = await hre.ethers.getContractFactory("HashProof");
  const hashProof = HashProof.attach(HASHPROOF_ADDRESS);

  const HashProofStaking = await hre.ethers.getContractFactory("HashProofStaking");
  const staking = HashProofStaking.attach(STAKING_ADDRESS);

  // Get balance before claiming
  const balanceBefore = await hashProof.balanceOf(user.address);
  console.log("💰 Balance Before:  ", hre.ethers.formatEther(balanceBefore), "HPROOF");

  // Check how many stakes user has
  const stakeCount = await staking.userStakeCount(user.address);

  if (stakeCount == 0n) {
    console.log("\n❌ You don't have any active stakes!");
    console.log("   Start staking with: npx hardhat run scripts/stake.js\n");
    return;
  }

  console.log("\n🔍 Checking stakes...");
  console.log("━".repeat(80));

  let totalClaimable = 0n;
  let stakesToClaim = [];

  // Loop through all user stakes to find claimable rewards
  for (let i = 0; i < stakeCount; i++) {
    const stakeInfo = await staking.userStakes(user.address, i);

    // Skip if stake is already withdrawn
    if (stakeInfo.withdrawn) {
      continue;
    }

    // Calculate pending rewards
    const pendingRewards = await staking.calculateRewards(user.address, i);

    if (pendingRewards > 0n) {
      totalClaimable += pendingRewards;
      stakesToClaim.push({
        index: i,
        amount: stakeInfo.amount,
        rewards: pendingRewards,
        poolId: stakeInfo.poolId,
        compounding: stakeInfo.compounding
      });

      // Get pool info
      const poolInfo = await staking.getPoolInfo(stakeInfo.poolId);
      const duration = poolInfo[0];
      const apy = Number(poolInfo[1]) / 100;
      const durationDays = Number(duration) / 86400;
      const poolName = durationDays === 0 ? "Flexible" :
                       durationDays === 30 ? "30-Day" :
                       durationDays === 90 ? "90-Day" :
                       durationDays === 180 ? "180-Day" : "365-Day";

      console.log(`Stake #${i}: ${poolName} (${apy}% APY) - ${hre.ethers.formatEther(pendingRewards)} HPROOF rewards`);
    }
  }

  if (totalClaimable == 0n) {
    console.log("\n❌ No rewards to claim yet!");
    console.log("   Rewards accumulate over time. Check back later!");
    console.log("   Use: npx hardhat run scripts/check-rewards.js\n");
    return;
  }

  console.log("━".repeat(80));
  console.log("💎 TOTAL CLAIMABLE:", hre.ethers.formatEther(totalClaimable), "HPROOF");

  // Calculate value
  const claimableValue = Number(hre.ethers.formatEther(totalClaimable));
  console.log("\n💰 VALUE AT DIFFERENT PRICES:");
  console.log("  At $0.10/token:  $" + (claimableValue * 0.10).toFixed(2));
  console.log("  At $0.50/token:  $" + (claimableValue * 0.50).toFixed(2));
  console.log("  At $1.00/token:  $" + (claimableValue * 1.00).toFixed(2));
  console.log("  At $10.0/token:  $" + (claimableValue * 10.0).toFixed(2));

  // Claim rewards for each stake
  console.log("\n🔧 CLAIMING REWARDS...");
  console.log("━".repeat(80));

  let successCount = 0;
  let totalClaimed = 0n;

  for (const stake of stakesToClaim) {
    try {
      console.log(`\nClaiming from Stake #${stake.index}...`);

      // Note: If compounding is enabled, rewards are auto-staked
      if (stake.compounding) {
        console.log("⚠️  Auto-compound enabled - rewards will be added to stake");
      }

      const tx = await staking.claimRewards(stake.index);
      const receipt = await tx.wait();

      console.log("✅ Claimed:", hre.ethers.formatEther(stake.rewards), "HPROOF");
      successCount++;
      totalClaimed += stake.rewards;
    } catch (error) {
      console.log("❌ Failed to claim from Stake #" + stake.index + ":", error.message);
    }
  }

  // Get balance after claiming
  const balanceAfter = await hashProof.balanceOf(user.address);
  const balanceIncrease = balanceAfter - balanceBefore;

  console.log("\n" + "=" .repeat(80));
  console.log("🎉 CLAIMING COMPLETE!");
  console.log("=" .repeat(80));

  console.log("\n📊 SUMMARY:");
  console.log("━".repeat(80));
  console.log("Stakes Claimed:    ", successCount, "/", stakesToClaim.length);
  console.log("Total Claimed:     ", hre.ethers.formatEther(totalClaimed), "HPROOF");
  console.log("Balance Before:    ", hre.ethers.formatEther(balanceBefore), "HPROOF");
  console.log("Balance After:     ", hre.ethers.formatEther(balanceAfter), "HPROOF");
  console.log("Balance Increase:  ", hre.ethers.formatEther(balanceIncrease), "HPROOF");

  if (balanceIncrease > 0n) {
    console.log("\n💵 VALUE GAINED:");
    const gainValue = Number(hre.ethers.formatEther(balanceIncrease));
    console.log("  At $0.10/token:  $" + (gainValue * 0.10).toFixed(2));
    console.log("  At $1.00/token:  $" + (gainValue * 1.00).toFixed(2));
    console.log("  At $10.0/token:  $" + (gainValue * 10.0).toFixed(2));
  }

  // Check if there are compounding stakes
  const compoundingStakes = stakesToClaim.filter(s => s.compounding);
  if (compoundingStakes.length > 0) {
    console.log("\n🔄 AUTO-COMPOUNDING STAKES:");
    console.log("━".repeat(80));
    console.log(compoundingStakes.length, "stake(s) with auto-compound enabled");
    console.log("Rewards were automatically added to your stake amount!");
    console.log("This increases your future reward rate 📈");
  }

  console.log("\n🎯 WHAT'S NEXT:");
  console.log("━".repeat(80));
  console.log("✅ Rewards claimed and in your wallet!");
  console.log("💰 Check your balance in MetaMask");
  console.log("📊 View rewards anytime: npx hardhat run scripts/check-rewards.js");
  console.log("➕ Stake more tokens:   npx hardhat run scripts/stake.js");
  console.log("🔄 Keep earning passive income!");

  console.log("\n💡 PRO TIP:");
  console.log("━".repeat(80));
  console.log("Consider re-staking your rewards for compound growth!");
  console.log("Or enable auto-compound when staking to maximize returns 🚀");

  console.log("\n" + "=" .repeat(80));
  console.log("💎 CONGRATULATIONS! YOU JUST EARNED PASSIVE INCOME! 🎊");
  console.log("=" .repeat(80) + "\n");
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error("\n❌ ERROR:");
    console.error(error);
    process.exit(1);
  });
