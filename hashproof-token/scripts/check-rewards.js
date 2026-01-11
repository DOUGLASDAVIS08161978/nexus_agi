const hre = require("hardhat");

/**
 * HASHPROOF REWARDS CHECKER
 * Check your pending staking rewards without claiming them!
 */

async function main() {
  console.log("=" .repeat(80));
  console.log("💰 HASHPROOF STAKING REWARDS CHECKER");
  console.log("=" .repeat(80));

  // Get signer
  const [user] = await hre.ethers.getSigners();
  console.log("\n💼 Checking rewards for:", user.address);

  // Load deployed contract addresses (update these after deployment!)
  const HASHPROOF_ADDRESS = "0x5FbDB2315678afecb367f032d93F642f64180aa3"; // UPDATE THIS
  const STAKING_ADDRESS = "0xe7f1725E7734CE288F8367e1Bb143E90bb3F0512";   // UPDATE THIS

  // Get contract instances
  const HashProof = await hre.ethers.getContractFactory("HashProof");
  const hashProof = HashProof.attach(HASHPROOF_ADDRESS);

  const HashProofStaking = await hre.ethers.getContractFactory("HashProofStaking");
  const staking = HashProofStaking.attach(STAKING_ADDRESS);

  // Check how many stakes user has
  const stakeCount = await staking.userStakeCount(user.address);

  if (stakeCount == 0n) {
    console.log("\n❌ You don't have any active stakes!");
    console.log("   Start staking with: npx hardhat run scripts/stake.js\n");
    return;
  }

  console.log("\n📊 ACTIVE STAKES:", stakeCount.toString());
  console.log("━".repeat(80));

  let totalRewards = 0n;
  let totalStaked = 0n;
  let totalValue = 0;

  // Loop through all user stakes
  for (let i = 0; i < stakeCount; i++) {
    const stakeInfo = await staking.userStakes(user.address, i);

    // Skip if stake is already withdrawn
    if (stakeInfo.withdrawn) {
      continue;
    }

    const amount = stakeInfo.amount;
    const poolId = stakeInfo.poolId;
    const startTime = stakeInfo.startTime;
    const compounding = stakeInfo.compounding;

    // Get pool info
    const poolInfo = await staking.getPoolInfo(poolId);
    const duration = poolInfo[0];
    const apy = Number(poolInfo[1]) / 100;

    // Calculate pending rewards
    const pendingRewards = await staking.calculateRewards(user.address, i);

    // Accumulate totals
    totalRewards += pendingRewards;
    totalStaked += amount;

    // Format pool name
    const durationDays = Number(duration) / 86400;
    const poolName = durationDays === 0 ? "Flexible" :
                     durationDays === 30 ? "30-Day" :
                     durationDays === 90 ? "90-Day" :
                     durationDays === 180 ? "180-Day" : "365-Day";

    // Calculate stake age
    const now = Math.floor(Date.now() / 1000);
    const stakeAge = now - Number(startTime);
    const daysStaked = Math.floor(stakeAge / 86400);
    const hoursStaked = Math.floor((stakeAge % 86400) / 3600);

    // Check if lock period expired
    const unlockTime = Number(startTime) + Number(duration);
    const isLocked = now < unlockTime;
    const timeUntilUnlock = unlockTime - now;
    const daysUntilUnlock = Math.floor(timeUntilUnlock / 86400);

    console.log(`\n📌 STAKE #${i}:`);
    console.log("━".repeat(80));
    console.log("  Pool:              ", poolName, `(${apy}% APY)`);
    console.log("  Staked Amount:     ", hre.ethers.formatEther(amount), "HPROOF");
    console.log("  Pending Rewards:   ", hre.ethers.formatEther(pendingRewards), "HPROOF");
    console.log("  Auto-Compound:     ", compounding ? "Yes ✅" : "No");
    console.log("  Stake Age:         ", `${daysStaked} days, ${hoursStaked} hours`);
    console.log("  Started:           ", new Date(Number(startTime) * 1000).toLocaleString());

    if (isLocked) {
      console.log("  Status:            ", "🔒 LOCKED");
      console.log("  Unlocks in:        ", daysUntilUnlock, "days");
      console.log("  Unlock Date:       ", new Date(unlockTime * 1000).toLocaleDateString());
      console.log("  Early Withdrawal:  ", "⚠️  10% penalty applies");
    } else {
      console.log("  Status:            ", "✅ UNLOCKED");
      console.log("  Unlocked Since:    ", new Date(unlockTime * 1000).toLocaleDateString());
      console.log("  Early Withdrawal:  ", "No penalty");
    }

    // Calculate reward rate
    const rewardValue = Number(hre.ethers.formatEther(pendingRewards));
    const dailyRate = (Number(hre.ethers.formatEther(amount)) * apy) / 100 / 365;

    console.log("\n  💵 Reward Rates:");
    console.log("     Daily:          ", dailyRate.toFixed(4), "HPROOF");
    console.log("     Weekly:         ", (dailyRate * 7).toFixed(4), "HPROOF");
    console.log("     Monthly:        ", (dailyRate * 30).toFixed(2), "HPROOF");
    console.log("     Yearly (est):   ", (dailyRate * 365).toFixed(2), "HPROOF");

    totalValue += rewardValue;
  }

  // Print summary
  console.log("\n" + "=" .repeat(80));
  console.log("📊 TOTAL SUMMARY");
  console.log("=" .repeat(80));
  console.log("Total Staked:      ", hre.ethers.formatEther(totalStaked), "HPROOF");
  console.log("Total Rewards:     ", hre.ethers.formatEther(totalRewards), "HPROOF");
  console.log("\n💰 REWARD VALUE (at different prices):");
  console.log("  At $0.10/token:  $" + (totalValue * 0.10).toFixed(2));
  console.log("  At $0.50/token:  $" + (totalValue * 0.50).toFixed(2));
  console.log("  At $1.00/token:  $" + (totalValue * 1.00).toFixed(2));
  console.log("  At $10.0/token:  $" + (totalValue * 10.0).toFixed(2));

  // Check HPROOF balance
  const balance = await hashProof.balanceOf(user.address);
  console.log("\n💼 Your Current Balance:", hre.ethers.formatEther(balance), "HPROOF");

  // Get total stats
  const totalSupply = await hashProof.totalSupply();
  const yourShare = (Number(hre.ethers.formatEther(balance)) / Number(hre.ethers.formatEther(totalSupply))) * 100;
  console.log("Your Share of Supply:  ", yourShare.toFixed(4) + "%");

  console.log("\n🎯 NEXT ACTIONS:");
  console.log("━".repeat(80));
  if (totalRewards > 0n) {
    console.log("✅ You have", hre.ethers.formatEther(totalRewards), "HPROOF in pending rewards!");
    console.log("💰 Claim rewards:  npx hardhat run scripts/claim-rewards.js");
  } else {
    console.log("⏰ Your rewards are accumulating...");
    console.log("   Check back later for more rewards!");
  }
  console.log("🔓 Unstake tokens: npx hardhat run scripts/unstake.js");
  console.log("➕ Stake more:     npx hardhat run scripts/stake.js");

  console.log("\n" + "=" .repeat(80));
  console.log("💎 KEEP STAKING, KEEP EARNING! 🚀");
  console.log("=" .repeat(80) + "\n");
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error("\n❌ ERROR:");
    console.error(error);
    process.exit(1);
  });
