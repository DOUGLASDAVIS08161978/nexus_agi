const hre = require("hardhat");

/**
 * HASHPROOF UNSTAKING SCRIPT
 * Withdraw your staked tokens (with early withdrawal penalty warning!)
 */

async function main() {
  console.log("=" .repeat(80));
  console.log("🔓 HASHPROOF UNSTAKING TOOL");
  console.log("=" .repeat(80));

  // Get signer
  const [user] = await hre.ethers.getSigners();
  console.log("\n💼 Unstaking for:", user.address);

  // Load deployed contract addresses (update these after deployment!)
  const HASHPROOF_ADDRESS = "0x5FbDB2315678afecb367f032d93F642f64180aa3"; // UPDATE THIS
  const STAKING_ADDRESS = "0xe7f1725E7734CE288F8367e1Bb143E90bb3F0512";   // UPDATE THIS

  // Get contract instances
  const HashProof = await hre.ethers.getContractFactory("HashProof");
  const hashProof = HashProof.attach(HASHPROOF_ADDRESS);

  const HashProofStaking = await hre.ethers.getContractFactory("HashProofStaking");
  const staking = HashProofStaking.attach(STAKING_ADDRESS);

  // Get balance before unstaking
  const balanceBefore = await hashProof.balanceOf(user.address);
  console.log("💰 Balance Before:  ", hre.ethers.formatEther(balanceBefore), "HPROOF");

  // Check how many stakes user has
  const stakeCount = await staking.userStakeCount(user.address);

  if (stakeCount == 0n) {
    console.log("\n❌ You don't have any active stakes!");
    console.log("   Start staking with: npx hardhat run scripts/stake.js\n");
    return;
  }

  console.log("\n📊 YOUR ACTIVE STAKES:");
  console.log("━".repeat(80));

  let activeStakes = [];

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

    // Format pool name
    const durationDays = Number(duration) / 86400;
    const poolName = durationDays === 0 ? "Flexible" :
                     durationDays === 30 ? "30-Day" :
                     durationDays === 90 ? "90-Day" :
                     durationDays === 180 ? "180-Day" : "365-Day";

    // Check if lock period expired
    const now = Math.floor(Date.now() / 1000);
    const unlockTime = Number(startTime) + Number(duration);
    const isLocked = now < unlockTime;
    const timeUntilUnlock = unlockTime - now;
    const daysUntilUnlock = Math.floor(timeUntilUnlock / 86400);

    // Calculate early withdrawal penalty
    let penalty = 0n;
    let willReceive = amount;
    if (isLocked) {
      penalty = (amount * 10n) / 100n; // 10% penalty
      willReceive = amount - penalty;
    }

    console.log(`\nStake #${i}: ${poolName} (${apy}% APY)`);
    console.log("  Amount Staked:     ", hre.ethers.formatEther(amount), "HPROOF");
    console.log("  Pending Rewards:   ", hre.ethers.formatEther(pendingRewards), "HPROOF");
    console.log("  Status:            ", isLocked ? "🔒 LOCKED" : "✅ UNLOCKED");

    if (isLocked) {
      console.log("  Unlocks in:        ", daysUntilUnlock, "days");
      console.log("  Unlock Date:       ", new Date(unlockTime * 1000).toLocaleDateString());
      console.log("  ⚠️  EARLY WITHDRAWAL PENALTY:");
      console.log("     Penalty:        ", hre.ethers.formatEther(penalty), "HPROOF (10%)");
      console.log("     You'll receive: ", hre.ethers.formatEther(willReceive), "HPROOF");
      console.log("     Loss:           $" + (Number(hre.ethers.formatEther(penalty)) * 0.10).toFixed(2) + " at $0.10/token");
    } else {
      console.log("  No Penalty:        ", "✅ Full amount returned");
    }

    activeStakes.push({
      index: i,
      amount: amount,
      rewards: pendingRewards,
      poolName: poolName,
      isLocked: isLocked,
      penalty: penalty,
      willReceive: willReceive,
      unlockTime: unlockTime
    });
  }

  if (activeStakes.length === 0) {
    console.log("\n❌ All your stakes have been withdrawn!");
    console.log("   Start new stake with: npx hardhat run scripts/stake.js\n");
    return;
  }

  console.log("━".repeat(80));

  // CONFIGURE WHICH STAKE TO UNSTAKE HERE
  const stakeIndexToUnstake = 0; // CHANGE THIS to unstake a different stake

  if (stakeIndexToUnstake >= activeStakes.length) {
    console.log("\n❌ Invalid stake index! You only have", activeStakes.length, "active stakes.");
    console.log("   Update the stakeIndexToUnstake variable in the script.\n");
    return;
  }

  const targetStake = activeStakes.find(s => s.index === stakeIndexToUnstake);

  if (!targetStake) {
    console.log("\n❌ Stake #" + stakeIndexToUnstake + " not found or already withdrawn!\n");
    return;
  }

  console.log("\n🎯 UNSTAKING STAKE #" + stakeIndexToUnstake);
  console.log("━".repeat(80));

  // Show what user will receive
  console.log("📊 UNSTAKE PREVIEW:");
  console.log("  Pool:              ", targetStake.poolName);
  console.log("  Staked Amount:     ", hre.ethers.formatEther(targetStake.amount), "HPROOF");
  console.log("  Pending Rewards:   ", hre.ethers.formatEther(targetStake.rewards), "HPROOF");

  if (targetStake.isLocked) {
    console.log("\n⚠️  WARNING: EARLY WITHDRAWAL!");
    console.log("━".repeat(80));
    console.log("🔒 This stake is still locked!");
    console.log("⚠️  You will lose 10% of your staked amount as penalty!");
    console.log("");
    console.log("  Penalty Amount:    ", hre.ethers.formatEther(targetStake.penalty), "HPROOF");
    console.log("  You will receive:  ", hre.ethers.formatEther(targetStake.willReceive), "HPROOF");
    console.log("  + Rewards:         ", hre.ethers.formatEther(targetStake.rewards), "HPROOF");
    console.log("  ─────────────────────────────────────");
    console.log("  Total Received:    ", hre.ethers.formatEther(targetStake.willReceive + targetStake.rewards), "HPROOF");
    console.log("");
    console.log("💡 RECOMMENDATION: Wait until", new Date(targetStake.unlockTime * 1000).toLocaleDateString());
    console.log("   to avoid penalty and get full amount!");
  } else {
    console.log("\n✅ NO PENALTY - STAKE IS UNLOCKED!");
    console.log("━".repeat(80));
    console.log("  You will receive:  ", hre.ethers.formatEther(targetStake.amount), "HPROOF");
    console.log("  + Rewards:         ", hre.ethers.formatEther(targetStake.rewards), "HPROOF");
    console.log("  ─────────────────────────────────────");
    console.log("  Total Received:    ", hre.ethers.formatEther(targetStake.amount + targetStake.rewards), "HPROOF");
  }

  // Calculate value
  const totalToReceive = targetStake.willReceive + targetStake.rewards;
  const receiveValue = Number(hre.ethers.formatEther(totalToReceive));

  console.log("\n💰 VALUE AT DIFFERENT PRICES:");
  console.log("  At $0.10/token:  $" + (receiveValue * 0.10).toFixed(2));
  console.log("  At $0.50/token:  $" + (receiveValue * 0.50).toFixed(2));
  console.log("  At $1.00/token:  $" + (receiveValue * 1.00).toFixed(2));
  console.log("  At $10.0/token:  $" + (receiveValue * 10.0).toFixed(2));

  // Execute unstake
  console.log("\n🔧 PROCEEDING WITH UNSTAKE...");
  console.log("━".repeat(80));

  try {
    const tx = await staking.unstake(stakeIndexToUnstake);
    console.log("⏳ Transaction submitted...");
    const receipt = await tx.wait();
    console.log("✅ Unstake successful!");

    // Get balance after unstaking
    const balanceAfter = await hashProof.balanceOf(user.address);
    const balanceIncrease = balanceAfter - balanceBefore;

    console.log("\n" + "=" .repeat(80));
    console.log("🎉 UNSTAKE COMPLETE!");
    console.log("=" .repeat(80));

    console.log("\n📊 SUMMARY:");
    console.log("━".repeat(80));
    console.log("Balance Before:    ", hre.ethers.formatEther(balanceBefore), "HPROOF");
    console.log("Balance After:     ", hre.ethers.formatEther(balanceAfter), "HPROOF");
    console.log("Tokens Received:   ", hre.ethers.formatEther(balanceIncrease), "HPROOF");

    if (targetStake.isLocked && targetStake.penalty > 0n) {
      console.log("Penalty Paid:      ", hre.ethers.formatEther(targetStake.penalty), "HPROOF ⚠️");
    }

    console.log("\n💵 VALUE RECEIVED:");
    const gainValue = Number(hre.ethers.formatEther(balanceIncrease));
    console.log("  At $0.10/token:  $" + (gainValue * 0.10).toFixed(2));
    console.log("  At $1.00/token:  $" + (gainValue * 1.00).toFixed(2));
    console.log("  At $10.0/token:  $" + (gainValue * 10.0).toFixed(2));

    console.log("\n🎯 WHAT'S NEXT:");
    console.log("━".repeat(80));
    console.log("✅ Tokens are back in your wallet!");
    console.log("💰 Check your balance in MetaMask");
    console.log("📊 Check remaining stakes: npx hardhat run scripts/check-rewards.js");
    console.log("➕ Re-stake for more rewards: npx hardhat run scripts/stake.js");
    console.log("💱 Trade on DEX or hold for price appreciation!");

    console.log("\n" + "=" .repeat(80));
    console.log("💎 THANKS FOR STAKING WITH HASHPROOF! 🚀");
    console.log("=" .repeat(80) + "\n");

  } catch (error) {
    console.log("\n❌ UNSTAKE FAILED!");
    console.error("Error:", error.message);
    console.log("\nPossible reasons:");
    console.log("  • Contract not authorized");
    console.log("  • Insufficient gas");
    console.log("  • Network issues");
    console.log("\nTry again or check the contract address.\n");
  }
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error("\n❌ ERROR:");
    console.error(error);
    process.exit(1);
  });
