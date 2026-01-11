const hre = require("hardhat");

/**
 * HASHPROOF GOVERNANCE - CREATE PROPOSAL
 * Submit proposals for community voting
 */

async function main() {
  console.log("=" .repeat(80));
  console.log("🗳️  HASHPROOF GOVERNANCE - CREATE PROPOSAL");
  console.log("=" .repeat(80));

  // Get signer
  const [proposer] = await hre.ethers.getSigners();
  console.log("\n💼 Proposer Address:", proposer.address);

  // Load deployed contract addresses (update these after deployment!)
  const HASHPROOF_ADDRESS = "0x5FbDB2315678afecb367f032d93F642f64180aa3"; // UPDATE THIS
  const GOVERNANCE_ADDRESS = "0x9fE46736679d2D9a65F0992F2272dE9f3c7fa6e0";  // UPDATE THIS

  // Get contract instances
  const HashProof = await hre.ethers.getContractFactory("HashProof");
  const hashProof = HashProof.attach(HASHPROOF_ADDRESS);

  const HashProofGovernance = await hre.ethers.getContractFactory("HashProofGovernance");
  const governance = HashProofGovernance.attach(GOVERNANCE_ADDRESS);

  // Check proposer's token balance
  const balance = await hashProof.balanceOf(proposer.address);
  console.log("💰 Your HPROOF Balance:", hre.ethers.formatEther(balance), "HPROOF");

  // Check proposal threshold
  const proposalThreshold = await governance.proposalThreshold();
  console.log("🎯 Proposal Threshold:", hre.ethers.formatEther(proposalThreshold), "HPROOF");

  if (balance < proposalThreshold) {
    console.log("\n❌ INSUFFICIENT TOKENS TO PROPOSE!");
    console.log("   You need:", hre.ethers.formatEther(proposalThreshold), "HPROOF");
    console.log("   You have:", hre.ethers.formatEther(balance), "HPROOF");
    console.log("   Missing:", hre.ethers.formatEther(proposalThreshold - balance), "HPROOF");
    console.log("\n   Buy or earn more HPROOF tokens to create proposals!\n");
    return;
  }

  console.log("✅ You have enough tokens to create proposals!");

  // Get governance parameters
  const votingPeriod = await governance.votingPeriod();
  const quorum = await governance.quorum();
  const totalSupply = await hashProof.totalSupply();
  const quorumPercentage = (Number(quorum) * 100) / Number(totalSupply);

  console.log("\n📊 GOVERNANCE PARAMETERS:");
  console.log("━".repeat(80));
  console.log("Voting Period:     ", Number(votingPeriod) / 86400, "days");
  console.log("Quorum Required:   ", hre.ethers.formatEther(quorum), "HPROOF");
  console.log("Quorum Percentage: ", quorumPercentage.toFixed(2) + "%");
  console.log("Total Supply:      ", hre.ethers.formatEther(totalSupply), "HPROOF");

  // CONFIGURE YOUR PROPOSAL HERE
  const proposalTitle = "Increase Staking Rewards for 365-Day Pool";

  const proposalDescription = `
# Proposal: Increase Staking Rewards for 365-Day Pool

## Summary
Increase the APY for the 365-Day staking pool from 30% to 35% to incentivize long-term holding and reduce selling pressure.

## Motivation
- Current 365-Day pool has lower participation than expected
- Competing projects offer 35-40% APY for similar lock periods
- Increasing rewards will attract more long-term holders
- Reduced circulation will support token price appreciation

## Implementation
1. Adjust the APY parameter in HashProofStaking contract
2. Allocate additional 5,000,000 HPROOF from team pool to staking rewards
3. Changes take effect immediately upon proposal execution

## Expected Outcomes
- 50% increase in 365-Day pool participation within 3 months
- Reduced token circulation by 10,000,000 HPROOF
- Stronger token price support
- Enhanced community confidence

## Risks
- Increased token emissions could create selling pressure
- Team pool allocation reduced by 5M tokens
- Higher reward obligations

## Vote Options
- FOR: Increase 365-Day APY to 35%
- AGAINST: Keep current 30% APY
- ABSTAIN: No preference

**Proposed by**: HashProof Community Member
**Voting Period**: 3 days from proposal creation
`.trim();

  console.log("\n📝 PROPOSAL DETAILS:");
  console.log("━".repeat(80));
  console.log("Title:       ", proposalTitle);
  console.log("Description: ", proposalDescription.substring(0, 200) + "...");
  console.log("\n(Full description will be stored on-chain)");

  // Get current proposal count
  const proposalCountBefore = await governance.proposalCount();
  console.log("\n📊 Current Proposals:", proposalCountBefore.toString());

  // Create proposal
  console.log("\n🔧 CREATING PROPOSAL...");
  console.log("━".repeat(80));

  try {
    const tx = await governance.propose(proposalTitle, proposalDescription);
    console.log("⏳ Transaction submitted...");
    console.log("   TX Hash:", tx.hash);

    const receipt = await tx.wait();
    console.log("✅ Proposal created successfully!");

    // Get the new proposal ID
    const proposalCountAfter = await governance.proposalCount();
    const newProposalId = proposalCountAfter - 1n; // Last proposal

    // Get proposal details
    const proposal = await governance.proposals(newProposalId);

    console.log("\n" + "=" .repeat(80));
    console.log("🎉 PROPOSAL CREATED!");
    console.log("=" .repeat(80));

    console.log("\n📋 PROPOSAL INFO:");
    console.log("━".repeat(80));
    console.log("Proposal ID:       ", newProposalId.toString());
    console.log("Title:             ", proposalTitle);
    console.log("Proposer:          ", proposal.proposer);
    console.log("Start Time:        ", new Date(Number(proposal.startTime) * 1000).toLocaleString());
    console.log("End Time:          ", new Date(Number(proposal.endTime) * 1000).toLocaleString());
    console.log("Status:            ", proposal.executed ? "Executed" : proposal.canceled ? "Canceled" : "Active 🔥");

    const now = Math.floor(Date.now() / 1000);
    const timeLeft = Number(proposal.endTime) - now;
    const daysLeft = Math.floor(timeLeft / 86400);
    const hoursLeft = Math.floor((timeLeft % 86400) / 3600);

    console.log("\n⏰ VOTING PERIOD:");
    console.log("━".repeat(80));
    console.log("Time Remaining:    ", daysLeft, "days,", hoursLeft, "hours");
    console.log("Ends:              ", new Date(Number(proposal.endTime) * 1000).toLocaleDateString());

    console.log("\n🗳️  CURRENT VOTES:");
    console.log("━".repeat(80));
    console.log("For:               ", hre.ethers.formatEther(proposal.forVotes), "HPROOF");
    console.log("Against:           ", hre.ethers.formatEther(proposal.againstVotes), "HPROOF");
    console.log("Abstain:           ", hre.ethers.formatEther(proposal.abstainVotes), "HPROOF");
    console.log("Total Votes:       ", hre.ethers.formatEther(proposal.forVotes + proposal.againstVotes + proposal.abstainVotes), "HPROOF");

    console.log("\n📊 QUORUM STATUS:");
    console.log("━".repeat(80));
    const totalVotes = proposal.forVotes + proposal.againstVotes + proposal.abstainVotes;
    const quorumProgress = (Number(totalVotes) / Number(quorum)) * 100;
    console.log("Votes Needed:      ", hre.ethers.formatEther(quorum), "HPROOF");
    console.log("Votes Cast:        ", hre.ethers.formatEther(totalVotes), "HPROOF");
    console.log("Progress:          ", quorumProgress.toFixed(2) + "%");

    console.log("\n🎯 NEXT STEPS:");
    console.log("━".repeat(80));
    console.log("1. Share your proposal with the community");
    console.log("2. Rally support and explain benefits");
    console.log("3. Community members vote with: npx hardhat run scripts/vote.js");
    console.log("4. Monitor votes: Check governance contract");
    console.log("5. If passed and quorum met: Execute with npx hardhat run scripts/execute-proposal.js");

    console.log("\n💡 CAMPAIGN TIPS:");
    console.log("━".repeat(80));
    console.log("• Post on social media (Twitter, Discord, Telegram)");
    console.log("• Explain the benefits clearly");
    console.log("• Address potential concerns");
    console.log("• Engage with voters and answer questions");
    console.log("• Remind people to vote before deadline!");

    console.log("\n📱 SHARE THIS:");
    console.log("━".repeat(80));
    console.log(`🗳️  NEW HASHPROOF PROPOSAL #${newProposalId}!`);
    console.log("");
    console.log(`📝 ${proposalTitle}`);
    console.log("");
    console.log(`⏰ Vote now! Ends in ${daysLeft} days`);
    console.log(`🎯 Need ${hre.ethers.formatEther(quorum)} HPROOF for quorum`);
    console.log("");
    console.log(`Vote with your HPROOF tokens and shape the future! 🚀`);

    console.log("\n" + "=" .repeat(80));
    console.log("🗳️  LET THE VOTING BEGIN! MAY THE BEST PROPOSAL WIN! 🏆");
    console.log("=" .repeat(80) + "\n");

  } catch (error) {
    console.log("\n❌ PROPOSAL CREATION FAILED!");
    console.error("Error:", error.message);
    console.log("\nPossible reasons:");
    console.log("  • Insufficient token balance (need", hre.ethers.formatEther(proposalThreshold), "HPROOF)");
    console.log("  • Governance contract not authorized");
    console.log("  • Network issues");
    console.log("  • Invalid proposal format");
    console.log("\nCheck your balance and try again.\n");
  }
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error("\n❌ ERROR:");
    console.error(error);
    process.exit(1);
  });
