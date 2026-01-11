const hre = require("hardhat");

/**
 * HASHPROOF GOVERNANCE - VOTE ON PROPOSAL
 * Cast your vote and shape the future!
 */

async function main() {
  console.log("=" .repeat(80));
  console.log("🗳️  HASHPROOF GOVERNANCE - VOTE ON PROPOSAL");
  console.log("=" .repeat(80));

  // Get signer
  const [voter] = await hre.ethers.getSigners();
  console.log("\n💼 Voter Address:", voter.address);

  // Load deployed contract addresses (update these after deployment!)
  const HASHPROOF_ADDRESS = "0x5FbDB2315678afecb367f032d93F642f64180aa3"; // UPDATE THIS
  const GOVERNANCE_ADDRESS = "0x9fE46736679d2D9a65F0992F2272dE9f3c7fa6e0";  // UPDATE THIS

  // Get contract instances
  const HashProof = await hre.ethers.getContractFactory("HashProof");
  const hashProof = HashProof.attach(HASHPROOF_ADDRESS);

  const HashProofGovernance = await hre.ethers.getContractFactory("HashProofGovernance");
  const governance = HashProofGovernance.attach(GOVERNANCE_ADDRESS);

  // Check voter's token balance
  const balance = await hashProof.balanceOf(voter.address);
  console.log("💰 Your HPROOF Balance:", hre.ethers.formatEther(balance), "HPROOF");
  console.log("🗳️  Voting Power:      ", hre.ethers.formatEther(balance), "HPROOF");

  if (balance == 0n) {
    console.log("\n❌ You don't have any HPROOF tokens!");
    console.log("   You need HPROOF tokens to vote.");
    console.log("   Buy tokens or earn through staking/mining.\n");
    return;
  }

  // Get total proposals
  const proposalCount = await governance.proposalCount();
  console.log("\n📊 Total Proposals:  ", proposalCount.toString());

  if (proposalCount == 0n) {
    console.log("\n❌ No proposals exist yet!");
    console.log("   Create one with: npx hardhat run scripts/propose.js\n");
    return;
  }

  console.log("\n📋 ACTIVE PROPOSALS:");
  console.log("=" .repeat(80));

  const now = Math.floor(Date.now() / 1000);
  let activeProposals = [];

  // List all active proposals
  for (let i = 0; i < proposalCount; i++) {
    const proposal = await governance.proposals(i);

    const isActive = !proposal.executed && !proposal.canceled && now < Number(proposal.endTime);

    if (isActive) {
      const timeLeft = Number(proposal.endTime) - now;
      const daysLeft = Math.floor(timeLeft / 86400);
      const hoursLeft = Math.floor((timeLeft % 86400) / 3600);

      console.log(`\n📌 Proposal #${i}:`);
      console.log("━".repeat(80));
      console.log("Title:        ", proposal.title);
      console.log("Proposer:     ", proposal.proposer);
      console.log("Status:       ", "🔥 ACTIVE - VOTE NOW!");
      console.log("Time Left:    ", daysLeft, "days,", hoursLeft, "hours");
      console.log("Ends:         ", new Date(Number(proposal.endTime) * 1000).toLocaleDateString());

      console.log("\n🗳️  Current Votes:");
      console.log("  For:        ", hre.ethers.formatEther(proposal.forVotes), "HPROOF");
      console.log("  Against:    ", hre.ethers.formatEther(proposal.againstVotes), "HPROOF");
      console.log("  Abstain:    ", hre.ethers.formatEther(proposal.abstainVotes), "HPROOF");

      const totalVotes = proposal.forVotes + proposal.againstVotes + proposal.abstainVotes;
      const quorum = await governance.quorum();
      const quorumProgress = (Number(totalVotes) / Number(quorum)) * 100;

      console.log("\n📊 Quorum Progress: ", quorumProgress.toFixed(2) + "%");
      console.log("   Need:            ", hre.ethers.formatEther(quorum), "HPROOF");
      console.log("   Have:            ", hre.ethers.formatEther(totalVotes), "HPROOF");

      // Check if user already voted
      const hasVoted = await governance.hasVoted(i, voter.address);
      if (hasVoted) {
        console.log("\n✅ You have already voted on this proposal");
      } else {
        console.log("\n❗ You have NOT voted yet!");
      }

      activeProposals.push({ id: i, proposal, hasVoted });
    }
  }

  if (activeProposals.length === 0) {
    console.log("\n❌ No active proposals to vote on!");
    console.log("   All proposals have ended or been executed.");
    console.log("   Create a new one with: npx hardhat run scripts/propose.js\n");
    return;
  }

  console.log("\n" + "=" .repeat(80));

  // CONFIGURE YOUR VOTE HERE
  const proposalId = 0;     // CHANGE THIS: Which proposal to vote on
  const voteChoice = 0;      // CHANGE THIS: 0 = Against, 1 = For, 2 = Abstain

  const voteChoiceText = voteChoice === 0 ? "AGAINST ❌" :
                         voteChoice === 1 ? "FOR ✅" : "ABSTAIN 🤷";

  console.log("\n🗳️  YOUR VOTE:");
  console.log("━".repeat(80));
  console.log("Proposal ID:  ", proposalId);
  console.log("Your Choice:  ", voteChoiceText);
  console.log("Voting Power: ", hre.ethers.formatEther(balance), "HPROOF");

  // Validate proposal exists
  if (proposalId >= proposalCount) {
    console.log("\n❌ Invalid proposal ID!");
    console.log("   Valid IDs: 0 to", (proposalCount - 1n).toString());
    console.log("   Update the proposalId variable in the script.\n");
    return;
  }

  // Get proposal details
  const targetProposal = await governance.proposals(proposalId);

  // Check if proposal is active
  const isActive = !targetProposal.executed && !targetProposal.canceled && now < Number(targetProposal.endTime);

  if (!isActive) {
    console.log("\n❌ This proposal is not active!");
    if (targetProposal.executed) {
      console.log("   Reason: Already executed");
    } else if (targetProposal.canceled) {
      console.log("   Reason: Canceled");
    } else if (now >= Number(targetProposal.endTime)) {
      console.log("   Reason: Voting period ended");
    }
    console.log("   Choose a different proposal.\n");
    return;
  }

  // Check if already voted
  const hasVoted = await governance.hasVoted(proposalId, voter.address);

  if (hasVoted) {
    console.log("\n⚠️  You have already voted on this proposal!");
    console.log("   Each address can only vote once per proposal.");
    console.log("   Your vote is already counted.\n");
    return;
  }

  console.log("\n📋 PROPOSAL DETAILS:");
  console.log("━".repeat(80));
  console.log("Title:        ", targetProposal.title);
  console.log("Description:");
  console.log(targetProposal.description);

  // Show vote impact
  const newForVotes = voteChoice === 1 ? targetProposal.forVotes + balance : targetProposal.forVotes;
  const newAgainstVotes = voteChoice === 0 ? targetProposal.againstVotes + balance : targetProposal.againstVotes;
  const newAbstainVotes = voteChoice === 2 ? targetProposal.abstainVotes + balance : targetProposal.abstainVotes;

  console.log("\n📊 VOTE IMPACT:");
  console.log("━".repeat(80));
  console.log("Current For:       ", hre.ethers.formatEther(targetProposal.forVotes), "HPROOF");
  console.log("After your vote:   ", hre.ethers.formatEther(newForVotes), "HPROOF");
  console.log("");
  console.log("Current Against:   ", hre.ethers.formatEther(targetProposal.againstVotes), "HPROOF");
  console.log("After your vote:   ", hre.ethers.formatEther(newAgainstVotes), "HPROOF");
  console.log("");
  console.log("Current Abstain:   ", hre.ethers.formatEther(targetProposal.abstainVotes), "HPROOF");
  console.log("After your vote:   ", hre.ethers.formatEther(newAbstainVotes), "HPROOF");

  const newTotalVotes = newForVotes + newAgainstVotes + newAbstainVotes;
  const quorum = await governance.quorum();
  const newQuorumProgress = (Number(newTotalVotes) / Number(quorum)) * 100;

  console.log("\n📊 QUORUM IMPACT:");
  console.log("━".repeat(80));
  const totalVotes = targetProposal.forVotes + targetProposal.againstVotes + targetProposal.abstainVotes;
  const currentQuorumProgress = (Number(totalVotes) / Number(quorum)) * 100;
  console.log("Current Progress:  ", currentQuorumProgress.toFixed(2) + "%");
  console.log("After your vote:   ", newQuorumProgress.toFixed(2) + "%");
  console.log("Quorum Required:   ", hre.ethers.formatEther(quorum), "HPROOF");

  if (newQuorumProgress >= 100) {
    console.log("\n🎉 Your vote will help reach quorum! ✅");
  } else {
    const remaining = quorum - newTotalVotes;
    console.log("\n⏳ Still need", hre.ethers.formatEther(remaining), "HPROOF more to reach quorum");
  }

  // Cast vote
  console.log("\n🔧 CASTING VOTE...");
  console.log("━".repeat(80));

  try {
    const tx = await governance.vote(proposalId, voteChoice);
    console.log("⏳ Transaction submitted...");
    console.log("   TX Hash:", tx.hash);

    const receipt = await tx.wait();
    console.log("✅ Vote cast successfully!");

    // Get updated proposal
    const updatedProposal = await governance.proposals(proposalId);

    console.log("\n" + "=" .repeat(80));
    console.log("🎉 YOUR VOTE HAS BEEN COUNTED!");
    console.log("=" .repeat(80));

    console.log("\n🗳️  UPDATED RESULTS:");
    console.log("━".repeat(80));
    console.log("For:          ", hre.ethers.formatEther(updatedProposal.forVotes), "HPROOF");
    console.log("Against:      ", hre.ethers.formatEther(updatedProposal.againstVotes), "HPROOF");
    console.log("Abstain:      ", hre.ethers.formatEther(updatedProposal.abstainVotes), "HPROOF");

    const updatedTotalVotes = updatedProposal.forVotes + updatedProposal.againstVotes + updatedProposal.abstainVotes;
    const finalQuorumProgress = (Number(updatedTotalVotes) / Number(quorum)) * 100;

    console.log("\n📊 QUORUM STATUS:");
    console.log("━".repeat(80));
    console.log("Progress:     ", finalQuorumProgress.toFixed(2) + "%");
    console.log("Total Votes:  ", hre.ethers.formatEther(updatedTotalVotes), "HPROOF");
    console.log("Quorum:       ", hre.ethers.formatEther(quorum), "HPROOF");

    if (finalQuorumProgress >= 100) {
      console.log("\n🎊 QUORUM REACHED! This proposal can be executed if it passes! ✅");
    } else {
      const stillNeeded = quorum - updatedTotalVotes;
      console.log("\n⏳ Need", hre.ethers.formatEther(stillNeeded), "HPROOF more votes for quorum");
    }

    // Determine current outcome
    if (updatedProposal.forVotes > updatedProposal.againstVotes) {
      console.log("\n📈 CURRENT OUTCOME: Proposal is PASSING ✅");
    } else if (updatedProposal.againstVotes > updatedProposal.forVotes) {
      console.log("\n📉 CURRENT OUTCOME: Proposal is FAILING ❌");
    } else {
      console.log("\n⚖️  CURRENT OUTCOME: Proposal is TIED");
    }

    const timeLeft = Number(updatedProposal.endTime) - now;
    const daysLeft = Math.floor(timeLeft / 86400);
    const hoursLeft = Math.floor((timeLeft % 86400) / 3600);

    console.log("\n⏰ TIME REMAINING:");
    console.log("━".repeat(80));
    console.log(daysLeft, "days,", hoursLeft, "hours until voting ends");
    console.log("Ends:", new Date(Number(updatedProposal.endTime) * 1000).toLocaleString());

    console.log("\n🎯 NEXT STEPS:");
    console.log("━".repeat(80));
    console.log("✅ Your vote has been counted!");
    console.log("📣 Share the proposal with others");
    console.log("👀 Monitor the vote progress");
    console.log("⏰ Wait for voting period to end");

    if (finalQuorumProgress >= 100 && updatedProposal.forVotes > updatedProposal.againstVotes) {
      console.log("🎊 If the vote passes, execute with: npx hardhat run scripts/execute-proposal.js");
    }

    console.log("\n💡 DEMOCRACY IN ACTION:");
    console.log("━".repeat(80));
    console.log("Your vote matters! You're helping shape the future of HashProof!");
    console.log("Encourage others to vote and participate in governance!");

    console.log("\n" + "=" .repeat(80));
    console.log("🗳️  THANK YOU FOR VOTING! POWER TO THE PEOPLE! 🎊");
    console.log("=" .repeat(80) + "\n");

  } catch (error) {
    console.log("\n❌ VOTING FAILED!");
    console.error("Error:", error.message);
    console.log("\nPossible reasons:");
    console.log("  • Already voted on this proposal");
    console.log("  • Proposal not active");
    console.log("  • Insufficient gas");
    console.log("  • Invalid vote choice (must be 0, 1, or 2)");
    console.log("\nCheck your vote parameters and try again.\n");
  }
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error("\n❌ ERROR:");
    console.error(error);
    process.exit(1);
  });
