const hre = require("hardhat");

/**
 * HASHPROOF GOVERNANCE - EXECUTE PROPOSAL
 * Execute passed proposals and implement changes
 */

async function main() {
  console.log("=" .repeat(80));
  console.log("⚡ HASHPROOF GOVERNANCE - EXECUTE PROPOSAL");
  console.log("=" .repeat(80));

  // Get signer
  const [executor] = await hre.ethers.getSigners();
  console.log("\n💼 Executor Address:", executor.address);

  // Load deployed contract addresses (update these after deployment!)
  const HASHPROOF_ADDRESS = "0x5FbDB2315678afecb367f032d93F642f64180aa3"; // UPDATE THIS
  const GOVERNANCE_ADDRESS = "0x9fE46736679d2D9a65F0992F2272dE9f3c7fa6e0";  // UPDATE THIS

  // Get contract instances
  const HashProof = await hre.ethers.getContractFactory("HashProof");
  const hashProof = HashProof.attach(HASHPROOF_ADDRESS);

  const HashProofGovernance = await hre.ethers.getContractFactory("HashProofGovernance");
  const governance = HashProofGovernance.attach(GOVERNANCE_ADDRESS);

  // Get total proposals
  const proposalCount = await governance.proposalCount();
  console.log("\n📊 Total Proposals:  ", proposalCount.toString());

  if (proposalCount == 0n) {
    console.log("\n❌ No proposals exist!");
    console.log("   Create one with: npx hardhat run scripts/propose.js\n");
    return;
  }

  console.log("\n📋 EXECUTABLE PROPOSALS:");
  console.log("=" .repeat(80));

  const now = Math.floor(Date.now() / 1000);
  const quorum = await governance.quorum();
  let executableProposals = [];

  // Find executable proposals
  for (let i = 0; i < proposalCount; i++) {
    const proposal = await governance.proposals(i);

    const votingEnded = now >= Number(proposal.endTime);
    const notExecuted = !proposal.executed;
    const notCanceled = !proposal.canceled;
    const totalVotes = proposal.forVotes + proposal.againstVotes + proposal.abstainVotes;
    const quorumMet = totalVotes >= quorum;
    const passed = proposal.forVotes > proposal.againstVotes;

    const isExecutable = votingEnded && notExecuted && notCanceled && quorumMet && passed;

    if (isExecutable) {
      console.log(`\n✅ Proposal #${i}: READY FOR EXECUTION`);
      console.log("━".repeat(80));
      console.log("Title:        ", proposal.title);
      console.log("Status:       ", "🎉 PASSED - Ready to Execute!");
      console.log("Ended:        ", new Date(Number(proposal.endTime) * 1000).toLocaleString());

      console.log("\n🗳️  Final Vote Results:");
      console.log("  For:        ", hre.ethers.formatEther(proposal.forVotes), "HPROOF");
      console.log("  Against:    ", hre.ethers.formatEther(proposal.againstVotes), "HPROOF");
      console.log("  Abstain:    ", hre.ethers.formatEther(proposal.abstainVotes), "HPROOF");

      const quorumProgress = (Number(totalVotes) / Number(quorum)) * 100;
      console.log("\n📊 Quorum:      ", quorumProgress.toFixed(2) + "% ✅");

      executableProposals.push({ id: i, proposal });
    } else {
      // Show why not executable
      let status = "";
      let reason = "";

      if (proposal.executed) {
        status = "✅ EXECUTED";
        reason = "Already executed";
      } else if (proposal.canceled) {
        status = "❌ CANCELED";
        reason = "Proposal was canceled";
      } else if (!votingEnded) {
        status = "⏰ VOTING IN PROGRESS";
        const timeLeft = Number(proposal.endTime) - now;
        const daysLeft = Math.floor(timeLeft / 86400);
        const hoursLeft = Math.floor((timeLeft % 86400) / 3600);
        reason = `Ends in ${daysLeft} days, ${hoursLeft} hours`;
      } else if (!quorumMet) {
        status = "❌ FAILED (No Quorum)";
        const quorumProgress = (Number(totalVotes) / Number(quorum)) * 100;
        reason = `Only ${quorumProgress.toFixed(2)}% quorum reached`;
      } else if (!passed) {
        status = "❌ FAILED (Rejected)";
        reason = "More votes against than for";
      }

      console.log(`\n❌ Proposal #${i}: ${status}`);
      console.log("━".repeat(80));
      console.log("Title:        ", proposal.title);
      console.log("Reason:       ", reason);
    }
  }

  if (executableProposals.length === 0) {
    console.log("\n" + "=" .repeat(80));
    console.log("\n❌ No proposals ready for execution!");
    console.log("\nProposals must meet ALL conditions:");
    console.log("  ✅ Voting period ended");
    console.log("  ✅ Not already executed");
    console.log("  ✅ Not canceled");
    console.log("  ✅ Quorum met (", hre.ethers.formatEther(quorum), "HPROOF)");
    console.log("  ✅ More FOR votes than AGAINST");
    console.log("\nWait for voting to end or encourage more participation!\n");
    return;
  }

  console.log("\n" + "=" .repeat(80));

  // CONFIGURE WHICH PROPOSAL TO EXECUTE
  const proposalIdToExecute = 0; // CHANGE THIS to execute a different proposal

  if (proposalIdToExecute >= proposalCount) {
    console.log("\n❌ Invalid proposal ID!");
    console.log("   Valid IDs: 0 to", (proposalCount - 1n).toString());
    console.log("   Update the proposalIdToExecute variable in the script.\n");
    return;
  }

  // Get target proposal
  const targetProposal = await governance.proposals(proposalIdToExecute);

  // Validate it's executable
  const votingEnded = now >= Number(targetProposal.endTime);
  const notExecuted = !targetProposal.executed;
  const notCanceled = !targetProposal.canceled;
  const totalVotes = targetProposal.forVotes + targetProposal.againstVotes + targetProposal.abstainVotes;
  const quorumMet = totalVotes >= quorum;
  const passed = targetProposal.forVotes > targetProposal.againstVotes;

  const isExecutable = votingEnded && notExecuted && notCanceled && quorumMet && passed;

  if (!isExecutable) {
    console.log("\n❌ This proposal cannot be executed!");
    console.log("\nReasons:");
    if (!votingEnded) console.log("  ❌ Voting period not ended yet");
    if (targetProposal.executed) console.log("  ❌ Already executed");
    if (targetProposal.canceled) console.log("  ❌ Proposal was canceled");
    if (!quorumMet) console.log("  ❌ Quorum not met");
    if (!passed) console.log("  ❌ More votes against than for");
    console.log("\nChoose a different proposal or wait for conditions to be met.\n");
    return;
  }

  console.log("\n⚡ EXECUTING PROPOSAL #" + proposalIdToExecute);
  console.log("=" .repeat(80));

  console.log("\n📋 PROPOSAL DETAILS:");
  console.log("━".repeat(80));
  console.log("Title:        ", targetProposal.title);
  console.log("Proposer:     ", targetProposal.proposer);
  console.log("Description:");
  console.log(targetProposal.description.substring(0, 500) + "...");

  console.log("\n🗳️  FINAL VOTE RESULTS:");
  console.log("━".repeat(80));
  console.log("For:          ", hre.ethers.formatEther(targetProposal.forVotes), "HPROOF");
  console.log("Against:      ", hre.ethers.formatEther(targetProposal.againstVotes), "HPROOF");
  console.log("Abstain:      ", hre.ethers.formatEther(targetProposal.abstainVotes), "HPROOF");
  console.log("Total Votes:  ", hre.ethers.formatEther(totalVotes), "HPROOF");

  const quorumProgress = (Number(totalVotes) / Number(quorum)) * 100;
  const passMargin = ((Number(targetProposal.forVotes) - Number(targetProposal.againstVotes)) / Number(totalVotes)) * 100;

  console.log("\n📊 VOTE STATISTICS:");
  console.log("━".repeat(80));
  console.log("Quorum:       ", quorumProgress.toFixed(2) + "% ✅");
  console.log("Pass Margin:  ", passMargin.toFixed(2) + "%");
  console.log("Participation:", ((Number(totalVotes) / Number(await hashProof.totalSupply())) * 100).toFixed(2) + "% of supply");

  console.log("\n⚠️  EXECUTION WARNING:");
  console.log("━".repeat(80));
  console.log("This will execute the proposal and implement the changes.");
  console.log("Execution is IRREVERSIBLE and changes take effect immediately.");
  console.log("Make sure you understand what this proposal does!");

  // Execute proposal
  console.log("\n🔧 EXECUTING PROPOSAL...");
  console.log("━".repeat(80));

  try {
    const tx = await governance.executeProposal(proposalIdToExecute);
    console.log("⏳ Transaction submitted...");
    console.log("   TX Hash:", tx.hash);

    const receipt = await tx.wait();
    console.log("✅ Proposal executed successfully!");

    // Get updated proposal
    const updatedProposal = await governance.proposals(proposalIdToExecute);

    console.log("\n" + "=" .repeat(80));
    console.log("🎉 PROPOSAL EXECUTED SUCCESSFULLY!");
    console.log("=" .repeat(80));

    console.log("\n📊 EXECUTION SUMMARY:");
    console.log("━".repeat(80));
    console.log("Proposal ID:      ", proposalIdToExecute);
    console.log("Title:            ", updatedProposal.title);
    console.log("Status:           ", "✅ EXECUTED");
    console.log("Executed At:      ", new Date().toLocaleString());
    console.log("Transaction:      ", tx.hash);

    console.log("\n🎯 WHAT HAPPENED:");
    console.log("━".repeat(80));
    console.log("✅ The proposal has been marked as executed");
    console.log("✅ Changes are now in effect (if applicable)");
    console.log("✅ Proposal cannot be executed again");
    console.log("✅ Results are permanent and recorded on-chain");

    console.log("\n💡 IMPLEMENTATION NOTES:");
    console.log("━".repeat(80));
    console.log("⚠️  This smart contract only records the execution.");
    console.log("⚠️  Actual implementation may require additional steps:");
    console.log("   • Update contract parameters manually");
    console.log("   • Deploy new contracts if needed");
    console.log("   • Coordinate with team for technical changes");
    console.log("   • Announce changes to community");

    console.log("\n📣 NEXT STEPS:");
    console.log("━".repeat(80));
    console.log("1. Announce execution to community");
    console.log("2. Implement any required technical changes");
    console.log("3. Update documentation if needed");
    console.log("4. Monitor for any issues");
    console.log("5. Celebrate democratic governance in action! 🎊");

    console.log("\n📱 ANNOUNCEMENT TEMPLATE:");
    console.log("━".repeat(80));
    console.log(`🎉 PROPOSAL #${proposalIdToExecute} EXECUTED!`);
    console.log("");
    console.log(`✅ ${updatedProposal.title}`);
    console.log("");
    console.log(`📊 Final Results:`);
    console.log(`   For: ${hre.ethers.formatEther(updatedProposal.forVotes)} HPROOF`);
    console.log(`   Against: ${hre.ethers.formatEther(updatedProposal.againstVotes)} HPROOF`);
    console.log(`   Quorum: ${quorumProgress.toFixed(2)}%`);
    console.log("");
    console.log(`The community has spoken! Changes are now in effect. 🚀`);
    console.log(`Thank you to everyone who participated in governance!`);

    console.log("\n" + "=" .repeat(80));
    console.log("🗳️  DEMOCRACY WORKS! THE PEOPLE HAVE DECIDED! 🎊");
    console.log("=" .repeat(80) + "\n");

  } catch (error) {
    console.log("\n❌ EXECUTION FAILED!");
    console.error("Error:", error.message);
    console.log("\nPossible reasons:");
    console.log("  • Proposal not executable (check conditions)");
    console.log("  • Already executed");
    console.log("  • Insufficient gas");
    console.log("  • Smart contract restrictions");
    console.log("\nVerify proposal status and try again.\n");
  }
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error("\n❌ ERROR:");
    console.error(error);
    process.exit(1);
  });
