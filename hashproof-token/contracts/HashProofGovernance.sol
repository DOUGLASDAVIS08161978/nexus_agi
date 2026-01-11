// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";

/**
 * @title HashProofGovernance
 * @dev Simple governance system for HPROOF token holders
 *
 * Features:
 * - Token holders can create proposals
 * - Voting power based on token balance
 * - Quorum requirements
 * - Execution of approved proposals
 */
contract HashProofGovernance is Ownable, ReentrancyGuard {
    IERC20 public hproofToken;

    enum ProposalStatus {
        Pending,
        Active,
        Succeeded,
        Defeated,
        Executed,
        Cancelled
    }

    struct Proposal {
        uint256 id;
        address proposer;
        string description;
        uint256 forVotes;
        uint256 againstVotes;
        uint256 abstainVotes;
        uint256 startTime;
        uint256 endTime;
        ProposalStatus status;
        mapping(address => bool) hasVoted;
        mapping(address => uint256) voteChoice; // 0=against, 1=for, 2=abstain
    }

    // Proposals
    mapping(uint256 => Proposal) public proposals;
    uint256 public proposalCount;

    // Governance parameters
    uint256 public votingPeriod = 3 days;
    uint256 public proposalThreshold = 10000 * 10**18; // Need 10k tokens to propose
    uint256 public quorumPercentage = 400; // 4% of total supply (in basis points)
    uint256 public constant BASIS_POINTS = 10000;

    // Events
    event ProposalCreated(uint256 indexed proposalId, address indexed proposer, string description);
    event VoteCast(address indexed voter, uint256 indexed proposalId, uint256 choice, uint256 weight);
    event ProposalExecuted(uint256 indexed proposalId);
    event ProposalCancelled(uint256 indexed proposalId);

    constructor(address _hproofToken, address initialOwner) Ownable(initialOwner) {
        hproofToken = IERC20(_hproofToken);
    }

    /**
     * @dev Create a new proposal
     */
    function propose(string calldata description) external returns (uint256) {
        require(
            hproofToken.balanceOf(msg.sender) >= proposalThreshold,
            "Insufficient tokens to propose"
        );

        uint256 proposalId = proposalCount++;
        Proposal storage newProposal = proposals[proposalId];

        newProposal.id = proposalId;
        newProposal.proposer = msg.sender;
        newProposal.description = description;
        newProposal.startTime = block.timestamp;
        newProposal.endTime = block.timestamp + votingPeriod;
        newProposal.status = ProposalStatus.Active;

        emit ProposalCreated(proposalId, msg.sender, description);

        return proposalId;
    }

    /**
     * @dev Cast a vote
     * @param proposalId The proposal to vote on
     * @param choice 0 = Against, 1 = For, 2 = Abstain
     */
    function castVote(uint256 proposalId, uint256 choice) external nonReentrant {
        require(choice <= 2, "Invalid vote choice");
        Proposal storage proposal = proposals[proposalId];

        require(proposal.status == ProposalStatus.Active, "Proposal not active");
        require(block.timestamp <= proposal.endTime, "Voting period ended");
        require(!proposal.hasVoted[msg.sender], "Already voted");

        uint256 weight = hproofToken.balanceOf(msg.sender);
        require(weight > 0, "No voting power");

        proposal.hasVoted[msg.sender] = true;
        proposal.voteChoice[msg.sender] = choice;

        if (choice == 0) {
            proposal.againstVotes += weight;
        } else if (choice == 1) {
            proposal.forVotes += weight;
        } else {
            proposal.abstainVotes += weight;
        }

        emit VoteCast(msg.sender, proposalId, choice, weight);
    }

    /**
     * @dev Finalize a proposal after voting ends
     */
    function finalizeProposal(uint256 proposalId) external {
        Proposal storage proposal = proposals[proposalId];

        require(proposal.status == ProposalStatus.Active, "Proposal not active");
        require(block.timestamp > proposal.endTime, "Voting period not ended");

        uint256 totalVotes = proposal.forVotes + proposal.againstVotes + proposal.abstainVotes;
        uint256 totalSupply = hproofToken.totalSupply();
        uint256 quorum = (totalSupply * quorumPercentage) / BASIS_POINTS;

        // Check if quorum reached
        if (totalVotes < quorum) {
            proposal.status = ProposalStatus.Defeated;
        } else if (proposal.forVotes > proposal.againstVotes) {
            proposal.status = ProposalStatus.Succeeded;
        } else {
            proposal.status = ProposalStatus.Defeated;
        }
    }

    /**
     * @dev Execute a succeeded proposal (owner only for now)
     */
    function executeProposal(uint256 proposalId) external onlyOwner {
        Proposal storage proposal = proposals[proposalId];
        require(proposal.status == ProposalStatus.Succeeded, "Proposal not succeeded");

        proposal.status = ProposalStatus.Executed;
        emit ProposalExecuted(proposalId);

        // Implementation of proposal execution would go here
        // For now, this is a simple governance system
    }

    /**
     * @dev Cancel a proposal (only proposer or owner)
     */
    function cancelProposal(uint256 proposalId) external {
        Proposal storage proposal = proposals[proposalId];
        require(
            msg.sender == proposal.proposer || msg.sender == owner(),
            "Not authorized"
        );
        require(
            proposal.status == ProposalStatus.Active || proposal.status == ProposalStatus.Pending,
            "Cannot cancel"
        );

        proposal.status = ProposalStatus.Cancelled;
        emit ProposalCancelled(proposalId);
    }

    /**
     * @dev Get proposal details
     */
    function getProposal(uint256 proposalId)
        external
        view
        returns (
            address proposer,
            string memory description,
            uint256 forVotes,
            uint256 againstVotes,
            uint256 abstainVotes,
            uint256 startTime,
            uint256 endTime,
            ProposalStatus status
        )
    {
        Proposal storage proposal = proposals[proposalId];
        return (
            proposal.proposer,
            proposal.description,
            proposal.forVotes,
            proposal.againstVotes,
            proposal.abstainVotes,
            proposal.startTime,
            proposal.endTime,
            proposal.status
        );
    }

    /**
     * @dev Check if address has voted on proposal
     */
    function hasVoted(uint256 proposalId, address voter) external view returns (bool) {
        return proposals[proposalId].hasVoted[voter];
    }

    /**
     * @dev Update voting period (owner only)
     */
    function setVotingPeriod(uint256 newPeriod) external onlyOwner {
        require(newPeriod >= 1 days && newPeriod <= 30 days, "Invalid period");
        votingPeriod = newPeriod;
    }

    /**
     * @dev Update proposal threshold (owner only)
     */
    function setProposalThreshold(uint256 newThreshold) external onlyOwner {
        proposalThreshold = newThreshold;
    }

    /**
     * @dev Update quorum percentage (owner only)
     */
    function setQuorumPercentage(uint256 newQuorum) external onlyOwner {
        require(newQuorum >= 100 && newQuorum <= 5000, "Invalid quorum"); // 1-50%
        quorumPercentage = newQuorum;
    }
}
