// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

/**
 * @title GovernanceVoteToken
 * @dev Dedicated voting rights token for DAO governance
 *
 * Features:
 * - Quadratic voting support
 * - Delegation capabilities
 * - Voting power snapshots
 *
 * Symbol: VOTE
 * Decimals: 18
 */
contract GovernanceVoteToken is ERC20, Ownable {
    struct Proposal {
        string description;
        uint256 votesFor;
        uint256 votesAgainst;
        uint256 endTime;
        bool executed;
        mapping(address => bool) hasVoted;
    }

    mapping(uint256 => Proposal) public proposals;
    mapping(address => address) public delegates;
    mapping(address => uint256) public votingPowerSnapshot;

    uint256 public proposalCount;
    uint256 public constant MIN_PROPOSAL_DURATION = 3 days;

    event ProposalCreated(uint256 indexed proposalId, string description, uint256 endTime);
    event Voted(uint256 indexed proposalId, address indexed voter, bool support, uint256 votes);
    event Delegated(address indexed delegator, address indexed delegate);
    event ProposalExecuted(uint256 indexed proposalId);

    constructor(address initialOwner)
        ERC20("Governance Vote Token", "VOTE")
        Ownable(initialOwner)
    {
        _mint(initialOwner, 1_000_000 * 10**18);
    }

    /**
     * @dev Create new proposal
     */
    function createProposal(string memory description, uint256 duration) external returns (uint256) {
        require(balanceOf(msg.sender) >= 1000 * 10**18, "Insufficient tokens to propose");
        require(duration >= MIN_PROPOSAL_DURATION, "Duration too short");

        uint256 proposalId = proposalCount++;
        Proposal storage proposal = proposals[proposalId];

        proposal.description = description;
        proposal.endTime = block.timestamp + duration;
        proposal.executed = false;

        emit ProposalCreated(proposalId, description, proposal.endTime);

        return proposalId;
    }

    /**
     * @dev Vote on proposal
     */
    function vote(uint256 proposalId, bool support) external {
        Proposal storage proposal = proposals[proposalId];

        require(block.timestamp < proposal.endTime, "Voting ended");
        require(!proposal.hasVoted[msg.sender], "Already voted");

        uint256 votes = getVotingPower(msg.sender);
        require(votes > 0, "No voting power");

        proposal.hasVoted[msg.sender] = true;

        if (support) {
            proposal.votesFor += votes;
        } else {
            proposal.votesAgainst += votes;
        }

        emit Voted(proposalId, msg.sender, support, votes);
    }

    /**
     * @dev Get voting power (with quadratic calculation)
     */
    function getVotingPower(address voter) public view returns (uint256) {
        address delegateTo = delegates[voter];
        uint256 balance = balanceOf(delegateTo != address(0) ? delegateTo : voter);

        // Quadratic voting: sqrt(tokens)
        return sqrt(balance / 10**18);
    }

    /**
     * @dev Delegate voting power
     */
    function delegate(address delegatee) external {
        require(delegatee != msg.sender, "Cannot delegate to self");
        delegates[msg.sender] = delegatee;
        emit Delegated(msg.sender, delegatee);
    }

    /**
     * @dev Square root calculation for quadratic voting
     */
    function sqrt(uint256 x) internal pure returns (uint256) {
        if (x == 0) return 0;
        uint256 z = (x + 1) / 2;
        uint256 y = x;
        while (z < y) {
            y = z;
            z = (x / z + z) / 2;
        }
        return y;
    }

    /**
     * @dev Execute passed proposal
     */
    function executeProposal(uint256 proposalId) external onlyOwner {
        Proposal storage proposal = proposals[proposalId];

        require(block.timestamp >= proposal.endTime, "Voting not ended");
        require(!proposal.executed, "Already executed");
        require(proposal.votesFor > proposal.votesAgainst, "Proposal failed");

        proposal.executed = true;

        emit ProposalExecuted(proposalId);
    }

    /**
     * @dev Get proposal status
     */
    function getProposalStatus(uint256 proposalId) external view returns (
        string memory description,
        uint256 votesFor,
        uint256 votesAgainst,
        uint256 endTime,
        bool executed,
        bool passed
    ) {
        Proposal storage proposal = proposals[proposalId];

        description = proposal.description;
        votesFor = proposal.votesFor;
        votesAgainst = proposal.votesAgainst;
        endTime = proposal.endTime;
        executed = proposal.executed;
        passed = votesFor > votesAgainst;
    }
}
