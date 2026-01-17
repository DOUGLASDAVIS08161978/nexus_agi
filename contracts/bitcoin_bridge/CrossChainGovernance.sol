// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title CrossChainGovernance
 * @dev Decentralized governance for Bitcoin bridge operations
 * Token-based voting system for bridge parameter updates
 */
contract CrossChainGovernance {

    // ============ State Variables ============

    address public owner;
    address public governanceToken;

    uint256 public proposalCount;
    uint256 public votingPeriod = 7 days;
    uint256 public executionDelay = 2 days;
    uint256 public proposalThreshold = 1000 ether; // Need 1000 tokens to propose
    uint256 public quorumVotes = 10000 ether; // Need 10k votes for quorum

    enum ProposalState {
        Pending,
        Active,
        Canceled,
        Defeated,
        Succeeded,
        Queued,
        Executed
    }

    enum ProposalType {
        BridgeFeeUpdate,
        OracleUpdate,
        OperatorChange,
        EmergencyAction,
        ParameterUpdate
    }

    struct Proposal {
        uint256 id;
        address proposer;
        ProposalType proposalType;
        string description;
        address targetContract;
        bytes callData;
        uint256 forVotes;
        uint256 againstVotes;
        uint256 abstainVotes;
        uint256 startBlock;
        uint256 endBlock;
        uint256 eta; // Execution time
        bool canceled;
        bool executed;
        mapping(address => Receipt) receipts;
    }

    struct Receipt {
        bool hasVoted;
        uint8 support; // 0=against, 1=for, 2=abstain
        uint256 votes;
    }

    mapping(uint256 => Proposal) public proposals;

    // Governance parameters
    mapping(address => uint256) public delegatedVotes;
    mapping(address => address) public delegates;

    // ============ Events ============

    event ProposalCreated(
        uint256 indexed proposalId,
        address indexed proposer,
        ProposalType proposalType,
        string description,
        uint256 startBlock,
        uint256 endBlock
    );

    event VoteCast(
        address indexed voter,
        uint256 indexed proposalId,
        uint8 support,
        uint256 votes,
        string reason
    );

    event ProposalCanceled(uint256 indexed proposalId);
    event ProposalQueued(uint256 indexed proposalId, uint256 eta);
    event ProposalExecuted(uint256 indexed proposalId);

    event DelegateChanged(
        address indexed delegator,
        address indexed fromDelegate,
        address indexed toDelegate
    );

    event DelegateVotesChanged(
        address indexed delegate,
        uint256 previousBalance,
        uint256 newBalance
    );

    // ============ Modifiers ============

    modifier onlyOwner() {
        require(msg.sender == owner, "Not owner");
        _;
    }

    // ============ Constructor ============

    constructor(address _governanceToken) {
        owner = msg.sender;
        governanceToken = _governanceToken;
    }

    // ============ Proposal Functions ============

    /**
     * @dev Create new governance proposal
     */
    function propose(
        ProposalType proposalType,
        string memory description,
        address targetContract,
        bytes memory callData
    ) external returns (uint256) {
        require(
            getVotes(msg.sender) >= proposalThreshold,
            "Insufficient voting power"
        );
        require(bytes(description).length > 0, "Empty description");

        proposalCount++;
        uint256 proposalId = proposalCount;

        Proposal storage newProposal = proposals[proposalId];
        newProposal.id = proposalId;
        newProposal.proposer = msg.sender;
        newProposal.proposalType = proposalType;
        newProposal.description = description;
        newProposal.targetContract = targetContract;
        newProposal.callData = callData;
        newProposal.startBlock = block.number;
        newProposal.endBlock = block.number + ((votingPeriod * 1 days) / 12 seconds);
        newProposal.forVotes = 0;
        newProposal.againstVotes = 0;
        newProposal.abstainVotes = 0;
        newProposal.canceled = false;
        newProposal.executed = false;

        emit ProposalCreated(
            proposalId,
            msg.sender,
            proposalType,
            description,
            newProposal.startBlock,
            newProposal.endBlock
        );

        return proposalId;
    }

    /**
     * @dev Cast vote on proposal
     * @param support 0=against, 1=for, 2=abstain
     */
    function castVote(uint256 proposalId, uint8 support) external {
        return _castVote(msg.sender, proposalId, support, "");
    }

    /**
     * @dev Cast vote with reason
     */
    function castVoteWithReason(
        uint256 proposalId,
        uint8 support,
        string calldata reason
    ) external {
        return _castVote(msg.sender, proposalId, support, reason);
    }

    /**
     * @dev Internal vote casting
     */
    function _castVote(
        address voter,
        uint256 proposalId,
        uint8 support,
        string memory reason
    ) internal {
        require(state(proposalId) == ProposalState.Active, "Not active");
        require(support <= 2, "Invalid vote type");

        Proposal storage proposal = proposals[proposalId];
        Receipt storage receipt = proposal.receipts[voter];

        require(!receipt.hasVoted, "Already voted");

        uint256 votes = getVotes(voter);
        require(votes > 0, "No voting power");

        if (support == 0) {
            proposal.againstVotes += votes;
        } else if (support == 1) {
            proposal.forVotes += votes;
        } else {
            proposal.abstainVotes += votes;
        }

        receipt.hasVoted = true;
        receipt.support = support;
        receipt.votes = votes;

        emit VoteCast(voter, proposalId, support, votes, reason);
    }

    /**
     * @dev Queue successful proposal for execution
     */
    function queue(uint256 proposalId) external {
        require(
            state(proposalId) == ProposalState.Succeeded,
            "Not succeeded"
        );

        Proposal storage proposal = proposals[proposalId];
        uint256 eta = block.timestamp + executionDelay;
        proposal.eta = eta;

        emit ProposalQueued(proposalId, eta);
    }

    /**
     * @dev Execute queued proposal
     */
    function execute(uint256 proposalId) external {
        require(
            state(proposalId) == ProposalState.Queued,
            "Not queued"
        );

        Proposal storage proposal = proposals[proposalId];
        require(block.timestamp >= proposal.eta, "Not ready");

        proposal.executed = true;

        // Execute proposal
        (bool success, ) = proposal.targetContract.call(proposal.callData);
        require(success, "Execution failed");

        emit ProposalExecuted(proposalId);
    }

    /**
     * @dev Cancel proposal (proposer or owner)
     */
    function cancel(uint256 proposalId) external {
        Proposal storage proposal = proposals[proposalId];
        require(
            msg.sender == proposal.proposer || msg.sender == owner,
            "Not authorized"
        );
        require(!proposal.executed, "Already executed");

        proposal.canceled = true;

        emit ProposalCanceled(proposalId);
    }

    // ============ Delegation Functions ============

    /**
     * @dev Delegate votes to another address
     */
    function delegate(address delegatee) external {
        address currentDelegate = delegates[msg.sender];
        uint256 delegatorBalance = _getTokenBalance(msg.sender);

        delegates[msg.sender] = delegatee;

        emit DelegateChanged(msg.sender, currentDelegate, delegatee);

        _moveDelegates(currentDelegate, delegatee, delegatorBalance);
    }

    /**
     * @dev Move delegated votes
     */
    function _moveDelegates(
        address srcRep,
        address dstRep,
        uint256 amount
    ) internal {
        if (srcRep != dstRep && amount > 0) {
            if (srcRep != address(0)) {
                uint256 srcRepOld = delegatedVotes[srcRep];
                uint256 srcRepNew = srcRepOld - amount;
                delegatedVotes[srcRep] = srcRepNew;

                emit DelegateVotesChanged(srcRep, srcRepOld, srcRepNew);
            }

            if (dstRep != address(0)) {
                uint256 dstRepOld = delegatedVotes[dstRep];
                uint256 dstRepNew = dstRepOld + amount;
                delegatedVotes[dstRep] = dstRepNew;

                emit DelegateVotesChanged(dstRep, dstRepOld, dstRepNew);
            }
        }
    }

    // ============ View Functions ============

    /**
     * @dev Get current proposal state
     */
    function state(uint256 proposalId) public view returns (ProposalState) {
        require(proposalId <= proposalCount && proposalId > 0, "Invalid proposal");

        Proposal storage proposal = proposals[proposalId];

        if (proposal.canceled) {
            return ProposalState.Canceled;
        } else if (block.number <= proposal.startBlock) {
            return ProposalState.Pending;
        } else if (block.number <= proposal.endBlock) {
            return ProposalState.Active;
        } else if (
            proposal.forVotes <= proposal.againstVotes ||
            proposal.forVotes < quorumVotes
        ) {
            return ProposalState.Defeated;
        } else if (proposal.eta == 0) {
            return ProposalState.Succeeded;
        } else if (proposal.executed) {
            return ProposalState.Executed;
        } else if (block.timestamp >= proposal.eta) {
            return ProposalState.Queued;
        } else {
            return ProposalState.Queued;
        }
    }

    /**
     * @dev Get voting power of address
     */
    function getVotes(address account) public view returns (uint256) {
        address delegate = delegates[account];
        if (delegate == address(0)) {
            return _getTokenBalance(account);
        }
        return delegatedVotes[delegate];
    }

    /**
     * @dev Get token balance (external call to governance token)
     */
    function _getTokenBalance(address account) internal view returns (uint256) {
        (bool success, bytes memory data) = governanceToken.staticcall(
            abi.encodeWithSignature("balanceOf(address)", account)
        );
        if (success && data.length >= 32) {
            return abi.decode(data, (uint256));
        }
        return 0;
    }

    /**
     * @dev Get proposal details
     */
    function getProposal(uint256 proposalId)
        external
        view
        returns (
            address proposer,
            ProposalType proposalType,
            string memory description,
            uint256 forVotes,
            uint256 againstVotes,
            uint256 abstainVotes,
            uint256 startBlock,
            uint256 endBlock,
            bool executed,
            bool canceled
        )
    {
        Proposal storage proposal = proposals[proposalId];
        return (
            proposal.proposer,
            proposal.proposalType,
            proposal.description,
            proposal.forVotes,
            proposal.againstVotes,
            proposal.abstainVotes,
            proposal.startBlock,
            proposal.endBlock,
            proposal.executed,
            proposal.canceled
        );
    }

    /**
     * @dev Get receipt for voter
     */
    function getReceipt(uint256 proposalId, address voter)
        external
        view
        returns (bool hasVoted, uint8 support, uint256 votes)
    {
        Receipt storage receipt = proposals[proposalId].receipts[voter];
        return (receipt.hasVoted, receipt.support, receipt.votes);
    }

    // ============ Admin Functions ============

    function setVotingPeriod(uint256 newPeriod) external onlyOwner {
        require(newPeriod >= 1 days, "Period too short");
        votingPeriod = newPeriod;
    }

    function setExecutionDelay(uint256 newDelay) external onlyOwner {
        require(newDelay >= 1 days, "Delay too short");
        executionDelay = newDelay;
    }

    function setProposalThreshold(uint256 newThreshold) external onlyOwner {
        proposalThreshold = newThreshold;
    }

    function setQuorumVotes(uint256 newQuorum) external onlyOwner {
        quorumVotes = newQuorum;
    }
}
