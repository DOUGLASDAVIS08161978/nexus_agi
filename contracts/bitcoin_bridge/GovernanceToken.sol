// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title GovernanceToken (BGOV)
 * @dev Governance token for Bitcoin Bridge ecosystem
 * Used for voting on bridge parameters and protocol upgrades
 */
contract GovernanceToken {

    // ============ ERC20 State ============

    string public constant name = "Bridge Governance Token";
    string public constant symbol = "BGOV";
    uint8 public constant decimals = 18;
    uint256 public totalSupply;

    uint256 public constant MAX_SUPPLY = 100_000_000 ether; // 100 million tokens

    mapping(address => uint256) public balanceOf;
    mapping(address => mapping(address => uint256)) public allowance;

    // ============ Governance State ============

    mapping(address => address) public delegates;
    mapping(address => uint256) public numCheckpoints;
    mapping(address => mapping(uint256 => Checkpoint)) public checkpoints;

    struct Checkpoint {
        uint32 fromBlock;
        uint256 votes;
    }

    // ============ Access Control ============

    address public owner;
    mapping(address => bool) public minters;

    bool public transfersEnabled;
    uint256 public launchTime;

    // ============ Staking & Rewards ============

    struct StakeInfo {
        uint256 amount;
        uint256 startTime;
        uint256 rewardDebt;
        bool active;
    }

    mapping(address => StakeInfo) public stakes;
    uint256 public totalStaked;
    uint256 public rewardRate = 1000; // 10% APY in basis points

    // ============ Events ============

    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);
    event DelegateChanged(address indexed delegator, address indexed fromDelegate, address indexed toDelegate);
    event DelegateVotesChanged(address indexed delegate, uint256 previousBalance, uint256 newBalance);
    event Staked(address indexed user, uint256 amount);
    event Unstaked(address indexed user, uint256 amount, uint256 reward);
    event TransfersEnabled(uint256 timestamp);

    // ============ Modifiers ============

    modifier onlyOwner() {
        require(msg.sender == owner, "Not owner");
        _;
    }

    modifier onlyMinter() {
        require(minters[msg.sender], "Not minter");
        _;
    }

    modifier whenTransfersEnabled() {
        require(transfersEnabled, "Transfers disabled");
        _;
    }

    // ============ Constructor ============

    constructor() {
        owner = msg.sender;
        minters[msg.sender] = true;
        launchTime = block.timestamp;

        // Mint initial supply to treasury
        _mint(msg.sender, 10_000_000 ether); // 10% initial supply
    }

    // ============ ERC20 Functions ============

    function transfer(address to, uint256 value)
        external
        whenTransfersEnabled
        returns (bool)
    {
        require(to != address(0), "Invalid recipient");
        require(balanceOf[msg.sender] >= value, "Insufficient balance");

        _transfer(msg.sender, to, value);
        return true;
    }

    function approve(address spender, uint256 value) external returns (bool) {
        require(spender != address(0), "Invalid spender");

        allowance[msg.sender][spender] = value;
        emit Approval(msg.sender, spender, value);

        return true;
    }

    function transferFrom(address from, address to, uint256 value)
        external
        whenTransfersEnabled
        returns (bool)
    {
        require(to != address(0), "Invalid recipient");
        require(balanceOf[from] >= value, "Insufficient balance");
        require(allowance[from][msg.sender] >= value, "Insufficient allowance");

        allowance[from][msg.sender] -= value;
        _transfer(from, to, value);

        return true;
    }

    function _transfer(address from, address to, uint256 value) internal {
        balanceOf[from] -= value;
        balanceOf[to] += value;

        emit Transfer(from, to, value);

        _moveDelegates(delegates[from], delegates[to], value);
    }

    // ============ Minting ============

    function mint(address to, uint256 amount) external onlyMinter returns (bool) {
        require(totalSupply + amount <= MAX_SUPPLY, "Exceeds max supply");
        _mint(to, amount);
        return true;
    }

    function _mint(address to, uint256 amount) internal {
        require(to != address(0), "Invalid recipient");

        totalSupply += amount;
        balanceOf[to] += amount;

        emit Transfer(address(0), to, amount);

        _moveDelegates(address(0), delegates[to], amount);
    }

    // ============ Delegation ============

    function delegate(address delegatee) external {
        return _delegate(msg.sender, delegatee);
    }

    function _delegate(address delegator, address delegatee) internal {
        address currentDelegate = delegates[delegator];
        uint256 delegatorBalance = balanceOf[delegator];

        delegates[delegator] = delegatee;

        emit DelegateChanged(delegator, currentDelegate, delegatee);

        _moveDelegates(currentDelegate, delegatee, delegatorBalance);
    }

    function _moveDelegates(address srcRep, address dstRep, uint256 amount) internal {
        if (srcRep != dstRep && amount > 0) {
            if (srcRep != address(0)) {
                uint32 srcRepNum = numCheckpoints[srcRep];
                uint256 srcRepOld = srcRepNum > 0 ? checkpoints[srcRep][srcRepNum - 1].votes : 0;
                uint256 srcRepNew = srcRepOld - amount;
                _writeCheckpoint(srcRep, srcRepNum, srcRepOld, srcRepNew);
            }

            if (dstRep != address(0)) {
                uint32 dstRepNum = numCheckpoints[dstRep];
                uint256 dstRepOld = dstRepNum > 0 ? checkpoints[dstRep][dstRepNum - 1].votes : 0;
                uint256 dstRepNew = dstRepOld + amount;
                _writeCheckpoint(dstRep, dstRepNum, dstRepOld, dstRepNew);
            }
        }
    }

    function _writeCheckpoint(
        address delegatee,
        uint32 nCheckpoints,
        uint256 oldVotes,
        uint256 newVotes
    ) internal {
        uint32 blockNumber = uint32(block.number);

        if (nCheckpoints > 0 && checkpoints[delegatee][nCheckpoints - 1].fromBlock == blockNumber) {
            checkpoints[delegatee][nCheckpoints - 1].votes = newVotes;
        } else {
            checkpoints[delegatee][nCheckpoints] = Checkpoint(blockNumber, newVotes);
            numCheckpoints[delegatee] = nCheckpoints + 1;
        }

        emit DelegateVotesChanged(delegatee, oldVotes, newVotes);
    }

    // ============ Staking ============

    function stake(uint256 amount) external {
        require(amount > 0, "Invalid amount");
        require(balanceOf[msg.sender] >= amount, "Insufficient balance");

        StakeInfo storage stakeInfo = stakes[msg.sender];

        // Claim pending rewards if already staking
        if (stakeInfo.active) {
            uint256 pending = _calculateReward(msg.sender);
            if (pending > 0) {
                _mint(msg.sender, pending);
            }
        }

        balanceOf[msg.sender] -= amount;
        stakeInfo.amount += amount;
        stakeInfo.startTime = block.timestamp;
        stakeInfo.active = true;
        totalStaked += amount;

        emit Staked(msg.sender, amount);
    }

    function unstake(uint256 amount) external {
        StakeInfo storage stakeInfo = stakes[msg.sender];
        require(stakeInfo.active, "Not staking");
        require(stakeInfo.amount >= amount, "Insufficient stake");

        uint256 reward = _calculateReward(msg.sender);

        stakeInfo.amount -= amount;
        if (stakeInfo.amount == 0) {
            stakeInfo.active = false;
        }
        totalStaked -= amount;

        balanceOf[msg.sender] += amount;

        if (reward > 0 && totalSupply + reward <= MAX_SUPPLY) {
            _mint(msg.sender, reward);
        }

        emit Unstaked(msg.sender, amount, reward);
    }

    function _calculateReward(address user) internal view returns (uint256) {
        StakeInfo memory stakeInfo = stakes[user];
        if (!stakeInfo.active) return 0;

        uint256 duration = block.timestamp - stakeInfo.startTime;
        uint256 reward = (stakeInfo.amount * rewardRate * duration) / (365 days * 10000);

        return reward;
    }

    function getStakeInfo(address user) external view returns (
        uint256 amount,
        uint256 pendingReward,
        uint256 stakedSince
    ) {
        StakeInfo memory stakeInfo = stakes[user];
        return (
            stakeInfo.amount,
            _calculateReward(user),
            stakeInfo.startTime
        );
    }

    // ============ Admin Functions ============

    function enableTransfers() external onlyOwner {
        transfersEnabled = true;
        emit TransfersEnabled(block.timestamp);
    }

    function addMinter(address minter) external onlyOwner {
        minters[minter] = true;
    }

    function removeMinter(address minter) external onlyOwner {
        minters[minter] = false;
    }

    function setRewardRate(uint256 newRate) external onlyOwner {
        require(newRate <= 5000, "Rate too high"); // Max 50% APY
        rewardRate = newRate;
    }

    // ============ View Functions ============

    function getCurrentVotes(address account) external view returns (uint256) {
        uint32 nCheckpoints = numCheckpoints[account];
        return nCheckpoints > 0 ? checkpoints[account][nCheckpoints - 1].votes : 0;
    }

    function getPriorVotes(address account, uint256 blockNumber)
        external
        view
        returns (uint256)
    {
        require(blockNumber < block.number, "Not yet determined");

        uint32 nCheckpoints = numCheckpoints[account];
        if (nCheckpoints == 0) {
            return 0;
        }

        if (checkpoints[account][nCheckpoints - 1].fromBlock <= blockNumber) {
            return checkpoints[account][nCheckpoints - 1].votes;
        }

        if (checkpoints[account][0].fromBlock > blockNumber) {
            return 0;
        }

        uint32 lower = 0;
        uint32 upper = nCheckpoints - 1;

        while (upper > lower) {
            uint32 center = upper - (upper - lower) / 2;
            Checkpoint memory cp = checkpoints[account][center];
            if (cp.fromBlock == blockNumber) {
                return cp.votes;
            } else if (cp.fromBlock < blockNumber) {
                lower = center;
            } else {
                upper = center - 1;
            }
        }

        return checkpoints[account][lower].votes;
    }
}
