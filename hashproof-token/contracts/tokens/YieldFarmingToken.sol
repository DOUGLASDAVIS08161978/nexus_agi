// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/token/ERC20/extensions/ERC20Burnable.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

/**
 * @title YieldFarmingToken
 * @dev Yield farming rewards for liquidity providers
 *
 * Features:
 * - Auto-compounding yield
 * - Multiple farming pools
 * - Boost mechanisms
 *
 * Symbol: YIELD
 * Decimals: 18
 */
contract YieldFarmingToken is ERC20, ERC20Burnable, Ownable {
    struct FarmingPool {
        string name;
        uint256 totalStaked;
        uint256 rewardRate;
        uint256 lastUpdateTime;
        uint256 rewardPerTokenStored;
        bool active;
    }

    struct UserStake {
        uint256 amount;
        uint256 rewardPerTokenPaid;
        uint256 rewards;
        uint256 startTime;
        uint256 boostMultiplier;
    }

    mapping(uint256 => FarmingPool) public farmingPools;
    mapping(uint256 => mapping(address => UserStake)) public userStakes;

    uint256 public poolCount;
    uint256 public constant BOOST_DURATION = 30 days;
    uint256 public constant MAX_BOOST = 200; // 2x

    event PoolCreated(uint256 indexed poolId, string name, uint256 rewardRate);
    event Staked(uint256 indexed poolId, address indexed user, uint256 amount);
    event Withdrawn(uint256 indexed poolId, address indexed user, uint256 amount);
    event RewardPaid(uint256 indexed poolId, address indexed user, uint256 reward);
    event BoostApplied(uint256 indexed poolId, address indexed user, uint256 multiplier);

    constructor(address initialOwner)
        ERC20("Yield Farming Token", "YIELD")
        Ownable(initialOwner)
    {
        _mint(initialOwner, 100_000_000 * 10**18);
    }

    /**
     * @dev Create new farming pool
     */
    function createPool(string memory name, uint256 rewardRate) external onlyOwner returns (uint256) {
        uint256 poolId = poolCount++;

        farmingPools[poolId] = FarmingPool({
            name: name,
            totalStaked: 0,
            rewardRate: rewardRate,
            lastUpdateTime: block.timestamp,
            rewardPerTokenStored: 0,
            active: true
        });

        emit PoolCreated(poolId, name, rewardRate);

        return poolId;
    }

    /**
     * @dev Stake tokens in farming pool
     */
    function stake(uint256 poolId, uint256 amount) external {
        require(farmingPools[poolId].active, "Pool not active");
        require(amount > 0, "Cannot stake 0");
        require(balanceOf(msg.sender) >= amount, "Insufficient balance");

        _updateReward(poolId, msg.sender);

        _transfer(msg.sender, address(this), amount);

        farmingPools[poolId].totalStaked += amount;
        userStakes[poolId][msg.sender].amount += amount;
        userStakes[poolId][msg.sender].startTime = block.timestamp;

        _calculateBoost(poolId, msg.sender);

        emit Staked(poolId, msg.sender, amount);
    }

    /**
     * @dev Withdraw staked tokens
     */
    function withdraw(uint256 poolId, uint256 amount) external {
        require(userStakes[poolId][msg.sender].amount >= amount, "Insufficient staked");

        _updateReward(poolId, msg.sender);

        farmingPools[poolId].totalStaked -= amount;
        userStakes[poolId][msg.sender].amount -= amount;

        _transfer(address(this), msg.sender, amount);

        emit Withdrawn(poolId, msg.sender, amount);
    }

    /**
     * @dev Claim farming rewards
     */
    function claimReward(uint256 poolId) external {
        _updateReward(poolId, msg.sender);

        uint256 reward = userStakes[poolId][msg.sender].rewards;
        require(reward > 0, "No rewards");

        userStakes[poolId][msg.sender].rewards = 0;

        _mint(msg.sender, reward);

        emit RewardPaid(poolId, msg.sender, reward);
    }

    /**
     * @dev Calculate boost multiplier based on stake duration
     */
    function _calculateBoost(uint256 poolId, address user) internal {
        uint256 duration = block.timestamp - userStakes[poolId][user].startTime;
        uint256 boost = 100 + (duration * 100 / BOOST_DURATION);

        if (boost > MAX_BOOST) {
            boost = MAX_BOOST;
        }

        userStakes[poolId][user].boostMultiplier = boost;

        emit BoostApplied(poolId, user, boost);
    }

    /**
     * @dev Update reward calculations
     */
    function _updateReward(uint256 poolId, address user) internal {
        FarmingPool storage pool = farmingPools[poolId];

        pool.rewardPerTokenStored = rewardPerToken(poolId);
        pool.lastUpdateTime = block.timestamp;

        if (user != address(0)) {
            userStakes[poolId][user].rewards = earned(poolId, user);
            userStakes[poolId][user].rewardPerTokenPaid = pool.rewardPerTokenStored;
        }
    }

    /**
     * @dev Calculate reward per token
     */
    function rewardPerToken(uint256 poolId) public view returns (uint256) {
        FarmingPool memory pool = farmingPools[poolId];

        if (pool.totalStaked == 0) {
            return pool.rewardPerTokenStored;
        }

        return pool.rewardPerTokenStored +
            ((block.timestamp - pool.lastUpdateTime) * pool.rewardRate * 1e18 / pool.totalStaked);
    }

    /**
     * @dev Calculate earned rewards for user
     */
    function earned(uint256 poolId, address user) public view returns (uint256) {
        UserStake memory stake = userStakes[poolId][user];

        uint256 baseReward = stake.amount * (rewardPerToken(poolId) - stake.rewardPerTokenPaid) / 1e18;
        uint256 boostedReward = baseReward * stake.boostMultiplier / 100;

        return boostedReward + stake.rewards;
    }

    /**
     * @dev Get user farming statistics
     */
    function getUserStats(uint256 poolId, address user) external view returns (
        uint256 staked,
        uint256 pendingRewards,
        uint256 boost,
        uint256 stakeDuration
    ) {
        UserStake memory stake = userStakes[poolId][user];
        staked = stake.amount;
        pendingRewards = earned(poolId, user);
        boost = stake.boostMultiplier;
        stakeDuration = block.timestamp - stake.startTime;
    }

    /**
     * @dev Get pool information
     */
    function getPoolInfo(uint256 poolId) external view returns (
        string memory name,
        uint256 totalStaked,
        uint256 rewardRate,
        bool active
    ) {
        FarmingPool memory pool = farmingPools[poolId];
        name = pool.name;
        totalStaked = pool.totalStaked;
        rewardRate = pool.rewardRate;
        active = pool.active;
    }

    /**
     * @dev Set pool status (owner only)
     */
    function setPoolStatus(uint256 poolId, bool active) external onlyOwner {
        farmingPools[poolId].active = active;
    }

    /**
     * @dev Update pool reward rate (owner only)
     */
    function setRewardRate(uint256 poolId, uint256 newRate) external onlyOwner {
        _updateReward(poolId, address(0));
        farmingPools[poolId].rewardRate = newRate;
    }
}
