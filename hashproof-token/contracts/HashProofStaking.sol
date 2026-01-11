// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";
import "@openzeppelin/contracts/utils/Pausable.sol";

interface IHashProof is IERC20 {
    function mintStakingReward(address staker, uint256 amount) external;
}

/**
 * @title HashProofStaking
 * @dev Staking contract for HPROOF tokens with compound interest
 *
 * Features:
 * - Stake HPROOF tokens to earn rewards
 * - Multiple staking pools with different APY and lock periods
 * - Auto-compounding rewards
 * - Emergency withdrawal (with penalty)
 * - Flexible staking (no lock) with lower APY
 */
contract HashProofStaking is Ownable, ReentrancyGuard, Pausable {
    using SafeERC20 for IERC20;

    IHashProof public hproofToken;

    // Staking pool structure
    struct StakingPool {
        uint256 duration;        // Lock period in seconds
        uint256 apy;            // Annual Percentage Yield in basis points (10000 = 100%)
        uint256 totalStaked;    // Total tokens staked in this pool
        bool active;            // Pool is active
    }

    // Stake info for each user
    struct StakeInfo {
        uint256 amount;         // Amount staked
        uint256 startTime;      // When stake started
        uint256 lastClaim;      // Last time rewards were claimed
        uint256 poolId;         // Which pool
        bool compounding;       // Auto-compound rewards
    }

    // Staking pools
    mapping(uint256 => StakingPool) public stakingPools;
    uint256 public poolCount;

    // User stakes (user => stake index => stake info)
    mapping(address => mapping(uint256 => StakeInfo)) public userStakes;
    mapping(address => uint256) public userStakeCount;

    // Total stats
    uint256 public totalStaked;
    uint256 public totalRewardsPaid;

    // Early withdrawal penalty (10% = 1000 basis points)
    uint256 public constant EARLY_WITHDRAWAL_PENALTY = 1000;
    uint256 public constant BASIS_POINTS = 10000;
    uint256 public constant SECONDS_PER_YEAR = 365 days;

    // Minimum stake amount
    uint256 public minStakeAmount = 100 * 10**18; // 100 HPROOF

    // Events
    event Staked(address indexed user, uint256 amount, uint256 poolId, uint256 stakeIndex);
    event Unstaked(address indexed user, uint256 amount, uint256 poolId, uint256 stakeIndex);
    event RewardsClaimed(address indexed user, uint256 amount, uint256 stakeIndex);
    event RewardsCompounded(address indexed user, uint256 amount, uint256 stakeIndex);
    event PoolCreated(uint256 poolId, uint256 duration, uint256 apy);
    event PoolUpdated(uint256 poolId, uint256 apy, bool active);
    event EmergencyWithdrawal(address indexed user, uint256 amount, uint256 penalty);

    constructor(address _hproofToken, address initialOwner) Ownable(initialOwner) {
        hproofToken = IHashProof(_hproofToken);

        // Create default staking pools
        _createPool(0, 500);           // Flexible: 0 lock, 5% APY
        _createPool(30 days, 1000);    // 30 days: 10% APY
        _createPool(90 days, 1500);    // 90 days: 15% APY
        _createPool(180 days, 2000);   // 180 days: 20% APY
        _createPool(365 days, 3000);   // 365 days: 30% APY
    }

    /**
     * @dev Create a new staking pool
     */
    function _createPool(uint256 duration, uint256 apy) internal {
        stakingPools[poolCount] = StakingPool({
            duration: duration,
            apy: apy,
            totalStaked: 0,
            active: true
        });

        emit PoolCreated(poolCount, duration, apy);
        poolCount++;
    }

    /**
     * @dev Stake tokens in a pool
     */
    function stake(uint256 amount, uint256 poolId, bool autoCompound)
        external
        nonReentrant
        whenNotPaused
    {
        require(amount >= minStakeAmount, "Amount below minimum");
        require(poolId < poolCount, "Invalid pool");
        require(stakingPools[poolId].active, "Pool not active");

        // Transfer tokens from user
        IERC20(address(hproofToken)).transferFrom(msg.sender, address(this), amount);

        // Create stake
        uint256 stakeIndex = userStakeCount[msg.sender];
        userStakes[msg.sender][stakeIndex] = StakeInfo({
            amount: amount,
            startTime: block.timestamp,
            lastClaim: block.timestamp,
            poolId: poolId,
            compounding: autoCompound
        });

        userStakeCount[msg.sender]++;
        stakingPools[poolId].totalStaked += amount;
        totalStaked += amount;

        emit Staked(msg.sender, amount, poolId, stakeIndex);
    }

    /**
     * @dev Calculate pending rewards for a stake
     */
    function calculateRewards(address user, uint256 stakeIndex)
        public
        view
        returns (uint256)
    {
        StakeInfo memory stakeInfo = userStakes[user][stakeIndex];
        if (stakeInfo.amount == 0) return 0;

        StakingPool memory pool = stakingPools[stakeInfo.poolId];

        uint256 stakingDuration = block.timestamp - stakeInfo.lastClaim;
        uint256 reward = (stakeInfo.amount * pool.apy * stakingDuration) /
            (BASIS_POINTS * SECONDS_PER_YEAR);

        return reward;
    }

    /**
     * @dev Claim rewards without unstaking
     */
    function claimRewards(uint256 stakeIndex)
        external
        nonReentrant
        whenNotPaused
    {
        StakeInfo storage stakeInfo = userStakes[msg.sender][stakeIndex];
        require(stakeInfo.amount > 0, "No stake found");

        uint256 reward = calculateRewards(msg.sender, stakeIndex);
        require(reward > 0, "No rewards to claim");

        stakeInfo.lastClaim = block.timestamp;

        if (stakeInfo.compounding) {
            // Add rewards to stake amount
            stakeInfo.amount += reward;
            stakingPools[stakeInfo.poolId].totalStaked += reward;
            totalStaked += reward;
            emit RewardsCompounded(msg.sender, reward, stakeIndex);
        } else {
            // Mint and transfer rewards
            hproofToken.mintStakingReward(msg.sender, reward);
            totalRewardsPaid += reward;
            emit RewardsClaimed(msg.sender, reward, stakeIndex);
        }
    }

    /**
     * @dev Unstake tokens
     */
    function unstake(uint256 stakeIndex)
        external
        nonReentrant
        whenNotPaused
    {
        StakeInfo storage stakeInfo = userStakes[msg.sender][stakeIndex];
        require(stakeInfo.amount > 0, "No stake found");

        StakingPool storage pool = stakingPools[stakeInfo.poolId];
        uint256 stakingDuration = block.timestamp - stakeInfo.startTime;

        // Check if lock period has passed
        bool isEarly = stakingDuration < pool.duration;

        // Calculate and claim any pending rewards first
        uint256 reward = calculateRewards(msg.sender, stakeIndex);
        if (reward > 0) {
            hproofToken.mintStakingReward(msg.sender, reward);
            totalRewardsPaid += reward;
        }

        uint256 amountToReturn = stakeInfo.amount;

        // Apply early withdrawal penalty if applicable
        if (isEarly && pool.duration > 0) {
            uint256 penalty = (amountToReturn * EARLY_WITHDRAWAL_PENALTY) / BASIS_POINTS;
            amountToReturn -= penalty;
            // Penalty stays in contract for future stakers
            emit EmergencyWithdrawal(msg.sender, amountToReturn, penalty);
        }

        // Update stats
        pool.totalStaked -= stakeInfo.amount;
        totalStaked -= stakeInfo.amount;

        // Clear stake
        delete userStakes[msg.sender][stakeIndex];

        // Return tokens
        IERC20(address(hproofToken)).transfer(msg.sender, amountToReturn);

        emit Unstaked(msg.sender, amountToReturn, stakeInfo.poolId, stakeIndex);
    }

    /**
     * @dev Get all user stakes
     */
    function getUserStakes(address user)
        external
        view
        returns (StakeInfo[] memory)
    {
        uint256 count = userStakeCount[user];
        StakeInfo[] memory stakes = new StakeInfo[](count);

        for (uint256 i = 0; i < count; i++) {
            stakes[i] = userStakes[user][i];
        }

        return stakes;
    }

    /**
     * @dev Get user total staked amount
     */
    function getUserTotalStaked(address user) external view returns (uint256) {
        uint256 total = 0;
        for (uint256 i = 0; i < userStakeCount[user]; i++) {
            total += userStakes[user][i].amount;
        }
        return total;
    }

    /**
     * @dev Get user total pending rewards
     */
    function getUserTotalRewards(address user) external view returns (uint256) {
        uint256 total = 0;
        for (uint256 i = 0; i < userStakeCount[user]; i++) {
            total += calculateRewards(user, i);
        }
        return total;
    }

    /**
     * @dev Update pool APY (owner only)
     */
    function updatePoolAPY(uint256 poolId, uint256 newAPY) external onlyOwner {
        require(poolId < poolCount, "Invalid pool");
        stakingPools[poolId].apy = newAPY;
        emit PoolUpdated(poolId, newAPY, stakingPools[poolId].active);
    }

    /**
     * @dev Toggle pool active status (owner only)
     */
    function togglePoolActive(uint256 poolId) external onlyOwner {
        require(poolId < poolCount, "Invalid pool");
        stakingPools[poolId].active = !stakingPools[poolId].active;
        emit PoolUpdated(
            poolId,
            stakingPools[poolId].apy,
            stakingPools[poolId].active
        );
    }

    /**
     * @dev Create new pool (owner only)
     */
    function createPool(uint256 duration, uint256 apy) external onlyOwner {
        _createPool(duration, apy);
    }

    /**
     * @dev Update minimum stake amount
     */
    function setMinStakeAmount(uint256 newAmount) external onlyOwner {
        minStakeAmount = newAmount;
    }

    /**
     * @dev Pause staking (emergency)
     */
    function pause() external onlyOwner {
        _pause();
    }

    /**
     * @dev Unpause staking
     */
    function unpause() external onlyOwner {
        _unpause();
    }

    /**
     * @dev Get pool details
     */
    function getPoolInfo(uint256 poolId)
        external
        view
        returns (
            uint256 duration,
            uint256 apy,
            uint256 poolTotalStaked,
            bool active
        )
    {
        StakingPool memory pool = stakingPools[poolId];
        return (pool.duration, pool.apy, pool.totalStaked, pool.active);
    }
}
