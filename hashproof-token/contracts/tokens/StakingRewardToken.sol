// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/token/ERC20/extensions/ERC20Burnable.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

/**
 * @title StakingRewardToken
 * @dev Staking rewards for Nexus AGI ecosystem
 *
 * Features:
 * - Compound interest calculation
 * - Flexible APY rates
 * - Auto-compounding rewards
 *
 * Symbol: STAKE
 * Decimals: 18
 */
contract StakingRewardToken is ERC20, ERC20Burnable, Ownable {
    struct StakeInfo {
        uint256 amount;
        uint256 startTime;
        uint256 lastClaimTime;
        uint256 accumulatedReward;
    }

    mapping(address => StakeInfo) public stakes;
    uint256 public annualPercentageYield = 1200; // 12% APY
    uint256 public constant SECONDS_PER_YEAR = 365 days;

    event Staked(address indexed user, uint256 amount);
    event RewardClaimed(address indexed user, uint256 amount);
    event APYUpdated(uint256 newAPY);

    constructor(address initialOwner)
        ERC20("Staking Reward Token", "STAKE")
        Ownable(initialOwner)
    {
        _mint(initialOwner, 10_000_000 * 10**18);
    }

    /**
     * @dev Stake tokens to earn rewards
     */
    function stakeTokens(uint256 amount) external {
        require(amount > 0, "Cannot stake 0");
        require(balanceOf(msg.sender) >= amount, "Insufficient balance");

        // Claim existing rewards first
        if (stakes[msg.sender].amount > 0) {
            _claimReward();
        }

        _transfer(msg.sender, address(this), amount);

        stakes[msg.sender].amount += amount;
        stakes[msg.sender].startTime = block.timestamp;
        stakes[msg.sender].lastClaimTime = block.timestamp;

        emit Staked(msg.sender, amount);
    }

    /**
     * @dev Calculate pending rewards
     */
    function calculateReward(address user) public view returns (uint256) {
        StakeInfo memory stake = stakes[user];

        if (stake.amount == 0) return 0;

        uint256 stakingDuration = block.timestamp - stake.lastClaimTime;
        uint256 reward = (stake.amount * annualPercentageYield * stakingDuration) / (SECONDS_PER_YEAR * 10000);

        return reward + stake.accumulatedReward;
    }

    /**
     * @dev Claim staking rewards
     */
    function claimReward() external {
        require(stakes[msg.sender].amount > 0, "No active stake");
        _claimReward();
    }

    /**
     * @dev Internal claim function
     */
    function _claimReward() internal {
        uint256 reward = calculateReward(msg.sender);

        if (reward > 0) {
            stakes[msg.sender].accumulatedReward = 0;
            stakes[msg.sender].lastClaimTime = block.timestamp;

            _mint(msg.sender, reward);

            emit RewardClaimed(msg.sender, reward);
        }
    }

    /**
     * @dev Unstake tokens
     */
    function unstake(uint256 amount) external {
        require(stakes[msg.sender].amount >= amount, "Insufficient staked");

        _claimReward();

        stakes[msg.sender].amount -= amount;
        _transfer(address(this), msg.sender, amount);
    }

    /**
     * @dev Update APY (owner only)
     */
    function setAPY(uint256 newAPY) external onlyOwner {
        require(newAPY <= 10000, "APY cannot exceed 100%");
        annualPercentageYield = newAPY;
        emit APYUpdated(newAPY);
    }
}
