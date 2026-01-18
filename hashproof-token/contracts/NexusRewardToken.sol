// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/token/ERC20/extensions/ERC20Burnable.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";

/**
 * @title NexusRewardToken
 * @dev Claimable reward token for Nexus AGI system
 *
 * Features:
 * - Public claimable rewards (faucet-style)
 * - Anti-spam protection (1 claim per hour)
 * - Burnable supply
 * - Owner-controlled reward amounts
 *
 * USE THIS CONTRACT FOR TESTING AND DEMONSTRATION
 * Deploy on Sepolia testnet first!
 */
contract NexusRewardToken is ERC20, ERC20Burnable, Ownable, ReentrancyGuard {

    // Reward amount per claim (in tokens, with decimals)
    uint256 public rewardAmount = 100 * 10**18;  // 100 tokens

    // Cooldown period between claims (1 hour)
    uint256 public cooldownPeriod = 1 hours;

    // Track last claim time for each address
    mapping(address => uint256) public lastClaimTime;

    // Total rewards claimed
    uint256 public totalRewardsClaimed;

    // Total unique claimers
    uint256 public totalClaimers;

    // Track if address has ever claimed
    mapping(address => bool) public hasClaimed;

    // Events
    event RewardClaimed(address indexed claimer, uint256 amount, uint256 timestamp);
    event RewardAmountUpdated(uint256 newAmount);
    event CooldownPeriodUpdated(uint256 newPeriod);
    event TokensDeposited(address indexed depositor, uint256 amount);

    /**
     * @dev Constructor
     * @param initialOwner Address of contract owner
     */
    constructor(address initialOwner)
        ERC20("Nexus Reward Token", "NREW")
        Ownable(initialOwner)
    {
        // Mint initial supply to owner for distribution
        _mint(initialOwner, 1_000_000 * 10**18);  // 1 million tokens
    }

    /**
     * @dev Claim reward tokens (public faucet)
     * Anyone can claim once per cooldown period
     */
    function claimReward() external nonReentrant {
        require(
            block.timestamp >= lastClaimTime[msg.sender] + cooldownPeriod,
            "Cooldown period not elapsed"
        );

        require(
            balanceOf(address(this)) >= rewardAmount,
            "Insufficient contract balance"
        );

        // Update state
        lastClaimTime[msg.sender] = block.timestamp;

        if (!hasClaimed[msg.sender]) {
            hasClaimed[msg.sender] = true;
            totalClaimers++;
        }

        totalRewardsClaimed += rewardAmount;

        // Transfer reward
        _transfer(address(this), msg.sender, rewardAmount);

        emit RewardClaimed(msg.sender, rewardAmount, block.timestamp);
    }

    /**
     * @dev Deposit tokens into contract for rewards
     * @param amount Amount to deposit
     */
    function depositRewards(uint256 amount) external {
        require(amount > 0, "Amount must be positive");
        require(balanceOf(msg.sender) >= amount, "Insufficient balance");

        _transfer(msg.sender, address(this), amount);

        emit TokensDeposited(msg.sender, amount);
    }

    /**
     * @dev Check time until next claim is available
     * @param account Address to check
     * @return Time remaining in seconds (0 if ready to claim)
     */
    function timeUntilNextClaim(address account) external view returns (uint256) {
        uint256 nextClaimTime = lastClaimTime[account] + cooldownPeriod;

        if (block.timestamp >= nextClaimTime) {
            return 0;
        }

        return nextClaimTime - block.timestamp;
    }

    /**
     * @dev Check if address can claim now
     * @param account Address to check
     * @return True if can claim
     */
    function canClaim(address account) external view returns (bool) {
        return block.timestamp >= lastClaimTime[account] + cooldownPeriod
            && balanceOf(address(this)) >= rewardAmount;
    }

    /**
     * @dev Get contract reward pool balance
     * @return Available rewards in contract
     */
    function getRewardPoolBalance() external view returns (uint256) {
        return balanceOf(address(this));
    }

    /**
     * @dev Get contract statistics
     */
    function getStats() external view returns (
        uint256 contractBalance,
        uint256 rewardsClaimedTotal,
        uint256 uniqueClaimers,
        uint256 currentRewardAmount,
        uint256 currentCooldown
    ) {
        contractBalance = balanceOf(address(this));
        rewardsClaimedTotal = totalRewardsClaimed;
        uniqueClaimers = totalClaimers;
        currentRewardAmount = rewardAmount;
        currentCooldown = cooldownPeriod;
    }

    /**
     * @dev Update reward amount (owner only)
     * @param newAmount New reward amount per claim
     */
    function setRewardAmount(uint256 newAmount) external onlyOwner {
        require(newAmount > 0, "Amount must be positive");
        rewardAmount = newAmount;
        emit RewardAmountUpdated(newAmount);
    }

    /**
     * @dev Update cooldown period (owner only)
     * @param newPeriod New cooldown period in seconds
     */
    function setCooldownPeriod(uint256 newPeriod) external onlyOwner {
        require(newPeriod >= 1 minutes, "Cooldown too short");
        require(newPeriod <= 7 days, "Cooldown too long");
        cooldownPeriod = newPeriod;
        emit CooldownPeriodUpdated(newPeriod);
    }

    /**
     * @dev Emergency withdraw (owner only)
     * @param amount Amount to withdraw
     */
    function emergencyWithdraw(uint256 amount) external onlyOwner {
        require(amount <= balanceOf(address(this)), "Insufficient balance");
        _transfer(address(this), owner(), amount);
    }

    /**
     * @dev Get user claim info
     */
    function getUserClaimInfo(address user) external view returns (
        uint256 lastClaim,
        uint256 nextClaimTime,
        bool hasClaimedBefore,
        bool canClaimNow
    ) {
        lastClaim = lastClaimTime[user];
        nextClaimTime = lastClaim + cooldownPeriod;
        hasClaimedBefore = hasClaimed[user];
        canClaimNow = block.timestamp >= nextClaimTime
            && balanceOf(address(this)) >= rewardAmount;
    }
}
