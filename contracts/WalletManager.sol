// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";
import "@openzeppelin/contracts/utils/Pausable.sol";

/**
 * @title WalletManager
 * @dev Smart contract for managing wallet interactions, token operations, and rewards
 *
 * Features:
 * - Deposit and withdraw ETH
 * - Deposit and withdraw ERC20 tokens
 * - Reward distribution system
 * - User balance tracking
 * - Emergency pause functionality
 */
contract WalletManager is Ownable, ReentrancyGuard, Pausable {

    // User balances for ETH
    mapping(address => uint256) public ethBalances;

    // User balances for ERC20 tokens
    mapping(address => mapping(address => uint256)) public tokenBalances;

    // Reward tracking
    mapping(address => uint256) public rewards;
    mapping(address => uint256) public lastClaimTime;

    // Reward rate (wei per second)
    uint256 public rewardRate = 1e15; // 0.001 ETH per second

    // Minimum deposit amount
    uint256 public minDepositAmount = 0.001 ether;

    // Total deposited by all users
    uint256 public totalDeposited;

    // Authorized operators
    mapping(address => bool) public operators;

    // Events
    event EthDeposited(address indexed user, uint256 amount, uint256 timestamp);
    event EthWithdrawn(address indexed user, uint256 amount, uint256 timestamp);
    event TokenDeposited(address indexed user, address indexed token, uint256 amount);
    event TokenWithdrawn(address indexed user, address indexed token, uint256 amount);
    event RewardsClaimed(address indexed user, uint256 amount);
    event RewardRateUpdated(uint256 newRate);
    event OperatorAdded(address indexed operator);
    event OperatorRemoved(address indexed operator);

    modifier onlyOperator() {
        require(operators[msg.sender] || owner() == msg.sender, "Not authorized");
        _;
    }

    constructor(address initialOwner) Ownable(initialOwner) {
        operators[initialOwner] = true;
    }

    // ========== ETH Operations ==========

    /**
     * @dev Deposit ETH into the contract
     */
    function depositETH() external payable whenNotPaused nonReentrant {
        require(msg.value >= minDepositAmount, "Deposit amount too low");

        ethBalances[msg.sender] += msg.value;
        totalDeposited += msg.value;

        // Update reward tracking
        _updateRewards(msg.sender);

        emit EthDeposited(msg.sender, msg.value, block.timestamp);
    }

    /**
     * @dev Withdraw ETH from the contract
     */
    function withdrawETH(uint256 amount) external nonReentrant {
        require(ethBalances[msg.sender] >= amount, "Insufficient balance");

        ethBalances[msg.sender] -= amount;
        totalDeposited -= amount;

        _updateRewards(msg.sender);

        (bool success, ) = msg.sender.call{value: amount}("");
        require(success, "ETH transfer failed");

        emit EthWithdrawn(msg.sender, amount, block.timestamp);
    }

    /**
     * @dev Withdraw all ETH balance
     */
    function withdrawAllETH() external nonReentrant {
        uint256 balance = ethBalances[msg.sender];
        require(balance > 0, "No balance to withdraw");

        ethBalances[msg.sender] = 0;
        totalDeposited -= balance;

        _updateRewards(msg.sender);

        (bool success, ) = msg.sender.call{value: balance}("");
        require(success, "ETH transfer failed");

        emit EthWithdrawn(msg.sender, balance, block.timestamp);
    }

    // ========== ERC20 Token Operations ==========

    /**
     * @dev Deposit ERC20 tokens into the contract
     */
    function depositToken(address token, uint256 amount)
        external
        whenNotPaused
        nonReentrant
    {
        require(token != address(0), "Invalid token address");
        require(amount > 0, "Amount must be > 0");

        IERC20 tokenContract = IERC20(token);
        require(
            tokenContract.transferFrom(msg.sender, address(this), amount),
            "Token transfer failed"
        );

        tokenBalances[msg.sender][token] += amount;

        emit TokenDeposited(msg.sender, token, amount);
    }

    /**
     * @dev Withdraw ERC20 tokens from the contract
     */
    function withdrawToken(address token, uint256 amount) external nonReentrant {
        require(tokenBalances[msg.sender][token] >= amount, "Insufficient token balance");

        tokenBalances[msg.sender][token] -= amount;

        IERC20 tokenContract = IERC20(token);
        require(tokenContract.transfer(msg.sender, amount), "Token transfer failed");

        emit TokenWithdrawn(msg.sender, token, amount);
    }

    /**
     * @dev Withdraw all tokens of a specific type
     */
    function withdrawAllTokens(address token) external nonReentrant {
        uint256 balance = tokenBalances[msg.sender][token];
        require(balance > 0, "No token balance to withdraw");

        tokenBalances[msg.sender][token] = 0;

        IERC20 tokenContract = IERC20(token);
        require(tokenContract.transfer(msg.sender, balance), "Token transfer failed");

        emit TokenWithdrawn(msg.sender, token, balance);
    }

    // ========== Rewards System ==========

    /**
     * @dev Calculate pending rewards for a user
     */
    function getPendingRewards(address user) public view returns (uint256) {
        if (lastClaimTime[user] == 0 || ethBalances[user] == 0) {
            return rewards[user];
        }

        uint256 timeElapsed = block.timestamp - lastClaimTime[user];
        uint256 newRewards = (ethBalances[user] * rewardRate * timeElapsed) / 1e18;

        return rewards[user] + newRewards;
    }

    /**
     * @dev Update rewards for a user
     */
    function _updateRewards(address user) internal {
        if (lastClaimTime[user] > 0) {
            rewards[user] = getPendingRewards(user);
        }
        lastClaimTime[user] = block.timestamp;
    }

    /**
     * @dev Claim accumulated rewards
     */
    function claimRewards() external nonReentrant {
        _updateRewards(msg.sender);

        uint256 reward = rewards[msg.sender];
        require(reward > 0, "No rewards to claim");
        require(address(this).balance >= reward, "Insufficient contract balance");

        rewards[msg.sender] = 0;

        (bool success, ) = msg.sender.call{value: reward}("");
        require(success, "Reward transfer failed");

        emit RewardsClaimed(msg.sender, reward);
    }

    // ========== Admin Functions ==========

    /**
     * @dev Update the reward rate (only owner)
     */
    function setRewardRate(uint256 newRate) external onlyOwner {
        rewardRate = newRate;
        emit RewardRateUpdated(newRate);
    }

    /**
     * @dev Set minimum deposit amount
     */
    function setMinDepositAmount(uint256 newAmount) external onlyOwner {
        minDepositAmount = newAmount;
    }

    /**
     * @dev Add an operator
     */
    function addOperator(address operator) external onlyOwner {
        operators[operator] = true;
        emit OperatorAdded(operator);
    }

    /**
     * @dev Remove an operator
     */
    function removeOperator(address operator) external onlyOwner {
        operators[operator] = false;
        emit OperatorRemoved(operator);
    }

    /**
     * @dev Fund the contract with ETH for rewards
     */
    function fundRewards() external payable onlyOwner {
        require(msg.value > 0, "Must send ETH");
    }

    /**
     * @dev Withdraw contract balance (only owner, for emergencies)
     */
    function emergencyWithdraw() external onlyOwner {
        uint256 balance = address(this).balance;
        (bool success, ) = owner().call{value: balance}("");
        require(success, "Withdrawal failed");
    }

    /**
     * @dev Pause contract operations
     */
    function pause() external onlyOwner {
        _pause();
    }

    /**
     * @dev Unpause contract operations
     */
    function unpause() external onlyOwner {
        _unpause();
    }

    // ========== View Functions ==========

    /**
     * @dev Get ETH balance for a user
     */
    function getETHBalance(address user) external view returns (uint256) {
        return ethBalances[user];
    }

    /**
     * @dev Get token balance for a user
     */
    function getTokenBalance(address user, address token) external view returns (uint256) {
        return tokenBalances[user][token];
    }

    /**
     * @dev Get contract's total ETH balance
     */
    function getContractBalance() external view returns (uint256) {
        return address(this).balance;
    }

    /**
     * @dev Check if address is an operator
     */
    function isOperator(address account) external view returns (bool) {
        return operators[account];
    }

    // ========== Fallback ==========

    /**
     * @dev Receive ETH directly
     */
    receive() external payable {
        ethBalances[msg.sender] += msg.value;
        totalDeposited += msg.value;
        _updateRewards(msg.sender);
        emit EthDeposited(msg.sender, msg.value, block.timestamp);
    }
}
