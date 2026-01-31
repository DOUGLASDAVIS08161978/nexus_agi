// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title WalletManagerSimple
 * @dev Simplified wallet manager without external dependencies
 */
contract WalletManagerSimple {
    address public owner;

    // User balances
    mapping(address => uint256) public ethBalances;

    // Reward tracking
    mapping(address => uint256) public rewards;
    mapping(address => uint256) public lastClaimTime;

    // Settings
    uint256 public rewardRate = 1e15; // 0.001 ETH per second
    uint256 public minDepositAmount = 0.001 ether;
    uint256 public totalDeposited;
    bool public paused;

    // Operators
    mapping(address => bool) public operators;

    // Events
    event EthDeposited(address indexed user, uint256 amount, uint256 timestamp);
    event EthWithdrawn(address indexed user, uint256 amount, uint256 timestamp);
    event RewardsClaimed(address indexed user, uint256 amount);
    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);

    modifier onlyOwner() {
        require(msg.sender == owner, "Not owner");
        _;
    }

    modifier whenNotPaused() {
        require(!paused, "Contract is paused");
        _;
    }

    constructor(address initialOwner) {
        owner = initialOwner;
        operators[initialOwner] = true;
    }

    // ========== ETH Operations ==========

    function depositETH() external payable whenNotPaused {
        require(msg.value >= minDepositAmount, "Amount too low");

        ethBalances[msg.sender] += msg.value;
        totalDeposited += msg.value;
        _updateRewards(msg.sender);

        emit EthDeposited(msg.sender, msg.value, block.timestamp);
    }

    function withdrawETH(uint256 amount) external {
        require(ethBalances[msg.sender] >= amount, "Insufficient balance");

        ethBalances[msg.sender] -= amount;
        totalDeposited -= amount;
        _updateRewards(msg.sender);

        (bool success, ) = msg.sender.call{value: amount}("");
        require(success, "Transfer failed");

        emit EthWithdrawn(msg.sender, amount, block.timestamp);
    }

    function withdrawAllETH() external {
        uint256 balance = ethBalances[msg.sender];
        require(balance > 0, "No balance");

        ethBalances[msg.sender] = 0;
        totalDeposited -= balance;
        _updateRewards(msg.sender);

        (bool success, ) = msg.sender.call{value: balance}("");
        require(success, "Transfer failed");

        emit EthWithdrawn(msg.sender, balance, block.timestamp);
    }

    // ========== Rewards System ==========

    function getPendingRewards(address user) public view returns (uint256) {
        if (lastClaimTime[user] == 0 || ethBalances[user] == 0) {
            return rewards[user];
        }

        uint256 timeElapsed = block.timestamp - lastClaimTime[user];
        uint256 newRewards = (ethBalances[user] * rewardRate * timeElapsed) / 1e18;

        return rewards[user] + newRewards;
    }

    function _updateRewards(address user) internal {
        if (lastClaimTime[user] > 0) {
            rewards[user] = getPendingRewards(user);
        }
        lastClaimTime[user] = block.timestamp;
    }

    function claimRewards() external {
        _updateRewards(msg.sender);

        uint256 reward = rewards[msg.sender];
        require(reward > 0, "No rewards");
        require(address(this).balance >= reward, "Insufficient contract balance");

        rewards[msg.sender] = 0;

        (bool success, ) = msg.sender.call{value: reward}("");
        require(success, "Transfer failed");

        emit RewardsClaimed(msg.sender, reward);
    }

    // ========== View Functions ==========

    function getETHBalance(address user) external view returns (uint256) {
        return ethBalances[user];
    }

    function getContractBalance() external view returns (uint256) {
        return address(this).balance;
    }

    function isOperator(address account) external view returns (bool) {
        return operators[account];
    }

    // ========== Admin Functions ==========

    function setRewardRate(uint256 newRate) external onlyOwner {
        rewardRate = newRate;
    }

    function setMinDepositAmount(uint256 newAmount) external onlyOwner {
        minDepositAmount = newAmount;
    }

    function addOperator(address operator) external onlyOwner {
        operators[operator] = true;
    }

    function removeOperator(address operator) external onlyOwner {
        operators[operator] = false;
    }

    function fundRewards() external payable onlyOwner {
        require(msg.value > 0, "Must send ETH");
    }

    function pause() external onlyOwner {
        paused = true;
    }

    function unpause() external onlyOwner {
        paused = false;
    }

    function transferOwnership(address newOwner) external onlyOwner {
        require(newOwner != address(0), "Invalid address");
        address oldOwner = owner;
        owner = newOwner;
        emit OwnershipTransferred(oldOwner, newOwner);
    }

    // ========== Fallback ==========

    receive() external payable {
        ethBalances[msg.sender] += msg.value;
        totalDeposited += msg.value;
        _updateRewards(msg.sender);
        emit EthDeposited(msg.sender, msg.value, block.timestamp);
    }
}
