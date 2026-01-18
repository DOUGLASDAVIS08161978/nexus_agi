// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/token/ERC20/extensions/ERC20Burnable.sol";
import "@openzeppelin/contracts/access/AccessControl.sol";

/**
 * @title HashPowerToken
 * @dev Represents mining hash power in the Nexus AGI network
 *
 * Features:
 * - Mintable by authorized miners
 * - Represents computational power contribution
 * - Burnable to claim mining rewards
 * - Role-based minting authorization
 *
 * Symbol: HASH
 * Decimals: 18
 */
contract HashPowerToken is ERC20, ERC20Burnable, AccessControl {
    bytes32 public constant MINER_ROLE = keccak256("MINER_ROLE");
    bytes32 public constant ADMIN_ROLE = keccak256("ADMIN_ROLE");

    uint256 public totalHashPowerMined;
    mapping(address => uint256) public minerContribution;
    mapping(address => uint256) public lastMiningTime;

    event HashPowerMinted(address indexed miner, uint256 amount, uint256 timestamp);
    event HashPowerBurned(address indexed miner, uint256 amount, uint256 reward);
    event MinerRegistered(address indexed miner);

    constructor(address admin) ERC20("Hash Power Token", "HASH") {
        _grantRole(DEFAULT_ADMIN_ROLE, admin);
        _grantRole(ADMIN_ROLE, admin);
    }

    /**
     * @dev Register a new miner
     */
    function registerMiner(address miner) external onlyRole(ADMIN_ROLE) {
        _grantRole(MINER_ROLE, miner);
        emit MinerRegistered(miner);
    }

    /**
     * @dev Mint hash power tokens for mining contribution
     */
    function mintHashPower(address miner, uint256 amount) external onlyRole(MINER_ROLE) {
        require(amount > 0, "Amount must be positive");

        _mint(miner, amount);

        totalHashPowerMined += amount;
        minerContribution[miner] += amount;
        lastMiningTime[miner] = block.timestamp;

        emit HashPowerMinted(miner, amount, block.timestamp);
    }

    /**
     * @dev Burn hash power to claim mining rewards
     */
    function claimReward(uint256 amount) external returns (uint256) {
        require(balanceOf(msg.sender) >= amount, "Insufficient hash power");

        burn(amount);

        uint256 reward = calculateReward(amount);
        minerContribution[msg.sender] -= amount;

        emit HashPowerBurned(msg.sender, amount, reward);

        return reward;
    }

    /**
     * @dev Calculate reward for burned hash power
     */
    function calculateReward(uint256 hashPower) public pure returns (uint256) {
        // 1 HASH = 0.001 reward units
        return hashPower / 1000;
    }

    /**
     * @dev Get miner statistics
     */
    function getMinerStats(address miner) external view returns (
        uint256 balance,
        uint256 contribution,
        uint256 lastMining,
        uint256 estimatedReward
    ) {
        balance = balanceOf(miner);
        contribution = minerContribution[miner];
        lastMining = lastMiningTime[miner];
        estimatedReward = calculateReward(balance);
    }

    /**
     * @dev Get total network hash power
     */
    function getTotalHashPower() external view returns (uint256) {
        return totalHashPowerMined;
    }
}
