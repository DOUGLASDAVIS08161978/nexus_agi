// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/token/ERC20/extensions/ERC20Burnable.sol";
import "@openzeppelin/contracts/token/ERC20/extensions/ERC20Pausable.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/token/ERC20/extensions/ERC20Permit.sol";

/**
 * @title NexusToken
 * @dev Main governance token for Nexus AGI ecosystem
 *
 * Features:
 * - Governance voting rights
 * - Staking capabilities
 * - Burnable supply
 * - Pausable for emergencies
 * - EIP-2612 Permit support
 *
 * Total Supply: 100,000,000 NEX
 * Decimals: 18
 */
contract NexusToken is ERC20, ERC20Burnable, ERC20Pausable, Ownable, ERC20Permit {
    uint256 public constant MAX_SUPPLY = 100_000_000 * 10**18;

    mapping(address => uint256) public stakingBalance;
    mapping(address => uint256) public stakingTimestamp;
    mapping(address => uint256) public votingPower;

    event Staked(address indexed user, uint256 amount);
    event Unstaked(address indexed user, uint256 amount);
    event VotingPowerUpdated(address indexed user, uint256 newPower);

    constructor(address initialOwner)
        ERC20("Nexus AGI Token", "NEX")
        Ownable(initialOwner)
        ERC20Permit("Nexus AGI Token")
    {
        _mint(initialOwner, MAX_SUPPLY);
    }

    /**
     * @dev Stake tokens to gain voting power
     */
    function stake(uint256 amount) external whenNotPaused {
        require(amount > 0, "Cannot stake 0");
        require(balanceOf(msg.sender) >= amount, "Insufficient balance");

        _transfer(msg.sender, address(this), amount);

        stakingBalance[msg.sender] += amount;
        stakingTimestamp[msg.sender] = block.timestamp;

        _updateVotingPower(msg.sender);

        emit Staked(msg.sender, amount);
    }

    /**
     * @dev Unstake tokens
     */
    function unstake(uint256 amount) external {
        require(amount > 0, "Cannot unstake 0");
        require(stakingBalance[msg.sender] >= amount, "Insufficient staked balance");

        stakingBalance[msg.sender] -= amount;
        _transfer(address(this), msg.sender, amount);

        _updateVotingPower(msg.sender);

        emit Unstaked(msg.sender, amount);
    }

    /**
     * @dev Update voting power based on staking
     */
    function _updateVotingPower(address user) internal {
        uint256 stakeDuration = block.timestamp - stakingTimestamp[user];
        uint256 timeBonus = stakeDuration / 1 days;

        votingPower[user] = stakingBalance[user] * (100 + timeBonus) / 100;

        emit VotingPowerUpdated(user, votingPower[user]);
    }

    /**
     * @dev Pause token transfers (emergency only)
     */
    function pause() external onlyOwner {
        _pause();
    }

    /**
     * @dev Unpause token transfers
     */
    function unpause() external onlyOwner {
        _unpause();
    }

    /**
     * @dev Override required by Solidity
     */
    function _update(address from, address to, uint256 value)
        internal
        override(ERC20, ERC20Pausable)
    {
        super._update(from, to, value);
    }

    /**
     * @dev Get user's total governance power
     */
    function getGovernancePower(address user) external view returns (uint256) {
        return balanceOf(user) + votingPower[user];
    }
}
