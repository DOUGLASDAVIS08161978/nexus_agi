// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/token/ERC20/extensions/ERC20Burnable.sol";
import "@openzeppelin/contracts/token/ERC20/extensions/ERC20Pausable.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";

/**
 * @title HashProof Token (HPROOF)
 * @dev Advanced ERC-20 token with deflationary mechanics and computational work rewards
 *
 * Features:
 * - 100,000,000 max supply (deflationary)
 * - 1% burn on every transfer (reduces supply over time)
 * - Mining reward distribution for computational work
 * - Anti-whale mechanics (max transaction limits)
 * - Pausable for emergency situations
 * - Owner-controlled minting for rewards
 *
 * Tokenomics:
 * - Initial Supply: 10,000,000 HPROOF (10%)
 * - Mining Rewards Pool: 50,000,000 HPROOF (50%)
 * - Staking Rewards: 30,000,000 HPROOF (30%)
 * - Team/Development: 10,000,000 HPROOF (10%)
 */
contract HashProof is ERC20, ERC20Burnable, ERC20Pausable, Ownable, ReentrancyGuard {
    // Maximum total supply (100 million tokens with 18 decimals)
    uint256 public constant MAX_SUPPLY = 100_000_000 * 10**18;

    // Burn rate: 1% = 100 basis points out of 10000
    uint256 public constant BURN_RATE = 100;
    uint256 public constant BASIS_POINTS = 10000;

    // Anti-whale: Max transaction size (1% of max supply)
    uint256 public maxTransactionAmount = 1_000_000 * 10**18;

    // Addresses exempt from fees and limits
    mapping(address => bool) public isExcludedFromFees;
    mapping(address => bool) public isExcludedFromLimits;

    // Mining reward tracker
    mapping(address => uint256) public miningRewards;
    mapping(address => uint256) public totalHashesSubmitted;

    // Pools for different allocations
    uint256 public miningRewardsPool = 50_000_000 * 10**18;
    uint256 public stakingRewardsPool = 30_000_000 * 10**18;
    uint256 public teamPool = 10_000_000 * 10**18;

    // Authorized minters (for staking contract, etc.)
    mapping(address => bool) public authorizedMinters;

    // Total burned tokens (for transparency)
    uint256 public totalBurned;

    // Events
    event TokensBurned(address indexed from, uint256 amount);
    event MiningRewardPaid(address indexed miner, uint256 amount, uint256 hashes);
    event ExcludedFromFees(address indexed account, bool excluded);
    event MaxTransactionUpdated(uint256 newAmount);
    event AuthorizedMinterAdded(address indexed minter);
    event AuthorizedMinterRemoved(address indexed minter);

    constructor(address initialOwner)
        ERC20("HashProof", "HPROOF")
        Ownable(initialOwner)
    {
        // Mint initial 10% supply to owner
        _mint(initialOwner, 10_000_000 * 10**18);

        // Exclude owner, this contract from fees and limits
        isExcludedFromFees[initialOwner] = true;
        isExcludedFromFees[address(this)] = true;
        isExcludedFromLimits[initialOwner] = true;
        isExcludedFromLimits[address(this)] = true;

        emit ExcludedFromFees(initialOwner, true);
        emit ExcludedFromFees(address(this), true);
    }

    /**
     * @dev Override transfer function to include burn mechanism
     */
    function _update(address from, address to, uint256 amount)
        internal
        override(ERC20, ERC20Pausable)
    {
        require(from != address(0), "ERC20: transfer from zero address");
        require(to != address(0), "ERC20: transfer to zero address");

        // Check max transaction amount (unless excluded)
        if (!isExcludedFromLimits[from] && !isExcludedFromLimits[to]) {
            require(amount <= maxTransactionAmount, "Transfer amount exceeds max");
        }

        // Apply burn if not excluded from fees
        if (!isExcludedFromFees[from] && !isExcludedFromFees[to] && from != address(0)) {
            uint256 burnAmount = (amount * BURN_RATE) / BASIS_POINTS;

            if (burnAmount > 0) {
                super._update(from, address(0), burnAmount); // Burn tokens
                totalBurned += burnAmount;
                emit TokensBurned(from, burnAmount);

                amount -= burnAmount; // Reduce transfer amount by burn
            }
        }

        super._update(from, to, amount);
    }

    /**
     * @dev Mint tokens for mining rewards
     * Can only be called by owner or authorized minters
     */
    function mintMiningReward(address miner, uint256 amount, uint256 hashes)
        external
        whenNotPaused
        nonReentrant
    {
        require(
            msg.sender == owner() || authorizedMinters[msg.sender],
            "Not authorized to mint"
        );
        require(amount <= miningRewardsPool, "Exceeds mining rewards pool");
        require(totalSupply() + amount <= MAX_SUPPLY, "Exceeds max supply");

        miningRewardsPool -= amount;
        miningRewards[miner] += amount;
        totalHashesSubmitted[miner] += hashes;

        _mint(miner, amount);
        emit MiningRewardPaid(miner, amount, hashes);
    }

    /**
     * @dev Mint tokens for staking rewards
     * Should be called by staking contract
     */
    function mintStakingReward(address staker, uint256 amount)
        external
        whenNotPaused
        nonReentrant
    {
        require(authorizedMinters[msg.sender], "Not authorized minter");
        require(amount <= stakingRewardsPool, "Exceeds staking rewards pool");
        require(totalSupply() + amount <= MAX_SUPPLY, "Exceeds max supply");

        stakingRewardsPool -= amount;
        _mint(staker, amount);
    }

    /**
     * @dev Mint team allocation tokens
     * Only owner can call
     */
    function mintTeamTokens(address recipient, uint256 amount)
        external
        onlyOwner
        nonReentrant
    {
        require(amount <= teamPool, "Exceeds team pool");
        require(totalSupply() + amount <= MAX_SUPPLY, "Exceeds max supply");

        teamPool -= amount;
        _mint(recipient, amount);
    }

    /**
     * @dev Add authorized minter (e.g., staking contract)
     */
    function addAuthorizedMinter(address minter) external onlyOwner {
        authorizedMinters[minter] = true;
        emit AuthorizedMinterAdded(minter);
    }

    /**
     * @dev Remove authorized minter
     */
    function removeAuthorizedMinter(address minter) external onlyOwner {
        authorizedMinters[minter] = false;
        emit AuthorizedMinterRemoved(minter);
    }

    /**
     * @dev Exclude/include address from fees
     */
    function setExcludedFromFees(address account, bool excluded) external onlyOwner {
        isExcludedFromFees[account] = excluded;
        emit ExcludedFromFees(account, excluded);
    }

    /**
     * @dev Exclude/include address from limits
     */
    function setExcludedFromLimits(address account, bool excluded) external onlyOwner {
        isExcludedFromLimits[account] = excluded;
    }

    /**
     * @dev Update max transaction amount
     */
    function setMaxTransactionAmount(uint256 newAmount) external onlyOwner {
        require(newAmount >= (MAX_SUPPLY / 1000), "Amount too low"); // Min 0.1%
        maxTransactionAmount = newAmount;
        emit MaxTransactionUpdated(newAmount);
    }

    /**
     * @dev Pause all transfers (emergency)
     */
    function pause() external onlyOwner {
        _pause();
    }

    /**
     * @dev Unpause transfers
     */
    function unpause() external onlyOwner {
        _unpause();
    }

    /**
     * @dev Get total supply remaining that can be minted
     */
    function remainingSupply() external view returns (uint256) {
        return MAX_SUPPLY - totalSupply();
    }

    /**
     * @dev Get circulating supply (total - burned)
     */
    function circulatingSupply() external view returns (uint256) {
        return totalSupply();
    }

    /**
     * @dev Get burn percentage
     */
    function getBurnRate() external pure returns (uint256) {
        return BURN_RATE;
    }
}
