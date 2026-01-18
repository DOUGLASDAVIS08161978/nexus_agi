// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

/**
 * @title LiquidityPoolToken
 * @dev LP token for AMM liquidity provision
 *
 * Features:
 * - Represents liquidity pool shares
 * - Fee distribution to LPs
 * - Automatic yield generation
 *
 * Symbol: LP
 * Decimals: 18
 */
contract LiquidityPoolToken is ERC20, Ownable {
    struct LiquidityProvider {
        uint256 provided;
        uint256 shares;
        uint256 lastRewardClaim;
        uint256 accumulatedFees;
    }

    mapping(address => LiquidityProvider) public liquidityProviders;

    uint256 public totalLiquidityProvided;
    uint256 public totalFeesCollected;
    uint256 public feePercentage = 30; // 0.3%

    event LiquidityAdded(address indexed provider, uint256 amount, uint256 shares);
    event LiquidityRemoved(address indexed provider, uint256 amount, uint256 shares);
    event FeesDistributed(address indexed provider, uint256 amount);
    event TradingFeeCollected(uint256 amount);

    constructor(address initialOwner)
        ERC20("Liquidity Pool Token", "LP")
        Ownable(initialOwner)
    {}

    /**
     * @dev Add liquidity to pool
     */
    function addLiquidity(uint256 amount) external {
        require(amount > 0, "Amount must be positive");

        uint256 shares;
        if (totalSupply() == 0) {
            shares = amount;
        } else {
            shares = (amount * totalSupply()) / totalLiquidityProvided;
        }

        _mint(msg.sender, shares);

        liquidityProviders[msg.sender].provided += amount;
        liquidityProviders[msg.sender].shares += shares;
        liquidityProviders[msg.sender].lastRewardClaim = block.timestamp;

        totalLiquidityProvided += amount;

        emit LiquidityAdded(msg.sender, amount, shares);
    }

    /**
     * @dev Remove liquidity from pool
     */
    function removeLiquidity(uint256 shares) external {
        require(balanceOf(msg.sender) >= shares, "Insufficient shares");

        uint256 amount = (shares * totalLiquidityProvided) / totalSupply();

        _burn(msg.sender, shares);

        liquidityProviders[msg.sender].provided -= amount;
        liquidityProviders[msg.sender].shares -= shares;

        totalLiquidityProvided -= amount;

        emit LiquidityRemoved(msg.sender, amount, shares);
    }

    /**
     * @dev Collect trading fees
     */
    function collectTradingFee(uint256 tradeAmount) external onlyOwner {
        uint256 fee = (tradeAmount * feePercentage) / 10000;

        totalFeesCollected += fee;

        emit TradingFeeCollected(fee);
    }

    /**
     * @dev Calculate pending fees for LP
     */
    function calculatePendingFees(address provider) public view returns (uint256) {
        if (totalSupply() == 0) return 0;

        uint256 lpShare = (balanceOf(provider) * 10000) / totalSupply();
        uint256 fees = (totalFeesCollected * lpShare) / 10000;

        return fees - liquidityProviders[provider].accumulatedFees;
    }

    /**
     * @dev Claim accumulated fees
     */
    function claimFees() external {
        uint256 pending = calculatePendingFees(msg.sender);
        require(pending > 0, "No fees to claim");

        liquidityProviders[msg.sender].accumulatedFees += pending;
        _mint(msg.sender, pending);

        emit FeesDistributed(msg.sender, pending);
    }

    /**
     * @dev Update fee percentage (owner only)
     */
    function setFeePercentage(uint256 newFee) external onlyOwner {
        require(newFee <= 100, "Fee too high");
        feePercentage = newFee;
    }

    /**
     * @dev Get LP stats
     */
    function getLPStats(address provider) external view returns (
        uint256 provided,
        uint256 shares,
        uint256 pendingFees,
        uint256 poolShare
    ) {
        provided = liquidityProviders[provider].provided;
        shares = balanceOf(provider);
        pendingFees = calculatePendingFees(provider);
        poolShare = totalSupply() > 0 ? (shares * 10000) / totalSupply() : 0;
    }
}
