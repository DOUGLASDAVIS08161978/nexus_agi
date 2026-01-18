// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title LiquidityToken (BLP - Bridge Liquidity Provider)
 * @dev LP token for providing liquidity to the bridge
 * Earn fees from all bridge transactions
 */
contract LiquidityToken {
    string public constant name = "Bridge Liquidity Provider Token";
    string public constant symbol = "BLP";
    uint8 public constant decimals = 18;
    uint256 public totalSupply;

    mapping(address => uint256) public balanceOf;
    mapping(address => mapping(address => uint256)) public allowance;

    uint256 public totalLiquidity; // Total BTC + ETH liquidity
    uint256 public accumulatedFees;
    uint256 public feePerShare;

    struct LPPosition {
        uint256 shares;
        uint256 feeDebt;
        uint256 depositTime;
    }

    mapping(address => LPPosition) public positions;

    address public owner;

    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);
    event LiquidityAdded(address indexed provider, uint256 amount, uint256 shares);
    event LiquidityRemoved(address indexed provider, uint256 amount, uint256 shares);
    event FeesCollected(uint256 amount);
    event FeesClaimed(address indexed user, uint256 amount);

    constructor() {
        owner = msg.sender;
    }

    function transfer(address to, uint256 value) external returns (bool) {
        _updateFees(msg.sender);
        _updateFees(to);
        _transfer(msg.sender, to, value);
        return true;
    }

    function approve(address spender, uint256 value) external returns (bool) {
        allowance[msg.sender][spender] = value;
        emit Approval(msg.sender, spender, value);
        return true;
    }

    function _transfer(address from, address to, uint256 value) internal {
        require(balanceOf[from] >= value, "Insufficient balance");
        balanceOf[from] -= value;
        balanceOf[to] += value;

        // Update positions
        positions[from].shares -= value;
        positions[to].shares += value;

        emit Transfer(from, to, value);
    }

    function addLiquidity(uint256 amount) external payable {
        require(amount > 0, "Amount must be positive");

        uint256 shares;
        if (totalSupply == 0) {
            shares = amount;
        } else {
            shares = (amount * totalSupply) / totalLiquidity;
        }

        totalSupply += shares;
        balanceOf[msg.sender] += shares;
        totalLiquidity += amount;

        LPPosition storage position = positions[msg.sender];
        position.shares += shares;
        position.depositTime = block.timestamp;
        position.feeDebt = (shares * feePerShare) / 1e18;

        emit LiquidityAdded(msg.sender, amount, shares);
        emit Transfer(address(0), msg.sender, shares);
    }

    function removeLiquidity(uint256 shares) external {
        require(balanceOf[msg.sender] >= shares, "Insufficient shares");
        _updateFees(msg.sender);

        uint256 amount = (shares * totalLiquidity) / totalSupply;

        totalSupply -= shares;
        balanceOf[msg.sender] -= shares;
        totalLiquidity -= amount;

        LPPosition storage position = positions[msg.sender];
        position.shares -= shares;

        emit LiquidityRemoved(msg.sender, amount, shares);
        emit Transfer(msg.sender, address(0), shares);
    }

    function collectBridgeFees(uint256 feeAmount) external {
        require(msg.sender == owner, "Not authorized");
        accumulatedFees += feeAmount;

        if (totalSupply > 0) {
            feePerShare += (feeAmount * 1e18) / totalSupply;
        }

        emit FeesCollected(feeAmount);
    }

    function _updateFees(address user) internal {
        LPPosition storage position = positions[user];
        if (position.shares > 0) {
            uint256 accumulatedUserFees = (position.shares * feePerShare) / 1e18;
            uint256 pendingFees = accumulatedUserFees - position.feeDebt;

            if (pendingFees > 0) {
                // Transfer fees to user (would be actual ETH/BTC in production)
                position.feeDebt = accumulatedUserFees;
            }
        }
    }

    function claimFees() external {
        _updateFees(msg.sender);
        LPPosition storage position = positions[msg.sender];

        uint256 fees = (position.shares * feePerShare) / 1e18 - position.feeDebt;
        require(fees > 0, "No fees to claim");

        position.feeDebt += fees;
        emit FeesClaimed(msg.sender, fees);
    }

    function getPendingFees(address user) external view returns (uint256) {
        LPPosition memory position = positions[user];
        uint256 accumulatedUserFees = (position.shares * feePerShare) / 1e18;
        return accumulatedUserFees - position.feeDebt;
    }

    function getAPY() external view returns (uint256) {
        if (totalLiquidity == 0) return 0;
        // Simplified APY calculation: (annual fees / total liquidity) * 100
        uint256 dailyFees = accumulatedFees / 365; // Approximate
        return (dailyFees * 365 * 100) / totalLiquidity;
    }
}
