// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title YieldToken (YIELD)
 * @dev Auto-compounding yield aggregator token
 * Earns yield from bridge fees and distributes to holders
 */
contract YieldToken {
    string public constant name = "Bridge Yield Token";
    string public constant symbol = "YIELD";
    uint8 public constant decimals = 18;
    uint256 public totalSupply;

    uint256 public constant MAX_SUPPLY = 1_000_000 ether; // 1M tokens
    uint256 public yieldPerToken; // Accumulated yield per token
    uint256 public lastDistributionTime;
    uint256 public constant YIELD_RATE = 500; // 5% APY in basis points

    mapping(address => uint256) public balanceOf;
    mapping(address => mapping(address => uint256)) public allowance;
    mapping(address => uint256) public yieldDebt; // For yield calculation
    mapping(address => uint256) public pendingYield;

    address public owner;
    address public yieldSource; // Bridge contract that provides yield

    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);
    event YieldDistributed(uint256 amount, uint256 timestamp);
    event YieldClaimed(address indexed user, uint256 amount);

    modifier onlyOwner() {
        require(msg.sender == owner, "Not owner");
        _;
    }

    constructor() {
        owner = msg.sender;
        lastDistributionTime = block.timestamp;
        // Initial mint to owner
        _mint(msg.sender, 100_000 ether); // 10% initial supply
    }

    function transfer(address to, uint256 value) external returns (bool) {
        _updateYield(msg.sender);
        _updateYield(to);
        _transfer(msg.sender, to, value);
        return true;
    }

    function transferFrom(address from, address to, uint256 value) external returns (bool) {
        require(allowance[from][msg.sender] >= value, "Insufficient allowance");
        _updateYield(from);
        _updateYield(to);
        allowance[from][msg.sender] -= value;
        _transfer(from, to, value);
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
        emit Transfer(from, to, value);
    }

    function _mint(address to, uint256 amount) internal {
        require(totalSupply + amount <= MAX_SUPPLY, "Max supply exceeded");
        totalSupply += amount;
        balanceOf[to] += amount;
        emit Transfer(address(0), to, amount);
    }

    function distributeYield() external {
        require(block.timestamp >= lastDistributionTime + 1 days, "Too soon");

        if (totalSupply > 0) {
            // Calculate yield: total supply * APY / 365 days
            uint256 dailyYield = (totalSupply * YIELD_RATE) / (365 * 10000);
            yieldPerToken += (dailyYield * 1e18) / totalSupply;

            lastDistributionTime = block.timestamp;
            emit YieldDistributed(dailyYield, block.timestamp);
        }
    }

    function _updateYield(address user) internal {
        if (balanceOf[user] > 0) {
            uint256 accumulated = (balanceOf[user] * yieldPerToken) / 1e18;
            if (accumulated > yieldDebt[user]) {
                pendingYield[user] += accumulated - yieldDebt[user];
            }
        }
        yieldDebt[user] = (balanceOf[user] * yieldPerToken) / 1e18;
    }

    function claimYield() external {
        _updateYield(msg.sender);
        uint256 yield = pendingYield[msg.sender];
        require(yield > 0, "No yield to claim");

        pendingYield[msg.sender] = 0;
        _mint(msg.sender, yield);

        emit YieldClaimed(msg.sender, yield);
    }

    function getClaimableYield(address user) external view returns (uint256) {
        uint256 accumulated = (balanceOf[user] * yieldPerToken) / 1e18;
        uint256 pending = pendingYield[user];
        if (accumulated > yieldDebt[user]) {
            pending += accumulated - yieldDebt[user];
        }
        return pending;
    }

    function setYieldSource(address source) external onlyOwner {
        yieldSource = source;
    }
}
