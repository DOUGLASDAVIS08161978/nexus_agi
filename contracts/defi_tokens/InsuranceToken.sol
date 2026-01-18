// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title InsuranceToken (INSURE)
 * @dev Insurance coverage for bridge operations
 * Holders provide insurance, earn premiums
 */
contract InsuranceToken {
    string public constant name = "Bridge Insurance Token";
    string public constant symbol = "INSURE";
    uint8 public constant decimals = 18;
    uint256 public totalSupply;
    uint256 public constant MAX_SUPPLY = 5_000_000 ether;

    mapping(address => uint256) public balanceOf;
    mapping(address => mapping(address => uint256)) public allowance;

    uint256 public premiumPool;
    uint256 public coverageProvided;
    uint256 public claimsPaid;

    struct Coverage {
        uint256 amount;
        uint256 premium;
        uint256 expiryTime;
        bool active;
    }

    mapping(address => Coverage) public coverages;
    mapping(address => uint256) public stakedAmount;

    address public owner;

    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);
    event CoveragePurchased(address indexed user, uint256 amount, uint256 premium);
    event ClaimPaid(address indexed user, uint256 amount);
    event PremiumEarned(uint256 amount);

    constructor() {
        owner = msg.sender;
        _mint(msg.sender, 500_000 ether); // 10% initial
    }

    function transfer(address to, uint256 value) external returns (bool) {
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
        emit Transfer(from, to, value);
    }

    function _mint(address to, uint256 amount) internal {
        require(totalSupply + amount <= MAX_SUPPLY, "Max supply");
        totalSupply += amount;
        balanceOf[to] += amount;
        emit Transfer(address(0), to, amount);
    }

    function stake(uint256 amount) external {
        require(balanceOf[msg.sender] >= amount, "Insufficient balance");
        balanceOf[msg.sender] -= amount;
        stakedAmount[msg.sender] += amount;
        coverageProvided += amount;
    }

    function purchaseCoverage(uint256 amount, uint256 duration) external payable {
        uint256 premium = (amount * 100) / 10000; // 1% premium
        require(msg.value >= premium, "Insufficient premium");

        coverages[msg.sender] = Coverage({
            amount: amount,
            premium: premium,
            expiryTime: block.timestamp + duration,
            active: true
        });

        premiumPool += premium;
        emit CoveragePurchased(msg.sender, amount, premium);
        emit PremiumEarned(premium);
    }

    function fileClaim(uint256 amount) external {
        Coverage storage coverage = coverages[msg.sender];
        require(coverage.active, "No active coverage");
        require(block.timestamp < coverage.expiryTime, "Coverage expired");
        require(amount <= coverage.amount, "Exceeds coverage");

        coverage.active = false;
        claimsPaid += amount;

        emit ClaimPaid(msg.sender, amount);
    }
}
