// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title FlashLoanToken (FLASH)
 * @dev Token for flash loan fees and liquidity
 */
contract FlashLoanToken {
    string public constant name = "Flash Loan Token";
    string public constant symbol = "FLASH";
    uint8 public constant decimals = 18;
    uint256 public totalSupply;
    uint256 public constant MAX_SUPPLY = 3_000_000 ether;

    mapping(address => uint256) public balanceOf;
    mapping(address => mapping(address => uint256)) public allowance;

    uint256 public flashLoanFee = 9; // 0.09% fee (9 basis points)
    uint256 public totalFlashLoans;
    uint256 public totalFeesEarned;

    address public owner;

    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);
    event FlashLoan(address indexed borrower, uint256 amount, uint256 fee);

    constructor() {
        owner = msg.sender;
        _mint(msg.sender, 300_000 ether);
    }

    function transfer(address to, uint256 value) external returns (bool) {
        require(balanceOf[msg.sender] >= value, "Insufficient balance");
        balanceOf[msg.sender] -= value;
        balanceOf[to] += value;
        emit Transfer(msg.sender, to, value);
        return true;
    }

    function approve(address spender, uint256 value) external returns (bool) {
        allowance[msg.sender][spender] = value;
        emit Approval(msg.sender, spender, value);
        return true;
    }

    function _mint(address to, uint256 amount) internal {
        require(totalSupply + amount <= MAX_SUPPLY, "Max supply");
        totalSupply += amount;
        balanceOf[to] += amount;
        emit Transfer(address(0), to, amount);
    }

    function executeFlashLoan(uint256 amount) external returns (bool) {
        uint256 fee = (amount * flashLoanFee) / 10000;
        uint256 repayAmount = amount + fee;

        // Flash loan logic here
        totalFlashLoans += amount;
        totalFeesEarned += fee;

        // Mint fee tokens to stakers
        if (totalSupply + fee <= MAX_SUPPLY) {
            _mint(msg.sender, fee / 100); // 1% of fee as reward
        }

        emit FlashLoan(msg.sender, amount, fee);
        return true;
    }

    function setFlashLoanFee(uint256 newFee) external {
        require(msg.sender == owner, "Not authorized");
        require(newFee <= 100, "Fee too high");
        flashLoanFee = newFee;
    }
}
