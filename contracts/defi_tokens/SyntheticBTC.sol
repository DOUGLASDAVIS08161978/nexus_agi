// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title SyntheticBTC (synBTC)
 * @dev Synthetic Bitcoin with leverage trading
 */
contract SyntheticBTC {
    string public constant name = "Synthetic Bitcoin";
    string public constant symbol = "synBTC";
    uint8 public constant decimals = 18;
    uint256 public totalSupply;

    mapping(address => uint256) public balanceOf;
    mapping(address => mapping(address => uint256)) public allowance;

    struct Position {
        uint256 collateral;
        uint256 debt;
        uint256 leverage; // 1x, 2x, 5x, 10x
        uint256 entryPrice;
        bool isLong;
    }

    mapping(address => Position) public positions;
    uint256 public btcPrice = 100_000 ether;

    address public owner;

    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);
    event PositionOpened(address indexed trader, uint256 leverage, bool isLong);
    event PositionClosed(address indexed trader, int256 pnl);

    constructor() {
        owner = msg.sender;
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

    function openPosition(uint256 collateral, uint256 leverage, bool isLong) external payable {
        require(leverage >= 1 && leverage <= 10, "Invalid leverage");
        require(collateral > 0, "No collateral");

        uint256 debt = collateral * (leverage - 1);

        positions[msg.sender] = Position({
            collateral: collateral,
            debt: debt,
            leverage: leverage,
            entryPrice: btcPrice,
            isLong: isLong
        });

        uint256 syntheticAmount = collateral * leverage;
        totalSupply += syntheticAmount;
        balanceOf[msg.sender] += syntheticAmount;

        emit PositionOpened(msg.sender, leverage, isLong);
        emit Transfer(address(0), msg.sender, syntheticAmount);
    }

    function closePosition() external {
        Position storage position = positions[msg.sender];
        require(position.collateral > 0, "No position");

        // Calculate P&L
        int256 priceDelta = int256(btcPrice) - int256(position.entryPrice);
        int256 pnl = (priceDelta * int256(position.collateral * position.leverage)) / int256(position.entryPrice);

        if (!position.isLong) {
            pnl = -pnl;
        }

        uint256 syntheticAmount = position.collateral * position.leverage;
        balanceOf[msg.sender] -= syntheticAmount;
        totalSupply -= syntheticAmount;

        emit PositionClosed(msg.sender, pnl);
        emit Transfer(msg.sender, address(0), syntheticAmount);

        delete positions[msg.sender];
    }

    function updateBTCPrice(uint256 newPrice) external {
        require(msg.sender == owner, "Not authorized");
        btcPrice = newPrice;
    }
}
