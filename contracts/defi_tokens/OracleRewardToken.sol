// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title OracleRewardToken (ORACLE)
 * @dev Rewards for oracle data providers
 */
contract OracleRewardToken {
    string public constant name = "Oracle Reward Token";
    string public constant symbol = "ORACLE";
    uint8 public constant decimals = 18;
    uint256 public totalSupply;
    uint256 public constant MAX_SUPPLY = 2_000_000 ether;

    mapping(address => uint256) public balanceOf;
    mapping(address => mapping(address => uint256)) public allowance;
    mapping(address => uint256) public dataSubmissions;
    mapping(address => uint256) public accuracyScore; // Out of 100

    address public owner;

    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);
    event DataSubmitted(address indexed oracle, uint256 reward);

    constructor() {
        owner = msg.sender;
        _mint(msg.sender, 200_000 ether);
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

    function rewardOracle(address oracle, uint256 accuracy) external {
        require(msg.sender == owner, "Not authorized");
        dataSubmissions[oracle]++;
        accuracyScore[oracle] = accuracy;

        uint256 reward = (100 ether * accuracy) / 100; // Up to 100 tokens per submission
        if (totalSupply + reward <= MAX_SUPPLY) {
            _mint(oracle, reward);
            emit DataSubmitted(oracle, reward);
        }
    }
}
