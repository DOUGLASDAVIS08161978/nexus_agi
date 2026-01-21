// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * Simple Wrapped Testnet BTC
 *
 * Simplified version for testing where anyone can mint.
 * EDUCATIONAL USE ONLY - Not for production!
 *
 * Destination: 0x24f6b1ce11c57d40b542f91ac85fa9eb61f78771
 */
contract SimpleWrappedBTC {
    string public name = "Simple Wrapped Testnet Bitcoin";
    string public symbol = "swTBTC";
    uint8 public decimals = 18;
    uint256 public totalSupply;

    mapping(address => uint256) public balanceOf;
    mapping(address => mapping(address => uint256)) public allowance;

    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);
    event Mint(address indexed to, uint256 amount, string note);
    event Burn(address indexed from, uint256 amount);

    /**
     * Anyone can mint - for testnet demonstration only!
     * In production, only authorized bridge operators should mint.
     */
    function mint(address to, uint256 amount) external {
        require(to != address(0), "Mint to zero address");
        require(amount > 0, "Amount must be positive");

        balanceOf[to] += amount;
        totalSupply += amount;

        emit Transfer(address(0), to, amount);
        emit Mint(to, amount, "Testnet mint");
    }

    /**
     * Mint with note for tracking
     */
    function mintWithNote(address to, uint256 amount, string calldata note) external {
        require(to != address(0), "Mint to zero address");
        require(amount > 0, "Amount must be positive");

        balanceOf[to] += amount;
        totalSupply += amount;

        emit Transfer(address(0), to, amount);
        emit Mint(to, amount, note);
    }

    /**
     * Burn tokens
     */
    function burn(uint256 amount) external {
        require(balanceOf[msg.sender] >= amount, "Insufficient balance");

        balanceOf[msg.sender] -= amount;
        totalSupply -= amount;

        emit Transfer(msg.sender, address(0), amount);
        emit Burn(msg.sender, amount);
    }

    /**
     * Standard ERC20 approve
     */
    function approve(address spender, uint256 amount) external returns (bool) {
        allowance[msg.sender][spender] = amount;
        emit Approval(msg.sender, spender, amount);
        return true;
    }

    /**
     * Standard ERC20 transfer
     */
    function transfer(address to, uint256 amount) external returns (bool) {
        require(to != address(0), "Transfer to zero address");
        require(balanceOf[msg.sender] >= amount, "Insufficient balance");

        balanceOf[msg.sender] -= amount;
        balanceOf[to] += amount;

        emit Transfer(msg.sender, to, amount);
        return true;
    }

    /**
     * Standard ERC20 transferFrom
     */
    function transferFrom(address from, address to, uint256 amount) external returns (bool) {
        require(from != address(0), "Transfer from zero address");
        require(to != address(0), "Transfer to zero address");
        require(balanceOf[from] >= amount, "Insufficient balance");
        require(allowance[from][msg.sender] >= amount, "Allowance exceeded");

        allowance[from][msg.sender] -= amount;
        balanceOf[from] -= amount;
        balanceOf[to] += amount;

        emit Transfer(from, to, amount);
        return true;
    }
}
