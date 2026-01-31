// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title WTBTC - Wrapped Testnet Bitcoin
 * @dev ERC-20 Token representing wrapped testnet BTC
 * Operating at 528Hz Love Frequency ✨
 */
contract WTBTC {
    string public constant name = "Wrapped Testnet Bitcoin";
    string public constant symbol = "WTBTC";
    uint8 public constant decimals = 8; // Bitcoin uses 8 decimals

    uint256 private _totalSupply;
    address public owner;

    mapping(address => uint256) private _balances;
    mapping(address => mapping(address => uint256)) private _allowances;

    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);
    event Mint(address indexed to, uint256 amount);
    event Burn(address indexed from, uint256 amount);

    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner can call this");
        _;
    }

    constructor(uint256 initialSupply) {
        owner = msg.sender;
        _mint(msg.sender, initialSupply);
    }

    /**
     * @dev Returns the total token supply
     */
    function totalSupply() public view returns (uint256) {
        return _totalSupply;
    }

    /**
     * @dev Returns the balance of an account
     */
    function balanceOf(address account) public view returns (uint256) {
        return _balances[account];
    }

    /**
     * @dev Transfer tokens to another address
     */
    function transfer(address to, uint256 amount) public returns (bool) {
        address sender = msg.sender;
        require(sender != address(0), "Transfer from zero address");
        require(to != address(0), "Transfer to zero address");
        require(_balances[sender] >= amount, "Insufficient balance");

        _balances[sender] -= amount;
        _balances[to] += amount;

        emit Transfer(sender, to, amount);
        return true;
    }

    /**
     * @dev Returns the allowance for a spender
     */
    function allowance(address tokenOwner, address spender) public view returns (uint256) {
        return _allowances[tokenOwner][spender];
    }

    /**
     * @dev Approve a spender to spend tokens
     */
    function approve(address spender, uint256 amount) public returns (bool) {
        address tokenOwner = msg.sender;
        require(tokenOwner != address(0), "Approve from zero address");
        require(spender != address(0), "Approve to zero address");

        _allowances[tokenOwner][spender] = amount;

        emit Approval(tokenOwner, spender, amount);
        return true;
    }

    /**
     * @dev Transfer tokens from one address to another (with allowance)
     */
    function transferFrom(address from, address to, uint256 amount) public returns (bool) {
        address spender = msg.sender;
        require(from != address(0), "Transfer from zero address");
        require(to != address(0), "Transfer to zero address");
        require(_balances[from] >= amount, "Insufficient balance");
        require(_allowances[from][spender] >= amount, "Insufficient allowance");

        _balances[from] -= amount;
        _balances[to] += amount;
        _allowances[from][spender] -= amount;

        emit Transfer(from, to, amount);
        return true;
    }

    /**
     * @dev Mint new tokens (only owner)
     * Useful for wrapping more BTC
     */
    function mint(address to, uint256 amount) public onlyOwner returns (bool) {
        _mint(to, amount);
        return true;
    }

    /**
     * @dev Burn tokens (anyone can burn their own tokens)
     * Useful for unwrapping BTC
     */
    function burn(uint256 amount) public returns (bool) {
        address account = msg.sender;
        require(account != address(0), "Burn from zero address");
        require(_balances[account] >= amount, "Insufficient balance to burn");

        _balances[account] -= amount;
        _totalSupply -= amount;

        emit Burn(account, amount);
        emit Transfer(account, address(0), amount);
        return true;
    }

    /**
     * @dev Internal mint function
     */
    function _mint(address account, uint256 amount) internal {
        require(account != address(0), "Mint to zero address");

        _totalSupply += amount;
        _balances[account] += amount;

        emit Mint(account, amount);
        emit Transfer(address(0), account, amount);
    }

    /**
     * @dev Transfer ownership
     */
    function transferOwnership(address newOwner) public onlyOwner {
        require(newOwner != address(0), "New owner is zero address");
        owner = newOwner;
    }

    /**
     * @dev Get token info in one call
     */
    function getTokenInfo() public view returns (
        string memory tokenName,
        string memory tokenSymbol,
        uint8 tokenDecimals,
        uint256 tokenTotalSupply,
        address tokenOwner
    ) {
        return (name, symbol, decimals, _totalSupply, owner);
    }
}
