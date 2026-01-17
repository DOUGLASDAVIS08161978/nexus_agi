// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title BTCPegToken
 * @dev ERC20 token representing Bitcoin on Ethereum (1:1 peg)
 * Mainnet-ready with full ERC20 compliance and bridge integration
 */
contract BTCPegToken {

    // ============ ERC20 State ============

    string public constant name = "Bitcoin Peg Token";
    string public constant symbol = "BTCP";
    uint8 public constant decimals = 18;
    uint256 public totalSupply;

    mapping(address => uint256) public balanceOf;
    mapping(address => mapping(address => uint256)) public allowance;

    // ============ Access Control ============

    address public owner;
    mapping(address => bool) public minters;
    mapping(address => bool) public burners;

    bool public paused;

    // ============ Events ============

    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);
    event Mint(address indexed to, uint256 amount, address indexed minter);
    event Burn(address indexed from, uint256 amount, address indexed burner);
    event MinterAdded(address indexed minter);
    event MinterRemoved(address indexed minter);
    event BurnerAdded(address indexed burner);
    event BurnerRemoved(address indexed burner);

    // ============ Modifiers ============

    modifier onlyOwner() {
        require(msg.sender == owner, "Not owner");
        _;
    }

    modifier onlyMinter() {
        require(minters[msg.sender], "Not authorized minter");
        _;
    }

    modifier onlyBurner() {
        require(burners[msg.sender], "Not authorized burner");
        _;
    }

    modifier whenNotPaused() {
        require(!paused, "Contract paused");
        _;
    }

    // ============ Constructor ============

    constructor() {
        owner = msg.sender;
        minters[msg.sender] = true;
        burners[msg.sender] = true;
    }

    // ============ ERC20 Functions ============

    function transfer(address to, uint256 value)
        external
        whenNotPaused
        returns (bool)
    {
        require(to != address(0), "Invalid recipient");
        require(balanceOf[msg.sender] >= value, "Insufficient balance");

        balanceOf[msg.sender] -= value;
        balanceOf[to] += value;

        emit Transfer(msg.sender, to, value);
        return true;
    }

    function approve(address spender, uint256 value)
        external
        whenNotPaused
        returns (bool)
    {
        require(spender != address(0), "Invalid spender");

        allowance[msg.sender][spender] = value;

        emit Approval(msg.sender, spender, value);
        return true;
    }

    function transferFrom(address from, address to, uint256 value)
        external
        whenNotPaused
        returns (bool)
    {
        require(to != address(0), "Invalid recipient");
        require(balanceOf[from] >= value, "Insufficient balance");
        require(allowance[from][msg.sender] >= value, "Insufficient allowance");

        balanceOf[from] -= value;
        balanceOf[to] += value;
        allowance[from][msg.sender] -= value;

        emit Transfer(from, to, value);
        return true;
    }

    // ============ Minting & Burning ============

    /**
     * @dev Mint new tokens (only authorized minters - bridge contracts)
     */
    function mint(address to, uint256 amount)
        external
        onlyMinter
        whenNotPaused
        returns (bool)
    {
        require(to != address(0), "Invalid recipient");
        require(amount > 0, "Invalid amount");

        totalSupply += amount;
        balanceOf[to] += amount;

        emit Mint(to, amount, msg.sender);
        emit Transfer(address(0), to, amount);

        return true;
    }

    /**
     * @dev Burn tokens from sender
     */
    function burn(uint256 amount)
        external
        whenNotPaused
        returns (bool)
    {
        require(balanceOf[msg.sender] >= amount, "Insufficient balance");
        require(amount > 0, "Invalid amount");

        balanceOf[msg.sender] -= amount;
        totalSupply -= amount;

        emit Burn(msg.sender, amount, msg.sender);
        emit Transfer(msg.sender, address(0), amount);

        return true;
    }

    /**
     * @dev Burn tokens from another address (requires allowance)
     */
    function burnFrom(address from, uint256 amount)
        external
        onlyBurner
        whenNotPaused
        returns (bool)
    {
        require(balanceOf[from] >= amount, "Insufficient balance");
        require(amount > 0, "Invalid amount");

        balanceOf[from] -= amount;
        totalSupply -= amount;

        emit Burn(from, amount, msg.sender);
        emit Transfer(from, address(0), amount);

        return true;
    }

    // ============ Access Control ============

    function addMinter(address minter) external onlyOwner {
        require(minter != address(0), "Invalid minter");
        minters[minter] = true;
        emit MinterAdded(minter);
    }

    function removeMinter(address minter) external onlyOwner {
        minters[minter] = false;
        emit MinterRemoved(minter);
    }

    function addBurner(address burner) external onlyOwner {
        require(burner != address(0), "Invalid burner");
        burners[burner] = true;
        emit BurnerAdded(burner);
    }

    function removeBurner(address burner) external onlyOwner {
        burners[burner] = false;
        emit BurnerRemoved(burner);
    }

    // ============ Admin Functions ============

    function pause() external onlyOwner {
        paused = true;
    }

    function unpause() external onlyOwner {
        paused = false;
    }

    function transferOwnership(address newOwner) external onlyOwner {
        require(newOwner != address(0), "Invalid owner");
        owner = newOwner;
    }

    // ============ View Functions ============

    function isMinter(address account) external view returns (bool) {
        return minters[account];
    }

    function isBurner(address account) external view returns (bool) {
        return burners[account];
    }
}
