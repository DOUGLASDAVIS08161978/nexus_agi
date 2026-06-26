// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/utils/Pausable.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";

/**
 * @title WTBTCBridge
 * @dev Bridge contract for moving WTBTC between Base Sepolia and Ethereum Mainnet
 */
contract WTBTCBridge is ERC20, Ownable, Pausable, ReentrancyGuard {

    // Base Sepolia contract address
    address public constant BASE_SEPOLIA_TOKEN = 0xE274570e000C32F5Cb2BC7c476D3BDC77Ed74dD5;

    // Bridge events
    event TokensBridgedIn(address indexed user, uint256 amount, string baseTxHash);
    event TokensBridgedOut(address indexed user, uint256 amount, address baseAddress);
    event BridgeOperatorUpdated(address indexed newOperator);

    address public bridgeOperator;

    // Mapping to track bridged transactions
    mapping(string => bool) public processedBaseTxs;
    mapping(address => uint256) public bridgedBalances;

    constructor() ERC20("Wrapped Testnet Bitcoin", "WTBTC") Ownable(msg.sender) {
        bridgeOperator = msg.sender;

        // Mint initial supply for liquidity
        _mint(msg.sender, 1000000 * 10**decimals()); // 1 million WTBTC
    }

    /**
     * @dev Bridge tokens FROM Base Sepolia TO Ethereum Mainnet
     * Called by bridge operator when tokens are locked on Base Sepolia
     */
    function bridgeIn(
        address user,
        uint256 amount,
        string calldata baseTxHash
    ) external onlyBridgeOperator whenNotPaused nonReentrant {
        require(!processedBaseTxs[baseTxHash], "Transaction already processed");
        require(user != address(0), "Invalid user address");
        require(amount > 0, "Amount must be greater than 0");

        processedBaseTxs[baseTxHash] = true;
        bridgedBalances[user] += amount;

        _mint(user, amount);

        emit TokensBridgedIn(user, amount, baseTxHash);
    }

    /**
     * @dev Bridge tokens FROM Ethereum Mainnet TO Base Sepolia
     * Burns tokens on Ethereum, operator will unlock on Base Sepolia
     */
    function bridgeOut(uint256 amount, address baseAddress) external whenNotPaused nonReentrant {
        require(amount > 0, "Amount must be greater than 0");
        require(balanceOf(msg.sender) >= amount, "Insufficient balance");
        require(baseAddress != address(0), "Invalid Base address");

        bridgedBalances[msg.sender] -= amount;
        _burn(msg.sender, amount);

        emit TokensBridgedOut(msg.sender, amount, baseAddress);
    }

    /**
     * @dev Emergency mint function (only owner)
     */
    function mint(address to, uint256 amount) external onlyOwner {
        _mint(to, amount);
    }

    /**
     * @dev Update bridge operator
     */
    function updateBridgeOperator(address newOperator) external onlyOwner {
        require(newOperator != address(0), "Invalid operator address");
        bridgeOperator = newOperator;
        emit BridgeOperatorUpdated(newOperator);
    }

    /**
     * @dev Pause bridge operations
     */
    function pause() external onlyOwner {
        _pause();
    }

    /**
     * @dev Unpause bridge operations
     */
    function unpause() external onlyOwner {
        _unpause();
    }

    /**
     * @dev Check if transaction has been processed
     */
    function isProcessed(string calldata baseTxHash) external view returns (bool) {
        return processedBaseTxs[baseTxHash];
    }

    /**
     * @dev Modifier to restrict access to bridge operator
     */
    modifier onlyBridgeOperator() {
        require(msg.sender == bridgeOperator, "Only bridge operator can call this");
        _;
    }
}
