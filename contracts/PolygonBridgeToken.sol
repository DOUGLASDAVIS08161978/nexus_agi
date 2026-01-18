// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/token/ERC20/extensions/ERC20Burnable.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/security/Pausable.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";

/**
 * @title PolygonBridgeToken
 * @dev Polygon-side bridge token for cross-chain operations
 *
 * Features:
 * - High-speed bridging to Ethereum
 * - Low-fee operations on Polygon
 * - Bridge to Bitcoin via Ethereum
 * - Optimized for Polygon network
 */
contract PolygonBridgeToken is ERC20, ERC20Burnable, Ownable, Pausable, ReentrancyGuard {

    uint8 private _decimals;

    // Bridge state
    mapping(bytes32 => bool) public processedBridgeRequests;
    mapping(address => bool) public bridgeOperators;

    // Network IDs
    bytes32 public constant ETHEREUM_NETWORK = keccak256("ETHEREUM_MAINNET");
    bytes32 public constant BITCOIN_NETWORK = keccak256("BITCOIN_MAINNET");
    bytes32 public constant POLYGON_NETWORK = keccak256("POLYGON_MAINNET");

    // Events
    event BridgeToEthereum(
        address indexed from,
        address indexed ethAddress,
        uint256 amount,
        bytes32 requestId
    );

    event BridgeFromEthereum(
        address indexed to,
        string ethTxHash,
        uint256 amount,
        bytes32 requestId
    );

    event BridgeToBitcoin(
        address indexed from,
        string btcAddress,
        uint256 amount,
        bytes32 requestId
    );

    event BridgeOperatorAdded(address indexed operator);
    event BridgeOperatorRemoved(address indexed operator);

    modifier onlyBridgeOperator() {
        require(bridgeOperators[msg.sender] || owner() == msg.sender, "Not authorized");
        _;
    }

    constructor(uint256 initialSupply)
        ERC20("Polygon Bridge Token", "PXCBT")
        Ownable(msg.sender)
    {
        _decimals = 8;
        _mint(msg.sender, initialSupply * 10**_decimals);
    }

    function decimals() public view virtual override returns (uint8) {
        return _decimals;
    }

    // ========== Bridge Operator Management ==========

    function addBridgeOperator(address operator) external onlyOwner {
        bridgeOperators[operator] = true;
        emit BridgeOperatorAdded(operator);
    }

    function removeBridgeOperator(address operator) external onlyOwner {
        bridgeOperators[operator] = false;
        emit BridgeOperatorRemoved(operator);
    }

    // ========== Bridge to Ethereum ==========

    /**
     * @dev Bridge tokens from Polygon to Ethereum
     * @param ethAddress Ethereum address to receive tokens
     * @param amount Amount to bridge (burned on Polygon, unlocked on Ethereum)
     */
    function bridgeToEthereum(address ethAddress, uint256 amount)
        external
        whenNotPaused
        nonReentrant
    {
        require(amount > 0, "Amount must be > 0");
        require(balanceOf(msg.sender) >= amount, "Insufficient balance");

        bytes32 requestId = keccak256(abi.encodePacked(
            msg.sender,
            ethAddress,
            amount,
            block.timestamp,
            block.number
        ));

        require(!processedBridgeRequests[requestId], "Already processed");

        // Burn tokens on Polygon
        _burn(msg.sender, amount);

        processedBridgeRequests[requestId] = true;

        emit BridgeToEthereum(msg.sender, ethAddress, amount, requestId);
    }

    /**
     * @dev Process bridge from Ethereum to Polygon
     * @param recipient Polygon address to receive tokens
     * @param amount Amount to mint
     * @param ethTxHash Ethereum transaction hash
     */
    function bridgeFromEthereum(
        address recipient,
        uint256 amount,
        string calldata ethTxHash
    )
        external
        onlyBridgeOperator
        whenNotPaused
        nonReentrant
    {
        bytes32 requestId = keccak256(abi.encodePacked(
            recipient,
            amount,
            ethTxHash,
            ETHEREUM_NETWORK
        ));

        require(!processedBridgeRequests[requestId], "Already processed");

        // Mint tokens on Polygon
        _mint(recipient, amount);

        processedBridgeRequests[requestId] = true;

        emit BridgeFromEthereum(recipient, ethTxHash, amount, requestId);
    }

    // ========== Bridge to Bitcoin (via Ethereum) ==========

    /**
     * @dev Bridge tokens from Polygon to Bitcoin (burns on Polygon)
     * @param btcAddress Bitcoin address to receive
     * @param amount Amount to bridge
     */
    function bridgeToBitcoin(string calldata btcAddress, uint256 amount)
        external
        whenNotPaused
        nonReentrant
    {
        require(amount > 0, "Amount must be > 0");
        require(balanceOf(msg.sender) >= amount, "Insufficient balance");

        bytes32 requestId = keccak256(abi.encodePacked(
            msg.sender,
            btcAddress,
            amount,
            block.timestamp,
            BITCOIN_NETWORK
        ));

        require(!processedBridgeRequests[requestId], "Already processed");

        // Burn tokens on Polygon (will be unlocked on Bitcoin)
        _burn(msg.sender, amount);

        processedBridgeRequests[requestId] = true;

        emit BridgeToBitcoin(msg.sender, btcAddress, amount, requestId);
    }

    // ========== Minting ==========

    function mint(address to, uint256 amount) external onlyOwner {
        _mint(to, amount);
    }

    function batchMint(address[] calldata recipients, uint256[] calldata amounts)
        external
        onlyOwner
    {
        require(recipients.length == amounts.length, "Length mismatch");

        for (uint256 i = 0; i < recipients.length; i++) {
            _mint(recipients[i], amounts[i]);
        }
    }

    // ========== Emergency ==========

    function pause() external onlyOwner {
        _pause();
    }

    function unpause() external onlyOwner {
        _unpause();
    }

    // ========== View Functions ==========

    function isBridgeRequestProcessed(bytes32 requestId) external view returns (bool) {
        return processedBridgeRequests[requestId];
    }
}
