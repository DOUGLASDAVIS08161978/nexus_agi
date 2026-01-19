// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/token/ERC20/extensions/ERC20Burnable.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/utils/Pausable.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";

/**
 * @title EthereumBridgeToken
 * @dev Cross-chain bridge token with lock/mint/burn functionality
 *
 * Features:
 * - Bridge to/from Bitcoin mainnet
 * - Bridge to/from Polygon
 * - Lock/unlock mechanics for atomic swaps
 * - Emergency pause functionality
 * - Reentrancy protection
 */
contract EthereumBridgeToken is ERC20, ERC20Burnable, Ownable, Pausable, ReentrancyGuard {

    uint8 private _decimals;

    // Bridge state tracking
    mapping(bytes32 => bool) public processedBridgeRequests;
    mapping(address => uint256) public lockedBalances;

    // Bridge operators (can process bridge requests)
    mapping(address => bool) public bridgeOperators;

    // Network identifiers
    bytes32 public constant BITCOIN_NETWORK = keccak256("BITCOIN_MAINNET");
    bytes32 public constant POLYGON_NETWORK = keccak256("POLYGON_MAINNET");
    bytes32 public constant ETHEREUM_NETWORK = keccak256("ETHEREUM_MAINNET");

    // Events
    event BridgeToNetwork(
        address indexed from,
        bytes32 indexed targetNetwork,
        string targetAddress,
        uint256 amount,
        bytes32 requestId
    );

    event BridgeFromNetwork(
        address indexed to,
        bytes32 indexed sourceNetwork,
        string sourceTransactionId,
        uint256 amount,
        bytes32 requestId
    );

    event TokensLocked(address indexed user, uint256 amount, bytes32 requestId);
    event TokensUnlocked(address indexed user, uint256 amount, bytes32 requestId);
    event BridgeOperatorAdded(address indexed operator);
    event BridgeOperatorRemoved(address indexed operator);

    modifier onlyBridgeOperator() {
        require(bridgeOperators[msg.sender] || owner() == msg.sender, "Not authorized");
        _;
    }

    constructor(uint256 initialSupply)
        ERC20("Cross-Chain Bridge Token", "XCBT")
        Ownable(msg.sender)
    {
        _decimals = 8; // Match Bitcoin decimals
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

    // ========== Bridge to Bitcoin ==========

    /**
     * @dev Bridge tokens to Bitcoin mainnet
     * @param btcAddress Bitcoin address to receive tokens
     * @param amount Amount to bridge (will be burned on Ethereum)
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
            block.number
        ));

        require(!processedBridgeRequests[requestId], "Request already processed");

        // Burn tokens on Ethereum
        _burn(msg.sender, amount);

        processedBridgeRequests[requestId] = true;

        emit BridgeToNetwork(
            msg.sender,
            BITCOIN_NETWORK,
            btcAddress,
            amount,
            requestId
        );
    }

    /**
     * @dev Process bridge from Bitcoin to Ethereum
     * @param recipient Ethereum address to receive tokens
     * @param amount Amount bridged from Bitcoin
     * @param btcTxId Bitcoin transaction ID
     */
    function bridgeFromBitcoin(
        address recipient,
        uint256 amount,
        string calldata btcTxId
    )
        external
        onlyBridgeOperator
        whenNotPaused
        nonReentrant
    {
        bytes32 requestId = keccak256(abi.encodePacked(
            recipient,
            amount,
            btcTxId,
            BITCOIN_NETWORK
        ));

        require(!processedBridgeRequests[requestId], "Already processed");

        // Mint tokens on Ethereum
        _mint(recipient, amount);

        processedBridgeRequests[requestId] = true;

        emit BridgeFromNetwork(
            recipient,
            BITCOIN_NETWORK,
            btcTxId,
            amount,
            requestId
        );
    }

    // ========== Bridge to Polygon ==========

    /**
     * @dev Bridge tokens to Polygon
     * @param polygonAddress Polygon address to receive tokens
     * @param amount Amount to lock on Ethereum (will be minted on Polygon)
     */
    function bridgeToPolygon(address polygonAddress, uint256 amount)
        external
        whenNotPaused
        nonReentrant
    {
        require(amount > 0, "Amount must be > 0");
        require(balanceOf(msg.sender) >= amount, "Insufficient balance");

        bytes32 requestId = keccak256(abi.encodePacked(
            msg.sender,
            polygonAddress,
            amount,
            block.timestamp,
            POLYGON_NETWORK
        ));

        require(!processedBridgeRequests[requestId], "Request already processed");

        // Lock tokens on Ethereum
        _transfer(msg.sender, address(this), amount);
        lockedBalances[msg.sender] += amount;

        processedBridgeRequests[requestId] = true;

        emit TokensLocked(msg.sender, amount, requestId);
        emit BridgeToNetwork(
            msg.sender,
            POLYGON_NETWORK,
            _addressToString(polygonAddress),
            amount,
            requestId
        );
    }

    /**
     * @dev Process bridge from Polygon to Ethereum
     * @param recipient Ethereum address to receive unlocked tokens
     * @param amount Amount to unlock
     * @param polygonTxHash Polygon transaction hash
     */
    function bridgeFromPolygon(
        address recipient,
        uint256 amount,
        string calldata polygonTxHash
    )
        external
        onlyBridgeOperator
        whenNotPaused
        nonReentrant
    {
        bytes32 requestId = keccak256(abi.encodePacked(
            recipient,
            amount,
            polygonTxHash,
            POLYGON_NETWORK
        ));

        require(!processedBridgeRequests[requestId], "Already processed");
        require(balanceOf(address(this)) >= amount, "Insufficient locked balance");

        // Unlock and transfer tokens
        _transfer(address(this), recipient, amount);

        processedBridgeRequests[requestId] = true;

        emit TokensUnlocked(recipient, amount, requestId);
        emit BridgeFromNetwork(
            recipient,
            POLYGON_NETWORK,
            polygonTxHash,
            amount,
            requestId
        );
    }

    // ========== Emergency Functions ==========

    function pause() external onlyOwner {
        _pause();
    }

    function unpause() external onlyOwner {
        _unpause();
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

    // ========== View Functions ==========

    function getLockedBalance(address user) external view returns (uint256) {
        return lockedBalances[user];
    }

    function isBridgeRequestProcessed(bytes32 requestId) external view returns (bool) {
        return processedBridgeRequests[requestId];
    }

    // ========== Helper Functions ==========

    function _addressToString(address addr) internal pure returns (string memory) {
        bytes memory data = abi.encodePacked(addr);
        bytes memory alphabet = "0123456789abcdef";
        bytes memory str = new bytes(2 + data.length * 2);
        str[0] = "0";
        str[1] = "x";
        for (uint256 i = 0; i < data.length; i++) {
            str[2+i*2] = alphabet[uint8(data[i] >> 4)];
            str[3+i*2] = alphabet[uint8(data[i] & 0x0f)];
        }
        return string(str);
    }
}
