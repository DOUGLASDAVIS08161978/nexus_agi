// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/utils/Pausable.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";

/**
 * @title TBTC - Testnet Bitcoin on Base Sepolia
 * @dev ERC-20 token with burn/mint bridge functionality
 * @dev 1:1 peg with Bitcoin testnet
 */
contract TBTC is ERC20, Ownable, Pausable, ReentrancyGuard {

    // Total supply: 1 million TBTC
    uint256 public constant MAX_SUPPLY = 1_000_000 * 10**18;

    // Bridge operator address
    address public bridgeOperator;

    // Mapping to track bridged transactions from Bitcoin
    mapping(string => bool) public processedBTCTxs;

    // Mapping to track burn requests
    mapping(uint256 => BurnRequest) public burnRequests;
    uint256 public burnRequestCounter;

    struct BurnRequest {
        address user;
        uint256 amount;
        string btcAddress;
        uint256 timestamp;
        bool processed;
    }

    // Events
    event BridgeOperatorUpdated(address indexed oldOperator, address indexed newOperator);
    event TokensMinted(address indexed to, uint256 amount, string btcTxHash);
    event TokensBurned(address indexed from, uint256 amount, string btcAddress, uint256 requestId);
    event BurnProcessed(uint256 indexed requestId, string btcTxHash);

    /**
     * @dev Constructor - Mints initial supply to deployer
     */
    constructor() ERC20("Testnet Bitcoin", "TBTC") Ownable(msg.sender) {
        bridgeOperator = msg.sender;

        // Mint initial supply (1 million TBTC)
        _mint(msg.sender, MAX_SUPPLY);

        emit BridgeOperatorUpdated(address(0), msg.sender);
    }

    /**
     * @dev Mint TBTC when BTC is locked on Bitcoin testnet
     * @param to Recipient address
     * @param amount Amount of TBTC to mint
     * @param btcTxHash Bitcoin transaction hash proving lock
     */
    function mint(
        address to,
        uint256 amount,
        string calldata btcTxHash
    ) external onlyBridgeOperator whenNotPaused nonReentrant {
        require(!processedBTCTxs[btcTxHash], "BTC transaction already processed");
        require(to != address(0), "Cannot mint to zero address");
        require(amount > 0, "Amount must be greater than 0");
        require(totalSupply() + amount <= MAX_SUPPLY, "Exceeds max supply");

        processedBTCTxs[btcTxHash] = true;
        _mint(to, amount);

        emit TokensMinted(to, amount, btcTxHash);
    }

    /**
     * @dev Burn TBTC to unlock BTC on Bitcoin testnet
     * @param amount Amount of TBTC to burn
     * @param btcAddress Bitcoin address to receive unlocked BTC
     */
    function burn(uint256 amount, string calldata btcAddress) external whenNotPaused nonReentrant {
        require(amount > 0, "Amount must be greater than 0");
        require(balanceOf(msg.sender) >= amount, "Insufficient balance");
        require(bytes(btcAddress).length > 0, "BTC address required");

        _burn(msg.sender, amount);

        burnRequestCounter++;
        burnRequests[burnRequestCounter] = BurnRequest({
            user: msg.sender,
            amount: amount,
            btcAddress: btcAddress,
            timestamp: block.timestamp,
            processed: false
        });

        emit TokensBurned(msg.sender, amount, btcAddress, burnRequestCounter);
    }

    /**
     * @dev Mark burn request as processed by bridge operator
     * @param requestId Burn request ID
     * @param btcTxHash Bitcoin transaction hash proving unlock
     */
    function processBurnRequest(
        uint256 requestId,
        string calldata btcTxHash
    ) external onlyBridgeOperator {
        require(requestId > 0 && requestId <= burnRequestCounter, "Invalid request ID");
        require(!burnRequests[requestId].processed, "Already processed");

        burnRequests[requestId].processed = true;

        emit BurnProcessed(requestId, btcTxHash);
    }

    /**
     * @dev Update bridge operator
     * @param newOperator New bridge operator address
     */
    function updateBridgeOperator(address newOperator) external onlyOwner {
        require(newOperator != address(0), "Invalid operator address");
        address oldOperator = bridgeOperator;
        bridgeOperator = newOperator;
        emit BridgeOperatorUpdated(oldOperator, newOperator);
    }

    /**
     * @dev Pause token operations
     */
    function pause() external onlyOwner {
        _pause();
    }

    /**
     * @dev Unpause token operations
     */
    function unpause() external onlyOwner {
        _unpause();
    }

    /**
     * @dev Check if Bitcoin transaction has been processed
     * @param btcTxHash Bitcoin transaction hash
     */
    function isProcessed(string calldata btcTxHash) external view returns (bool) {
        return processedBTCTxs[btcTxHash];
    }

    /**
     * @dev Get burn request details
     * @param requestId Burn request ID
     */
    function getBurnRequest(uint256 requestId) external view returns (
        address user,
        uint256 amount,
        string memory btcAddress,
        uint256 timestamp,
        bool processed
    ) {
        BurnRequest memory request = burnRequests[requestId];
        return (
            request.user,
            request.amount,
            request.btcAddress,
            request.timestamp,
            request.processed
        );
    }

    /**
     * @dev Emergency withdrawal (owner only)
     * @param token Token address (use address(0) for ETH)
     * @param amount Amount to withdraw
     */
    function emergencyWithdraw(address token, uint256 amount) external onlyOwner {
        if (token == address(0)) {
            payable(owner()).transfer(amount);
        } else {
            IERC20(token).transfer(owner(), amount);
        }
    }

    /**
     * @dev Modifier to restrict access to bridge operator
     */
    modifier onlyBridgeOperator() {
        require(msg.sender == bridgeOperator, "Only bridge operator");
        _;
    }

    /**
     * @dev Override transfer to add pause functionality
     */
    function _update(address from, address to, uint256 value) internal virtual override whenNotPaused {
        super._update(from, to, value);
    }
}
