// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title BitcoinOracleRegistry
 * @dev Oracle system for Bitcoin price feeds and blockchain validation
 * Aggregates data from multiple oracle providers for security
 */
contract BitcoinOracleRegistry {

    // ============ State Variables ============

    address public owner;

    // Price oracle data
    struct PriceData {
        uint256 price; // BTC price in USD (18 decimals)
        uint256 timestamp;
        address provider;
        bool verified;
    }

    mapping(uint256 => PriceData) public priceHistory; // roundId => PriceData
    uint256 public currentRound;
    uint256 public heartbeat = 1 hours; // Price update frequency

    // Oracle providers
    struct OracleProvider {
        address providerAddress;
        string name;
        bool active;
        uint256 totalUpdates;
        uint256 lastUpdateTime;
        uint256 reputation; // 0-100
    }

    mapping(address => OracleProvider) public oracleProviders;
    address[] public activeProviders;

    // Bitcoin transaction validation
    struct BTCTransactionProof {
        bytes32 txHash;
        uint256 blockHeight;
        bytes32 blockHash;
        bytes merkleProof;
        uint256 confirmations;
        bool validated;
        address validator;
        uint256 timestamp;
    }

    mapping(bytes32 => BTCTransactionProof) public txProofs;

    // Consensus mechanism
    uint256 public minOracleConsensus = 3; // Minimum agreeing oracles
    uint256 public priceDeviationThreshold = 100; // 1% = 100 basis points

    // ============ Events ============

    event PriceUpdated(
        uint256 indexed roundId,
        uint256 price,
        address indexed provider,
        uint256 timestamp
    );

    event OracleProviderAdded(
        address indexed provider,
        string name
    );

    event OracleProviderRemoved(
        address indexed provider
    );

    event BTCTransactionValidated(
        bytes32 indexed txHash,
        uint256 blockHeight,
        bytes32 blockHash,
        address indexed validator
    );

    event ConsensusReached(
        uint256 indexed roundId,
        uint256 finalPrice,
        uint256 participatingOracles
    );

    // ============ Modifiers ============

    modifier onlyOwner() {
        require(msg.sender == owner, "Not owner");
        _;
    }

    modifier onlyOracle() {
        require(oracleProviders[msg.sender].active, "Not authorized oracle");
        _;
    }

    // ============ Constructor ============

    constructor() {
        owner = msg.sender;
    }

    // ============ Price Oracle Functions ============

    /**
     * @dev Submit price update from oracle
     */
    function submitPrice(uint256 price) external onlyOracle {
        require(price > 0, "Invalid price");
        require(
            block.timestamp >= oracleProviders[msg.sender].lastUpdateTime + heartbeat,
            "Too soon to update"
        );

        currentRound++;

        priceHistory[currentRound] = PriceData({
            price: price,
            timestamp: block.timestamp,
            provider: msg.sender,
            verified: false
        });

        oracleProviders[msg.sender].totalUpdates++;
        oracleProviders[msg.sender].lastUpdateTime = block.timestamp;

        emit PriceUpdated(currentRound, price, msg.sender, block.timestamp);

        // Check for consensus
        _checkPriceConsensus(currentRound);
    }

    /**
     * @dev Internal function to check price consensus
     */
    function _checkPriceConsensus(uint256 roundId) internal {
        uint256 recentPrices = 0;
        uint256 totalPrice = 0;

        // Look at last N submissions
        uint256 lookbackRounds = minOracleConsensus * 2;
        uint256 startRound = roundId > lookbackRounds ? roundId - lookbackRounds : 0;

        for (uint256 i = startRound; i <= roundId; i++) {
            PriceData memory data = priceHistory[i];
            if (data.timestamp > block.timestamp - heartbeat) {
                totalPrice += data.price;
                recentPrices++;
            }
        }

        if (recentPrices >= minOracleConsensus) {
            uint256 avgPrice = totalPrice / recentPrices;
            priceHistory[roundId].verified = true;

            emit ConsensusReached(roundId, avgPrice, recentPrices);
        }
    }

    /**
     * @dev Get latest verified price
     */
    function getLatestPrice() external view returns (
        uint256 price,
        uint256 timestamp,
        bool verified
    ) {
        for (uint256 i = currentRound; i > 0; i--) {
            if (priceHistory[i].verified) {
                return (
                    priceHistory[i].price,
                    priceHistory[i].timestamp,
                    true
                );
            }
        }
        return (0, 0, false);
    }

    /**
     * @dev Get price at specific round
     */
    function getPriceAtRound(uint256 roundId) external view returns (
        uint256 price,
        uint256 timestamp,
        address provider,
        bool verified
    ) {
        PriceData memory data = priceHistory[roundId];
        return (data.price, data.timestamp, data.provider, data.verified);
    }

    // ============ Bitcoin Transaction Validation ============

    /**
     * @dev Submit Bitcoin transaction proof
     */
    function submitBTCTransactionProof(
        bytes32 txHash,
        uint256 blockHeight,
        bytes32 blockHash,
        bytes calldata merkleProof,
        uint256 confirmations
    ) external onlyOracle {
        require(txHash != bytes32(0), "Invalid tx hash");
        require(blockHash != bytes32(0), "Invalid block hash");
        require(confirmations >= 6, "Insufficient confirmations");

        txProofs[txHash] = BTCTransactionProof({
            txHash: txHash,
            blockHeight: blockHeight,
            blockHash: blockHash,
            merkleProof: merkleProof,
            confirmations: confirmations,
            validated: true,
            validator: msg.sender,
            timestamp: block.timestamp
        });

        emit BTCTransactionValidated(txHash, blockHeight, blockHash, msg.sender);
    }

    /**
     * @dev Verify Bitcoin transaction
     */
    function verifyBTCTransaction(bytes32 txHash)
        external
        view
        returns (bool validated, uint256 confirmations, bytes32 blockHash)
    {
        BTCTransactionProof memory proof = txProofs[txHash];
        return (proof.validated, proof.confirmations, proof.blockHash);
    }

    /**
     * @dev Get full transaction proof
     */
    function getBTCTransactionProof(bytes32 txHash)
        external
        view
        returns (
            uint256 blockHeight,
            bytes32 blockHash,
            bytes memory merkleProof,
            uint256 confirmations,
            bool validated
        )
    {
        BTCTransactionProof memory proof = txProofs[txHash];
        return (
            proof.blockHeight,
            proof.blockHash,
            proof.merkleProof,
            proof.confirmations,
            proof.validated
        );
    }

    // ============ Oracle Provider Management ============

    /**
     * @dev Register new oracle provider
     */
    function addOracleProvider(address provider, string calldata name)
        external
        onlyOwner
    {
        require(provider != address(0), "Invalid provider");
        require(!oracleProviders[provider].active, "Already registered");

        oracleProviders[provider] = OracleProvider({
            providerAddress: provider,
            name: name,
            active: true,
            totalUpdates: 0,
            lastUpdateTime: 0,
            reputation: 100
        });

        activeProviders.push(provider);

        emit OracleProviderAdded(provider, name);
    }

    /**
     * @dev Remove oracle provider
     */
    function removeOracleProvider(address provider) external onlyOwner {
        require(oracleProviders[provider].active, "Not active provider");

        oracleProviders[provider].active = false;

        // Remove from active providers array
        for (uint256 i = 0; i < activeProviders.length; i++) {
            if (activeProviders[i] == provider) {
                activeProviders[i] = activeProviders[activeProviders.length - 1];
                activeProviders.pop();
                break;
            }
        }

        emit OracleProviderRemoved(provider);
    }

    /**
     * @dev Update oracle reputation
     */
    function updateOracleReputation(address provider, uint256 newReputation)
        external
        onlyOwner
    {
        require(newReputation <= 100, "Invalid reputation");
        require(oracleProviders[provider].active, "Not active provider");

        oracleProviders[provider].reputation = newReputation;
    }

    // ============ Admin Functions ============

    function setHeartbeat(uint256 newHeartbeat) external onlyOwner {
        require(newHeartbeat >= 5 minutes, "Heartbeat too short");
        heartbeat = newHeartbeat;
    }

    function setMinConsensus(uint256 newMin) external onlyOwner {
        require(newMin >= 2, "Min consensus too low");
        minOracleConsensus = newMin;
    }

    function setPriceDeviationThreshold(uint256 newThreshold) external onlyOwner {
        require(newThreshold <= 1000, "Threshold too high"); // Max 10%
        priceDeviationThreshold = newThreshold;
    }

    // ============ View Functions ============

    function getActiveProvidersCount() external view returns (uint256) {
        return activeProviders.length;
    }

    function getActiveProviders() external view returns (address[] memory) {
        return activeProviders;
    }

    function getOracleStats(address provider)
        external
        view
        returns (
            string memory name,
            bool active,
            uint256 totalUpdates,
            uint256 lastUpdateTime,
            uint256 reputation
        )
    {
        OracleProvider memory oracle = oracleProviders[provider];
        return (
            oracle.name,
            oracle.active,
            oracle.totalUpdates,
            oracle.lastUpdateTime,
            oracle.reputation
        );
    }
}
