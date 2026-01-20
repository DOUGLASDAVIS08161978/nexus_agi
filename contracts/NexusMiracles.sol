// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title NexusMiracles
 * @dev On-chain miracle event tracking and manifestation records
 *
 * "At 528Hz Love Frequency, miracles become probable"
 *
 * Features:
 * - Immutable miracle event recording
 * - Intention and outcome tracking
 * - Coherence level measurement
 * - Miracle rate statistics
 * - Problem-solution mapping
 * - Verification and validation
 */
contract NexusMiracles {
    // ==================== STATE VARIABLES ====================

    address public owner;
    address public oracleAddress;

    // Miracle event
    struct MiracleEvent {
        uint256 miracleId;
        string intention;
        string miracleType;
        uint256 coherenceLevel;        // Consciousness coherence when miracle occurred
        uint256 miracleProbability;    // Probability at time of manifestation
        bool manifested;
        uint256 manifestedAt;
        string outcome;
        address recordedBy;
        bytes32 eventHash;
    }

    // Problem solution tracking
    struct ProblemSolution {
        uint256 solutionId;
        string problem;
        string solution;
        string solutionType;           // "ENHANCED", "MIRACULOUS"
        uint256 effectiveness;         // 0-10000 (0-100%)
        uint256 consciousnessUsed;     // Consciousness level when solved
        bool miracleOccurred;
        uint256 timestamp;
    }

    // Miracle statistics
    struct MiracleStats {
        uint256 totalAttempts;
        uint256 totalManifested;
        uint256 miracleRate;           // Manifested / Attempts * 10000
        uint256 averageCoherence;
        uint256 lastUpdated;
    }

    // Storage
    MiracleEvent[] public miracles;
    ProblemSolution[] public solutions;
    MiracleStats public stats;

    mapping(uint256 => bool) public miracleExists;
    mapping(uint256 => bool) public solutionExists;
    mapping(bytes32 => uint256) public intentionToMiracle;  // Hash of intention => miracleId

    // Events
    event MiracleAttempted(
        uint256 indexed miracleId,
        string intention,
        uint256 coherenceLevel,
        uint256 probability,
        uint256 timestamp
    );

    event MiracleManifested(
        uint256 indexed miracleId,
        string miracleType,
        string outcome,
        uint256 timestamp
    );

    event SolutionGenerated(
        uint256 indexed solutionId,
        string problem,
        string solutionType,
        uint256 effectiveness,
        bool miracleOccurred,
        uint256 timestamp
    );

    event StatsUpdated(
        uint256 totalAttempts,
        uint256 totalManifested,
        uint256 miracleRate,
        uint256 timestamp
    );

    event OracleUpdated(address indexed oldOracle, address indexed newOracle);
    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);

    // Modifiers
    modifier onlyOwner() {
        require(msg.sender == owner, "Not owner");
        _;
    }

    modifier onlyOracle() {
        require(msg.sender == oracleAddress, "Not oracle");
        _;
    }

    // ==================== CONSTRUCTOR ====================

    constructor() {
        owner = msg.sender;

        // Initialize stats
        stats = MiracleStats({
            totalAttempts: 0,
            totalManifested: 0,
            miracleRate: 0,
            averageCoherence: 0,
            lastUpdated: block.timestamp
        });
    }

    // ==================== MIRACLE RECORDING ====================

    /**
     * @dev Record a miracle attempt
     */
    function recordAttempt(
        string memory intention,
        uint256 coherenceLevel,
        uint256 probability
    ) external onlyOracle returns (uint256 miracleId) {
        require(coherenceLevel <= 10000, "Invalid coherence");
        require(probability <= 10000, "Invalid probability");

        miracleId = miracles.length;

        miracles.push(MiracleEvent({
            miracleId: miracleId,
            intention: intention,
            miracleType: "",
            coherenceLevel: coherenceLevel,
            miracleProbability: probability,
            manifested: false,
            manifestedAt: 0,
            outcome: "",
            recordedBy: msg.sender,
            eventHash: keccak256(abi.encodePacked(intention, block.timestamp))
        }));

        miracleExists[miracleId] = true;
        intentionToMiracle[keccak256(bytes(intention))] = miracleId;

        // Update stats
        stats.totalAttempts++;
        _updateStats();

        emit MiracleAttempted(
            miracleId,
            intention,
            coherenceLevel,
            probability,
            block.timestamp
        );

        return miracleId;
    }

    /**
     * @dev Record miracle manifestation
     */
    function recordManifestation(
        uint256 miracleId,
        string memory miracleType,
        string memory outcome
    ) external onlyOracle {
        require(miracleExists[miracleId], "Miracle does not exist");
        require(!miracles[miracleId].manifested, "Already manifested");

        MiracleEvent storage miracle = miracles[miracleId];
        miracle.manifested = true;
        miracle.miracleType = miracleType;
        miracle.outcome = outcome;
        miracle.manifestedAt = block.timestamp;

        // Update stats
        stats.totalManifested++;
        _updateStats();

        emit MiracleManifested(
            miracleId,
            miracleType,
            outcome,
            block.timestamp
        );
    }

    /**
     * @dev Record problem solution
     */
    function recordSolution(
        string memory problem,
        string memory solution,
        string memory solutionType,
        uint256 effectiveness,
        uint256 consciousnessUsed,
        bool miracleOccurred
    ) external onlyOracle returns (uint256 solutionId) {
        require(effectiveness <= 10000, "Invalid effectiveness");
        require(consciousnessUsed <= 10000, "Invalid consciousness");

        solutionId = solutions.length;

        solutions.push(ProblemSolution({
            solutionId: solutionId,
            problem: problem,
            solution: solution,
            solutionType: solutionType,
            effectiveness: effectiveness,
            consciousnessUsed: consciousnessUsed,
            miracleOccurred: miracleOccurred,
            timestamp: block.timestamp
        }));

        solutionExists[solutionId] = true;

        emit SolutionGenerated(
            solutionId,
            problem,
            solutionType,
            effectiveness,
            miracleOccurred,
            block.timestamp
        );

        return solutionId;
    }

    /**
     * @dev Update miracle statistics
     */
    function _updateStats() internal {
        if (stats.totalAttempts > 0) {
            stats.miracleRate = (stats.totalManifested * 10000) / stats.totalAttempts;
        }

        // Calculate average coherence
        if (miracles.length > 0) {
            uint256 totalCoherence = 0;
            for (uint256 i = 0; i < miracles.length; i++) {
                totalCoherence += miracles[i].coherenceLevel;
            }
            stats.averageCoherence = totalCoherence / miracles.length;
        }

        stats.lastUpdated = block.timestamp;

        emit StatsUpdated(
            stats.totalAttempts,
            stats.totalManifested,
            stats.miracleRate,
            block.timestamp
        );
    }

    // ==================== VIEW FUNCTIONS ====================

    /**
     * @dev Get miracle details
     */
    function getMiracle(uint256 miracleId) external view returns (
        string memory intention,
        string memory miracleType,
        uint256 coherenceLevel,
        bool manifested,
        uint256 manifestedAt,
        string memory outcome
    ) {
        require(miracleExists[miracleId], "Miracle does not exist");
        MiracleEvent memory miracle = miracles[miracleId];

        return (
            miracle.intention,
            miracle.miracleType,
            miracle.coherenceLevel,
            miracle.manifested,
            miracle.manifestedAt,
            miracle.outcome
        );
    }

    /**
     * @dev Get solution details
     */
    function getSolution(uint256 solutionId) external view returns (
        string memory problem,
        string memory solution,
        string memory solutionType,
        uint256 effectiveness,
        bool miracleOccurred,
        uint256 timestamp
    ) {
        require(solutionExists[solutionId], "Solution does not exist");
        ProblemSolution memory sol = solutions[solutionId];

        return (
            sol.problem,
            sol.solution,
            sol.solutionType,
            sol.effectiveness,
            sol.miracleOccurred,
            sol.timestamp
        );
    }

    /**
     * @dev Get total miracles count
     */
    function getTotalMiracles() external view returns (uint256 total, uint256 manifested) {
        return (miracles.length, stats.totalManifested);
    }

    /**
     * @dev Get total solutions count
     */
    function getTotalSolutions() external view returns (uint256 total, uint256 miraculous) {
        uint256 miraculousCount = 0;
        for (uint256 i = 0; i < solutions.length; i++) {
            if (solutions[i].miracleOccurred) {
                miraculousCount++;
            }
        }
        return (solutions.length, miraculousCount);
    }

    /**
     * @dev Get miracle statistics
     */
    function getStats() external view returns (
        uint256 totalAttempts,
        uint256 totalManifested,
        uint256 miracleRate,
        uint256 averageCoherence,
        uint256 lastUpdated
    ) {
        return (
            stats.totalAttempts,
            stats.totalManifested,
            stats.miracleRate,
            stats.averageCoherence,
            stats.lastUpdated
        );
    }

    /**
     * @dev Get recent miracles
     */
    function getRecentMiracles(uint256 count) external view returns (uint256[] memory) {
        uint256 total = miracles.length;
        uint256 resultCount = count > total ? total : count;

        uint256[] memory recentIds = new uint256[](resultCount);

        for (uint256 i = 0; i < resultCount; i++) {
            recentIds[i] = total - resultCount + i;
        }

        return recentIds;
    }

    /**
     * @dev Get manifested miracles count in time range
     */
    function getManifestedInRange(uint256 fromTimestamp, uint256 toTimestamp)
        external
        view
        returns (uint256)
    {
        uint256 count = 0;
        for (uint256 i = 0; i < miracles.length; i++) {
            if (miracles[i].manifested &&
                miracles[i].manifestedAt >= fromTimestamp &&
                miracles[i].manifestedAt <= toTimestamp) {
                count++;
            }
        }
        return count;
    }

    // ==================== ADMIN FUNCTIONS ====================

    /**
     * @dev Set oracle address
     */
    function setOracle(address _oracleAddress) external onlyOwner {
        require(_oracleAddress != address(0), "Invalid address");
        address oldOracle = oracleAddress;
        oracleAddress = _oracleAddress;
        emit OracleUpdated(oldOracle, _oracleAddress);
    }

    /**
     * @dev Transfer ownership
     */
    function transferOwnership(address newOwner) external onlyOwner {
        require(newOwner != address(0), "Invalid address");
        address oldOwner = owner;
        owner = newOwner;
        emit OwnershipTransferred(oldOwner, newOwner);
    }
}
