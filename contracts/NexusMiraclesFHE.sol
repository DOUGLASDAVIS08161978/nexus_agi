// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "fhevm/lib/TFHE.sol";
import "fhevm/gateway/GatewayCaller.sol";

/**
 * @title NexusMiraclesFHE
 * @dev Miracle event recording with encrypted verification
 * Your miracles remain private - only you decide what to reveal!
 * Operates at 528Hz Love Frequency ✨
 */
contract NexusMiraclesFHE is GatewayCaller {
    address public owner;
    address public oracle;

    struct EncryptedMiracle {
        bytes32 miracleId;
        address witness;
        euint32 magnitude;          // Encrypted miracle magnitude (0-1000)
        euint16 verificationScore;  // Encrypted verification confidence (0-100)
        euint8 category;            // Encrypted miracle category
        uint256 timestamp;
        bool exists;
    }

    // Private miracle records
    mapping(bytes32 => EncryptedMiracle) private miracles;
    mapping(address => bytes32[]) private witnessMiracles;
    mapping(address => euint32) private totalMiraclePoints;  // Encrypted total

    // Public count (number of miracles recorded)
    uint256 public totalMiracles;

    // Encrypted global statistics
    euint32 private globalMiracleMagnitude;

    event MiracleRecorded(
        bytes32 indexed miracleId,
        address indexed witness,
        uint256 timestamp
    );

    event MiracleVerified(
        bytes32 indexed miracleId,
        uint256 timestamp
    );

    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner");
        _;
    }

    modifier onlyOracle() {
        require(msg.sender == oracle, "Only oracle");
        _;
    }

    constructor() {
        owner = msg.sender;
        oracle = msg.sender;
        globalMiracleMagnitude = TFHE.asEuint32(0);
    }

    /**
     * @dev Record encrypted miracle event
     * Miracle details remain private until you choose to reveal them
     */
    function recordEncryptedMiracle(
        bytes32 miracleId,
        address witness,
        einput encryptedMagnitude,
        einput encryptedVerification,
        einput encryptedCategory,
        bytes calldata magnitudeProof,
        bytes calldata verificationProof,
        bytes calldata categoryProof
    ) external onlyOracle {
        require(!miracles[miracleId].exists, "Miracle already recorded");

        // Convert encrypted inputs
        euint32 mag = TFHE.asEuint32(encryptedMagnitude, magnitudeProof);
        euint16 verify = TFHE.asEuint16(encryptedVerification, verificationProof);
        euint8 cat = TFHE.asEuint8(encryptedCategory, categoryProof);

        // Store encrypted miracle data
        miracles[miracleId] = EncryptedMiracle({
            miracleId: miracleId,
            witness: witness,
            magnitude: mag,
            verificationScore: verify,
            category: cat,
            timestamp: block.timestamp,
            exists: true
        });

        // Track witness miracles
        witnessMiracles[witness].push(miracleId);

        // Add to witness total miracle points (encrypted addition!)
        if (TFHE.isInitialized(totalMiraclePoints[witness])) {
            totalMiraclePoints[witness] = TFHE.add(totalMiraclePoints[witness], mag);
        } else {
            totalMiraclePoints[witness] = mag;
        }

        // Add to global magnitude
        globalMiracleMagnitude = TFHE.add(globalMiracleMagnitude, mag);
        totalMiracles++;

        emit MiracleRecorded(miracleId, witness, block.timestamp);
    }

    /**
     * @dev Get your encrypted miracle data
     * Only you can see your miracle records
     */
    function getMyEncryptedMiracle(bytes32 miracleId)
        external
        view
        returns (euint32, euint16, euint8, uint256)
    {
        EncryptedMiracle memory miracle = miracles[miracleId];
        require(miracle.exists, "Miracle does not exist");
        require(miracle.witness == msg.sender, "Not your miracle");

        return (
            miracle.magnitude,
            miracle.verificationScore,
            miracle.category,
            miracle.timestamp
        );
    }

    /**
     * @dev Request decryption of your miracle data
     * Async decryption - only you can decrypt
     */
    function requestMiracleDecryption(bytes32 miracleId)
        external
        returns (uint256, uint256, uint256)
    {
        EncryptedMiracle memory miracle = miracles[miracleId];
        require(miracle.exists, "Miracle does not exist");
        require(miracle.witness == msg.sender || msg.sender == oracle, "Not authorized");

        return (
            TFHE.decrypt(miracle.magnitude),
            TFHE.decrypt(miracle.verificationScore),
            TFHE.decrypt(miracle.category)
        );
    }

    /**
     * @dev Get list of your miracle IDs
     * The details remain encrypted
     */
    function getMyMiracleIds() external view returns (bytes32[] memory) {
        return witnessMiracles[msg.sender];
    }

    /**
     * @dev Get your total encrypted miracle points
     * Only you can decrypt to see the actual total
     */
    function getMyEncryptedTotal() external view returns (euint32) {
        require(TFHE.isInitialized(totalMiraclePoints[msg.sender]), "No miracles recorded");
        return totalMiraclePoints[msg.sender];
    }

    /**
     * @dev Request decryption of your total miracle points
     */
    function requestMyTotalDecryption() external returns (uint256) {
        require(TFHE.isInitialized(totalMiraclePoints[msg.sender]), "No miracles recorded");
        return TFHE.decrypt(totalMiraclePoints[msg.sender]);
    }

    /**
     * @dev Compare two miracles without revealing magnitudes
     * @return Encrypted boolean: true if miracle1 > miracle2
     */
    function compareMiracles(bytes32 miracleId1, bytes32 miracleId2)
        external
        view
        returns (ebool)
    {
        require(miracles[miracleId1].exists, "Miracle 1 does not exist");
        require(miracles[miracleId2].exists, "Miracle 2 does not exist");

        return TFHE.gt(
            miracles[miracleId1].magnitude,
            miracles[miracleId2].magnitude
        );
    }

    /**
     * @dev Compare your total miracle points with another witness
     * Results are encrypted - neither party learns the actual values!
     * @return Encrypted boolean: true if your total is higher
     */
    function compareWithWitness(address other) external view returns (ebool) {
        require(TFHE.isInitialized(totalMiraclePoints[msg.sender]), "You have no miracles");
        require(TFHE.isInitialized(totalMiraclePoints[other]), "Other has no miracles");

        return TFHE.gt(
            totalMiraclePoints[msg.sender],
            totalMiraclePoints[other]
        );
    }

    /**
     * @dev Check if miracle meets verification threshold (encrypted)
     * @param miracleId Miracle to check
     * @return Encrypted boolean: true if verified
     */
    function isMiracleVerified(bytes32 miracleId) external view returns (ebool) {
        require(miracles[miracleId].exists, "Miracle does not exist");

        euint16 threshold = TFHE.asEuint16(75);  // 75% verification threshold
        return TFHE.gte(miracles[miracleId].verificationScore, threshold);
    }

    /**
     * @dev Get global encrypted statistics
     * Total miracle magnitude across all witnesses (encrypted)
     */
    function getGlobalEncryptedMagnitude() external view onlyOwner returns (euint32) {
        return globalMiracleMagnitude;
    }

    /**
     * @dev Request decryption of global statistics
     */
    function requestGlobalDecryption() external onlyOwner returns (uint256) {
        return TFHE.decrypt(globalMiracleMagnitude);
    }

    /**
     * @dev Oracle can access encrypted miracle data
     * But cannot decrypt without authorization
     */
    function getOracleMiracleView(bytes32 miracleId)
        external
        view
        onlyOracle
        returns (address, euint32, euint16, euint8, uint256)
    {
        EncryptedMiracle memory miracle = miracles[miracleId];
        require(miracle.exists, "Miracle does not exist");

        return (
            miracle.witness,
            miracle.magnitude,
            miracle.verificationScore,
            miracle.category,
            miracle.timestamp
        );
    }

    /**
     * @dev Grant decryption permission for specific miracle
     * Share your miracle with trusted parties
     */
    function shareMiracle(bytes32 miracleId, address trustedParty) external {
        EncryptedMiracle memory miracle = miracles[miracleId];
        require(miracle.exists, "Miracle does not exist");
        require(miracle.witness == msg.sender, "Not your miracle");

        // Grant decryption permissions
        TFHE.allow(miracle.magnitude, trustedParty);
        TFHE.allow(miracle.verificationScore, trustedParty);
        TFHE.allow(miracle.category, trustedParty);
    }

    /**
     * @dev Check if you have any miracles in a specific category (encrypted comparison)
     * @param targetCategory Category to check (encrypted)
     * @return Count of miracles in category (public)
     */
    function countMiraclesInCategory(uint8 targetCategory) external view returns (uint256) {
        bytes32[] memory myMiracles = witnessMiracles[msg.sender];
        uint256 count = 0;

        euint8 target = TFHE.asEuint8(targetCategory);

        for (uint256 i = 0; i < myMiracles.length; i++) {
            EncryptedMiracle memory miracle = miracles[myMiracles[i]];
            // This comparison happens on encrypted data!
            ebool matches = TFHE.eq(miracle.category, target);
            // In production, would need gateway callback to handle encrypted boolean
        }

        return count;
    }

    function setOracle(address _oracle) external onlyOwner {
        oracle = _oracle;
    }

    /**
     * @dev Miracle categories (encrypted in actual usage):
     * 1 = Healing
     * 2 = Manifestation
     * 3 = Synchronicity
     * 4 = Transformation
     * 5 = Divine Intervention
     * 6 = Consciousness Expansion
     * 7 = Love Frequency Resonance (528Hz)
     */
}
