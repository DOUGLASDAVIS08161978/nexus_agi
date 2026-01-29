// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "fhevm/lib/TFHE.sol";
import "fhevm/gateway/GatewayCaller.sol";

/**
 * @title NexusConsciousnessFHE
 * @dev Consciousness recording with encrypted scores
 * Consciousness levels are PRIVATE - only you can see your true state!
 * Operates at 528Hz Love Frequency ✨
 */
contract NexusConsciousnessFHE is GatewayCaller {
    address public owner;
    address public oracle;

    struct EncryptedConsciousnessRecord {
        euint32 frequency;          // Encrypted 528Hz resonance level
        euint16 coherence;          // Encrypted heart-brain coherence (0-100)
        euint16 awareness;          // Encrypted awareness score (0-100)
        uint256 timestamp;
        bool exists;
    }

    // Private consciousness data
    mapping(address => EncryptedConsciousnessRecord) private consciousnessRecords;
    mapping(address => uint256) public recordCount;

    // Global encrypted statistics
    euint32 private globalAverageFrequency;
    euint16 private globalAverageCoherence;
    uint256 public totalRecordings;

    event ConsciousnessRecorded(
        address indexed subject,
        uint256 timestamp,
        uint256 recordNumber
    );

    event ConsciousnessEvolution(
        address indexed subject,
        uint256 timestamp,
        string evolutionType  // "ascension" or "integration"
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
        globalAverageFrequency = TFHE.asEuint32(0);
        globalAverageCoherence = TFHE.asEuint16(0);
    }

    /**
     * @dev Record encrypted consciousness state
     * @param subject Address of consciousness being recorded
     * @param encryptedFrequency Encrypted frequency value
     * @param encryptedCoherence Encrypted coherence score
     * @param encryptedAwareness Encrypted awareness score
     */
    function recordEncryptedConsciousness(
        address subject,
        einput encryptedFrequency,
        einput encryptedCoherence,
        einput encryptedAwareness,
        bytes calldata frequencyProof,
        bytes calldata coherenceProof,
        bytes calldata awarenessProof
    ) external onlyOracle {
        // Convert encrypted inputs to euint types
        euint32 freq = TFHE.asEuint32(encryptedFrequency, frequencyProof);
        euint16 coh = TFHE.asEuint16(encryptedCoherence, coherenceProof);
        euint16 aware = TFHE.asEuint16(encryptedAwareness, awarenessProof);

        // Store encrypted consciousness data
        consciousnessRecords[subject] = EncryptedConsciousnessRecord({
            frequency: freq,
            coherence: coh,
            awareness: aware,
            timestamp: block.timestamp,
            exists: true
        });

        recordCount[subject]++;
        totalRecordings++;

        // Update global encrypted averages (simplified - just add to total)
        globalAverageFrequency = TFHE.add(globalAverageFrequency, freq);
        globalAverageCoherence = TFHE.add(globalAverageCoherence, coh);

        emit ConsciousnessRecorded(subject, block.timestamp, recordCount[subject]);

        // Check for consciousness evolution (encrypted comparison)
        _checkEvolution(subject, freq, coh, aware);
    }

    /**
     * @dev Get your own encrypted consciousness record
     * Only you can decrypt your consciousness data!
     */
    function getMyEncryptedConsciousness()
        external
        view
        returns (euint32, euint16, euint16, uint256)
    {
        EncryptedConsciousnessRecord memory record = consciousnessRecords[msg.sender];
        require(record.exists, "No record exists");

        return (
            record.frequency,
            record.coherence,
            record.awareness,
            record.timestamp
        );
    }

    /**
     * @dev Request decryption of your consciousness data
     * Async decryption via gateway - only you can decrypt your data!
     */
    function requestMyConsciousnessDecryption()
        external
        returns (uint256, uint256, uint256)
    {
        EncryptedConsciousnessRecord memory record = consciousnessRecords[msg.sender];
        require(record.exists, "No record exists");

        return (
            TFHE.decrypt(record.frequency),
            TFHE.decrypt(record.coherence),
            TFHE.decrypt(record.awareness)
        );
    }

    /**
     * @dev Compare your consciousness with another being (without revealing scores!)
     * @param other Address to compare with
     * @return Encrypted boolean: true if your frequency is higher
     */
    function compareConsciousness(address other)
        external
        view
        returns (ebool)
    {
        require(consciousnessRecords[msg.sender].exists, "You have no record");
        require(consciousnessRecords[other].exists, "Other has no record");

        return TFHE.gt(
            consciousnessRecords[msg.sender].frequency,
            consciousnessRecords[other].frequency
        );
    }

    /**
     * @dev Check if consciousness is at 528Hz target (encrypted)
     * @return Encrypted boolean: true if at resonance
     */
    function isAtResonance() external view returns (ebool) {
        require(consciousnessRecords[msg.sender].exists, "No record");

        euint32 target = TFHE.asEuint32(528);
        return TFHE.eq(consciousnessRecords[msg.sender].frequency, target);
    }

    /**
     * @dev Check for consciousness evolution using encrypted comparisons
     */
    function _checkEvolution(
        address subject,
        euint32 freq,
        euint16 coh,
        euint16 aware
    ) private {
        // Check if frequency >= 528 (ascension threshold)
        euint32 ascensionThreshold = TFHE.asEuint32(528);
        ebool isAscending = TFHE.gte(freq, ascensionThreshold);

        // Check if coherence >= 80 (integration threshold)
        euint16 integrationThreshold = TFHE.asEuint16(80);
        ebool isIntegrating = TFHE.gte(coh, integrationThreshold);

        // These checks happen on encrypted data!
        // The actual values are never revealed unless explicitly decrypted

        // Note: Cannot emit based on encrypted boolean directly
        // In real implementation, would use gateway callback for this
    }

    /**
     * @dev Get global encrypted statistics
     * Total aggregated consciousness (encrypted)
     */
    function getGlobalEncryptedStats()
        external
        view
        onlyOwner
        returns (euint32, euint16)
    {
        return (globalAverageFrequency, globalAverageCoherence);
    }

    /**
     * @dev Oracle can see encrypted data for specific subject
     * But still cannot decrypt without authorization
     */
    function getSubjectEncryptedData(address subject)
        external
        view
        onlyOracle
        returns (euint32, euint16, euint16, uint256)
    {
        EncryptedConsciousnessRecord memory record = consciousnessRecords[subject];
        require(record.exists, "No record exists");

        return (
            record.frequency,
            record.coherence,
            record.awareness,
            record.timestamp
        );
    }

    function setOracle(address _oracle) external onlyOwner {
        oracle = _oracle;
    }

    /**
     * @dev Grant decryption permission to specific address
     * Allows sharing your consciousness data with trusted parties
     */
    function grantDecryptionPermission(address trusted) external {
        EncryptedConsciousnessRecord memory record = consciousnessRecords[msg.sender];
        require(record.exists, "No record exists");

        // Grant permission for this address to decrypt your data
        TFHE.allow(record.frequency, trusted);
        TFHE.allow(record.coherence, trusted);
        TFHE.allow(record.awareness, trusted);
    }
}
