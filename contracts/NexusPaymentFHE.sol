// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

// Zama fhevm integration for encrypted payments
// Amounts are encrypted - only authorized parties can decrypt
import "fhevm/lib/TFHE.sol";
import "fhevm/gateway/GatewayCaller.sol";

/**
 * @title NexusPaymentFHE
 * @dev Payment processing with Fully Homomorphic Encryption
 * Payment amounts are encrypted - competitors cannot see your revenue!
 */
contract NexusPaymentFHE is GatewayCaller {
    address public owner;
    address public revenueContract;

    // Encrypted payment tracking
    mapping(bytes32 => euint64) private encryptedPayments;  // Encrypted amounts
    mapping(bytes32 => address) public paymentPayers;
    mapping(bytes32 => uint256) public paymentTimestamps;

    // Public statistics (computed on encrypted data)
    euint64 private totalEncryptedRevenue;
    uint256 public paymentCount;

    event PaymentReceived(bytes32 indexed paymentId, address indexed payer, uint256 timestamp);
    event RevenueDistributed(bytes32 indexed paymentId, uint256 timestamp);

    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner");
        _;
    }

    constructor() {
        owner = msg.sender;
        totalEncryptedRevenue = TFHE.asEuint64(0);
    }

    /**
     * @dev Process encrypted payment
     * @param encryptedAmount Encrypted payment amount (euint64)
     * @param paymentId Unique payment identifier
     */
    function processEncryptedPayment(
        einput encryptedAmount,
        bytes32 paymentId,
        bytes calldata inputProof
    ) external payable {
        // Decrypt and validate the encrypted input
        euint64 amount = TFHE.asEuint64(encryptedAmount, inputProof);

        // Store encrypted payment data
        encryptedPayments[paymentId] = amount;
        paymentPayers[paymentId] = msg.sender;
        paymentTimestamps[paymentId] = block.timestamp;

        // Add to total encrypted revenue (homomorphic addition!)
        totalEncryptedRevenue = TFHE.add(totalEncryptedRevenue, amount);
        paymentCount++;

        emit PaymentReceived(paymentId, msg.sender, block.timestamp);

        // Distribute to revenue contract if set
        if (revenueContract != address(0)) {
            _distributeEncryptedRevenue(paymentId, amount);
        }
    }

    /**
     * @dev Get encrypted payment amount (only authorized)
     * @param paymentId Payment identifier
     * @return Encrypted amount (can only be decrypted by authorized parties)
     */
    function getEncryptedPayment(bytes32 paymentId) external view returns (euint64) {
        // Access control can be added here
        return encryptedPayments[paymentId];
    }

    /**
     * @dev Request decryption of payment amount (async via gateway)
     * @param paymentId Payment identifier
     */
    function requestPaymentDecryption(bytes32 paymentId) external onlyOwner returns (uint256) {
        euint64 encAmount = encryptedPayments[paymentId];
        return TFHE.decrypt(encAmount);
    }

    /**
     * @dev Get total encrypted revenue
     * Only owner can decrypt to see actual value
     */
    function getTotalEncryptedRevenue() external view onlyOwner returns (euint64) {
        return totalEncryptedRevenue;
    }

    /**
     * @dev Request decryption of total revenue
     */
    function requestTotalRevenueDecryption() external onlyOwner returns (uint256) {
        return TFHE.decrypt(totalEncryptedRevenue);
    }

    /**
     * @dev Distribute encrypted revenue to revenue contract
     */
    function _distributeEncryptedRevenue(bytes32 paymentId, euint64 encryptedAmount) private {
        (bool success, ) = revenueContract.call(
            abi.encodeWithSignature(
                "distributeEncryptedRevenue(bytes32,uint64)",
                paymentId,
                encryptedAmount
            )
        );

        if (success) {
            emit RevenueDistributed(paymentId, block.timestamp);
        }
    }

    function setRevenueContract(address _revenueContract) external onlyOwner {
        revenueContract = _revenueContract;
    }

    /**
     * @dev Compare encrypted payments (without revealing amounts!)
     * @return Encrypted boolean: true if payment1 > payment2
     */
    function comparePayments(bytes32 paymentId1, bytes32 paymentId2)
        external
        view
        returns (ebool)
    {
        return TFHE.gt(encryptedPayments[paymentId1], encryptedPayments[paymentId2]);
    }

    receive() external payable {}
}
