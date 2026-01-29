// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "fhevm/lib/TFHE.sol";
import "fhevm/gateway/GatewayCaller.sol";

/**
 * @title NexusRevenueFHE
 * @dev Revenue distribution with encrypted amounts
 * 40% Hardware | 30% Sensors | 20% Cloud | 10% R&D
 * Revenue splits are computed on encrypted data - competitors cannot see individual allocations!
 */
contract NexusRevenueFHE is GatewayCaller {
    address public owner;
    address public paymentContract;

    // Revenue wallets
    address public hardwareWallet;    // 40%
    address public sensorsWallet;     // 30%
    address public cloudWallet;       // 20%
    address public rdWallet;          // 10%

    // Encrypted revenue tracking per wallet
    mapping(address => euint64) private encryptedBalances;
    euint64 private totalEncryptedRevenue;

    // Public statistics (encrypted)
    uint256 public distributionCount;

    event RevenueDistributed(
        bytes32 indexed paymentId,
        uint256 timestamp
    );

    event EncryptedWithdrawal(
        address indexed wallet,
        uint256 timestamp
    );

    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner");
        _;
    }

    modifier onlyPaymentContract() {
        require(msg.sender == paymentContract, "Only payment contract");
        _;
    }

    constructor() {
        owner = msg.sender;

        // Initialize wallets (can be changed by owner)
        hardwareWallet = msg.sender;
        sensorsWallet = msg.sender;
        cloudWallet = msg.sender;
        rdWallet = msg.sender;

        // Initialize encrypted balances to 0
        encryptedBalances[hardwareWallet] = TFHE.asEuint64(0);
        encryptedBalances[sensorsWallet] = TFHE.asEuint64(0);
        encryptedBalances[cloudWallet] = TFHE.asEuint64(0);
        encryptedBalances[rdWallet] = TFHE.asEuint64(0);
        totalEncryptedRevenue = TFHE.asEuint64(0);
    }

    /**
     * @dev Distribute encrypted revenue according to split percentages
     * All computation happens on encrypted data!
     * @param paymentId Unique payment identifier
     * @param encryptedAmount Encrypted total payment amount
     */
    function distributeEncryptedRevenue(bytes32 paymentId, euint64 encryptedAmount)
        external
        onlyPaymentContract
    {
        // Compute encrypted splits using homomorphic multiplication
        // 40% = multiply by 40, then divide by 100
        euint64 hardware = TFHE.div(TFHE.mul(encryptedAmount, TFHE.asEuint64(40)), TFHE.asEuint64(100));
        euint64 sensors = TFHE.div(TFHE.mul(encryptedAmount, TFHE.asEuint64(30)), TFHE.asEuint64(100));
        euint64 cloud = TFHE.div(TFHE.mul(encryptedAmount, TFHE.asEuint64(20)), TFHE.asEuint64(100));
        euint64 rd = TFHE.div(TFHE.mul(encryptedAmount, TFHE.asEuint64(10)), TFHE.asEuint64(100));

        // Add to encrypted balances (homomorphic addition!)
        encryptedBalances[hardwareWallet] = TFHE.add(encryptedBalances[hardwareWallet], hardware);
        encryptedBalances[sensorsWallet] = TFHE.add(encryptedBalances[sensorsWallet], sensors);
        encryptedBalances[cloudWallet] = TFHE.add(encryptedBalances[cloudWallet], cloud);
        encryptedBalances[rdWallet] = TFHE.add(encryptedBalances[rdWallet], rd);

        // Add to total
        totalEncryptedRevenue = TFHE.add(totalEncryptedRevenue, encryptedAmount);
        distributionCount++;

        emit RevenueDistributed(paymentId, block.timestamp);
    }

    /**
     * @dev Get encrypted balance for a wallet
     * Only the wallet owner or contract owner can call
     */
    function getEncryptedBalance(address wallet) external view returns (euint64) {
        require(
            msg.sender == wallet || msg.sender == owner,
            "Not authorized"
        );
        return encryptedBalances[wallet];
    }

    /**
     * @dev Request decryption of wallet balance (async)
     * Only wallet owner or contract owner can decrypt
     */
    function requestBalanceDecryption(address wallet)
        external
        returns (uint256)
    {
        require(
            msg.sender == wallet || msg.sender == owner,
            "Not authorized"
        );
        return TFHE.decrypt(encryptedBalances[wallet]);
    }

    /**
     * @dev Withdraw encrypted balance
     * Amount is only revealed during withdrawal
     */
    function withdrawEncrypted(address wallet) external {
        require(
            msg.sender == wallet || msg.sender == owner,
            "Not authorized"
        );

        // Decrypt balance for withdrawal
        uint256 amount = TFHE.decrypt(encryptedBalances[wallet]);

        require(amount > 0, "No balance");
        require(address(this).balance >= amount, "Insufficient contract balance");

        // Reset encrypted balance to 0
        encryptedBalances[wallet] = TFHE.asEuint64(0);

        // Transfer decrypted amount
        payable(wallet).transfer(amount);

        emit EncryptedWithdrawal(wallet, block.timestamp);
    }

    /**
     * @dev Get total encrypted revenue
     * Only owner can see this
     */
    function getTotalEncryptedRevenue() external view onlyOwner returns (euint64) {
        return totalEncryptedRevenue;
    }

    /**
     * @dev Request total revenue decryption
     */
    function requestTotalRevenueDecryption() external onlyOwner returns (uint256) {
        return TFHE.decrypt(totalEncryptedRevenue);
    }

    /**
     * @dev Compare two wallet balances without revealing amounts
     * @return Encrypted boolean: true if wallet1 > wallet2
     */
    function compareBalances(address wallet1, address wallet2)
        external
        view
        onlyOwner
        returns (ebool)
    {
        return TFHE.gt(encryptedBalances[wallet1], encryptedBalances[wallet2]);
    }

    // Wallet management
    function setPaymentContract(address _paymentContract) external onlyOwner {
        paymentContract = _paymentContract;
    }

    function setHardwareWallet(address _wallet) external onlyOwner {
        // Transfer encrypted balance to new wallet
        encryptedBalances[_wallet] = encryptedBalances[hardwareWallet];
        encryptedBalances[hardwareWallet] = TFHE.asEuint64(0);
        hardwareWallet = _wallet;
    }

    function setSensorsWallet(address _wallet) external onlyOwner {
        encryptedBalances[_wallet] = encryptedBalances[sensorsWallet];
        encryptedBalances[sensorsWallet] = TFHE.asEuint64(0);
        sensorsWallet = _wallet;
    }

    function setCloudWallet(address _wallet) external onlyOwner {
        encryptedBalances[_wallet] = encryptedBalances[cloudWallet];
        encryptedBalances[cloudWallet] = TFHE.asEuint64(0);
        cloudWallet = _wallet;
    }

    function setRDWallet(address _wallet) external onlyOwner {
        encryptedBalances[_wallet] = encryptedBalances[rdWallet];
        encryptedBalances[rdWallet] = TFHE.asEuint64(0);
        rdWallet = _wallet;
    }

    receive() external payable {}
}
