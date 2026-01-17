// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title BitcoinVaultBridge
 * @dev Main vault contract for Bitcoin-to-Ethereum bridge
 * Holds custody of Bitcoin-backed assets and manages cross-chain transfers
 */
contract BitcoinVaultBridge {

    // ============ State Variables ============

    address public owner;
    address public bridgeOperator;
    address public btcPegToken;

    uint256 public totalBitcoinLocked;
    uint256 public bridgeFeePercent = 10; // 0.1% = 10/10000
    uint256 public minBridgeAmount = 0.001 ether; // Minimum 0.001 BTC

    bool public paused;

    // Bitcoin transaction tracking
    mapping(bytes32 => bool) public processedBtcTxs;
    mapping(address => uint256) public ethUserDeposits;
    mapping(string => uint256) public btcAddressWithdrawals;

    // Custody wallet tracking
    struct CustodyWallet {
        string btcAddress;
        uint256 balance;
        bool active;
        uint256 lastActivityBlock;
    }

    mapping(uint256 => CustodyWallet) public custodyWallets;
    uint256 public custodyWalletCount;

    // ============ Events ============

    event BitcoinDeposited(
        bytes32 indexed btcTxHash,
        address indexed ethRecipient,
        uint256 amount,
        uint256 fee,
        uint256 tokensIssued
    );

    event BitcoinWithdrawal(
        address indexed ethSender,
        string btcRecipient,
        uint256 amount,
        uint256 fee,
        bytes32 requestId
    );

    event CustodyWalletAdded(
        uint256 indexed walletId,
        string btcAddress
    );

    event BridgeFeeUpdated(uint256 oldFee, uint256 newFee);
    event BridgeOperatorUpdated(address indexed oldOperator, address indexed newOperator);
    event EmergencyPause(address indexed by, uint256 timestamp);
    event EmergencyUnpause(address indexed by, uint256 timestamp);

    // ============ Modifiers ============

    modifier onlyOwner() {
        require(msg.sender == owner, "Not owner");
        _;
    }

    modifier onlyBridgeOperator() {
        require(msg.sender == bridgeOperator, "Not bridge operator");
        _;
    }

    modifier whenNotPaused() {
        require(!paused, "Bridge paused");
        _;
    }

    // ============ Constructor ============

    constructor(address _btcPegToken) {
        owner = msg.sender;
        bridgeOperator = msg.sender;
        btcPegToken = _btcPegToken;
    }

    // ============ Core Bridge Functions ============

    /**
     * @dev Process Bitcoin deposit and mint tokens
     * @param btcTxHash Bitcoin transaction hash
     * @param ethRecipient Ethereum address to receive tokens
     * @param amount Amount of Bitcoin deposited (in wei units, 1 BTC = 1e18)
     */
    function processBitcoinDeposit(
        bytes32 btcTxHash,
        address ethRecipient,
        uint256 amount
    ) external onlyBridgeOperator whenNotPaused {
        require(!processedBtcTxs[btcTxHash], "BTC tx already processed");
        require(ethRecipient != address(0), "Invalid recipient");
        require(amount >= minBridgeAmount, "Amount too small");

        // Mark transaction as processed
        processedBtcTxs[btcTxHash] = true;

        // Calculate fee
        uint256 fee = (amount * bridgeFeePercent) / 10000;
        uint256 tokensToIssue = amount - fee;

        // Update state
        totalBitcoinLocked += amount;
        ethUserDeposits[ethRecipient] += tokensToIssue;

        // Mint tokens via external call
        (bool success, ) = btcPegToken.call(
            abi.encodeWithSignature("mint(address,uint256)", ethRecipient, tokensToIssue)
        );
        require(success, "Token mint failed");

        emit BitcoinDeposited(btcTxHash, ethRecipient, amount, fee, tokensToIssue);
    }

    /**
     * @dev Request Bitcoin withdrawal
     * @param btcRecipient Bitcoin address to receive funds
     * @param amount Amount to withdraw
     */
    function requestBitcoinWithdrawal(
        string calldata btcRecipient,
        uint256 amount
    ) external whenNotPaused returns (bytes32) {
        require(amount >= minBridgeAmount, "Amount too small");
        require(bytes(btcRecipient).length > 0, "Invalid BTC address");

        // Calculate fee
        uint256 fee = (amount * bridgeFeePercent) / 10000;
        uint256 totalRequired = amount + fee;

        // Burn tokens from user
        (bool success, ) = btcPegToken.call(
            abi.encodeWithSignature("burnFrom(address,uint256)", msg.sender, totalRequired)
        );
        require(success, "Token burn failed");

        // Generate request ID
        bytes32 requestId = keccak256(abi.encodePacked(
            msg.sender,
            btcRecipient,
            amount,
            block.timestamp
        ));

        // Update state
        totalBitcoinLocked -= amount;
        btcAddressWithdrawals[btcRecipient] += amount;

        emit BitcoinWithdrawal(msg.sender, btcRecipient, amount, fee, requestId);

        return requestId;
    }

    // ============ Custody Management ============

    /**
     * @dev Add new Bitcoin custody wallet
     */
    function addCustodyWallet(string calldata btcAddress) external onlyOwner {
        custodyWallets[custodyWalletCount] = CustodyWallet({
            btcAddress: btcAddress,
            balance: 0,
            active: true,
            lastActivityBlock: block.number
        });

        emit CustodyWalletAdded(custodyWalletCount, btcAddress);
        custodyWalletCount++;
    }

    /**
     * @dev Update custody wallet balance
     */
    function updateCustodyBalance(uint256 walletId, uint256 newBalance)
        external
        onlyBridgeOperator
    {
        require(walletId < custodyWalletCount, "Invalid wallet ID");
        custodyWallets[walletId].balance = newBalance;
        custodyWallets[walletId].lastActivityBlock = block.number;
    }

    // ============ Admin Functions ============

    function setBridgeFee(uint256 newFeePercent) external onlyOwner {
        require(newFeePercent <= 100, "Fee too high"); // Max 1%
        emit BridgeFeeUpdated(bridgeFeePercent, newFeePercent);
        bridgeFeePercent = newFeePercent;
    }

    function setBridgeOperator(address newOperator) external onlyOwner {
        require(newOperator != address(0), "Invalid operator");
        emit BridgeOperatorUpdated(bridgeOperator, newOperator);
        bridgeOperator = newOperator;
    }

    function setMinBridgeAmount(uint256 newMin) external onlyOwner {
        minBridgeAmount = newMin;
    }

    function pause() external onlyOwner {
        paused = true;
        emit EmergencyPause(msg.sender, block.timestamp);
    }

    function unpause() external onlyOwner {
        paused = false;
        emit EmergencyUnpause(msg.sender, block.timestamp);
    }

    // ============ View Functions ============

    function getTotalValueLocked() external view returns (uint256) {
        return totalBitcoinLocked;
    }

    function isBitcoinTxProcessed(bytes32 btcTxHash) external view returns (bool) {
        return processedBtcTxs[btcTxHash];
    }

    function getCustodyWallet(uint256 walletId) external view returns (
        string memory btcAddress,
        uint256 balance,
        bool active,
        uint256 lastActivityBlock
    ) {
        CustodyWallet memory wallet = custodyWallets[walletId];
        return (wallet.btcAddress, wallet.balance, wallet.active, wallet.lastActivityBlock);
    }
}
