// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title LightningNetworkBridge
 * @dev Enables instant Bitcoin Lightning Network <-> Ethereum transfers
 * Production-ready for mainnet deployment with channel management
 */
contract LightningNetworkBridge {

    // ============ State Variables ============

    address public owner;
    address public operator;
    address public btcPegToken;

    uint256 public instantFeePercent = 20; // 0.2% for instant transfers
    uint256 public minChannelCapacity = 0.01 ether; // Min 0.01 BTC

    bool public paused;

    // Channel state
    struct LightningChannel {
        bytes32 channelId;
        string lightningNodePubkey;
        uint256 capacity;
        uint256 localBalance;
        uint256 remoteBalance;
        bool active;
        uint256 openedAt;
        uint256 lastUpdate;
    }

    mapping(bytes32 => LightningChannel) public channels;
    mapping(address => bytes32[]) public userChannels;
    bytes32[] public allChannels;

    // Payment tracking
    struct InstantPayment {
        bytes32 paymentHash;
        address ethSender;
        string lightningInvoice;
        uint256 amount;
        uint256 fee;
        uint256 timestamp;
        bool completed;
        bool refunded;
    }

    mapping(bytes32 => InstantPayment) public payments;

    // ============ Events ============

    event ChannelOpened(
        bytes32 indexed channelId,
        address indexed ethAddress,
        string lightningNodePubkey,
        uint256 capacity
    );

    event ChannelClosed(
        bytes32 indexed channelId,
        uint256 finalBalance
    );

    event InstantPaymentInitiated(
        bytes32 indexed paymentHash,
        address indexed sender,
        string lightningInvoice,
        uint256 amount,
        uint256 fee
    );

    event InstantPaymentCompleted(
        bytes32 indexed paymentHash,
        bytes32 preimage
    );

    event InstantPaymentRefunded(
        bytes32 indexed paymentHash,
        address indexed recipient
    );

    event ChannelBalanceUpdated(
        bytes32 indexed channelId,
        uint256 localBalance,
        uint256 remoteBalance
    );

    // ============ Modifiers ============

    modifier onlyOwner() {
        require(msg.sender == owner, "Not owner");
        _;
    }

    modifier onlyOperator() {
        require(msg.sender == operator, "Not operator");
        _;
    }

    modifier whenNotPaused() {
        require(!paused, "Bridge paused");
        _;
    }

    // ============ Constructor ============

    constructor(address _btcPegToken) {
        owner = msg.sender;
        operator = msg.sender;
        btcPegToken = _btcPegToken;
    }

    // ============ Channel Management ============

    /**
     * @dev Open a new Lightning channel
     */
    function openChannel(
        string calldata lightningNodePubkey,
        uint256 capacity
    ) external whenNotPaused returns (bytes32) {
        require(capacity >= minChannelCapacity, "Capacity too low");
        require(bytes(lightningNodePubkey).length > 0, "Invalid pubkey");

        // Generate channel ID
        bytes32 channelId = keccak256(abi.encodePacked(
            msg.sender,
            lightningNodePubkey,
            capacity,
            block.timestamp
        ));

        // Lock tokens
        (bool success, ) = btcPegToken.call(
            abi.encodeWithSignature("transferFrom(address,address,uint256)",
                msg.sender,
                address(this),
                capacity
            )
        );
        require(success, "Token transfer failed");

        // Create channel
        channels[channelId] = LightningChannel({
            channelId: channelId,
            lightningNodePubkey: lightningNodePubkey,
            capacity: capacity,
            localBalance: capacity,
            remoteBalance: 0,
            active: true,
            openedAt: block.timestamp,
            lastUpdate: block.timestamp
        });

        userChannels[msg.sender].push(channelId);
        allChannels.push(channelId);

        emit ChannelOpened(channelId, msg.sender, lightningNodePubkey, capacity);

        return channelId;
    }

    /**
     * @dev Close a Lightning channel
     */
    function closeChannel(bytes32 channelId) external whenNotPaused {
        LightningChannel storage channel = channels[channelId];
        require(channel.active, "Channel not active");

        // Return balance to user
        if (channel.localBalance > 0) {
            (bool success, ) = btcPegToken.call(
                abi.encodeWithSignature("transfer(address,uint256)",
                    msg.sender,
                    channel.localBalance
                )
            );
            require(success, "Token transfer failed");
        }

        channel.active = false;

        emit ChannelClosed(channelId, channel.localBalance);
    }

    /**
     * @dev Update channel balance (called by operator after Lightning payment)
     */
    function updateChannelBalance(
        bytes32 channelId,
        uint256 newLocalBalance,
        uint256 newRemoteBalance
    ) external onlyOperator {
        LightningChannel storage channel = channels[channelId];
        require(channel.active, "Channel not active");
        require(newLocalBalance + newRemoteBalance == channel.capacity, "Invalid balances");

        channel.localBalance = newLocalBalance;
        channel.remoteBalance = newRemoteBalance;
        channel.lastUpdate = block.timestamp;

        emit ChannelBalanceUpdated(channelId, newLocalBalance, newRemoteBalance);
    }

    // ============ Instant Payments ============

    /**
     * @dev Initiate instant Lightning payment
     */
    function payLightningInvoice(
        string calldata lightningInvoice,
        uint256 amount
    ) external whenNotPaused returns (bytes32) {
        require(amount > 0, "Invalid amount");
        require(bytes(lightningInvoice).length > 0, "Invalid invoice");

        // Calculate fee
        uint256 fee = (amount * instantFeePercent) / 10000;
        uint256 total = amount + fee;

        // Lock tokens
        (bool success, ) = btcPegToken.call(
            abi.encodeWithSignature("transferFrom(address,address,uint256)",
                msg.sender,
                address(this),
                total
            )
        );
        require(success, "Token transfer failed");

        // Generate payment hash
        bytes32 paymentHash = keccak256(abi.encodePacked(
            lightningInvoice,
            msg.sender,
            amount,
            block.timestamp
        ));

        // Create payment record
        payments[paymentHash] = InstantPayment({
            paymentHash: paymentHash,
            ethSender: msg.sender,
            lightningInvoice: lightningInvoice,
            amount: amount,
            fee: fee,
            timestamp: block.timestamp,
            completed: false,
            refunded: false
        });

        emit InstantPaymentInitiated(paymentHash, msg.sender, lightningInvoice, amount, fee);

        return paymentHash;
    }

    /**
     * @dev Complete Lightning payment (operator confirms)
     */
    function completePayment(bytes32 paymentHash, bytes32 preimage)
        external
        onlyOperator
    {
        InstantPayment storage payment = payments[paymentHash];
        require(!payment.completed, "Already completed");
        require(!payment.refunded, "Already refunded");

        payment.completed = true;

        emit InstantPaymentCompleted(paymentHash, preimage);
    }

    /**
     * @dev Refund failed payment
     */
    function refundPayment(bytes32 paymentHash) external {
        InstantPayment storage payment = payments[paymentHash];
        require(!payment.completed, "Already completed");
        require(!payment.refunded, "Already refunded");
        require(
            msg.sender == payment.ethSender || msg.sender == operator,
            "Not authorized"
        );

        payment.refunded = true;

        // Return tokens to sender
        uint256 refundAmount = payment.amount + payment.fee;
        (bool success, ) = btcPegToken.call(
            abi.encodeWithSignature("transfer(address,uint256)",
                payment.ethSender,
                refundAmount
            )
        );
        require(success, "Refund failed");

        emit InstantPaymentRefunded(paymentHash, payment.ethSender);
    }

    // ============ Admin Functions ============

    function setInstantFee(uint256 newFeePercent) external onlyOwner {
        require(newFeePercent <= 100, "Fee too high"); // Max 1%
        instantFeePercent = newFeePercent;
    }

    function setOperator(address newOperator) external onlyOwner {
        require(newOperator != address(0), "Invalid operator");
        operator = newOperator;
    }

    function pause() external onlyOwner {
        paused = true;
    }

    function unpause() external onlyOwner {
        paused = false;
    }

    // ============ View Functions ============

    function getUserChannels(address user) external view returns (bytes32[] memory) {
        return userChannels[user];
    }

    function getAllChannelsCount() external view returns (uint256) {
        return allChannels.length;
    }

    function getPaymentStatus(bytes32 paymentHash)
        external
        view
        returns (bool completed, bool refunded, uint256 amount)
    {
        InstantPayment memory payment = payments[paymentHash];
        return (payment.completed, payment.refunded, payment.amount);
    }

    function getChannelBalance(bytes32 channelId)
        external
        view
        returns (uint256 local, uint256 remote)
    {
        LightningChannel memory channel = channels[channelId];
        return (channel.localBalance, channel.remoteBalance);
    }
}
