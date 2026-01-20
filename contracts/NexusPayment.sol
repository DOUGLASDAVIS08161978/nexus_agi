// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title NexusPayment
 * @dev Payment processing contract for Nexus AGI subscriptions
 *
 * Features:
 * - Three subscription tiers (Starter, Professional, Enterprise)
 * - Automatic revenue allocation (40% hardware, 30% sensors, 20% cloud, 10% R&D)
 * - Payment history tracking
 * - Upgradeable tier system
 * - Emergency pause functionality
 */
contract NexusPayment {
    // ==================== STATE VARIABLES ====================

    address public owner;
    address public revenueContract;
    bool public paused;

    uint256 public totalPayments;
    uint256 public totalRevenue;
    uint256 public totalCustomers;

    // Subscription tiers
    enum Tier { Free, Starter, Professional, Enterprise }

    struct TierInfo {
        uint256 price;          // Price in wei
        uint256 requests;       // Monthly request limit
        bool active;
    }

    struct Subscription {
        Tier tier;
        uint256 expiresAt;
        uint256 totalPaid;
        bool active;
    }

    struct Payment {
        address customer;
        Tier tier;
        uint256 amount;
        uint256 timestamp;
        bytes32 txHash;
    }

    // Mappings
    mapping(Tier => TierInfo) public tiers;
    mapping(address => Subscription) public subscriptions;
    mapping(address => Payment[]) public paymentHistory;
    mapping(address => bool) public isCustomer;

    // Events
    event PaymentReceived(
        address indexed customer,
        Tier tier,
        uint256 amount,
        uint256 timestamp
    );

    event SubscriptionCreated(
        address indexed customer,
        Tier tier,
        uint256 expiresAt
    );

    event SubscriptionUpgraded(
        address indexed customer,
        Tier fromTier,
        Tier toTier
    );

    event TierUpdated(Tier tier, uint256 newPrice, uint256 newRequests);
    event RevenueAllocated(uint256 amount, address revenueContract);
    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);
    event Paused(address account);
    event Unpaused(address account);

    // Modifiers
    modifier onlyOwner() {
        require(msg.sender == owner, "Not owner");
        _;
    }

    modifier whenNotPaused() {
        require(!paused, "Contract is paused");
        _;
    }

    // ==================== CONSTRUCTOR ====================

    constructor() {
        owner = msg.sender;
        paused = false;

        // Initialize tier pricing (approximate ETH values)
        tiers[Tier.Free] = TierInfo({
            price: 0,
            requests: 100,
            active: true
        });

        tiers[Tier.Starter] = TierInfo({
            price: 0.01 ether,  // ~$29 at $2900 ETH
            requests: 10000,
            active: true
        });

        tiers[Tier.Professional] = TierInfo({
            price: 0.03 ether,  // ~$99 at $2900 ETH
            requests: 100000,
            active: true
        });

        tiers[Tier.Enterprise] = TierInfo({
            price: 0.15 ether,  // ~$499 at $2900 ETH
            requests: 1000000,
            active: true
        });
    }

    // ==================== PAYMENT FUNCTIONS ====================

    /**
     * @dev Purchase a subscription tier
     * @param tier The tier to purchase
     */
    function subscribe(Tier tier) external payable whenNotPaused {
        require(tier != Tier.Free, "Cannot purchase free tier");
        require(tiers[tier].active, "Tier not active");
        require(msg.value == tiers[tier].price, "Incorrect payment amount");

        Subscription storage sub = subscriptions[msg.sender];

        // If new customer
        if (!isCustomer[msg.sender]) {
            isCustomer[msg.sender] = true;
            totalCustomers++;
        }

        // Calculate expiration (30 days from now or from current expiration)
        uint256 expiresAt;
        if (sub.expiresAt > block.timestamp) {
            expiresAt = sub.expiresAt + 30 days;
        } else {
            expiresAt = block.timestamp + 30 days;
        }

        // Update subscription
        Tier oldTier = sub.tier;
        sub.tier = tier;
        sub.expiresAt = expiresAt;
        sub.totalPaid += msg.value;
        sub.active = true;

        // Record payment
        paymentHistory[msg.sender].push(Payment({
            customer: msg.sender,
            tier: tier,
            amount: msg.value,
            timestamp: block.timestamp,
            txHash: blockhash(block.number - 1)
        }));

        // Update totals
        totalPayments++;
        totalRevenue += msg.value;

        // Emit events
        if (oldTier == Tier.Free) {
            emit SubscriptionCreated(msg.sender, tier, expiresAt);
        } else {
            emit SubscriptionUpgraded(msg.sender, oldTier, tier);
        }

        emit PaymentReceived(msg.sender, tier, msg.value, block.timestamp);

        // Allocate revenue if revenue contract is set
        if (revenueContract != address(0)) {
            _allocateRevenue(msg.value);
        }
    }

    /**
     * @dev Renew existing subscription
     */
    function renew() external payable whenNotPaused {
        Subscription storage sub = subscriptions[msg.sender];
        require(sub.tier != Tier.Free, "Free tier cannot be renewed");
        require(msg.value == tiers[sub.tier].price, "Incorrect payment amount");

        // Extend expiration
        if (sub.expiresAt > block.timestamp) {
            sub.expiresAt += 30 days;
        } else {
            sub.expiresAt = block.timestamp + 30 days;
        }

        sub.totalPaid += msg.value;
        sub.active = true;

        // Record payment
        paymentHistory[msg.sender].push(Payment({
            customer: msg.sender,
            tier: sub.tier,
            amount: msg.value,
            timestamp: block.timestamp,
            txHash: blockhash(block.number - 1)
        }));

        totalPayments++;
        totalRevenue += msg.value;

        emit PaymentReceived(msg.sender, sub.tier, msg.value, block.timestamp);

        // Allocate revenue
        if (revenueContract != address(0)) {
            _allocateRevenue(msg.value);
        }
    }

    /**
     * @dev Internal function to allocate revenue to revenue contract
     */
    function _allocateRevenue(uint256 amount) internal {
        (bool success, ) = revenueContract.call{value: amount}(
            abi.encodeWithSignature("allocate()")
        );
        require(success, "Revenue allocation failed");
        emit RevenueAllocated(amount, revenueContract);
    }

    // ==================== VIEW FUNCTIONS ====================

    /**
     * @dev Check if subscription is active
     */
    function isSubscriptionActive(address customer) external view returns (bool) {
        Subscription memory sub = subscriptions[customer];
        return sub.active && sub.expiresAt > block.timestamp;
    }

    /**
     * @dev Get subscription details
     */
    function getSubscription(address customer) external view returns (
        Tier tier,
        uint256 expiresAt,
        uint256 totalPaid,
        bool active
    ) {
        Subscription memory sub = subscriptions[customer];
        return (sub.tier, sub.expiresAt, sub.totalPaid, sub.active);
    }

    /**
     * @dev Get tier information
     */
    function getTierInfo(Tier tier) external view returns (
        uint256 price,
        uint256 requests,
        bool active
    ) {
        TierInfo memory info = tiers[tier];
        return (info.price, info.requests, info.active);
    }

    /**
     * @dev Get payment history count
     */
    function getPaymentCount(address customer) external view returns (uint256) {
        return paymentHistory[customer].length;
    }

    /**
     * @dev Get specific payment
     */
    function getPayment(address customer, uint256 index) external view returns (
        Tier tier,
        uint256 amount,
        uint256 timestamp
    ) {
        require(index < paymentHistory[customer].length, "Invalid index");
        Payment memory payment = paymentHistory[customer][index];
        return (payment.tier, payment.amount, payment.timestamp);
    }

    /**
     * @dev Get contract metrics
     */
    function getMetrics() external view returns (
        uint256 _totalPayments,
        uint256 _totalRevenue,
        uint256 _totalCustomers,
        uint256 _contractBalance
    ) {
        return (
            totalPayments,
            totalRevenue,
            totalCustomers,
            address(this).balance
        );
    }

    // ==================== ADMIN FUNCTIONS ====================

    /**
     * @dev Set revenue contract address
     */
    function setRevenueContract(address _revenueContract) external onlyOwner {
        require(_revenueContract != address(0), "Invalid address");
        revenueContract = _revenueContract;
    }

    /**
     * @dev Update tier pricing
     */
    function updateTier(
        Tier tier,
        uint256 price,
        uint256 requests,
        bool active
    ) external onlyOwner {
        require(tier != Tier.Free, "Cannot modify free tier");
        tiers[tier] = TierInfo(price, requests, active);
        emit TierUpdated(tier, price, requests);
    }

    /**
     * @dev Pause contract
     */
    function pause() external onlyOwner {
        paused = true;
        emit Paused(msg.sender);
    }

    /**
     * @dev Unpause contract
     */
    function unpause() external onlyOwner {
        paused = false;
        emit Unpaused(msg.sender);
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

    /**
     * @dev Emergency withdraw (only if revenue contract not set)
     */
    function emergencyWithdraw() external onlyOwner {
        require(revenueContract == address(0), "Revenue contract active");
        payable(owner).transfer(address(this).balance);
    }

    // ==================== RECEIVE FUNCTION ====================

    receive() external payable {
        revert("Use subscribe() or renew()");
    }
}
