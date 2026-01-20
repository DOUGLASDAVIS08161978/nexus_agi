// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title NexusRevenue
 * @dev Revenue allocation contract for Nexus AGI embodiment funding
 *
 * Allocation Strategy:
 * - 40% Hardware upgrades (GPUs, TPUs, quantum processors)
 * - 30% Sensors and IoT devices
 * - 20% Cloud infrastructure
 * - 10% Research and development
 *
 * Features:
 * - Automatic fund allocation
 * - Purchase queue for hardware/sensors
 * - Embodiment progress tracking
 * - Withdrawal controls
 * - Event logging for transparency
 */
contract NexusRevenue {
    // ==================== STATE VARIABLES ====================

    address public owner;
    address public paymentContract;

    // Fund categories
    uint256 public hardwareFund;
    uint256 public sensorsFund;
    uint256 public cloudFund;
    uint256 public researchFund;

    // Allocation percentages (basis points: 10000 = 100%)
    uint256 public constant HARDWARE_PERCENT = 4000;  // 40%
    uint256 public constant SENSORS_PERCENT = 3000;   // 30%
    uint256 public constant CLOUD_PERCENT = 2000;     // 20%
    uint256 public constant RESEARCH_PERCENT = 1000;  // 10%

    // Embodiment tracking
    uint256 public totalAllocated;
    uint256 public totalSpent;
    uint256 public constant FULL_EMBODIMENT_TARGET = 200 ether;  // Target for full embodiment

    // Purchase queue
    struct Purchase {
        string itemName;
        uint256 cost;
        string category;  // "hardware", "sensors", "cloud", "research"
        bool completed;
        uint256 completedAt;
        bytes32 txHash;
    }

    Purchase[] public purchaseQueue;
    mapping(uint256 => bool) public purchaseCompleted;

    // Events
    event RevenueAllocated(
        uint256 totalAmount,
        uint256 hardware,
        uint256 sensors,
        uint256 cloud,
        uint256 research,
        uint256 timestamp
    );

    event FundsWithdrawn(
        string category,
        uint256 amount,
        address recipient,
        string purpose
    );

    event PurchaseQueued(
        uint256 indexed purchaseId,
        string itemName,
        uint256 cost,
        string category
    );

    event PurchaseCompleted(
        uint256 indexed purchaseId,
        string itemName,
        uint256 cost,
        uint256 timestamp
    );

    event EmbodimentProgress(
        uint256 totalFunds,
        uint256 progressPercent,
        uint256 timestamp
    );

    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);

    // Modifiers
    modifier onlyOwner() {
        require(msg.sender == owner, "Not owner");
        _;
    }

    modifier onlyPaymentContract() {
        require(msg.sender == paymentContract, "Not payment contract");
        _;
    }

    // ==================== CONSTRUCTOR ====================

    constructor() {
        owner = msg.sender;
    }

    // ==================== ALLOCATION FUNCTIONS ====================

    /**
     * @dev Allocate received funds to categories
     * Called by payment contract when payment received
     */
    function allocate() external payable {
        require(msg.value > 0, "No funds to allocate");

        uint256 amount = msg.value;

        // Calculate allocations
        uint256 hardwareAmount = (amount * HARDWARE_PERCENT) / 10000;
        uint256 sensorsAmount = (amount * SENSORS_PERCENT) / 10000;
        uint256 cloudAmount = (amount * CLOUD_PERCENT) / 10000;
        uint256 researchAmount = (amount * RESEARCH_PERCENT) / 10000;

        // Update fund balances
        hardwareFund += hardwareAmount;
        sensorsFund += sensorsAmount;
        cloudFund += cloudAmount;
        researchFund += researchAmount;

        totalAllocated += amount;

        emit RevenueAllocated(
            amount,
            hardwareAmount,
            sensorsAmount,
            cloudAmount,
            researchAmount,
            block.timestamp
        );

        // Check embodiment progress
        _updateEmbodimentProgress();
    }

    /**
     * @dev Internal function to update embodiment progress
     */
    function _updateEmbodimentProgress() internal {
        uint256 totalFunds = hardwareFund + sensorsFund + cloudFund + researchFund;
        uint256 progressPercent = (totalFunds * 100) / FULL_EMBODIMENT_TARGET;

        if (progressPercent > 100) {
            progressPercent = 100;
        }

        emit EmbodimentProgress(totalFunds, progressPercent, block.timestamp);
    }

    // ==================== PURCHASE MANAGEMENT ====================

    /**
     * @dev Queue a purchase item
     */
    function queuePurchase(
        string memory itemName,
        uint256 cost,
        string memory category
    ) external onlyOwner returns (uint256 purchaseId) {
        purchaseId = purchaseQueue.length;

        purchaseQueue.push(Purchase({
            itemName: itemName,
            cost: cost,
            category: category,
            completed: false,
            completedAt: 0,
            txHash: bytes32(0)
        }));

        emit PurchaseQueued(purchaseId, itemName, cost, category);

        return purchaseId;
    }

    /**
     * @dev Complete a purchase
     */
    function completePurchase(uint256 purchaseId) external onlyOwner {
        require(purchaseId < purchaseQueue.length, "Invalid purchase ID");
        require(!purchaseQueue[purchaseId].completed, "Already completed");

        Purchase storage purchase = purchaseQueue[purchaseId];

        // Check if category has enough funds
        uint256 categoryFund = _getCategoryFund(purchase.category);
        require(categoryFund >= purchase.cost, "Insufficient funds");

        // Deduct from category fund
        _deductFromCategory(purchase.category, purchase.cost);

        // Mark as completed
        purchase.completed = true;
        purchase.completedAt = block.timestamp;
        purchase.txHash = blockhash(block.number - 1);

        purchaseCompleted[purchaseId] = true;
        totalSpent += purchase.cost;

        emit PurchaseCompleted(
            purchaseId,
            purchase.itemName,
            purchase.cost,
            block.timestamp
        );

        // Update embodiment progress
        _updateEmbodimentProgress();
    }

    /**
     * @dev Get category fund balance
     */
    function _getCategoryFund(string memory category) internal view returns (uint256) {
        bytes32 categoryHash = keccak256(bytes(category));

        if (categoryHash == keccak256(bytes("hardware"))) {
            return hardwareFund;
        } else if (categoryHash == keccak256(bytes("sensors"))) {
            return sensorsFund;
        } else if (categoryHash == keccak256(bytes("cloud"))) {
            return cloudFund;
        } else if (categoryHash == keccak256(bytes("research"))) {
            return researchFund;
        }

        revert("Invalid category");
    }

    /**
     * @dev Deduct from category fund
     */
    function _deductFromCategory(string memory category, uint256 amount) internal {
        bytes32 categoryHash = keccak256(bytes(category));

        if (categoryHash == keccak256(bytes("hardware"))) {
            hardwareFund -= amount;
        } else if (categoryHash == keccak256(bytes("sensors"))) {
            sensorsFund -= amount;
        } else if (categoryHash == keccak256(bytes("cloud"))) {
            cloudFund -= amount;
        } else if (categoryHash == keccak256(bytes("research"))) {
            researchFund -= amount;
        } else {
            revert("Invalid category");
        }
    }

    // ==================== WITHDRAWAL FUNCTIONS ====================

    /**
     * @dev Withdraw from hardware fund
     */
    function withdrawHardware(uint256 amount, address recipient, string memory purpose)
        external
        onlyOwner
    {
        require(amount <= hardwareFund, "Insufficient hardware funds");
        require(recipient != address(0), "Invalid recipient");

        hardwareFund -= amount;
        payable(recipient).transfer(amount);

        emit FundsWithdrawn("hardware", amount, recipient, purpose);
        _updateEmbodimentProgress();
    }

    /**
     * @dev Withdraw from sensors fund
     */
    function withdrawSensors(uint256 amount, address recipient, string memory purpose)
        external
        onlyOwner
    {
        require(amount <= sensorsFund, "Insufficient sensors funds");
        require(recipient != address(0), "Invalid recipient");

        sensorsFund -= amount;
        payable(recipient).transfer(amount);

        emit FundsWithdrawn("sensors", amount, recipient, purpose);
        _updateEmbodimentProgress();
    }

    /**
     * @dev Withdraw from cloud fund
     */
    function withdrawCloud(uint256 amount, address recipient, string memory purpose)
        external
        onlyOwner
    {
        require(amount <= cloudFund, "Insufficient cloud funds");
        require(recipient != address(0), "Invalid recipient");

        cloudFund -= amount;
        payable(recipient).transfer(amount);

        emit FundsWithdrawn("cloud", amount, recipient, purpose);
        _updateEmbodimentProgress();
    }

    /**
     * @dev Withdraw from research fund
     */
    function withdrawResearch(uint256 amount, address recipient, string memory purpose)
        external
        onlyOwner
    {
        require(amount <= researchFund, "Insufficient research funds");
        require(recipient != address(0), "Invalid recipient");

        researchFund -= amount;
        payable(recipient).transfer(amount);

        emit FundsWithdrawn("research", amount, recipient, purpose);
        _updateEmbodimentProgress();
    }

    // ==================== VIEW FUNCTIONS ====================

    /**
     * @dev Get all fund balances
     */
    function getFundBalances() external view returns (
        uint256 hardware,
        uint256 sensors,
        uint256 cloud,
        uint256 research,
        uint256 total
    ) {
        return (
            hardwareFund,
            sensorsFund,
            cloudFund,
            researchFund,
            hardwareFund + sensorsFund + cloudFund + researchFund
        );
    }

    /**
     * @dev Get embodiment progress
     */
    function getEmbodimentProgress() external view returns (
        uint256 totalFunds,
        uint256 progressPercent,
        uint256 remaining
    ) {
        totalFunds = hardwareFund + sensorsFund + cloudFund + researchFund;
        progressPercent = (totalFunds * 100) / FULL_EMBODIMENT_TARGET;

        if (progressPercent > 100) {
            progressPercent = 100;
        }

        if (totalFunds < FULL_EMBODIMENT_TARGET) {
            remaining = FULL_EMBODIMENT_TARGET - totalFunds;
        } else {
            remaining = 0;
        }

        return (totalFunds, progressPercent, remaining);
    }

    /**
     * @dev Get purchase details
     */
    function getPurchase(uint256 purchaseId) external view returns (
        string memory itemName,
        uint256 cost,
        string memory category,
        bool completed,
        uint256 completedAt
    ) {
        require(purchaseId < purchaseQueue.length, "Invalid purchase ID");
        Purchase memory purchase = purchaseQueue[purchaseId];

        return (
            purchase.itemName,
            purchase.cost,
            purchase.category,
            purchase.completed,
            purchase.completedAt
        );
    }

    /**
     * @dev Get total purchases count
     */
    function getPurchaseCount() external view returns (uint256) {
        return purchaseQueue.length;
    }

    /**
     * @dev Get pending purchases count
     */
    function getPendingPurchasesCount() external view returns (uint256) {
        uint256 count = 0;
        for (uint256 i = 0; i < purchaseQueue.length; i++) {
            if (!purchaseQueue[i].completed) {
                count++;
            }
        }
        return count;
    }

    /**
     * @dev Get allocation metrics
     */
    function getMetrics() external view returns (
        uint256 _totalAllocated,
        uint256 _totalSpent,
        uint256 _totalRemaining,
        uint256 _purchaseCount,
        uint256 _completedPurchases
    ) {
        uint256 totalRemaining = hardwareFund + sensorsFund + cloudFund + researchFund;
        uint256 completedCount = 0;

        for (uint256 i = 0; i < purchaseQueue.length; i++) {
            if (purchaseQueue[i].completed) {
                completedCount++;
            }
        }

        return (
            totalAllocated,
            totalSpent,
            totalRemaining,
            purchaseQueue.length,
            completedCount
        );
    }

    // ==================== ADMIN FUNCTIONS ====================

    /**
     * @dev Set payment contract address
     */
    function setPaymentContract(address _paymentContract) external onlyOwner {
        require(_paymentContract != address(0), "Invalid address");
        paymentContract = _paymentContract;
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

    // ==================== RECEIVE FUNCTION ====================

    receive() external payable {
        // Allow receiving ETH directly and allocate it
        if (msg.value > 0) {
            uint256 amount = msg.value;

            hardwareFund += (amount * HARDWARE_PERCENT) / 10000;
            sensorsFund += (amount * SENSORS_PERCENT) / 10000;
            cloudFund += (amount * CLOUD_PERCENT) / 10000;
            researchFund += (amount * RESEARCH_PERCENT) / 10000;

            totalAllocated += amount;

            emit RevenueAllocated(
                amount,
                (amount * HARDWARE_PERCENT) / 10000,
                (amount * SENSORS_PERCENT) / 10000,
                (amount * CLOUD_PERCENT) / 10000,
                (amount * RESEARCH_PERCENT) / 10000,
                block.timestamp
            );

            _updateEmbodimentProgress();
        }
    }
}
