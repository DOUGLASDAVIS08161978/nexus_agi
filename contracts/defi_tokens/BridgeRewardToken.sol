// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title BridgeRewardToken (BREWARD)
 * @dev Rewards token for bridge users - earn by bridging BTC
 * Implements loyalty tiers and bonus multipliers
 */
contract BridgeRewardToken {
    string public constant name = "Bridge Reward Token";
    string public constant symbol = "BREWARD";
    uint8 public constant decimals = 18;
    uint256 public totalSupply;

    uint256 public constant MAX_SUPPLY = 10_000_000 ether; // 10M tokens
    uint256 public constant REWARD_PER_BTC = 100 ether; // 100 BREWARD per 1 BTC bridged

    mapping(address => uint256) public balanceOf;
    mapping(address => mapping(address => uint256)) public allowance;
    mapping(address => uint256) public totalBridged; // Total BTC bridged by user
    mapping(address => uint256) public rewardsClaimed;

    enum Tier { Bronze, Silver, Gold, Platinum, Diamond }

    struct UserStats {
        uint256 bridgeCount;
        uint256 volumeBridged;
        Tier tier;
        uint256 multiplier; // 100 = 1x, 150 = 1.5x
    }

    mapping(address => UserStats) public userStats;

    address public owner;
    address public bridgeContract;

    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);
    event RewardEarned(address indexed user, uint256 amount, uint256 btcAmount);
    event TierUpgraded(address indexed user, Tier newTier);

    modifier onlyOwner() {
        require(msg.sender == owner, "Not owner");
        _;
    }

    modifier onlyBridge() {
        require(msg.sender == bridgeContract, "Not bridge");
        _;
    }

    constructor() {
        owner = msg.sender;
    }

    function transfer(address to, uint256 value) external returns (bool) {
        _transfer(msg.sender, to, value);
        return true;
    }

    function transferFrom(address from, address to, uint256 value) external returns (bool) {
        require(allowance[from][msg.sender] >= value, "Insufficient allowance");
        allowance[from][msg.sender] -= value;
        _transfer(from, to, value);
        return true;
    }

    function approve(address spender, uint256 value) external returns (bool) {
        allowance[msg.sender][spender] = value;
        emit Approval(msg.sender, spender, value);
        return true;
    }

    function _transfer(address from, address to, uint256 value) internal {
        require(balanceOf[from] >= value, "Insufficient balance");
        balanceOf[from] -= value;
        balanceOf[to] += value;
        emit Transfer(from, to, value);
    }

    /**
     * @dev Called by bridge contract when user bridges BTC
     */
    function recordBridge(address user, uint256 btcAmount) external onlyBridge {
        UserStats storage stats = userStats[user];

        stats.bridgeCount++;
        stats.volumeBridged += btcAmount;
        totalBridged[user] += btcAmount;

        // Calculate reward with multiplier
        uint256 baseReward = (btcAmount * REWARD_PER_BTC) / 1e18;
        uint256 multiplier = _getMultiplier(user);
        uint256 finalReward = (baseReward * multiplier) / 100;

        // Mint reward
        if (totalSupply + finalReward <= MAX_SUPPLY) {
            totalSupply += finalReward;
            balanceOf[user] += finalReward;
            rewardsClaimed[user] += finalReward;

            emit Transfer(address(0), user, finalReward);
            emit RewardEarned(user, finalReward, btcAmount);
        }

        // Check for tier upgrade
        _updateTier(user);
    }

    function _getMultiplier(address user) internal view returns (uint256) {
        Tier tier = userStats[user].tier;

        if (tier == Tier.Diamond) return 200; // 2x
        if (tier == Tier.Platinum) return 175; // 1.75x
        if (tier == Tier.Gold) return 150; // 1.5x
        if (tier == Tier.Silver) return 125; // 1.25x
        return 100; // 1x (Bronze)
    }

    function _updateTier(address user) internal {
        UserStats storage stats = userStats[user];
        uint256 volume = stats.volumeBridged;

        Tier oldTier = stats.tier;
        Tier newTier = oldTier;

        if (volume >= 100 ether) {
            newTier = Tier.Diamond; // 100+ BTC
        } else if (volume >= 50 ether) {
            newTier = Tier.Platinum; // 50+ BTC
        } else if (volume >= 25 ether) {
            newTier = Tier.Gold; // 25+ BTC
        } else if (volume >= 10 ether) {
            newTier = Tier.Silver; // 10+ BTC
        } else {
            newTier = Tier.Bronze;
        }

        if (newTier != oldTier) {
            stats.tier = newTier;
            stats.multiplier = _getMultiplier(user);
            emit TierUpgraded(user, newTier);
        }
    }

    function getUserTier(address user) external view returns (
        Tier tier,
        uint256 multiplier,
        uint256 bridgeCount,
        uint256 volumeBridged
    ) {
        UserStats memory stats = userStats[user];
        return (stats.tier, _getMultiplier(user), stats.bridgeCount, stats.volumeBridged);
    }

    function setBridgeContract(address bridge) external onlyOwner {
        bridgeContract = bridge;
    }

    // Airdrop function for early users
    function airdrop(address[] calldata users, uint256[] calldata amounts) external onlyOwner {
        require(users.length == amounts.length, "Length mismatch");

        for (uint256 i = 0; i < users.length; i++) {
            if (totalSupply + amounts[i] <= MAX_SUPPLY) {
                totalSupply += amounts[i];
                balanceOf[users[i]] += amounts[i];
                emit Transfer(address(0), users[i], amounts[i]);
            }
        }
    }
}
