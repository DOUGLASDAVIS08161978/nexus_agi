// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/access/AccessControl.sol";

/**
 * @title OracleDataToken
 * @dev Access token for oracle data feeds
 *
 * Features:
 * - Pay-per-use oracle data
 * - Data provider rewards
 * - Quality staking mechanism
 *
 * Symbol: DATA
 * Decimals: 18
 */
contract OracleDataToken is ERC20, AccessControl {
    bytes32 public constant ORACLE_ROLE = keccak256("ORACLE_ROLE");
    bytes32 public constant DATA_PROVIDER_ROLE = keccak256("DATA_PROVIDER_ROLE");

    struct DataFeed {
        string name;
        address provider;
        uint256 pricePerQuery;
        uint256 totalQueries;
        uint256 reputation;
    }

    struct DataProvider {
        uint256 stakedAmount;
        uint256 totalEarned;
        uint256 successfulQueries;
        uint256 failedQueries;
    }

    mapping(bytes32 => DataFeed) public dataFeeds;
    mapping(address => DataProvider) public dataProviders;
    mapping(address => uint256) public queryCredits;

    uint256 public constant MIN_STAKE = 1000 * 10**18;
    uint256 public totalStaked;

    event DataFeedRegistered(bytes32 indexed feedId, string name, address provider);
    event DataQueried(bytes32 indexed feedId, address indexed user, uint256 cost);
    event ProviderStaked(address indexed provider, uint256 amount);
    event ProviderRewarded(address indexed provider, uint256 amount);
    event ReputationUpdated(bytes32 indexed feedId, uint256 newReputation);

    constructor(address admin) ERC20("Oracle Data Token", "DATA") {
        _grantRole(DEFAULT_ADMIN_ROLE, admin);
        _mint(admin, 20_000_000 * 10**18);
    }

    /**
     * @dev Register as data provider
     */
    function registerDataProvider(uint256 stakeAmount) external {
        require(stakeAmount >= MIN_STAKE, "Insufficient stake");
        require(balanceOf(msg.sender) >= stakeAmount, "Insufficient balance");

        _transfer(msg.sender, address(this), stakeAmount);

        dataProviders[msg.sender].stakedAmount += stakeAmount;
        totalStaked += stakeAmount;

        _grantRole(DATA_PROVIDER_ROLE, msg.sender);

        emit ProviderStaked(msg.sender, stakeAmount);
    }

    /**
     * @dev Register data feed
     */
    function registerDataFeed(
        string memory name,
        uint256 pricePerQuery
    ) external onlyRole(DATA_PROVIDER_ROLE) returns (bytes32) {
        bytes32 feedId = keccak256(abi.encodePacked(name, msg.sender, block.timestamp));

        dataFeeds[feedId] = DataFeed({
            name: name,
            provider: msg.sender,
            pricePerQuery: pricePerQuery,
            totalQueries: 0,
            reputation: 100
        });

        emit DataFeedRegistered(feedId, name, msg.sender);

        return feedId;
    }

    /**
     * @dev Query data feed
     */
    function queryDataFeed(bytes32 feedId) external returns (bool) {
        DataFeed storage feed = dataFeeds[feedId];
        require(feed.provider != address(0), "Feed not found");

        uint256 cost = feed.pricePerQuery;
        require(balanceOf(msg.sender) >= cost, "Insufficient balance");

        _transfer(msg.sender, feed.provider, cost);

        feed.totalQueries++;
        dataProviders[feed.provider].totalEarned += cost;
        dataProviders[feed.provider].successfulQueries++;

        emit DataQueried(feedId, msg.sender, cost);

        return true;
    }

    /**
     * @dev Update feed reputation
     */
    function updateReputation(bytes32 feedId, bool success) external onlyRole(ORACLE_ROLE) {
        DataFeed storage feed = dataFeeds[feedId];
        require(feed.provider != address(0), "Feed not found");

        if (success) {
            feed.reputation = feed.reputation < 100 ? feed.reputation + 1 : 100;
        } else {
            feed.reputation = feed.reputation > 0 ? feed.reputation - 5 : 0;
            dataProviders[feed.provider].failedQueries++;
        }

        emit ReputationUpdated(feedId, feed.reputation);
    }

    /**
     * @dev Purchase query credits
     */
    function purchaseCredits(uint256 amount) external {
        require(balanceOf(msg.sender) >= amount, "Insufficient balance");

        _burn(msg.sender, amount);
        queryCredits[msg.sender] += amount;
    }

    /**
     * @dev Withdraw staked tokens
     */
    function withdrawStake(uint256 amount) external {
        require(dataProviders[msg.sender].stakedAmount >= amount, "Insufficient staked");

        dataProviders[msg.sender].stakedAmount -= amount;
        totalStaked -= amount;

        _transfer(address(this), msg.sender, amount);
    }

    /**
     * @dev Get provider statistics
     */
    function getProviderStats(address provider) external view returns (
        uint256 staked,
        uint256 earned,
        uint256 successful,
        uint256 failed,
        uint256 successRate
    ) {
        DataProvider memory p = dataProviders[provider];
        staked = p.stakedAmount;
        earned = p.totalEarned;
        successful = p.successfulQueries;
        failed = p.failedQueries;
        successRate = (successful + failed) > 0 ? (successful * 100) / (successful + failed) : 0;
    }
}
