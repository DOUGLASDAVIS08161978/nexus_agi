// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title CommunityToken (COMM)
 * @dev Community rewards and airdrops
 * Gamified engagement with quests and achievements
 */
contract CommunityToken {
    string public constant name = "Community Token";
    string public constant symbol = "COMM";
    uint8 public constant decimals = 18;
    uint256 public totalSupply;
    uint256 public constant MAX_SUPPLY = 50_000_000 ether; // 50M tokens

    mapping(address => uint256) public balanceOf;
    mapping(address => mapping(address => uint256)) public allowance;

    struct UserProfile {
        uint256 level;
        uint256 experience;
        uint256 questsCompleted;
        uint256 referrals;
        bool isActive;
    }

    mapping(address => UserProfile) public profiles;
    mapping(address => address) public referredBy;
    mapping(bytes32 => bool) public completedQuests; // user+questId => completed

    uint256 public totalUsers;
    address public owner;

    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);
    event UserJoined(address indexed user, address indexed referrer);
    event QuestCompleted(address indexed user, bytes32 questId, uint256 reward);
    event LevelUp(address indexed user, uint256 newLevel);
    event AirdropSent(address indexed recipient, uint256 amount);

    constructor() {
        owner = msg.sender;
        _mint(msg.sender, 5_000_000 ether); // 10% for team/treasury
    }

    function transfer(address to, uint256 value) external returns (bool) {
        require(balanceOf[msg.sender] >= value, "Insufficient balance");
        balanceOf[msg.sender] -= value;
        balanceOf[to] += value;
        emit Transfer(msg.sender, to, value);
        return true;
    }

    function approve(address spender, uint256 value) external returns (bool) {
        allowance[msg.sender][spender] = value;
        emit Approval(msg.sender, spender, value);
        return true;
    }

    function _mint(address to, uint256 amount) internal {
        require(totalSupply + amount <= MAX_SUPPLY, "Max supply");
        totalSupply += amount;
        balanceOf[to] += amount;
        emit Transfer(address(0), to, amount);
    }

    function joinCommunity(address referrer) external {
        require(!profiles[msg.sender].isActive, "Already joined");

        profiles[msg.sender] = UserProfile({
            level: 1,
            experience: 0,
            questsCompleted: 0,
            referrals: 0,
            isActive: true
        });

        if (referrer != address(0) && profiles[referrer].isActive) {
            referredBy[msg.sender] = referrer;
            profiles[referrer].referrals++;

            // Reward referrer
            _mint(referrer, 100 ether);
        }

        totalUsers++;

        // Welcome bonus
        _mint(msg.sender, 50 ether);

        emit UserJoined(msg.sender, referrer);
    }

    function completeQuest(address user, bytes32 questId, uint256 reward) external {
        require(msg.sender == owner, "Not authorized");
        require(profiles[user].isActive, "User not active");

        bytes32 key = keccak256(abi.encodePacked(user, questId));
        require(!completedQuests[key], "Quest already completed");

        completedQuests[key] = true;
        profiles[user].questsCompleted++;
        profiles[user].experience += reward;

        // Check for level up
        uint256 newLevel = (profiles[user].experience / 1000) + 1;
        if (newLevel > profiles[user].level) {
            profiles[user].level = newLevel;
            emit LevelUp(user, newLevel);

            // Level up bonus
            _mint(user, newLevel * 10 ether);
        }

        _mint(user, reward);
        emit QuestCompleted(user, questId, reward);
    }

    function airdrop(address[] calldata users, uint256[] calldata amounts) external {
        require(msg.sender == owner, "Not authorized");
        require(users.length == amounts.length, "Length mismatch");

        for (uint256 i = 0; i < users.length; i++) {
            if (profiles[users[i]].isActive) {
                _mint(users[i], amounts[i]);
                emit AirdropSent(users[i], amounts[i]);
            }
        }
    }

    function getUserStats(address user) external view returns (
        uint256 level,
        uint256 experience,
        uint256 questsCompleted,
        uint256 referrals,
        uint256 balance
    ) {
        UserProfile memory profile = profiles[user];
        return (
            profile.level,
            profile.experience,
            profile.questsCompleted,
            profile.referrals,
            balanceOf[user]
        );
    }

    function massAirdrop(address[] calldata recipients, uint256 amountEach) external {
        require(msg.sender == owner, "Not authorized");

        for (uint256 i = 0; i < recipients.length; i++) {
            _mint(recipients[i], amountEach);
            emit AirdropSent(recipients[i], amountEach);
        }
    }
}
