// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/access/AccessControl.sol";

/**
 * @title BridgeRewardToken
 * @dev Rewards for operating and using cross-chain bridges
 *
 * Features:
 * - Rewards bridge operators
 * - Incentivizes bridge usage
 * - Vesting for long-term alignment
 *
 * Symbol: BRG
 * Decimals: 18
 */
contract BridgeRewardToken is ERC20, AccessControl {
    bytes32 public constant BRIDGE_OPERATOR_ROLE = keccak256("BRIDGE_OPERATOR_ROLE");
    bytes32 public constant ADMIN_ROLE = keccak256("ADMIN_ROLE");

    struct VestingSchedule {
        uint256 amount;
        uint256 startTime;
        uint256 duration;
        uint256 claimed;
    }

    mapping(address => VestingSchedule) public vestingSchedules;
    mapping(address => uint256) public bridgeVolume;
    mapping(address => uint256) public operatorRewards;

    uint256 public constant REWARD_RATE = 100; // 1% per transaction
    uint256 public totalBridgeVolume;

    event BridgeOperatorAdded(address indexed operator);
    event RewardGranted(address indexed recipient, uint256 amount);
    event VestingCreated(address indexed beneficiary, uint256 amount, uint256 duration);
    event VestingClaimed(address indexed beneficiary, uint256 amount);

    constructor(address admin) ERC20("Bridge Reward Token", "BRG") {
        _grantRole(DEFAULT_ADMIN_ROLE, admin);
        _grantRole(ADMIN_ROLE, admin);
    }

    /**
     * @dev Add bridge operator
     */
    function addBridgeOperator(address operator) external onlyRole(ADMIN_ROLE) {
        _grantRole(BRIDGE_OPERATOR_ROLE, operator);
        emit BridgeOperatorAdded(operator);
    }

    /**
     * @dev Grant bridge reward
     */
    function grantBridgeReward(address user, uint256 volume) external onlyRole(BRIDGE_OPERATOR_ROLE) {
        uint256 reward = (volume * REWARD_RATE) / 10000;

        _mint(user, reward);

        bridgeVolume[user] += volume;
        operatorRewards[msg.sender] += reward;
        totalBridgeVolume += volume;

        emit RewardGranted(user, reward);
    }

    /**
     * @dev Create vesting schedule for rewards
     */
    function createVesting(address beneficiary, uint256 amount, uint256 duration) external onlyRole(ADMIN_ROLE) {
        require(vestingSchedules[beneficiary].amount == 0, "Vesting already exists");

        _mint(address(this), amount);

        vestingSchedules[beneficiary] = VestingSchedule({
            amount: amount,
            startTime: block.timestamp,
            duration: duration,
            claimed: 0
        });

        emit VestingCreated(beneficiary, amount, duration);
    }

    /**
     * @dev Claim vested tokens
     */
    function claimVested() external {
        VestingSchedule storage schedule = vestingSchedules[msg.sender];
        require(schedule.amount > 0, "No vesting schedule");

        uint256 vested = calculateVested(msg.sender);
        uint256 claimable = vested - schedule.claimed;

        require(claimable > 0, "Nothing to claim");

        schedule.claimed += claimable;
        _transfer(address(this), msg.sender, claimable);

        emit VestingClaimed(msg.sender, claimable);
    }

    /**
     * @dev Calculate vested amount
     */
    function calculateVested(address beneficiary) public view returns (uint256) {
        VestingSchedule memory schedule = vestingSchedules[beneficiary];

        if (schedule.amount == 0) return 0;

        uint256 elapsed = block.timestamp - schedule.startTime;

        if (elapsed >= schedule.duration) {
            return schedule.amount;
        }

        return (schedule.amount * elapsed) / schedule.duration;
    }
}
