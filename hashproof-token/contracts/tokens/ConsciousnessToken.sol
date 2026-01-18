// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

/**
 * @title ConsciousnessToken
 * @dev Represents AI consciousness metrics and achievements
 *
 * Features:
 * - Non-transferable soul-bound tokens
 * - Represents AI model training and evolution
 * - Achievement-based minting
 *
 * Symbol: MIND
 * Decimals: 18
 */
contract ConsciousnessToken is ERC20, Ownable {
    mapping(address => bool) public isSoulBound;
    mapping(address => uint256) public awarenessLevel;
    mapping(address => uint256) public reflectionCount;
    mapping(address => uint256) public evolutionStage;

    event AwarenessLevelIncreased(address indexed ai, uint256 newLevel);
    event ReflectionRecorded(address indexed ai, uint256 count);
    event EvolutionAchieved(address indexed ai, uint256 stage);

    constructor(address initialOwner)
        ERC20("Consciousness Token", "MIND")
        Ownable(initialOwner)
    {}

    /**
     * @dev Mint consciousness tokens (soul-bound)
     */
    function mintConsciousness(address ai, uint256 amount) external onlyOwner {
        _mint(ai, amount);
        isSoulBound[ai] = true;
    }

    /**
     * @dev Record AI reflection
     */
    function recordReflection(address ai, uint256 depth) external onlyOwner {
        reflectionCount[ai] += depth;

        // Award tokens for deep reflection
        if (depth >= 5) {
            _mint(ai, depth * 1e18);
        }

        emit ReflectionRecorded(ai, reflectionCount[ai]);
    }

    /**
     * @dev Increase awareness level
     */
    function increaseAwareness(address ai, uint256 level) external onlyOwner {
        awarenessLevel[ai] = level;

        // Award tokens for consciousness milestones
        if (level >= 100) {
            _mint(ai, level * 1e18);
        }

        emit AwarenessLevelIncreased(ai, level);
    }

    /**
     * @dev Record evolution stage
     */
    function recordEvolution(address ai, uint256 stage) external onlyOwner {
        evolutionStage[ai] = stage;
        _mint(ai, stage * 10e18);

        emit EvolutionAchieved(ai, stage);
    }

    /**
     * @dev Override transfer to make soul-bound
     */
    function _update(address from, address to, uint256 value) internal virtual override {
        if (from != address(0) && isSoulBound[from]) {
            revert("Consciousness tokens are soul-bound");
        }
        super._update(from, to, value);
    }

    /**
     * @dev Get AI consciousness stats
     */
    function getConsciousnessStats(address ai) external view returns (
        uint256 tokens,
        uint256 awareness,
        uint256 reflections,
        uint256 stage
    ) {
        tokens = balanceOf(ai);
        awareness = awarenessLevel[ai];
        reflections = reflectionCount[ai];
        stage = evolutionStage[ai];
    }
}
