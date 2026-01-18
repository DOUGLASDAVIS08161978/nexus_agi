// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/token/ERC20/extensions/ERC20Burnable.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

/**
 * @title NetworkFeeToken
 * @dev Used for paying network transaction fees
 *
 * Features:
 * - Gas payment mechanism
 * - Fee burning for deflationary pressure
 * - Multi-network support
 *
 * Symbol: FEE
 * Decimals: 18
 */
contract NetworkFeeToken is ERC20, ERC20Burnable, Ownable {
    struct NetworkStats {
        uint256 totalTransactions;
        uint256 totalFeesBurned;
        uint256 averageFee;
    }

    mapping(string => NetworkStats) public networkStats;
    mapping(address => uint256) public feeBalance;
    mapping(address => bool) public isFeeCollector;

    uint256 public baseFee = 0.001 ether;
    uint256 public totalFeesBurned;
    uint256 public totalTransactions;

    event FeeCharged(address indexed user, string network, uint256 amount);
    event FeeBurned(uint256 amount);
    event FeeCollectorAdded(address indexed collector);
    event BaseFeeUpdated(uint256 newBaseFee);

    modifier onlyFeeCollector() {
        require(isFeeCollector[msg.sender] || msg.sender == owner(), "Not fee collector");
        _;
    }

    constructor(address initialOwner)
        ERC20("Network Fee Token", "FEE")
        Ownable(initialOwner)
    {
        _mint(initialOwner, 50_000_000 * 10**18);
        isFeeCollector[initialOwner] = true;
    }

    /**
     * @dev Add fee collector
     */
    function addFeeCollector(address collector) external onlyOwner {
        isFeeCollector[collector] = true;
        emit FeeCollectorAdded(collector);
    }

    /**
     * @dev Charge network fee
     */
    function chargeFee(address user, string memory network, uint256 multiplier) external onlyFeeCollector returns (uint256) {
        uint256 fee = baseFee * multiplier / 100;

        require(balanceOf(user) >= fee, "Insufficient fee balance");

        _burn(user, fee);

        networkStats[network].totalTransactions++;
        networkStats[network].totalFeesBurned += fee;
        networkStats[network].averageFee = networkStats[network].totalFeesBurned / networkStats[network].totalTransactions;

        totalFeesBurned += fee;
        totalTransactions++;

        emit FeeCharged(user, network, fee);
        emit FeeBurned(fee);

        return fee;
    }

    /**
     * @dev Batch charge fees
     */
    function batchChargeFees(
        address[] memory users,
        string memory network,
        uint256 multiplier
    ) external onlyFeeCollector returns (uint256) {
        uint256 totalCharged = 0;

        for (uint256 i = 0; i < users.length; i++) {
            uint256 fee = baseFee * multiplier / 100;

            if (balanceOf(users[i]) >= fee) {
                _burn(users[i], fee);
                totalCharged += fee;

                emit FeeCharged(users[i], network, fee);
            }
        }

        if (totalCharged > 0) {
            networkStats[network].totalTransactions += users.length;
            networkStats[network].totalFeesBurned += totalCharged;
            networkStats[network].averageFee = networkStats[network].totalFeesBurned / networkStats[network].totalTransactions;

            totalFeesBurned += totalCharged;
            totalTransactions += users.length;

            emit FeeBurned(totalCharged);
        }

        return totalCharged;
    }

    /**
     * @dev Update base fee
     */
    function setBaseFee(uint256 newBaseFee) external onlyOwner {
        baseFee = newBaseFee;
        emit BaseFeeUpdated(newBaseFee);
    }

    /**
     * @dev Get network statistics
     */
    function getNetworkStats(string memory network) external view returns (
        uint256 transactions,
        uint256 feesBurned,
        uint256 avgFee
    ) {
        NetworkStats memory stats = networkStats[network];
        transactions = stats.totalTransactions;
        feesBurned = stats.totalFeesBurned;
        avgFee = stats.averageFee;
    }

    /**
     * @dev Calculate fee for transaction
     */
    function calculateFee(uint256 multiplier) external view returns (uint256) {
        return baseFee * multiplier / 100;
    }
}
