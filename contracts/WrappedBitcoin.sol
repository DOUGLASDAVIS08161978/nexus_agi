// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

/**
 * @title WrappedBitcoin (WBTC) Token for Monad Testnet
 * @notice ERC20 token representing Bitcoin bridged to Monad
 * @dev Minted when Bitcoin is locked, burned when redeemed
 */
contract WrappedBitcoin is ERC20, Ownable {

    // Events
    event BridgeDeposit(
        address indexed to,
        uint256 amount,
        string btcTxId,
        uint256 timestamp
    );

    event BridgeWithdrawal(
        address indexed from,
        uint256 amount,
        string btcAddress,
        uint256 timestamp
    );

    // Bridge tracking
    mapping(string => bool) public processedBtcTxIds;
    mapping(address => uint256) public totalBridged;

    constructor() ERC20("Wrapped Bitcoin", "WBTC") Ownable(msg.sender) {
        // 8 decimals to match Bitcoin
        _setupDecimals(8);
    }

    function _setupDecimals(uint8 decimals_) private {
        // Note: This is handled in the decimals() override
    }

    function decimals() public pure override returns (uint8) {
        return 8;
    }

    /**
     * @notice Bridge Bitcoin to WBTC
     * @param to Address to receive WBTC
     * @param amount Amount of WBTC to mint (in satoshis, 8 decimals)
     * @param btcTxId Bitcoin transaction ID as proof
     */
    function bridgeFromBitcoin(
        address to,
        uint256 amount,
        string memory btcTxId
    ) external onlyOwner {
        require(to != address(0), "Invalid address");
        require(amount > 0, "Amount must be positive");
        require(!processedBtcTxIds[btcTxId], "BTC transaction already processed");

        // Mark transaction as processed
        processedBtcTxIds[btcTxId] = true;

        // Mint WBTC to user
        _mint(to, amount);

        // Track bridged amount
        totalBridged[to] += amount;

        emit BridgeDeposit(to, amount, btcTxId, block.timestamp);
    }

    /**
     * @notice Burn WBTC to redeem Bitcoin
     * @param amount Amount of WBTC to burn
     * @param btcAddress Bitcoin address to receive BTC
     */
    function bridgeToBitcoin(
        uint256 amount,
        string memory btcAddress
    ) external {
        require(amount > 0, "Amount must be positive");
        require(balanceOf(msg.sender) >= amount, "Insufficient balance");
        require(bytes(btcAddress).length > 0, "Invalid BTC address");

        // Burn WBTC
        _burn(msg.sender, amount);

        emit BridgeWithdrawal(msg.sender, amount, btcAddress, block.timestamp);
    }

    /**
     * @notice Check if Bitcoin transaction has been processed
     */
    function isBtcTxProcessed(string memory btcTxId) external view returns (bool) {
        return processedBtcTxIds[btcTxId];
    }

    /**
     * @notice Get total amount bridged by an address
     */
    function getTotalBridged(address account) external view returns (uint256) {
        return totalBridged[account];
    }
}
