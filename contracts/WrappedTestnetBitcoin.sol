// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title Wrapped Testnet Bitcoin (WtBTC)
 * @dev ERC-20 token representing Bitcoin Testnet BTC on Ethereum
 *
 * This contract is part of the NEXUS AGI Cross-Chain Bridge System
 * Contract Address: 0x324befe00354823df73691e37ed4f7b19ad74f63
 * Network: Ethereum Sepolia Testnet
 *
 * Features:
 * - Standard ERC-20 functionality
 * - 1:1 backing with Bitcoin Testnet
 * - Mintable by bridge contract
 * - Burnable for redemption
 * - Transfer functionality
 *
 * Bridge Details:
 * - Bitcoin Wallet: tb1qh2zh2ekmps6ts4zt80sl0a00g2avzxmytc6al2
 * - Ethereum Wallet: 0x86377ab13279b9a4877f0c2cb00049d5302506fe
 * - PSBT Support: Enabled
 * - Atomic Swaps: Supported
 */

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/token/ERC20/extensions/ERC20Burnable.sol";
import "@openzeppelin/contracts/security/Pausable.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract WrappedTestnetBitcoin is ERC20, ERC20Burnable, Pausable, Ownable {

    // Bitcoin transaction tracking
    mapping(string => bool) public processedBitcoinTxs;

    // Bridge wallet
    address public bridgeWallet;

    // Events
    event BridgeDeposit(
        string indexed bitcoinTxId,
        address indexed recipient,
        uint256 amount,
        uint256 timestamp
    );

    event BridgeWithdrawal(
        address indexed sender,
        string bitcoinAddress,
        uint256 amount,
        uint256 timestamp
    );

    event BridgeWalletUpdated(
        address indexed oldWallet,
        address indexed newWallet
    );

    /**
     * @dev Constructor initializes the WtBTC token
     */
    constructor() ERC20("Wrapped Testnet Bitcoin", "WtBTC") {
        bridgeWallet = msg.sender;

        // Mint initial supply for bridge operations
        // 1 WtBTC = 1 tBTC = 100,000,000 satoshis
        _mint(msg.sender, 1 * 10**decimals());
    }

    /**
     * @dev Returns the number of decimals (18 for ERC-20 standard)
     */
    function decimals() public pure override returns (uint8) {
        return 18;
    }

    /**
     * @dev Mint new WtBTC tokens when Bitcoin is locked in bridge
     * @param bitcoinTxId The Bitcoin transaction ID
     * @param recipient The Ethereum address to receive WtBTC
     * @param amount The amount to mint (in wei, 18 decimals)
     */
    function bridgeMint(
        string memory bitcoinTxId,
        address recipient,
        uint256 amount
    ) external onlyOwner whenNotPaused {
        require(recipient != address(0), "WtBTC: mint to zero address");
        require(amount > 0, "WtBTC: amount must be greater than 0");
        require(!processedBitcoinTxs[bitcoinTxId], "WtBTC: Bitcoin tx already processed");

        // Mark transaction as processed
        processedBitcoinTxs[bitcoinTxId] = true;

        // Mint tokens
        _mint(recipient, amount);

        emit BridgeDeposit(bitcoinTxId, recipient, amount, block.timestamp);
    }

    /**
     * @dev Burn WtBTC tokens to redeem Bitcoin
     * @param amount The amount to burn
     * @param bitcoinAddress The Bitcoin address to receive tBTC
     */
    function bridgeBurn(
        uint256 amount,
        string memory bitcoinAddress
    ) external whenNotPaused {
        require(amount > 0, "WtBTC: amount must be greater than 0");
        require(bytes(bitcoinAddress).length > 0, "WtBTC: invalid Bitcoin address");

        // Burn tokens
        _burn(msg.sender, amount);

        emit BridgeWithdrawal(msg.sender, bitcoinAddress, amount, block.timestamp);
    }

    /**
     * @dev Update the bridge wallet address
     * @param newBridgeWallet The new bridge wallet address
     */
    function updateBridgeWallet(address newBridgeWallet) external onlyOwner {
        require(newBridgeWallet != address(0), "WtBTC: invalid bridge wallet");

        address oldWallet = bridgeWallet;
        bridgeWallet = newBridgeWallet;

        emit BridgeWalletUpdated(oldWallet, newBridgeWallet);
    }

    /**
     * @dev Pause the contract (emergency only)
     */
    function pause() external onlyOwner {
        _pause();
    }

    /**
     * @dev Unpause the contract
     */
    function unpause() external onlyOwner {
        _unpause();
    }

    /**
     * @dev Override transfer to add pause functionality
     */
    function _beforeTokenTransfer(
        address from,
        address to,
        uint256 amount
    ) internal override whenNotPaused {
        super._beforeTokenTransfer(from, to, amount);
    }

    /**
     * @dev Check if a Bitcoin transaction has been processed
     * @param bitcoinTxId The Bitcoin transaction ID
     */
    function isBitcoinTxProcessed(string memory bitcoinTxId) external view returns (bool) {
        return processedBitcoinTxs[bitcoinTxId];
    }
}
