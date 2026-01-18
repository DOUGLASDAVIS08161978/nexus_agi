// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title TestnetWBTC - Wrapped Bitcoin Token for Testing
 * @dev Simple ERC-20 token that can be deployed to Sepolia testnet
 * @notice This is a testnet token with no real value - for educational purposes only
 */

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/token/ERC20/extensions/ERC20Burnable.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract TestnetWBTC is ERC20, ERC20Burnable, Ownable {

    uint8 private _decimals;

    /**
     * @dev Constructor that gives msg.sender all of initial supply
     * @param initialSupply Initial supply in whole tokens (before decimals)
     */
    constructor(uint256 initialSupply) ERC20("Testnet Wrapped Bitcoin", "tWBTC") Ownable(msg.sender) {
        _decimals = 8; // Same as Bitcoin
        _mint(msg.sender, initialSupply * 10**_decimals);
    }

    /**
     * @dev Returns the number of decimals used for token amounts
     */
    function decimals() public view virtual override returns (uint8) {
        return _decimals;
    }

    /**
     * @dev Mint new tokens (only owner)
     * @param to Address to receive tokens
     * @param amount Amount of tokens to mint (in smallest unit)
     */
    function mint(address to, uint256 amount) public onlyOwner {
        _mint(to, amount);
    }

    /**
     * @dev Mint tokens to multiple addresses
     * @param recipients Array of recipient addresses
     * @param amounts Array of amounts (must match recipients length)
     */
    function batchMint(address[] calldata recipients, uint256[] calldata amounts) public onlyOwner {
        require(recipients.length == amounts.length, "Arrays must have same length");

        for (uint256 i = 0; i < recipients.length; i++) {
            _mint(recipients[i], amounts[i]);
        }
    }

    /**
     * @dev Airdrop same amount to multiple addresses
     * @param recipients Array of recipient addresses
     * @param amount Amount to send to each recipient
     */
    function airdrop(address[] calldata recipients, uint256 amount) public onlyOwner {
        for (uint256 i = 0; i < recipients.length; i++) {
            _mint(recipients[i], amount);
        }
    }

    /**
     * @dev Emergency function to recover accidentally sent ERC20 tokens
     * @param token ERC20 token address
     * @param amount Amount to recover
     */
    function recoverERC20(address token, uint256 amount) public onlyOwner {
        IERC20(token).transfer(owner(), amount);
    }

    /**
     * @dev Get token information
     */
    function getTokenInfo() public view returns (
        string memory tokenName,
        string memory tokenSymbol,
        uint8 tokenDecimals,
        uint256 tokenTotalSupply,
        address tokenOwner
    ) {
        return (
            name(),
            symbol(),
            decimals(),
            totalSupply(),
            owner()
        );
    }
}
