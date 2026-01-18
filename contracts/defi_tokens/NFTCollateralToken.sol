// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title NFTCollateralToken (NFTC)
 * @dev Fractionalized NFT collateral for lending
 */
contract NFTCollateralToken {
    string public constant name = "NFT Collateral Token";
    string public constant symbol = "NFTC";
    uint8 public constant decimals = 18;
    uint256 public totalSupply;

    mapping(address => uint256) public balanceOf;
    mapping(address => mapping(address => uint256)) public allowance;

    struct NFTVault {
        address nftContract;
        uint256 tokenId;
        uint256 valuation;
        uint256 fractionsIssued;
        bool active;
    }

    mapping(uint256 => NFTVault) public vaults;
    uint256 public vaultCount;

    address public owner;

    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);
    event NFTDeposited(uint256 indexed vaultId, address indexed nftContract, uint256 tokenId);
    event FractionsCreated(uint256 indexed vaultId, uint256 amount);

    constructor() {
        owner = msg.sender;
    }

    function transfer(address to, uint256 value) external returns (bool) {
        _transfer(msg.sender, to, value);
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

    function depositNFT(address nftContract, uint256 tokenId, uint256 valuation) external returns (uint256) {
        vaultCount++;
        vaults[vaultCount] = NFTVault({
            nftContract: nftContract,
            tokenId: tokenId,
            valuation: valuation,
            fractionsIssued: valuation,
            active: true
        });

        totalSupply += valuation;
        balanceOf[msg.sender] += valuation;

        emit NFTDeposited(vaultCount, nftContract, tokenId);
        emit FractionsCreated(vaultCount, valuation);
        emit Transfer(address(0), msg.sender, valuation);

        return vaultCount;
    }
}
