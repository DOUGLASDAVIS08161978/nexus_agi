// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title StableBTC (sBTC)
 * @dev Algorithmic stablecoin pegged to $1, backed by Bitcoin collateral
 * Maintains peg through collateralization ratio and stability mechanisms
 */
contract StableBTC {
    string public constant name = "Stable Bitcoin USD";
    string public constant symbol = "sBTC";
    uint8 public constant decimals = 18;
    uint256 public totalSupply;

    uint256 public constant MIN_COLLATERAL_RATIO = 150; // 150% = 1.5x overcollateralized
    uint256 public constant LIQUIDATION_RATIO = 130; // 130% = liquidation threshold
    uint256 public constant STABILITY_FEE = 50; // 0.5% annual fee in basis points

    mapping(address => uint256) public balanceOf;
    mapping(address => mapping(address => uint256)) public allowance;

    struct Vault {
        uint256 collateral; // BTC amount in wei
        uint256 debt; // sBTC minted
        uint256 lastUpdate;
    }

    mapping(address => Vault) public vaults;
    uint256 public totalCollateral;
    uint256 public totalDebt;

    address public owner;
    address public priceOracle; // Oracle for BTC/USD price
    uint256 public btcPrice = 100_000 ether; // $100,000 default (18 decimals)

    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);
    event VaultOpened(address indexed user, uint256 collateral, uint256 debt);
    event VaultClosed(address indexed user, uint256 collateral, uint256 debt);
    event Liquidation(address indexed user, address indexed liquidator, uint256 collateral);
    event PriceUpdated(uint256 newPrice);

    modifier onlyOwner() {
        require(msg.sender == owner, "Not owner");
        _;
    }

    constructor() {
        owner = msg.sender;
    }

    // ERC20 Functions
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

    // Vault Functions
    function openVault(uint256 collateralAmount, uint256 debtAmount) external payable {
        require(collateralAmount > 0, "No collateral");
        require(debtAmount > 0, "No debt");

        // Check collateralization ratio
        uint256 collateralValue = (collateralAmount * btcPrice) / 1e18;
        uint256 ratio = (collateralValue * 100) / debtAmount;
        require(ratio >= MIN_COLLATERAL_RATIO, "Insufficient collateral");

        Vault storage vault = vaults[msg.sender];
        vault.collateral += collateralAmount;
        vault.debt += debtAmount;
        vault.lastUpdate = block.timestamp;

        totalCollateral += collateralAmount;
        totalDebt += debtAmount;

        // Mint sBTC
        totalSupply += debtAmount;
        balanceOf[msg.sender] += debtAmount;

        emit VaultOpened(msg.sender, collateralAmount, debtAmount);
        emit Transfer(address(0), msg.sender, debtAmount);
    }

    function closeVault(uint256 repayAmount) external {
        Vault storage vault = vaults[msg.sender];
        require(vault.debt > 0, "No vault");
        require(balanceOf[msg.sender] >= repayAmount, "Insufficient balance");

        // Calculate stability fee
        uint256 timeElapsed = block.timestamp - vault.lastUpdate;
        uint256 fee = (vault.debt * STABILITY_FEE * timeElapsed) / (365 days * 10000);
        uint256 totalRepay = repayAmount + fee;

        require(balanceOf[msg.sender] >= totalRepay, "Insufficient for fee");

        // Burn sBTC
        balanceOf[msg.sender] -= totalRepay;
        totalSupply -= repayAmount;

        // Calculate collateral to return
        uint256 collateralReturn = (vault.collateral * repayAmount) / vault.debt;

        vault.debt -= repayAmount;
        vault.collateral -= collateralReturn;
        totalDebt -= repayAmount;
        totalCollateral -= collateralReturn;

        if (vault.debt == 0) {
            delete vaults[msg.sender];
        }

        emit VaultClosed(msg.sender, collateralReturn, repayAmount);
        emit Transfer(msg.sender, address(0), totalRepay);
    }

    function liquidate(address user) external {
        Vault storage vault = vaults[user];
        require(vault.debt > 0, "No vault");

        // Check if undercollateralized
        uint256 collateralValue = (vault.collateral * btcPrice) / 1e18;
        uint256 ratio = (collateralValue * 100) / vault.debt;
        require(ratio < LIQUIDATION_RATIO, "Vault healthy");

        // Liquidator must repay debt
        require(balanceOf[msg.sender] >= vault.debt, "Insufficient balance");

        // Burn liquidator's sBTC
        balanceOf[msg.sender] -= vault.debt;
        totalSupply -= vault.debt;

        // Transfer collateral to liquidator (with bonus)
        uint256 bonus = vault.collateral / 10; // 10% liquidation bonus
        uint256 totalCollateralReturn = vault.collateral + bonus;

        totalDebt -= vault.debt;
        totalCollateral -= vault.collateral;

        emit Liquidation(user, msg.sender, totalCollateralReturn);
        emit Transfer(msg.sender, address(0), vault.debt);

        delete vaults[user];
    }

    function getCollateralizationRatio(address user) external view returns (uint256) {
        Vault memory vault = vaults[user];
        if (vault.debt == 0) return 0;

        uint256 collateralValue = (vault.collateral * btcPrice) / 1e18;
        return (collateralValue * 100) / vault.debt;
    }

    function updateBTCPrice(uint256 newPrice) external onlyOwner {
        btcPrice = newPrice;
        emit PriceUpdated(newPrice);
    }

    function setPriceOracle(address oracle) external onlyOwner {
        priceOracle = oracle;
    }
}
