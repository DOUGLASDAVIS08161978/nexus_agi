const { ethers } = require('ethers');
const fs = require('fs');
const path = require('path');

// Read config from project directory
const config = JSON.parse(fs.readFileSync(path.join(__dirname, '..', 'pool_config.json'), 'utf8'));

const POSITION_MANAGER_ABI = [
  'function createAndInitializePoolIfNecessary(address token0, address token1, uint24 fee, uint160 sqrtPriceX96) external payable returns (address pool)',
  'function mint((address token0, address token1, uint24 fee, int24 tickLower, int24 tickUpper, uint256 amount0Desired, uint256 amount1Desired, uint256 amount0Min, uint256 amount1Min, address recipient, uint256 deadline)) external payable returns (uint256 tokenId, uint128 liquidity, uint256 amount0, uint256 amount1)'
];

const ERC20_ABI = [
  'function approve(address spender, uint256 amount) external returns (bool)'
];

async function main() {
  try {
    // Use a provider with increased timeout
    const provider = new ethers.JsonRpcProvider(config.rpcUrl, {
      name: 'base-sepolia',
      chainId: 84532
    }, {
      staticNetwork: true,
      polling: true,
      pollingInterval: 5000
    });

    // Set timeout
    provider.pollingInterval = 5000;

    const wallet = new ethers.Wallet(config.privateKey, provider);

    console.log('Wallet connected:', wallet.address);

    // Calculate sqrtPriceX96
    const ratio = parseFloat(config.ratio);
    const sqrtPrice = Math.sqrt(ratio);
    const sqrtPriceX96 = BigInt(Math.floor(sqrtPrice * (2 ** 96)));

    // Determine token order
    const token0 = config.tbtcAddress.toLowerCase() < config.wethAddress.toLowerCase()
      ? config.tbtcAddress : config.wethAddress;
    const token1 = config.tbtcAddress.toLowerCase() < config.wethAddress.toLowerCase()
      ? config.wethAddress : config.tbtcAddress;
    const token0IsTbtc = token0.toLowerCase() === config.tbtcAddress.toLowerCase();

    // Adjust price if needed
    let finalSqrtPriceX96 = sqrtPriceX96;
    if (!token0IsTbtc) {
      finalSqrtPriceX96 = BigInt(Math.floor((2 ** 96) / Math.sqrt(ratio)));
    }

    console.log('Token0:', token0, token0IsTbtc ? '(TBTC)' : '(WETH)');
    console.log('Token1:', token1, token0IsTbtc ? '(WETH)' : '(TBTC)');
    console.log('Price (sqrtPriceX96):', finalSqrtPriceX96.toString());

    // Save pool info
    const poolInfo = {
      token0,
      token1,
      sqrtPriceX96: finalSqrtPriceX96.toString(),
      ratio: ratio
    };

    fs.writeFileSync(path.join(__dirname, '..', 'pool_info.json'), JSON.stringify(poolInfo, null, 2));

    console.log('\n✅ Pool configuration ready!');
    console.log('   Run the transaction script to create the pool');

  } catch (error) {
    console.error('Error:', error.message);
    process.exit(1);
  }
}

main();
