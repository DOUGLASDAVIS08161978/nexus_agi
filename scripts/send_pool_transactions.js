/**
 * Send Pool Creation Transactions
 * Reads pool configuration and sends transactions to create Uniswap V3 pool
 * Uses minimal network calls - optimized for slow connections
 */

const { ethers } = require('ethers');
const fs = require('fs');
const path = require('path');

// Load environment
require('dotenv').config({ path: path.join(__dirname, '..', '.env') });

// Read pool configuration
const poolInfo = JSON.parse(fs.readFileSync(path.join(__dirname, '..', 'pool_info.json'), 'utf8'));

const POSITION_MANAGER = '0x27F971cb582BF9E50F397e4d29a5C7A34f11faA2';
const FACTORY = '0x4752ba5dbc23f44d87826276bf6fd6b1c372ad24';

const ERC20_ABI = [
  'function approve(address spender, uint256 amount) external returns (bool)',
  'function balanceOf(address account) external view returns (uint256)'
];

const WETH_ABI = [
  'function deposit() external payable',
  'function balanceOf(address) external view returns (uint256)'
];

const POSITION_MANAGER_ABI = [
  'function createAndInitializePoolIfNecessary(address token0, address token1, uint24 fee, uint160 sqrtPriceX96) external payable returns (address pool)',
  'function mint((address token0, address token1, uint24 fee, int24 tickLower, int24 tickUpper, uint256 amount0Desired, uint256 amount1Desired, uint256 amount0Min, uint256 amount1Min, address recipient, uint256 deadline)) external payable returns (uint256 tokenId, uint128 liquidity, uint256 amount0, uint256 amount1)'
];

const FACTORY_ABI = [
  'function getPool(address tokenA, address tokenB, uint24 fee) external view returns (address pool)'
];

// Colors
const colors = {
  reset: '\x1b[0m',
  green: '\x1b[32m',
  red: '\x1b[31m',
  yellow: '\x1b[33m',
  cyan: '\x1b[36m',
  magenta: '\x1b[35m',
  bold: '\x1b[1m'
};

function log(message, color = 'reset') {
  console.log(`${colors[color]}${message}${colors.reset}`);
}

async function main() {
  try {
    log('\n🚀 CREATING UNISWAP V3 POOL - SENDING TRANSACTIONS\n', 'bold');

    // Setup provider with aggressive timeout settings
    const provider = new ethers.JsonRpcProvider(
      process.env.BASE_SEPOLIA_RPC_URL,
      { name: 'base-sepolia', chainId: 84532 },
      { staticNetwork: true }
    );

    const wallet = new ethers.Wallet(process.env.BASE_SEPOLIA_PRIVATE_KEY, provider);

    log(`Wallet: ${wallet.address}`, 'cyan');
    log(`Token0: ${poolInfo.token0}`, 'cyan');
    log(`Token1: ${poolInfo.token1}`, 'cyan');
    log(`Price: ${poolInfo.ratio} ETH per TBTC\n`, 'cyan');

    const token0 = poolInfo.token0;
    const token1 = poolInfo.token1;
    const token0IsTbtc = token1.toLowerCase() !== '0x4200000000000000000000000000000000000006'.toLowerCase();

    // Determine TBTC and WETH addresses
    const TBTC_ADDRESS = token0IsTbtc ? token0 : token1;
    const WETH_ADDRESS = token0IsTbtc ? token1 : token0;

    // Step 1: Wrap ETH to WETH
    log('════════════════════════════════════════════════════════', 'bold');
    log('STEP 1: WRAP ETH TO WETH', 'bold');
    log('════════════════════════════════════════════════════════\n', 'bold');

    const weth = new ethers.Contract(WETH_ADDRESS, WETH_ABI, wallet);
    const wethAmount = ethers.parseEther('0.001'); // Wrap 0.001 ETH for now

    log('Wrapping 0.001 ETH to WETH...', 'cyan');
    const wrapTx = await weth.deposit({ value: wethAmount, gasLimit: 100000 });
    log(`   TX: ${wrapTx.hash}`, 'yellow');
    log('   Waiting for confirmation...', 'yellow');
    await wrapTx.wait();
    log('✅ ETH wrapped to WETH\n', 'green');

    // Step 2: Approve TBTC
    log('════════════════════════════════════════════════════════', 'bold');
    log('STEP 2: APPROVE TOKENS', 'bold');
    log('════════════════════════════════════════════════════════\n', 'bold');

    const tbtc = new ethers.Contract(TBTC_ADDRESS, ERC20_ABI, wallet);

    log('Approving TBTC...', 'cyan');
    const approveTbtcTx = await tbtc.approve(POSITION_MANAGER, ethers.MaxUint256, { gasLimit: 100000 });
    log(`   TX: ${approveTbtcTx.hash}`, 'yellow');
    await approveTbtcTx.wait();
    log('✅ TBTC approved\n', 'green');

    log('Approving WETH...', 'cyan');
    const approveWethTx = await weth.approve(POSITION_MANAGER, ethers.MaxUint256, { gasLimit: 100000 });
    log(`   TX: ${approveWethTx.hash}`, 'yellow');
    await approveWethTx.wait();
    log('✅ WETH approved\n', 'green');

    // Step 3: Create pool
    log('════════════════════════════════════════════════════════', 'bold');
    log('STEP 3: CREATE UNISWAP V3 POOL', 'bold');
    log('════════════════════════════════════════════════════════\n', 'bold');

    const positionManager = new ethers.Contract(POSITION_MANAGER, POSITION_MANAGER_ABI, wallet);

    log('Creating pool...', 'cyan');
    try {
      const createTx = await positionManager.createAndInitializePoolIfNecessary(
        token0,
        token1,
        3000, // 0.3% fee
        poolInfo.sqrtPriceX96,
        { gasLimit: 500000 }
      );
      log(`   TX: ${createTx.hash}`, 'yellow');
      await createTx.wait();
      log('✅ Pool created!\n', 'green');
    } catch (error) {
      if (error.message.includes('Already initialized')) {
        log('⚠️  Pool already exists\n', 'yellow');
      } else {
        throw error;
      }
    }

    // Get pool address
    const factory = new ethers.Contract(FACTORY, FACTORY_ABI, provider);
    const poolAddress = await factory.getPool(token0, token1, 3000);

    log(`🏊 Pool Address: ${poolAddress}\n`, 'bold');

    // Step 4: Add liquidity
    log('════════════════════════════════════════════════════════', 'bold');
    log('STEP 4: ADD LIQUIDITY', 'bold');
    log('════════════════════════════════════════════════════════\n', 'bold');

    const tbtcAmount = ethers.parseEther('1'); // Add 1 TBTC
    const amount0Desired = token0IsTbtc ? tbtcAmount : wethAmount;
    const amount1Desired = token0IsTbtc ? wethAmount : tbtcAmount;

    log('Adding liquidity...', 'cyan');
    log(`   Amount0: ${ethers.formatEther(amount0Desired)} ${token0IsTbtc ? 'TBTC' : 'WETH'}`, 'cyan');
    log(`   Amount1: ${ethers.formatEther(amount1Desired)} ${token0IsTbtc ? 'WETH' : 'TBTC'}`, 'cyan');

    const mintParams = {
      token0,
      token1,
      fee: 3000,
      tickLower: -887220, // Full range
      tickUpper: 887220,
      amount0Desired,
      amount1Desired,
      amount0Min: 0,
      amount1Min: 0,
      recipient: wallet.address,
      deadline: Math.floor(Date.now() / 1000) + 3600
    };

    const mintTx = await positionManager.mint(mintParams, { gasLimit: 700000 });
    log(`   TX: ${mintTx.hash}`, 'yellow');
    await mintTx.wait();
    log('✅ Liquidity added!\n', 'green');

    // Success!
    log('════════════════════════════════════════════════════════', 'bold');
    log('✅ SUCCESS - POOL CREATED!', 'bold');
    log('════════════════════════════════════════════════════════\n', 'bold');

    log('📊 POOL DETAILS:', 'cyan');
    log(`   Pool: ${poolAddress}`, 'cyan');
    log(`   TBTC: ${TBTC_ADDRESS}`, 'cyan');
    log(`   WETH: ${WETH_ADDRESS}`, 'cyan');
    log(`   Fee: 0.3%`, 'cyan');
    log(`   Price: 1 TBTC = ${poolInfo.ratio} ETH`, 'cyan');
    log(`   Network: Base Sepolia\n`, 'cyan');

    log('🔗 VIEW POOL:', 'cyan');
    log(`   https://sepolia.basescan.org/address/${poolAddress}`, 'yellow');
    log(`   https://app.uniswap.org/pools/${poolAddress}\n`, 'yellow');

    log('✨ Your TBTC is now tradeable on Uniswap! ✨\n', 'magenta');

  } catch (error) {
    log('\n❌ Error:', 'red');
    console.error(error.message);

    if (error.message.includes('timeout') || error.message.includes('TIMEOUT')) {
      log('\n💡 Network timeout - try again with better connection', 'yellow');
    } else if (error.message.includes('insufficient funds')) {
      log('\n💡 Insufficient ETH for gas - get more from faucet', 'yellow');
    }

    process.exit(1);
  }
}

main();
