/**
 * Create Uniswap V3 Liquidity Pool for TBTC
 *
 * Sets initial price to match Bitcoin's current market price
 * Creates TBTC/WETH pool on Base Sepolia
 */

const { ethers } = require('ethers');
const axios = require('axios');
const fs = require('fs');
const path = require('path');

// Load .env from project root
const envPath = path.join(__dirname, '..', '.env');
const envResult = require('dotenv').config({ path: envPath });

if (envResult.error) {
  console.error('Failed to load .env:', envResult.error);
  console.error('Tried path:', envPath);
  process.exit(1);
}

// ANSI colors
const colors = {
  reset: '\x1b[0m',
  green: '\x1b[32m',
  red: '\x1b[31m',
  yellow: '\x1b[33m',
  cyan: '\x1b[36m',
  bright: '\x1b[1m',
  magenta: '\x1b[35m',
};

function log(message, color = 'reset') {
  console.log(`${colors[color]}${message}${colors.reset}`);
}

function header(text) {
  const line = '═'.repeat(80);
  console.log('\n' + line);
  console.log(`║  ${text}`.padEnd(79) + '║');
  console.log(line + '\n');
}

// Uniswap V3 addresses on Base Sepolia
const UNISWAP_V3_FACTORY = '0x4752ba5dbc23f44d87826276bf6fd6b1c372ad24';
const UNISWAP_V3_POSITION_MANAGER = '0x27F971cb582BF9E50F397e4d29a5C7A34f11faA2';
const WETH_BASE_SEPOLIA = '0x4200000000000000000000000000000000000006'; // Canonical WETH on Base

// ABIs
const ERC20_ABI = [
  'function approve(address spender, uint256 amount) external returns (bool)',
  'function balanceOf(address account) external view returns (uint256)',
  'function decimals() external view returns (uint8)',
  'function symbol() external view returns (string)',
  'function transfer(address to, uint256 amount) external returns (bool)'
];

const WETH_ABI = [
  'function deposit() external payable',
  'function withdraw(uint256) external',
  'function balanceOf(address) external view returns (uint256)',
  'function approve(address spender, uint256 amount) external returns (bool)'
];

const UNISWAP_V3_FACTORY_ABI = [
  'function createPool(address tokenA, address tokenB, uint24 fee) external returns (address pool)',
  'function getPool(address tokenA, address tokenB, uint24 fee) external view returns (address pool)'
];

const UNISWAP_V3_POOL_ABI = [
  'function initialize(uint160 sqrtPriceX96) external',
  'function slot0() external view returns (uint160 sqrtPriceX96, int24 tick, uint16 observationIndex, uint16 observationCardinality, uint16 observationCardinalityNext, uint8 feeProtocol, bool unlocked)'
];

const POSITION_MANAGER_ABI = [
  'function createAndInitializePoolIfNecessary(address token0, address token1, uint24 fee, uint160 sqrtPriceX96) external payable returns (address pool)',
  'function mint((address token0, address token1, uint24 fee, int24 tickLower, int24 tickUpper, uint256 amount0Desired, uint256 amount1Desired, uint256 amount0Min, uint256 amount1Min, address recipient, uint256 deadline)) external payable returns (uint256 tokenId, uint128 liquidity, uint256 amount0, uint256 amount1)'
];

async function getBitcoinPrice() {
  try {
    log('📊 Fetching Bitcoin price...', 'cyan');

    // Try Binance first (more reliable, no API key needed)
    try {
      const [btcResponse, ethResponse] = await Promise.all([
        axios.get('https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT', { timeout: 5000 }),
        axios.get('https://api.binance.com/api/v3/ticker/price?symbol=ETHUSDT', { timeout: 5000 })
      ]);

      const btcPrice = parseFloat(btcResponse.data.price);
      const ethPrice = parseFloat(ethResponse.data.price);

      log(`   Bitcoin: $${btcPrice.toLocaleString()}`, 'green');
      log(`   Ethereum: $${ethPrice.toLocaleString()}`, 'green');

      return { btcPrice, ethPrice };
    } catch (binanceError) {
      log('   Binance failed, trying CoinGecko...', 'yellow');

      // Fallback to CoinGecko
      const response = await axios.get(
        'https://api.coingecko.com/api/v3/simple/price?ids=bitcoin,ethereum&vs_currencies=usd',
        { timeout: 5000 }
      );
      const btcPrice = response.data.bitcoin.usd;
      const ethPrice = response.data.ethereum.usd;

      log(`   Bitcoin: $${btcPrice.toLocaleString()}`, 'green');
      log(`   Ethereum: $${ethPrice.toLocaleString()}`, 'green');

      return { btcPrice, ethPrice };
    }
  } catch (error) {
    // If all APIs fail, use approximate values
    log('   ⚠️  APIs unavailable, using approximate prices', 'yellow');
    const btcPrice = 78000; // Approximate current BTC price
    const ethPrice = 2330; // Approximate current ETH price

    log(`   Bitcoin: ~$${btcPrice.toLocaleString()}`, 'yellow');
    log(`   Ethereum: ~$${ethPrice.toLocaleString()}`, 'yellow');

    return { btcPrice, ethPrice };
  }
}

function calculateSqrtPriceX96(btcPrice, ethPrice) {
  // Price ratio: how many ETH for 1 BTC
  const ratio = btcPrice / ethPrice;

  log(`\n💱 Price Calculation:`, 'bright');
  log(`   1 TBTC = ${ratio.toFixed(4)} ETH`, 'cyan');
  log(`   1 ETH = ${(1/ratio).toFixed(6)} TBTC`, 'cyan');

  // Uniswap V3 uses sqrtPriceX96 = sqrt(price) * 2^96
  // price = token1/token0
  // We need to determine token0 and token1 order (sorted by address)

  const sqrtPrice = Math.sqrt(ratio);
  const sqrtPriceX96 = BigInt(Math.floor(sqrtPrice * (2 ** 96)));

  return { ratio, sqrtPriceX96 };
}

async function main() {
  header('🦄 CREATE UNISWAP V3 POOL - TBTC PEGGED TO BITCOIN PRICE');

  // Load TBTC deployment
  const deploymentPath = path.join(__dirname, '..', 'tbtc_base_sepolia_deployment.json');

  if (!fs.existsSync(deploymentPath)) {
    log('❌ TBTC not deployed on Base Sepolia', 'red');
    log('   Run: ./LAUNCH_TBTC.sh first', 'yellow');
    return;
  }

  const deployment = JSON.parse(fs.readFileSync(deploymentPath, 'utf8'));
  const TBTC_ADDRESS = deployment.contracts.TBTC.address;

  log(`TBTC Address: ${TBTC_ADDRESS}`, 'cyan');
  log(`WETH Address: ${WETH_BASE_SEPOLIA}`, 'cyan');
  log(`Network: Base Sepolia`, 'cyan');

  // Get Bitcoin price
  const { btcPrice, ethPrice } = await getBitcoinPrice();
  const { ratio, sqrtPriceX96 } = calculateSqrtPriceX96(btcPrice, ethPrice);

  // Setup provider and wallet
  const rpcUrl = process.env.BASE_SEPOLIA_RPC_URL;

  if (!rpcUrl) {
    log('❌ RPC URL not found in environment', 'red');
    log('   Set BASE_SEPOLIA_RPC_URL in .env', 'yellow');
    return;
  }

  log(`\n🔗 Connecting to: ${rpcUrl}`, 'cyan');

  // Create provider with Base Sepolia network config
  const network = {
    name: 'base-sepolia',
    chainId: 84532
  };

  const provider = new ethers.JsonRpcProvider(rpcUrl, network);

  // Get private key (try both env var names)
  let privateKey = process.env.BASE_SEPOLIA_PRIVATE_KEY || process.env.PRIVATE_KEY;

  if (!privateKey) {
    log('❌ Private key not found in environment', 'red');
    log('   Set BASE_SEPOLIA_PRIVATE_KEY or PRIVATE_KEY in .env', 'yellow');
    return;
  }

  // Ensure private key has 0x prefix
  if (!privateKey.startsWith('0x')) {
    privateKey = '0x' + privateKey;
  }

  const wallet = new ethers.Wallet(privateKey, provider);

  log(`\n💼 Your Address: ${wallet.address}`, 'cyan');

  // Check balances
  header('STEP 1: CHECK BALANCES');

  const ethBalance = await provider.getBalance(wallet.address);
  log(`ETH Balance: ${ethers.formatEther(ethBalance)} ETH`, 'cyan');

  const tbtc = new ethers.Contract(TBTC_ADDRESS, ERC20_ABI, wallet);
  const tbtcBalance = await tbtc.balanceOf(wallet.address);
  log(`TBTC Balance: ${ethers.formatEther(tbtcBalance)} TBTC`, 'cyan');

  if (tbtcBalance === 0n) {
    log('\n❌ You need TBTC tokens to create a pool!', 'red');
    log('   Run: ./LAUNCH_TBTC.sh to mint tokens', 'yellow');
    return;
  }

  // Wrap ETH to WETH
  header('STEP 2: WRAP ETH TO WETH');

  const weth = new ethers.Contract(WETH_BASE_SEPOLIA, WETH_ABI, wallet);
  const wethBalance = await weth.balanceOf(wallet.address);

  log(`Current WETH Balance: ${ethers.formatEther(wethBalance)} WETH`, 'cyan');

  // Calculate how much WETH we need based on TBTC amount
  const tbtcAmount = ethers.parseEther('1'); // Use 1 TBTC for the pool
  const wethNeeded = (BigInt(ethers.parseEther(ratio.toString())) * tbtcAmount) / ethers.parseEther('1');

  log(`\nWrapping ${ethers.formatEther(wethNeeded)} ETH to WETH...`, 'cyan');

  const wrapTx = await weth.deposit({ value: wethNeeded });
  log(`   Transaction: ${wrapTx.hash}`, 'yellow');
  await wrapTx.wait();
  log('✅ ETH wrapped to WETH', 'green');

  // Approve tokens
  header('STEP 3: APPROVE TOKENS FOR UNISWAP');

  log('Approving TBTC...', 'cyan');
  const approveTbtc = await tbtc.approve(UNISWAP_V3_POSITION_MANAGER, ethers.MaxUint256);
  await approveTbtc.wait();
  log('✅ TBTC approved', 'green');

  log('Approving WETH...', 'cyan');
  const approveWeth = await weth.approve(UNISWAP_V3_POSITION_MANAGER, ethers.MaxUint256);
  await approveWeth.wait();
  log('✅ WETH approved', 'green');

  // Determine token0 and token1 (Uniswap requires sorted addresses)
  const token0 = TBTC_ADDRESS.toLowerCase() < WETH_BASE_SEPOLIA.toLowerCase() ? TBTC_ADDRESS : WETH_BASE_SEPOLIA;
  const token1 = TBTC_ADDRESS.toLowerCase() < WETH_BASE_SEPOLIA.toLowerCase() ? WETH_BASE_SEPOLIA : TBTC_ADDRESS;

  const token0IsTbtc = token0.toLowerCase() === TBTC_ADDRESS.toLowerCase();

  log(`\n📋 Pool Configuration:`, 'bright');
  log(`   Token0: ${token0IsTbtc ? 'TBTC' : 'WETH'}`, 'cyan');
  log(`   Token1: ${token0IsTbtc ? 'WETH' : 'TBTC'}`, 'cyan');
  log(`   Fee Tier: 0.3% (3000)`, 'cyan');

  // Adjust sqrtPriceX96 if token order is inverted
  let finalSqrtPriceX96 = sqrtPriceX96;
  if (!token0IsTbtc) {
    // If WETH is token0, we need to invert the price
    finalSqrtPriceX96 = BigInt(Math.floor((2 ** 96) / Math.sqrt(ratio)));
  }

  // Create and initialize pool
  header('STEP 4: CREATE UNISWAP V3 POOL');

  const positionManager = new ethers.Contract(UNISWAP_V3_POSITION_MANAGER, POSITION_MANAGER_ABI, wallet);

  log(`Creating pool with initial price (sqrtPriceX96: ${finalSqrtPriceX96.toString()})...`, 'cyan');

  try {
    const createPoolTx = await positionManager.createAndInitializePoolIfNecessary(
      token0,
      token1,
      3000, // 0.3% fee tier
      finalSqrtPriceX96
    );

    log(`   Transaction: ${createPoolTx.hash}`, 'yellow');
    const receipt = await createPoolTx.wait();
    log('✅ Pool created and initialized!', 'green');
  } catch (error) {
    if (error.message.includes('Already initialized')) {
      log('⚠️  Pool already exists', 'yellow');
    } else {
      throw error;
    }
  }

  // Get pool address
  const factory = new ethers.Contract(UNISWAP_V3_FACTORY, UNISWAP_V3_FACTORY_ABI, provider);
  const poolAddress = await factory.getPool(token0, token1, 3000);

  log(`\n🏊 Pool Address: ${poolAddress}`, 'bright');

  // Add liquidity
  header('STEP 5: ADD LIQUIDITY');

  const amount0Desired = token0IsTbtc ? tbtcAmount : wethNeeded;
  const amount1Desired = token0IsTbtc ? wethNeeded : tbtcAmount;

  // Full range liquidity (tick range -887220 to 887220)
  const tickLower = -887220;
  const tickUpper = 887220;

  log('Adding liquidity to pool...', 'cyan');
  log(`   Amount0: ${ethers.formatEther(amount0Desired)} ${token0IsTbtc ? 'TBTC' : 'WETH'}`, 'cyan');
  log(`   Amount1: ${ethers.formatEther(amount1Desired)} ${token0IsTbtc ? 'WETH' : 'TBTC'}`, 'cyan');
  log(`   Range: Full range`, 'cyan');

  const mintParams = {
    token0,
    token1,
    fee: 3000,
    tickLower,
    tickUpper,
    amount0Desired,
    amount1Desired,
    amount0Min: 0,
    amount1Min: 0,
    recipient: wallet.address,
    deadline: Math.floor(Date.now() / 1000) + 3600
  };

  const mintTx = await positionManager.mint(mintParams);
  log(`   Transaction: ${mintTx.hash}`, 'yellow');
  const mintReceipt = await mintTx.wait();
  log('✅ Liquidity added!', 'green');

  // Summary
  header('✅ UNISWAP V3 POOL CREATED SUCCESSFULLY!');

  log('📊 POOL DETAILS:', 'bright');
  log(`   Pool Address: ${poolAddress}`, 'cyan');
  log(`   TBTC Address: ${TBTC_ADDRESS}`, 'cyan');
  log(`   WETH Address: ${WETH_BASE_SEPOLIA}`, 'cyan');
  log(`   Fee Tier: 0.3%`, 'cyan');
  log(`   Initial Price: 1 TBTC = ${ratio.toFixed(4)} ETH`, 'green');
  log(`   Network: Base Sepolia`, 'cyan');

  log('\n🔗 VIEW ON BASE SEPOLIA:', 'bright');
  log(`   Pool: https://sepolia.basescan.org/address/${poolAddress}`, 'yellow');
  log(`   Uniswap: https://app.uniswap.org/pools/${poolAddress}`, 'yellow');

  log('\n⚠️  IMPORTANT:', 'red');
  log('   • Price will change as people trade!', 'yellow');
  log('   • This is just initial price discovery', 'yellow');
  log('   • Your proof-of-reserves maintains the 1:1 BTC backing', 'yellow');
  log('   • Market price may differ from backing value', 'yellow');

  log('\n✨ Your TBTC is now tradeable on Uniswap! ✨\n', 'magenta');
}

main().catch((error) => {
  log('\n❌ Error:', 'red');
  console.error(error);
  process.exit(1);
});
