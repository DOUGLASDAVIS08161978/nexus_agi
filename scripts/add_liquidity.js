/**
 * Add Liquidity to DEX for WTBTC Token
 *
 * This script adds liquidity to QuickSwap (Uniswap V2 fork on Polygon)
 *
 * Requirements:
 * 1. Deployed WTBTC contract
 * 2. MATIC in your wallet for pairing
 * 3. Gas fees (MATIC)
 */

const { ethers } = require('ethers');
const fs = require('fs');
const path = require('path');
require('dotenv').config();

// QuickSwap Router on Polygon
const QUICKSWAP_ROUTER = '0xa5E0829CaCEd8fFDD4De3c43696c57F7D7A678ff';

// QuickSwap Router ABI (minimal)
const ROUTER_ABI = [
  'function addLiquidityETH(address token, uint amountTokenDesired, uint amountTokenMin, uint amountETHMin, address to, uint deadline) external payable returns (uint amountToken, uint amountETH, uint liquidity)',
  'function getAmountsOut(uint amountIn, address[] memory path) public view returns (uint[] memory amounts)',
  'function WETH() external pure returns (address)'
];

// ERC20 ABI (minimal)
const ERC20_ABI = [
  'function approve(address spender, uint256 amount) external returns (bool)',
  'function balanceOf(address account) external view returns (uint256)',
  'function decimals() external view returns (uint8)',
  'function symbol() external view returns (string)'
];

// ANSI colors
const colors = {
  reset: '\x1b[0m',
  green: '\x1b[32m',
  red: '\x1b[31m',
  yellow: '\x1b[33m',
  cyan: '\x1b[36m',
  bright: '\x1b[1m',
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

async function main() {
  header('💧 ADD LIQUIDITY TO QUICKSWAP (POLYGON)');

  // Read deployment info
  const deploymentPath = path.join(__dirname, '..', 'deployment_info.json');
  if (!fs.existsSync(deploymentPath)) {
    log('❌ deployment_info.json not found. Deploy contracts first!', 'red');
    log('   Run: ./deploy_quick.sh', 'yellow');
    process.exit(1);
  }

  const deployment = JSON.parse(fs.readFileSync(deploymentPath, 'utf8'));
  const tokenAddress = deployment.contracts.TestnetWBTC.address;

  log(`Token address: ${tokenAddress}`, 'cyan');

  // Setup provider and wallet
  const provider = new ethers.JsonRpcProvider(process.env.POLYGON_RPC_URL);
  const wallet = new ethers.Wallet('0x' + process.env.PRIVATE_KEY, provider);

  log(`Wallet: ${wallet.address}`, 'cyan');

  // Check balance
  const balance = await provider.getBalance(wallet.address);
  const balanceMATIC = Number(ethers.formatEther(balance));
  log(`MATIC Balance: ${balanceMATIC.toFixed(4)} MATIC`, balanceMATIC > 0.1 ? 'green' : 'red');

  if (balanceMATIC < 0.1) {
    log('\n⚠️  You need at least 0.1 MATIC to add liquidity!', 'yellow');
    log('   - Some for the liquidity pool', 'yellow');
    log('   - Some for gas fees', 'yellow');
    return;
  }

  // Connect to token contract
  const token = new ethers.Contract(tokenAddress, ERC20_ABI, wallet);
  const tokenBalance = await token.balanceOf(wallet.address);
  const decimals = await token.decimals();
  const symbol = await token.symbol();
  const tokenAmount = Number(ethers.formatUnits(tokenBalance, decimals));

  log(`\nToken: ${symbol}`, 'cyan');
  log(`Token Balance: ${tokenAmount.toFixed(8)} ${symbol}`, tokenAmount > 0 ? 'green' : 'red');

  if (tokenAmount === 0) {
    log('\n⚠️  You need to mint some tokens first!', 'yellow');
    log('   Run: node scripts/mint_tokens.js', 'yellow');
    return;
  }

  // Liquidity amounts
  const tokenAmountToAdd = ethers.parseUnits('10', decimals); // Add 10 tokens
  const maticAmountToAdd = ethers.parseEther('0.01'); // Add 0.01 MATIC

  log(`\n📋 Liquidity to add:`, 'bright');
  log(`   ${ethers.formatUnits(tokenAmountToAdd, decimals)} ${symbol}`, 'yellow');
  log(`   ${ethers.formatEther(maticAmountToAdd)} MATIC`, 'yellow');

  // Connect to QuickSwap Router
  const router = new ethers.Contract(QUICKSWAP_ROUTER, ROUTER_ABI, wallet);

  header('STEP 1: APPROVE TOKEN SPENDING');

  log('Approving QuickSwap Router to spend tokens...', 'cyan');
  const approveTx = await token.approve(QUICKSWAP_ROUTER, tokenAmountToAdd);
  log(`Transaction hash: ${approveTx.hash}`, 'yellow');
  log('Waiting for confirmation...', 'cyan');
  await approveTx.wait();
  log('✅ Approval confirmed!', 'green');

  header('STEP 2: ADD LIQUIDITY');

  log('Adding liquidity to QuickSwap...', 'cyan');

  // Set deadline (10 minutes from now)
  const deadline = Math.floor(Date.now() / 1000) + 600;

  // Allow 1% slippage
  const minTokenAmount = (tokenAmountToAdd * 99n) / 100n;
  const minMaticAmount = (maticAmountToAdd * 99n) / 100n;

  try {
    const liquidityTx = await router.addLiquidityETH(
      tokenAddress,
      tokenAmountToAdd,
      minTokenAmount,
      minMaticAmount,
      wallet.address,
      deadline,
      { value: maticAmountToAdd }
    );

    log(`Transaction hash: ${liquidityTx.hash}`, 'yellow');
    log('Waiting for confirmation...', 'cyan');

    const receipt = await liquidityTx.wait();
    log('✅ Liquidity added successfully!', 'green');

    header('✅ SUCCESS!');

    log('Your liquidity has been added to QuickSwap!', 'green');
    log(`\nView on PolygonScan:`, 'cyan');
    log(`https://polygonscan.com/tx/${liquidityTx.hash}`, 'yellow');

    log(`\nView your liquidity pool:`, 'cyan');
    log(`https://quickswap.exchange/#/pool`, 'yellow');

    log(`\nYour LP tokens represent your share of the pool.`, 'cyan');
    log(`You can remove liquidity anytime on QuickSwap.`, 'cyan');

  } catch (error) {
    log('\n❌ Failed to add liquidity:', 'red');
    if (error.message.includes('INSUFFICIENT')) {
      log('   Insufficient token or MATIC balance', 'yellow');
    } else if (error.message.includes('EXPIRED')) {
      log('   Transaction deadline expired, try again', 'yellow');
    } else {
      console.error(error);
    }
  }
}

main().catch((error) => {
  log('\n❌ Error:', 'red');
  console.error(error);
  process.exit(1);
});
