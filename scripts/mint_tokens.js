/**
 * Mint WTBTC Tokens
 *
 * This script mints tokens to your wallet so you can add liquidity
 */

const { ethers } = require('ethers');
const fs = require('fs');
const path = require('path');
require('dotenv').config();

// TestnetWBTC ABI (minimal - just what we need)
const TOKEN_ABI = [
  'function mint(address to, uint256 amount) external',
  'function balanceOf(address account) external view returns (uint256)',
  'function decimals() external view returns (uint8)',
  'function symbol() external view returns (string)',
  'function name() external view returns (string)',
  'function owner() external view returns (address)'
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
  header('🪙 MINT WTBTC TOKENS');

  // Read deployment info
  const deploymentPath = path.join(__dirname, '..', 'deployment_info.json');
  if (!fs.existsSync(deploymentPath)) {
    log('❌ deployment_info.json not found. Deploy contracts first!', 'red');
    log('   Run: ./deploy_quick.sh', 'yellow');
    process.exit(1);
  }

  const deployment = JSON.parse(fs.readFileSync(deploymentPath, 'utf8'));
  const tokenAddress = deployment.contracts.TestnetWBTC.address;

  // Setup provider and wallet
  const provider = new ethers.JsonRpcProvider(process.env.POLYGON_RPC_URL);
  const wallet = new ethers.Wallet('0x' + process.env.PRIVATE_KEY, provider);

  log(`Wallet: ${wallet.address}`, 'cyan');

  // Connect to token contract
  const token = new ethers.Contract(tokenAddress, TOKEN_ABI, wallet);

  const name = await token.name();
  const symbol = await token.symbol();
  const decimals = await token.decimals();

  log(`Token: ${name} (${symbol})`, 'cyan');
  log(`Address: ${tokenAddress}`, 'cyan');
  log(`Decimals: ${decimals}`, 'cyan');

  // Check current balance
  const balanceBefore = await token.balanceOf(wallet.address);
  log(`\nCurrent balance: ${ethers.formatUnits(balanceBefore, decimals)} ${symbol}`, 'yellow');

  // Amount to mint (1000 tokens)
  const amountToMint = ethers.parseUnits('1000', decimals);

  header('MINTING TOKENS');

  log(`Minting ${ethers.formatUnits(amountToMint, decimals)} ${symbol} to ${wallet.address}...`, 'cyan');

  try {
    const tx = await token.mint(wallet.address, amountToMint);
    log(`Transaction hash: ${tx.hash}`, 'yellow');
    log('Waiting for confirmation...', 'cyan');

    await tx.wait();
    log('✅ Tokens minted successfully!', 'green');

    // Check new balance
    const balanceAfter = await token.balanceOf(wallet.address);
    log(`\nNew balance: ${ethers.formatUnits(balanceAfter, decimals)} ${symbol}`, 'green');

    header('✅ SUCCESS!');

    log(`View transaction:`, 'cyan');
    log(`https://polygonscan.com/tx/${tx.hash}`, 'yellow');

    log(`\n📋 Next steps:`, 'bright');
    log(`   1. Add liquidity: node scripts/add_liquidity.js`, 'cyan');
    log(`   2. Or check balance: cast balance ${tokenAddress} --rpc-url ${process.env.POLYGON_RPC_URL}`, 'cyan');

  } catch (error) {
    log('\n❌ Failed to mint tokens:', 'red');
    if (error.message.includes('Ownable')) {
      log('   Only the contract owner can mint tokens', 'yellow');
      log(`   Owner might be: ${await token.owner()}`, 'yellow');
      log(`   Your address: ${wallet.address}`, 'yellow');
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
