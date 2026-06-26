/**
 * Bridge Bitcoin to TBTC on Base Sepolia
 *
 * This script mints TBTC when Bitcoin is locked on testnet
 *
 * Usage: node scripts/bridge_btc_to_tbtc.js
 */

const { ethers } = require('ethers');
const fs = require('fs');
const path = require('path');
const readline = require('readline');
require('dotenv').config();

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

// TBTC ABI (minimal)
const TBTC_ABI = [
  'function mint(address to, uint256 amount, string calldata btcTxHash) external',
  'function balanceOf(address account) external view returns (uint256)',
  'function isProcessed(string calldata btcTxHash) external view returns (bool)',
  'function bridgeOperator() external view returns (address)',
  'function symbol() external view returns (string)',
  'function decimals() external view returns (uint8)'
];

// Create readline interface
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});

function question(prompt) {
  return new Promise((resolve) => {
    rl.question(prompt, resolve);
  });
}

async function main() {
  header('🌉 BRIDGE BITCOIN TO TBTC');

  // Read deployment info
  const deploymentPath = path.join(__dirname, '..', 'tbtc_base_sepolia_deployment.json');
  if (!fs.existsSync(deploymentPath)) {
    log('❌ tbtc_base_sepolia_deployment.json not found. Deploy TBTC first!', 'red');
    log('   Run: ./LAUNCH_TBTC.sh', 'yellow');
    rl.close();
    process.exit(1);
  }

  const deployment = JSON.parse(fs.readFileSync(deploymentPath, 'utf8'));
  const tbtcAddress = deployment.contracts.TBTC.address;

  log(`TBTC Contract: ${tbtcAddress}`, 'cyan');
  log(`Network: Base Sepolia (Chain ID: 84532)`, 'cyan');

  // Setup provider and wallet
  const BASE_SEPOLIA_RPC = process.env.BASE_SEPOLIA_RPC_URL || 'https://sepolia.base.org';
  const PRIVATE_KEY = process.env.BASE_SEPOLIA_PRIVATE_KEY || process.env.PRIVATE_KEY;

  if (!PRIVATE_KEY || PRIVATE_KEY.length !== 64) {
    log('❌ Private key not found in .env', 'red');
    rl.close();
    process.exit(1);
  }

  const provider = new ethers.JsonRpcProvider(BASE_SEPOLIA_RPC);
  const wallet = new ethers.Wallet('0x' + PRIVATE_KEY, provider);

  log(`\nYour wallet: ${wallet.address}`, 'cyan');

  // Connect to TBTC contract
  const tbtc = new ethers.Contract(tbtcAddress, TBTC_ABI, wallet);

  // Check bridge operator
  const operator = await tbtc.bridgeOperator();
  if (operator.toLowerCase() !== wallet.address.toLowerCase()) {
    log('\n⚠️  WARNING: You are not the bridge operator!', 'yellow');
    log(`   Bridge operator: ${operator}`, 'yellow');
    log(`   Your address: ${wallet.address}`, 'yellow');
    log('\n   Only the bridge operator can mint TBTC.', 'red');
    rl.close();
    return;
  }

  log('✅ You are the bridge operator!', 'green');

  header('ENTER BITCOIN TRANSACTION DETAILS');

  // Get Bitcoin TX hash
  log('Enter the Bitcoin testnet transaction hash:', 'cyan');
  log('(Example: 64a80c8...)', 'yellow');
  const btcTxHash = await question('Bitcoin TX Hash: ');

  if (!btcTxHash || btcTxHash.length < 10) {
    log('❌ Invalid Bitcoin transaction hash', 'red');
    rl.close();
    return;
  }

  // Check if already processed
  const isProcessed = await tbtc.isProcessed(btcTxHash);
  if (isProcessed) {
    log('\n✅ This Bitcoin transaction has already been bridged!', 'green');
    log('   No action needed.', 'yellow');
    rl.close();
    return;
  }

  // Get recipient address
  log('\nEnter recipient address for TBTC:', 'cyan');
  log('(Leave empty to use your address)', 'yellow');
  let recipient = await question(`Recipient [${wallet.address}]: `);

  if (!recipient || recipient.trim() === '') {
    recipient = wallet.address;
  }

  if (!ethers.isAddress(recipient)) {
    log('❌ Invalid Ethereum address', 'red');
    rl.close();
    return;
  }

  // Get amount
  log('\nEnter amount of BTC locked (in BTC, e.g., 0.01):', 'cyan');
  const amountStr = await question('Amount: ');

  let amount;
  try {
    amount = ethers.parseUnits(amountStr, 18); // TBTC has 18 decimals
  } catch (error) {
    log('❌ Invalid amount', 'red');
    rl.close();
    return;
  }

  header('MINTING TBTC');

  log(`\n📋 Bridge Details:`, 'bright');
  log(`   Bitcoin TX: ${btcTxHash}`, 'yellow');
  log(`   Recipient: ${recipient}`, 'yellow');
  log(`   Amount: ${amountStr} TBTC`, 'yellow');

  log('\nProcessing bridge transaction...', 'cyan');

  try {
    const tx = await tbtc.mint(recipient, amount, btcTxHash);
    log(`Transaction hash: ${tx.hash}`, 'yellow');
    log('Waiting for confirmation...', 'cyan');

    await tx.wait();
    log('✅ TBTC minted successfully!', 'green');

    // Check balance
    const balance = await tbtc.balanceOf(recipient);
    log(`\nRecipient balance: ${ethers.formatEther(balance)} TBTC`, 'green');

    header('✅ BRIDGE SUCCESSFUL!');

    log('Your Bitcoin has been bridged to Base Sepolia!', 'green');
    log(`\nView on BaseScan:`, 'cyan');
    log(`https://sepolia.basescan.org/tx/${tx.hash}`, 'yellow');

    log(`\nNext steps:`, 'bright');
    log(`  • Add TBTC to MetaMask`, 'cyan');
    log(`  • Trade on Uniswap`, 'cyan');
    log(`  • Burn to get Bitcoin back`, 'cyan');

    log('\n✨ Welcome to Base Sepolia! ✨\n', 'magenta');

  } catch (error) {
    log('\n❌ Bridge failed:', 'red');
    if (error.message.includes('Only bridge operator')) {
      log('   You are not the bridge operator', 'yellow');
    } else if (error.message.includes('already processed')) {
      log('   Transaction already bridged', 'yellow');
    } else if (error.message.includes('Exceeds max supply')) {
      log('   Cannot mint beyond max supply (1,000,000 TBTC)', 'yellow');
    } else {
      console.error(error);
    }
  }

  rl.close();
}

main().catch((error) => {
  log('\n❌ Error:', 'red');
  console.error(error);
  rl.close();
  process.exit(1);
});
