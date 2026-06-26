/**
 * Bridge WTBTC from Base Sepolia to Ethereum Mainnet
 *
 * This script processes your Base Sepolia transaction and mints equivalent tokens on Ethereum
 */

const { ethers } = require('ethers');
const fs = require('fs');
const path = require('path');
require('dotenv').config();

// Your Base Sepolia transaction
const BASE_TX_HASH = '0xab880c7a9bc9a6ef6110e093c2632dcc6144ccf53f86f0059542a28a4a0d78cc';
const BASE_AMOUNT = '1'; // 1 WTBTC transferred

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

// Bridge ABI (minimal)
const BRIDGE_ABI = [
  'function bridgeIn(address user, uint256 amount, string calldata baseTxHash) external',
  'function balanceOf(address account) external view returns (uint256)',
  'function isProcessed(string calldata baseTxHash) external view returns (bool)',
  'function bridgeOperator() external view returns (address)'
];

async function main() {
  header('🌉 BRIDGE WTBTC FROM BASE SEPOLIA TO ETHEREUM');

  // Read deployment info
  const deploymentPath = path.join(__dirname, '..', 'ethereum_bridge_deployment.json');
  if (!fs.existsSync(deploymentPath)) {
    log('❌ ethereum_bridge_deployment.json not found. Deploy bridge first!', 'red');
    log('   Run: ./deploy_ethereum_mainnet_bridge.sh', 'yellow');
    process.exit(1);
  }

  const deployment = JSON.parse(fs.readFileSync(deploymentPath, 'utf8'));
  const bridgeAddress = deployment.contracts.WTBTCBridge.address;

  log(`Bridge Address: ${bridgeAddress}`, 'cyan');
  log(`Base TX Hash: ${BASE_TX_HASH}`, 'cyan');

  // Setup provider and wallet
  const ETHEREUM_RPC = process.env.ETHEREUM_RPC_URL || `https://mainnet.infura.io/v3/5f5c1ddd0f2b469f83dc4b6a1cfa4057`;
  const provider = new ethers.JsonRpcProvider(ETHEREUM_RPC);
  const wallet = new ethers.Wallet('0x' + process.env.PRIVATE_KEY, provider);

  log(`\nYour wallet: ${wallet.address}`, 'cyan');

  // Connect to bridge
  const bridge = new ethers.Contract(bridgeAddress, BRIDGE_ABI, wallet);

  // Check if already processed
  const isProcessed = await bridge.isProcessed(BASE_TX_HASH);
  if (isProcessed) {
    log('\n✅ This transaction has already been bridged!', 'green');
    const balance = await bridge.balanceOf(wallet.address);
    log(`Your balance: ${ethers.formatEther(balance)} WTBTC`, 'green');
    return;
  }

  // Check bridge operator
  const operator = await bridge.bridgeOperator();
  if (operator.toLowerCase() !== wallet.address.toLowerCase()) {
    log('\n⚠️  WARNING: You are not the bridge operator!', 'yellow');
    log(`   Bridge operator: ${operator}`, 'yellow');
    log(`   Your address: ${wallet.address}`, 'yellow');
    log('\n   Only the bridge operator can process bridge transactions.', 'red');
    return;
  }

  header('PROCESSING BRIDGE TRANSACTION');

  log('Converting Base Sepolia WTBTC to Ethereum Mainnet WTBTC...', 'cyan');

  const amount = ethers.parseUnits(BASE_AMOUNT, 18);

  log(`\n📋 Bridge Details:`, 'bright');
  log(`   From: Base Sepolia`, 'yellow');
  log(`   To: Ethereum Mainnet`, 'yellow');
  log(`   Amount: ${BASE_AMOUNT} WTBTC`, 'yellow');
  log(`   User: ${wallet.address}`, 'yellow');

  try {
    const tx = await bridge.bridgeIn(wallet.address, amount, BASE_TX_HASH);
    log(`\nTransaction hash: ${tx.hash}`, 'yellow');
    log('Waiting for confirmation...', 'cyan');

    await tx.wait();
    log('✅ Bridge transaction confirmed!', 'green');

    // Check balance
    const balance = await bridge.balanceOf(wallet.address);
    log(`\nYour new balance: ${ethers.formatEther(balance)} WTBTC`, 'green');

    header('✅ BRIDGE SUCCESSFUL!');

    log('Your tokens have been bridged to Ethereum Mainnet!', 'green');
    log(`\nView on Etherscan:`, 'cyan');
    log(`https://etherscan.io/tx/${tx.hash}`, 'yellow');

    log(`\nNext steps:`, 'bright');
    log(`  1. Add liquidity on Uniswap`, 'cyan');
    log(`  2. List on DEX aggregators`, 'cyan');
    log(`  3. Add to MetaMask: ${bridgeAddress}`, 'cyan');

    log('\n✨ Welcome to Ethereum Mainnet! ✨\n', 'magenta');

  } catch (error) {
    log('\n❌ Bridge failed:', 'red');
    if (error.message.includes('Only bridge operator')) {
      log('   You are not the bridge operator', 'yellow');
    } else if (error.message.includes('already processed')) {
      log('   Transaction already bridged', 'yellow');
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
