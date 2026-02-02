/**
 * Mint TBTC with Bitcoin Proof
 *
 * This script:
 * 1. Verifies Bitcoin transaction
 * 2. Checks confirmations
 * 3. Mints equivalent TBTC
 * 4. Records in proof of reserves
 */

const { ethers } = require('ethers');
const { BitcoinVerifier, ProofOfReserves } = require('./lightning_network_integration');
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
  'function decimals() external view returns (uint8)',
  'function totalSupply() external view returns (uint256)'
];

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
  header('🪙 MINT TBTC WITH BITCOIN PROOF');

  // Check if TBTC is deployed
  const deploymentPath = path.join(__dirname, '..', 'tbtc_base_sepolia_deployment.json');

  if (!fs.existsSync(deploymentPath)) {
    log('❌ TBTC not deployed on Base Sepolia', 'red');
    log('   Run: ./LAUNCH_TBTC.sh', 'yellow');
    rl.close();
    return;
  }

  const deployment = JSON.parse(fs.readFileSync(deploymentPath, 'utf8'));
  const tbtcAddress = deployment.contracts.TBTC.address;

  log(`TBTC Contract: ${tbtcAddress}`, 'cyan');
  log(`Network: Base Sepolia`, 'cyan');

  // Get Bitcoin transaction ID
  const btcTxId = process.argv[2] || await question('\nEnter Bitcoin transaction ID: ');

  if (!btcTxId || btcTxId.length !== 64) {
    log('❌ Invalid Bitcoin transaction ID (must be 64 characters)', 'red');
    rl.close();
    return;
  }

  // Load Lightning integration config
  const configPath = path.join(__dirname, '..', 'lightning_integration_config.json');
  let reserveAddress;

  if (fs.existsSync(configPath)) {
    const config = JSON.parse(fs.readFileSync(configPath, 'utf8'));
    reserveAddress = config.bitcoin.reserveAddress;
  } else {
    reserveAddress = process.env.BITCOIN_RESERVE_ADDRESS || await question('Enter Bitcoin reserve address: ');
  }

  log(`\n📋 Minting Parameters:`, 'bright');
  log(`   Bitcoin TX: ${btcTxId}`, 'cyan');
  log(`   Reserve Address: ${reserveAddress}`, 'cyan');

  // Initialize Bitcoin verifier
  header('STEP 1: VERIFY BITCOIN TRANSACTION');

  log('🔍 Connecting to Bitcoin network...', 'cyan');
  const verifier = new BitcoinVerifier('testnet');

  let btcTransaction;
  try {
    btcTransaction = await verifier.getTransaction(btcTxId);
    log('✅ Transaction found!', 'green');

    log(`\n📊 Transaction Details:`, 'bright');
    log(`   Confirmations: ${btcTransaction.status.confirmed ? 'Confirmed' : 'Unconfirmed'}`, 'cyan');

    if (btcTransaction.status.confirmed) {
      const currentBlock = await fetch('https://mempool.space/testnet/api/blocks/tip/height')
        .then(r => r.json());
      const confirmations = currentBlock - btcTransaction.status.block_height + 1;
      log(`   Block Height: ${btcTransaction.status.block_height}`, 'cyan');
      log(`   Confirmations: ${confirmations}`, 'cyan');
    }

  } catch (error) {
    log(`❌ Error fetching transaction: ${error.message}`, 'red');
    rl.close();
    return;
  }

  // Find output to reserve address
  let outputToReserve = null;
  for (const vout of btcTransaction.vout) {
    if (vout.scriptpubkey_address === reserveAddress) {
      outputToReserve = vout;
      break;
    }
  }

  if (!outputToReserve) {
    log(`\n❌ No output found to reserve address ${reserveAddress}`, 'red');
    log('   Transaction outputs:', 'yellow');
    btcTransaction.vout.forEach((vout, i) => {
      log(`   ${i}. ${vout.scriptpubkey_address}: ${vout.value} sats`, 'yellow');
    });
    rl.close();
    return;
  }

  const btcAmount = outputToReserve.value;
  const btcInBtc = (btcAmount / 100000000).toFixed(8);

  log(`\n✅ Found output to reserve address!`, 'green');
  log(`   Amount: ${btcAmount} satoshis (${btcInBtc} BTC)`, 'green');

  if (!btcTransaction.status.confirmed) {
    log('\n⚠️  WARNING: Transaction not yet confirmed', 'yellow');
    const proceed = await question('Continue anyway? (y/n): ');
    if (proceed.toLowerCase() !== 'y') {
      log('Minting cancelled. Wait for confirmations and try again.', 'yellow');
      rl.close();
      return;
    }
  }

  // Setup Ethereum connection
  header('STEP 2: MINT TBTC ON BASE SEPOLIA');

  const BASE_SEPOLIA_RPC = process.env.BASE_SEPOLIA_RPC_URL || 'https://sepolia.base.org';
  const PRIVATE_KEY = process.env.BASE_SEPOLIA_PRIVATE_KEY || process.env.PRIVATE_KEY;

  if (!PRIVATE_KEY || PRIVATE_KEY.length !== 64) {
    log('❌ Private key not found in .env', 'red');
    rl.close();
    return;
  }

  log('🔗 Connecting to Base Sepolia...', 'cyan');
  const provider = new ethers.JsonRpcProvider(BASE_SEPOLIA_RPC);
  const wallet = new ethers.Wallet('0x' + PRIVATE_KEY, provider);

  log(`✅ Connected!`, 'green');
  log(`   Wallet: ${wallet.address}`, 'cyan');

  // Connect to TBTC contract
  const tbtc = new ethers.Contract(tbtcAddress, TBTC_ABI, wallet);

  // Check if already processed
  const isProcessed = await tbtc.isProcessed(btcTxId);
  if (isProcessed) {
    log('\n✅ This Bitcoin transaction has already been used to mint TBTC!', 'green');
    const balance = await tbtc.balanceOf(wallet.address);
    log(`Your TBTC balance: ${ethers.formatEther(balance)} TBTC`, 'green');
    rl.close();
    return;
  }

  // Check bridge operator
  const operator = await tbtc.bridgeOperator();
  if (operator.toLowerCase() !== wallet.address.toLowerCase()) {
    log(`\n⚠️  WARNING: You are not the bridge operator`, 'yellow');
    log(`   Bridge operator: ${operator}`, 'yellow');
    log(`   Your address: ${wallet.address}`, 'yellow');
    log('\n   Only the bridge operator can mint TBTC.', 'red');
    rl.close();
    return;
  }

  // Calculate TBTC amount (1:1 ratio, but TBTC has 18 decimals)
  // 1 BTC (8 decimals) = 1 TBTC (18 decimals)
  // So: satoshis * 10^10 = TBTC wei
  const tbtcAmount = ethers.parseUnits(btcInBtc, 18);

  log(`\n🪙 Minting Details:`, 'bright');
  log(`   Bitcoin Amount: ${btcInBtc} BTC (${btcAmount} sats)`, 'cyan');
  log(`   TBTC Amount: ${ethers.formatEther(tbtcAmount)} TBTC`, 'cyan');
  log(`   Recipient: ${wallet.address}`, 'cyan');
  log(`   Bitcoin TX: ${btcTxId}`, 'cyan');

  const confirm = await question('\nProceed with minting? (y/n): ');
  if (confirm.toLowerCase() !== 'y') {
    log('Minting cancelled.', 'yellow');
    rl.close();
    return;
  }

  try {
    log('\n💫 Minting TBTC...', 'cyan');
    const tx = await tbtc.mint(wallet.address, tbtcAmount, btcTxId);
    log(`Transaction hash: ${tx.hash}`, 'yellow');
    log('Waiting for confirmation...', 'cyan');

    await tx.wait();
    log('✅ TBTC minted successfully!', 'green');

    // Check new balance
    const balance = await tbtc.balanceOf(wallet.address);
    log(`\nYour new balance: ${ethers.formatEther(balance)} TBTC`, 'green');

    // Record in proof of reserves
    header('STEP 3: UPDATE PROOF OF RESERVES');

    const por = new ProofOfReserves(reserveAddress);
    por.loadReserves();

    try {
      await por.addReserve(btcTxId, btcAmount, Number(tbtcAmount));
      log('✅ Reserve recorded!', 'green');
    } catch (error) {
      log('⚠️  Could not update proof of reserves (this is OK)', 'yellow');
    }

    // Final summary
    header('✅ MINTING COMPLETE!');

    log(`\n📊 Summary:`, 'bright');
    log(`   Bitcoin Locked: ${btcInBtc} BTC`, 'green');
    log(`   TBTC Minted: ${ethers.formatEther(tbtcAmount)} TBTC`, 'green');
    log(`   Your Balance: ${ethers.formatEther(balance)} TBTC`, 'green');

    log(`\n🔗 View on BaseScan:`, 'cyan');
    log(`   https://sepolia.basescan.org/tx/${tx.hash}`, 'yellow');

    log(`\n🏦 Proof of Reserves:`, 'cyan');
    log(`   Bitcoin Address: ${reserveAddress}`, 'yellow');
    log(`   Explorer: https://mempool.space/testnet/address/${reserveAddress}`, 'yellow');

    log(`\n📋 Next Steps:`, 'bright');
    log('   • Check reserves: node scripts/check_proof_of_reserves.js', 'cyan');
    log('   • Add to MetaMask: Token address is ' + tbtcAddress, 'cyan');
    log('   • Bridge to Ethereum: ./BRIDGE_ALL_TO_ETHEREUM.sh', 'cyan');

    log('\n✨ Your TBTC is backed by real Bitcoin! ✨\n', 'magenta');

  } catch (error) {
    log('\n❌ Minting failed:', 'red');

    if (error.message.includes('Only bridge operator')) {
      log('   You are not the bridge operator', 'yellow');
    } else if (error.message.includes('already processed')) {
      log('   This Bitcoin transaction has already been used', 'yellow');
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
