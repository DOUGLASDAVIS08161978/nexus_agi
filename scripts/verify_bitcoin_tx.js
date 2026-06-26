/**
 * Verify Bitcoin Transaction
 *
 * Verifies that Bitcoin has been locked before minting TBTC
 */

const { BitcoinVerifier } = require('./lightning_network_integration');
const readline = require('readline');

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
  const line = '═'.repeat(80);
  console.log('\n' + line);
  console.log('║  🔍 VERIFY BITCOIN TRANSACTION'.padEnd(79) + '║');
  console.log(line + '\n');

  // Get transaction ID
  const txid = process.argv[2] || await question('Enter Bitcoin transaction ID: ');

  if (!txid || txid.length !== 64) {
    log('❌ Invalid transaction ID (must be 64 characters)', 'red');
    rl.close();
    return;
  }

  // Get expected address
  const address = process.argv[3] || await question('Enter expected Bitcoin address: ');

  if (!address) {
    log('❌ Address is required', 'red');
    rl.close();
    return;
  }

  // Get expected amount
  const amountStr = process.argv[4] || await question('Enter expected amount (in satoshis): ');
  const amount = parseInt(amountStr);

  if (isNaN(amount) || amount <= 0) {
    log('❌ Invalid amount', 'red');
    rl.close();
    return;
  }

  log(`\n📋 Verification Parameters:`, 'bright');
  log(`   Transaction: ${txid}`, 'cyan');
  log(`   Address: ${address}`, 'cyan');
  log(`   Amount: ${amount} satoshis (${(amount / 100000000).toFixed(8)} BTC)`, 'cyan');

  // Initialize verifier
  log('\n🔗 Connecting to Bitcoin network...', 'cyan');
  const verifier = new BitcoinVerifier('testnet');

  try {
    const result = await verifier.verifyTransaction(txid, address, amount);

    if (result.verified) {
      log('\n✅ VERIFICATION SUCCESSFUL!', 'green');
      log(`   Amount: ${result.amount} satoshis`, 'green');
      log(`   Confirmations: ${result.confirmations}`, 'green');
      log(`   Timestamp: ${new Date(result.timestamp * 1000).toISOString()}`, 'green');

      log('\n🪙 You can now mint TBTC:', 'bright');
      log(`   Amount: ${(result.amount / 100000000).toFixed(8)} TBTC`, 'cyan');
      log(`   TX Hash: ${result.txid}`, 'cyan');

      log('\n📝 Next step:', 'bright');
      log(`   node scripts/mint_with_bitcoin_proof.js ${txid}`, 'yellow');
    } else {
      log('\n❌ VERIFICATION FAILED', 'red');
      log(`   Reason: ${result.reason}`, 'yellow');

      if (result.confirmations === 0) {
        log('\n💡 Tip: Wait for confirmations and try again', 'cyan');
      }
    }
  } catch (error) {
    log('\n❌ Error:', 'red');
    log(`   ${error.message}`, 'yellow');
  }

  rl.close();
}

main().catch((error) => {
  log('\n❌ Error:', 'red');
  console.error(error);
  rl.close();
  process.exit(1);
});
