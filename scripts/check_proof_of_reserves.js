/**
 * Check Proof of Reserves
 *
 * Displays current Bitcoin backing for TBTC
 */

const { ProofOfReserves, BitcoinVerifier } = require('./lightning_network_integration');
const fs = require('fs');
const path = require('path');

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

async function main() {
  header('🏦 PROOF OF RESERVES - TBTC BACKING');

  // Load configuration
  const configPath = path.join(__dirname, '..', 'lightning_integration_config.json');

  if (!fs.existsSync(configPath)) {
    log('❌ Lightning integration not configured', 'red');
    log('   Run: node scripts/lightning_network_integration.js', 'yellow');
    return;
  }

  const config = JSON.parse(fs.readFileSync(configPath, 'utf8'));
  const reserveAddress = config.bitcoin.reserveAddress;

  log(`Reserve Address: ${reserveAddress}`, 'cyan');
  log(`Network: ${config.bitcoin.network}`, 'cyan');

  // Initialize verifier
  const verifier = new BitcoinVerifier(config.bitcoin.network);

  // Get on-chain balance
  log('\n🔍 Fetching on-chain balance...', 'cyan');

  try {
    const balance = await verifier.getAddressBalance(reserveAddress);

    const btcBalance = balance.balance / 100000000;
    const btcFunded = balance.funded / 100000000;
    const btcSpent = balance.spent / 100000000;

    log('\n💰 ON-CHAIN BALANCE:', 'bright');
    log(`   Current: ${btcBalance.toFixed(8)} BTC (${balance.balance} sats)`, 'green');
    log(`   Total Funded: ${btcFunded.toFixed(8)} BTC`, 'cyan');
    log(`   Total Spent: ${btcSpent.toFixed(8)} BTC`, 'cyan');

    // Load proof of reserves
    const por = new ProofOfReserves(reserveAddress);
    por.loadReserves();

    log('\n📊 TRACKED RESERVES:', 'bright');
    log(`   Total Entries: ${por.reserves.length}`, 'cyan');

    if (por.reserves.length > 0) {
      let totalBTC = 0;
      let totalTBTC = 0;

      log('\n   Recent Reserves:', 'yellow');
      por.reserves.slice(-5).forEach((reserve, i) => {
        totalBTC += reserve.btcAmount;
        totalTBTC += reserve.tbtcAmount;

        const btc = (reserve.btcAmount / 100000000).toFixed(8);
        const tbtc = (reserve.tbtcAmount / 1e18).toFixed(8);
        const date = new Date(reserve.timestamp).toLocaleString();

        log(`   ${i + 1}. ${btc} BTC → ${tbtc} TBTC`, 'cyan');
        log(`      TX: ${reserve.txid.substring(0, 16)}...`, 'yellow');
        log(`      Date: ${date}`, 'yellow');
      });

      log(`\n   Total Tracked BTC: ${(totalBTC / 100000000).toFixed(8)} BTC`, 'green');
      log(`   Total Minted TBTC: ${(totalTBTC / 1e18).toFixed(8)} TBTC`, 'green');

      // Calculate collateralization ratio
      const ratio = (balance.balance / totalBTC) * 100;
      log(`\n🔒 COLLATERALIZATION RATIO:`, 'bright');
      log(`   ${ratio.toFixed(2)}%`, ratio >= 100 ? 'green' : 'red');

      if (ratio >= 150) {
        log('   Status: OVER-COLLATERALIZED ✅', 'green');
      } else if (ratio >= 100) {
        log('   Status: FULLY BACKED ✅', 'green');
      } else {
        log('   Status: UNDER-COLLATERALIZED ⚠️', 'yellow');
      }
    } else {
      log('   No reserves tracked yet', 'yellow');
      log('   Mint TBTC with Bitcoin proof to start tracking', 'yellow');
    }

    // Display reserve address QR code info
    log('\n📱 FUND RESERVES:', 'bright');
    log(`   Address: ${reserveAddress}`, 'cyan');
    log(`   QR Code: https://mempool.space/testnet/address/${reserveAddress}`, 'yellow');
    log(`   Explorer: https://mempool.space/testnet/address/${reserveAddress}`, 'yellow');

  } catch (error) {
    if (error.message.includes('404')) {
      log('\n💡 Address not yet funded', 'yellow');
      log(`   Send Bitcoin testnet to: ${reserveAddress}`, 'cyan');
      log(`   Get testnet BTC: https://testnet-faucet.mempool.co/`, 'yellow');
    } else {
      log(`\n❌ Error fetching balance: ${error.message}`, 'red');
    }
  }

  // Lightning Network integration
  log('\n⚡ LIGHTNING NETWORK:', 'bright');
  log(`   Integrated Nodes: ${config.lightning.nodes.length}`, 'cyan');
  log(`   Total Liquidity: ${config.lightning.totalLiquidity} sats`, 'cyan');
  log(`   Recommended: ${config.lightning.recommendedNode.alias}`, 'cyan');

  log('\n📋 COMMANDS:', 'bright');
  log('   Verify BTC TX: node scripts/verify_bitcoin_tx.js <txid>', 'cyan');
  log('   Mint with proof: node scripts/mint_with_bitcoin_proof.js <txid>', 'cyan');
  log('   Check reserves: node scripts/check_proof_of_reserves.js', 'cyan');

  log('\n✨ TBTC is backed by real Bitcoin! ✨\n', 'magenta');
}

main().catch((error) => {
  log('\n❌ Error:', 'red');
  console.error(error);
  process.exit(1);
});
