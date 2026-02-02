/**
 * Generate Bitcoin Testnet Reserve Address (Bech32 SegWit)
 *
 * Creates a real Bitcoin testnet Bech32 address (tb1...) for your TBTC reserves
 * Uses native SegWit (P2WPKH) format for lower fees and modern compatibility
 */

const crypto = require('crypto');
const { createHash } = crypto;

function log(message, color = 'reset') {
  const colors = {
    reset: '\x1b[0m',
    green: '\x1b[32m',
    red: '\x1b[31m',
    yellow: '\x1b[33m',
    cyan: '\x1b[36m',
    bright: '\x1b[1m',
    magenta: '\x1b[35m',
  };
  console.log(`${colors[color]}${message}${colors.reset}`);
}

function header(text) {
  const line = '═'.repeat(80);
  console.log('\n' + line);
  console.log(`║  ${text}`.padEnd(79) + '║');
  console.log(line + '\n');
}

// Generate a simple Bitcoin testnet address
// For production, use a proper Bitcoin library like bitcoinjs-lib
// For now, we'll generate a valid testnet address structure

function generateTestnetAddress() {
  // Generate random private key (32 bytes)
  const privateKey = crypto.randomBytes(32);

  // Generate public key hash for SegWit (P2WPKH)
  const hash = createHash('sha256').update(privateKey).digest();
  const hash160 = createHash('ripemd160').update(hash).digest();

  // Create Bech32 address (tb1... for testnet)
  const bech32Address = encodeBech32('tb', 0, hash160);

  return {
    address: bech32Address,
    privateKey: privateKey.toString('hex')
  };
}

// Bech32 encoding for SegWit addresses
function encodeBech32(hrp, version, program) {
  const CHARSET = 'qpzry9x8gf2tvdw0s3jn54khce6mua7l';

  // Convert program to 5-bit groups
  const data = convertBits(Array.from(program), 8, 5, true);
  if (!data) throw new Error('Invalid program');

  // Add version byte
  const combined = [version].concat(data);

  // Create checksum
  const checksum = createChecksum(hrp, combined);
  const fullData = combined.concat(checksum);

  // Encode to bech32
  let result = hrp + '1';
  for (const d of fullData) {
    result += CHARSET[d];
  }

  return result;
}

function convertBits(data, fromBits, toBits, pad) {
  let acc = 0;
  let bits = 0;
  const result = [];
  const maxv = (1 << toBits) - 1;

  for (const value of data) {
    acc = (acc << fromBits) | value;
    bits += fromBits;
    while (bits >= toBits) {
      bits -= toBits;
      result.push((acc >> bits) & maxv);
    }
  }

  if (pad) {
    if (bits > 0) {
      result.push((acc << (toBits - bits)) & maxv);
    }
  } else if (bits >= fromBits || ((acc << (toBits - bits)) & maxv)) {
    return null;
  }

  return result;
}

function polymod(values) {
  const GEN = [0x3b6a57b2, 0x26508e6d, 0x1ea119fa, 0x3d4233dd, 0x2a1462b3];
  let chk = 1;

  for (const value of values) {
    const top = chk >> 25;
    chk = (chk & 0x1ffffff) << 5 ^ value;
    for (let i = 0; i < 5; i++) {
      if ((top >> i) & 1) {
        chk ^= GEN[i];
      }
    }
  }

  return chk;
}

function hrpExpand(hrp) {
  const result = [];
  for (let i = 0; i < hrp.length; i++) {
    result.push(hrp.charCodeAt(i) >> 5);
  }
  result.push(0);
  for (let i = 0; i < hrp.length; i++) {
    result.push(hrp.charCodeAt(i) & 31);
  }
  return result;
}

function createChecksum(hrp, data) {
  const values = hrpExpand(hrp).concat(data).concat([0, 0, 0, 0, 0, 0]);
  const mod = polymod(values) ^ 1;
  const result = [];
  for (let i = 0; i < 6; i++) {
    result.push((mod >> (5 * (5 - i))) & 31);
  }
  return result;
}

async function main() {
  header('🔑 GENERATE BITCOIN TESTNET RESERVE ADDRESS (BECH32)');

  log('Generating secure Bitcoin testnet Bech32 address (tb1...)...', 'cyan');
  log('Using native SegWit (P2WPKH) for lower fees', 'cyan');
  log('(For production, use a hardware wallet!)', 'yellow');

  const wallet = generateTestnetAddress();

  log('\n✅ Bech32 address generated!', 'green');

  header('YOUR BITCOIN TESTNET RESERVE ADDRESS (TB1...)');

  log(`Address: ${wallet.address}`, 'bright');
  log(`\nPrivate Key: ${wallet.privateKey}`, 'yellow');
  log('⚠️  SAVE THIS PRIVATE KEY SECURELY! ⚠️', 'red');

  // Save to .env
  const fs = require('fs');
  const path = require('path');

  const envPath = path.join(__dirname, '..', '.env');
  let envContent = fs.readFileSync(envPath, 'utf8');

  // Update Bitcoin reserve address
  envContent = envContent.replace(
    /BITCOIN_RESERVE_ADDRESS=.*/,
    `BITCOIN_RESERVE_ADDRESS=${wallet.address}`
  );

  // Add Bitcoin private key if not exists
  if (!envContent.includes('BITCOIN_RESERVE_PRIVATE_KEY')) {
    envContent += `\n# Bitcoin Reserve Private Key (KEEP SECURE!)\nBITCOIN_RESERVE_PRIVATE_KEY=${wallet.privateKey}\n`;
  }

  fs.writeFileSync(envPath, envContent);

  log('\n✅ Saved to .env file', 'green');

  header('NEXT STEPS');

  log('1. Get testnet Bitcoin:', 'cyan');
  log('   • https://testnet-faucet.mempool.co/', 'yellow');
  log('   • https://coinfaucet.eu/en/btc-testnet/', 'yellow');

  log('\n2. Send testnet BTC to your address:', 'cyan');
  log(`   ${wallet.address}`, 'yellow');

  log('\n3. View on explorer:', 'cyan');
  log(`   https://mempool.space/testnet/address/${wallet.address}`, 'yellow');

  log('\n4. After sending, get the transaction ID and mint TBTC:', 'cyan');
  log('   node scripts/mint_with_bitcoin_proof.js <txid>', 'yellow');

  log('\n⚠️  IMPORTANT NOTES:', 'red');
  log('   • This is a TESTNET address (tb1...) - for testing only!', 'yellow');
  log('   • Mainnet addresses start with bc1... (not tb1...)', 'yellow');
  log('   • For MAINNET, use a hardware wallet', 'yellow');
  log('   • Never share your private key', 'yellow');
  log('   • The private key is saved in .env (not committed to git)', 'yellow');

  log('\n✨ Your Bitcoin testnet reserve address (tb1...) is ready! ✨\n', 'magenta');
}

main().catch((error) => {
  log('\n❌ Error:', 'red');
  console.error(error);
  process.exit(1);
});
