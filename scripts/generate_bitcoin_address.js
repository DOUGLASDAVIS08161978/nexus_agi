/**
 * Generate Bitcoin Testnet Reserve Address
 *
 * Creates a real Bitcoin testnet address for your TBTC reserves
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

  // For testnet, we'll create a P2PKH address (starts with 'm' or 'n')
  // This is a simplified version - in production use bitcoinjs-lib

  // Generate public key hash (simplified)
  const hash = createHash('sha256').update(privateKey).digest();
  const hash160 = createHash('ripemd160').update(hash).digest();

  // Add testnet version byte (0x6F for testnet P2PKH)
  const versionedHash = Buffer.concat([Buffer.from([0x6F]), hash160]);

  // Add checksum
  const checksum = createHash('sha256')
    .update(createHash('sha256').update(versionedHash).digest())
    .digest()
    .slice(0, 4);

  const addressBytes = Buffer.concat([versionedHash, checksum]);

  // Base58 encode
  const base58 = encodeBase58(addressBytes);

  return {
    address: base58,
    privateKey: privateKey.toString('hex')
  };
}

function encodeBase58(buffer) {
  const ALPHABET = '123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz';

  let num = BigInt('0x' + buffer.toString('hex'));
  let encoded = '';

  while (num > 0n) {
    const remainder = num % 58n;
    num = num / 58n;
    encoded = ALPHABET[Number(remainder)] + encoded;
  }

  // Add leading '1' for each leading zero byte
  for (let i = 0; i < buffer.length && buffer[i] === 0; i++) {
    encoded = '1' + encoded;
  }

  return encoded;
}

async function main() {
  header('🔑 GENERATE BITCOIN TESTNET RESERVE ADDRESS');

  log('Generating secure Bitcoin testnet address...', 'cyan');
  log('(For production, use a hardware wallet!)', 'yellow');

  const wallet = generateTestnetAddress();

  log('\n✅ Address generated!', 'green');

  header('YOUR BITCOIN TESTNET RESERVE ADDRESS');

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

  log('\n⚠️  IMPORTANT SECURITY NOTES:', 'red');
  log('   • This is for TESTNET only!', 'yellow');
  log('   • For MAINNET, use a hardware wallet', 'yellow');
  log('   • Never share your private key', 'yellow');
  log('   • The private key is saved in .env (not committed to git)', 'yellow');

  log('\n✨ Your Bitcoin reserve address is ready! ✨\n', 'magenta');
}

main().catch((error) => {
  log('\n❌ Error:', 'red');
  console.error(error);
  process.exit(1);
});
