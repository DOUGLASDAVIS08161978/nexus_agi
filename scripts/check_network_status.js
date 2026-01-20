/**
 * Network Status Checker
 * Checks connection status and balances across all configured networks
 *
 * Run with: node scripts/check_network_status.js
 */

const https = require('https');
require('dotenv').config();

const RECIPIENT_ADDRESS = process.env.RECIPIENT_ADDRESS || '0x24f6b1ce11c57d40b542f91ac85fa9eb61f78771';

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

/**
 * Make JSON-RPC call
 */
function rpcCall(url, method, params = []) {
  return new Promise((resolve, reject) => {
    const data = JSON.stringify({
      jsonrpc: '2.0',
      method: method,
      params: params,
      id: 1
    });

    const urlObj = new URL(url);
    const options = {
      hostname: urlObj.hostname,
      port: urlObj.port || 443,
      path: urlObj.pathname + urlObj.search,
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Content-Length': data.length
      }
    };

    const req = https.request(options, (res) => {
      let responseData = '';

      res.on('data', (chunk) => {
        responseData += chunk;
      });

      res.on('end', () => {
        try {
          const json = JSON.parse(responseData);
          resolve(json);
        } catch (error) {
          reject(new Error('Failed to parse response: ' + error.message));
        }
      });
    });

    req.on('error', (error) => {
      reject(error);
    });

    req.write(data);
    req.end();
  });
}

/**
 * Get balance for address
 */
async function getBalance(url, address) {
  try {
    const response = await rpcCall(url, 'eth_getBalance', [address, 'latest']);
    if (response.result) {
      const balanceWei = BigInt(response.result);
      const balanceEther = Number(balanceWei) / 1e18;
      return { success: true, balance: balanceEther, balanceWei: balanceWei.toString() };
    }
    return { success: false, error: 'No result in response' };
  } catch (error) {
    return { success: false, error: error.message };
  }
}

/**
 * Get latest block number
 */
async function getBlockNumber(url) {
  try {
    const response = await rpcCall(url, 'eth_blockNumber', []);
    if (response.result) {
      return { success: true, blockNumber: parseInt(response.result, 16) };
    }
    return { success: false, error: 'No result in response' };
  } catch (error) {
    return { success: false, error: error.message };
  }
}

/**
 * Check network status
 */
async function checkNetwork(name, rpcUrl, nativeToken) {
  log(`\n🔍 Checking ${name}...`, 'cyan');
  log(`   RPC: ${rpcUrl}`);

  // Check connection
  const blockResult = await getBlockNumber(rpcUrl);
  if (!blockResult.success) {
    log(`   ❌ Connection failed: ${blockResult.error}`, 'red');
    return false;
  }

  log(`   ✅ Connected! Latest block: ${blockResult.blockNumber.toLocaleString()}`, 'green');

  // Check balance
  const balanceResult = await getBalance(rpcUrl, RECIPIENT_ADDRESS);
  if (!balanceResult.success) {
    log(`   ⚠️  Failed to check balance: ${balanceResult.error}`, 'yellow');
    return true;
  }

  if (balanceResult.balance > 0) {
    log(`   💰 Balance: ${balanceResult.balance.toFixed(6)} ${nativeToken}`, 'green');
  } else {
    log(`   ⚠️  Balance: 0 ${nativeToken} (Need gas for deployment!)`, 'yellow');
  }

  return true;
}

/**
 * Main function
 */
async function main() {
  header('🌐 NETWORK STATUS CHECKER');

  log('📋 Configuration:', 'bright');
  log(`   Recipient Address: ${RECIPIENT_ADDRESS}`, 'cyan');
  log(`   Alchemy API Key: ${process.env.ALCHEMY_API_KEY ? '✅ Set' : '❌ Not set'}`,
      process.env.ALCHEMY_API_KEY ? 'green' : 'red');
  log(`   Infura API Key: ${process.env.INFURA_API_KEY ? '✅ Set' : '❌ Not set'}`,
      process.env.INFURA_API_KEY ? 'green' : 'red');
  log(`   Private Key: ${process.env.PRIVATE_KEY && process.env.PRIVATE_KEY !== 'your_private_key_here_without_0x_prefix' ? '✅ Set' : '❌ Not set'}`,
      process.env.PRIVATE_KEY && process.env.PRIVATE_KEY !== 'your_private_key_here_without_0x_prefix' ? 'green' : 'red');

  // Check each network
  const networks = [
    {
      name: 'Ethereum Sepolia (Testnet)',
      rpcUrl: process.env.SEPOLIA_RPC_URL || 'https://eth-sepolia.g.alchemy.com/v2/nF8V5Ycxcvl6zfy0NGPZF',
      nativeToken: 'ETH'
    },
    {
      name: 'Polygon Mainnet',
      rpcUrl: process.env.POLYGON_RPC_URL || 'https://polygon-mainnet.infura.io/v3/38f2c0df20264c98b108d04914464e12',
      nativeToken: 'MATIC'
    },
    {
      name: 'Polygon Mumbai (Testnet)',
      rpcUrl: process.env.POLYGON_MUMBAI_RPC_URL || 'https://polygon-mumbai.infura.io/v3/38f2c0df20264c98b108d04914464e12',
      nativeToken: 'MATIC'
    },
    {
      name: 'Ethereum Mainnet',
      rpcUrl: process.env.MAINNET_RPC_URL || 'https://eth-mainnet.g.alchemy.com/v2/nF8V5Ycxcvl6zfy0NGPZF',
      nativeToken: 'ETH'
    }
  ];

  let allConnected = true;
  for (const network of networks) {
    const connected = await checkNetwork(network.name, network.rpcUrl, network.nativeToken);
    if (!connected) allConnected = false;
  }

  header(allConnected ? '✅ ALL NETWORKS CONNECTED!' : '⚠️ SOME NETWORKS FAILED');

  log('\n📚 NEXT STEPS:\n', 'bright');

  if (!process.env.PRIVATE_KEY || process.env.PRIVATE_KEY === 'your_private_key_here_without_0x_prefix') {
    log('   1. ❌ Add your private key to .env file', 'red');
    log('      Edit: /home/user/nexus_agi/.env', 'yellow');
    log('      Line: PRIVATE_KEY=your_64_character_key', 'yellow');
  } else {
    log('   1. ✅ Private key is set', 'green');
  }

  log('\n   2. Get testnet tokens for gas:', 'cyan');
  log('      Sepolia ETH: https://www.alchemy.com/faucets/ethereum-sepolia', 'cyan');
  log('      Mumbai MATIC: https://faucet.polygon.technology/', 'cyan');

  log('\n   3. Deploy to testnet:', 'cyan');
  log('      cd /home/user/nexus_agi', 'yellow');
  log('      npx hardhat run scripts/deploy_testnet_bridge.js --network sepolia', 'yellow');

  log('\n   4. Or deploy to Polygon:', 'cyan');
  log('      npx hardhat run scripts/deploy_testnet_bridge.js --network polygon', 'yellow');
  log('      (Requires MATIC for gas)\n', 'red');
}

main().catch((error) => {
  log('\n❌ Error: ' + error.message, 'red');
  console.error(error);
  process.exit(1);
});
