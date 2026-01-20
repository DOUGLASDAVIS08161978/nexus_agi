/**
 * Bridge Interaction Script with Nexus AGI Directory Integration
 * Allows interaction with deployed bridge contracts
 * Integrates with 133+ APIs from Nexus AGI Directory
 *
 * Run with: node scripts/interact_bridge.js
 */

const https = require('https');
const fs = require('fs');
require('dotenv').config();

// Nexus AGI Directory URL
const NEXUS_AGI_DIRECTORY_URL = 'nexus-agi.com/.well-known/seeds-public.json';
const RECIPIENT_ADDRESS = process.env.RECIPIENT_ADDRESS || '0x24f6b1ce11c57d40b542f91ac85fa9eb61f78771';

// ANSI color codes for terminal output
const colors = {
  reset: '\x1b[0m',
  bright: '\x1b[1m',
  green: '\x1b[32m',
  blue: '\x1b[34m',
  yellow: '\x1b[33m',
  red: '\x1b[31m',
  cyan: '\x1b[36m',
  magenta: '\x1b[35m'
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

function subheader(text) {
  const line = '─'.repeat(80);
  console.log('\n' + line);
  console.log(`  ${text}`);
  console.log(line + '\n');
}

/**
 * Fetch Nexus AGI Directory
 */
async function fetchNexusAGIDirectory() {
  return new Promise((resolve, reject) => {
    https.get(`https://${NEXUS_AGI_DIRECTORY_URL}`, (res) => {
      let data = '';

      res.on('data', (chunk) => {
        data += chunk;
      });

      res.on('end', () => {
        try {
          const apis = JSON.parse(data);
          resolve(apis);
        } catch (error) {
          reject(new Error('Failed to parse API directory: ' + error.message));
        }
      });
    }).on('error', (error) => {
      reject(new Error('Failed to fetch API directory: ' + error.message));
    });
  });
}

/**
 * Display available blockchain and crypto APIs
 */
function displayBlockchainAPIs(apis) {
  subheader('AVAILABLE BLOCKCHAIN & CRYPTO APIs FROM NEXUS AGI DIRECTORY');

  const blockchainKeywords = ['blockchain', 'crypto', 'ethereum', 'bitcoin', 'web3', 'defi', 'nft'];
  const relevantAPIs = apis.filter(api => {
    const searchText = `${api.name} ${api.id} ${(api.capabilities || []).join(' ')}`.toLowerCase();
    return blockchainKeywords.some(keyword => searchText.includes(keyword));
  });

  if (relevantAPIs.length === 0) {
    log('ℹ️  No specific blockchain APIs found. Showing all available APIs...', 'yellow');
    log(`\n📊 Total APIs in directory: ${apis.length}`, 'cyan');

    // Show first 10 APIs as examples
    log('\n🔍 Sample APIs (first 10):', 'bright');
    apis.slice(0, 10).forEach((api, index) => {
      log(`\n${index + 1}. ${api.name}`, 'green');
      log(`   ID: ${api.id}`, 'cyan');
      log(`   Endpoint: ${api.endpoint}`, 'blue');
      if (api.capabilities) {
        log(`   Capabilities: ${api.capabilities.join(', ')}`, 'yellow');
      }
      if (api.status) {
        log(`   Status: ${api.status}`, 'magenta');
      }
    });
  } else {
    log(`📊 Found ${relevantAPIs.length} blockchain-related APIs\n`, 'cyan');

    relevantAPIs.forEach((api, index) => {
      log(`${index + 1}. ${api.name}`, 'green');
      log(`   ID: ${api.id}`, 'cyan');
      log(`   Endpoint: ${api.endpoint}`, 'blue');
      if (api.capabilities) {
        log(`   Capabilities: ${api.capabilities.join(', ')}`, 'yellow');
      }
      if (api.status) {
        log(`   Status: ${api.status}`, 'magenta');
      }
      console.log('');
    });
  }
}

/**
 * Display bridge contract information
 */
function displayBridgeInfo() {
  subheader('BRIDGE CONTRACT INFORMATION');

  // Try to load latest deployment info
  const deploymentFiles = fs.readdirSync('.').filter(f => f.startsWith('deployment_sepolia_'));

  if (deploymentFiles.length === 0) {
    log('⚠️  No deployment found. Please run deployment first:', 'yellow');
    log('   npx hardhat run scripts/deploy_testnet_bridge.js --network sepolia\n', 'cyan');
    return null;
  }

  // Load most recent deployment
  const latestDeployment = deploymentFiles.sort().reverse()[0];
  const deployment = JSON.parse(fs.readFileSync(latestDeployment, 'utf8'));

  log('🪙 TestnetWBTC (tWBTC):', 'bright');
  log(`   Address: ${deployment.contracts.testnetWBTC.address}`, 'green');
  log(`   Your Balance: ${deployment.contracts.testnetWBTC.recipientBalance} tWBTC`, 'cyan');
  log(`   Etherscan: ${deployment.contracts.testnetWBTC.etherscan}`, 'blue');

  log('\n🌉 EthereumBridgeToken (XCBT):', 'bright');
  log(`   Address: ${deployment.contracts.ethereumBridgeToken.address}`, 'green');
  log(`   Your Balance: ${deployment.contracts.ethereumBridgeToken.recipientBalance} XCBT`, 'cyan');
  log(`   Etherscan: ${deployment.contracts.ethereumBridgeToken.etherscan}`, 'blue');
  log(`   Bridge Operator: ${deployment.contracts.ethereumBridgeToken.bridgeOperator}`, 'magenta');

  log('\n👤 Your Wallet:', 'bright');
  log(`   Address: ${deployment.recipient}`, 'green');
  log(`   Network: Sepolia Testnet`, 'cyan');

  return deployment;
}

/**
 * Display bridge operations menu
 */
function displayBridgeOperations(deployment) {
  subheader('AVAILABLE BRIDGE OPERATIONS');

  log('1️⃣  Bridge to Bitcoin Testnet', 'green');
  log('   Function: bridgeToBitcoin(btcAddress, amount)');
  log('   Description: Burns tokens on Ethereum and initiates bridge to Bitcoin\n');

  log('2️⃣  Bridge to Polygon', 'green');
  log('   Function: bridgeToPolygon(polygonAddress, amount)');
  log('   Description: Locks tokens on Ethereum for minting on Polygon\n');

  log('3️⃣  Bridge from Bitcoin (Operator)', 'green');
  log('   Function: bridgeFromBitcoin(recipient, amount, btcTxId)');
  log('   Description: Mints tokens on Ethereum from Bitcoin transaction\n');

  log('4️⃣  Bridge from Polygon (Operator)', 'green');
  log('   Function: bridgeFromPolygon(recipient, amount, polygonTxHash)');
  log('   Description: Unlocks tokens on Ethereum from Polygon bridge\n');

  log('5️⃣  Check Bridge Request Status', 'green');
  log('   Function: isBridgeRequestProcessed(requestId)');
  log('   Description: Verify if a bridge request has been processed\n');
}

/**
 * Generate example bridge transaction code
 */
function generateExampleCode(deployment) {
  subheader('EXAMPLE: BRIDGE TO BITCOIN TESTNET');

  const exampleCode = `
// Example: Bridge 1 XCBT to Bitcoin Testnet
const { ethers } = require('hardhat');
require('dotenv').config();

async function bridgeToBitcoin() {
  // Connect to contract
  const bridgeAddress = '${deployment.contracts.ethereumBridgeToken.address}';
  const EthereumBridgeToken = await ethers.getContractFactory('EthereumBridgeToken');
  const bridge = await EthereumBridgeToken.attach(bridgeAddress);

  // Your Bitcoin testnet address
  const btcAddress = 'tb1q... or mnopqr...'; // Replace with your Bitcoin testnet address

  // Amount to bridge (1 XCBT = 1 * 10^8 base units)
  const amount = ethers.parseUnits('1', 8);

  // Execute bridge transaction
  console.log('Bridging to Bitcoin...');
  const tx = await bridge.bridgeToBitcoin(btcAddress, amount);
  console.log('Transaction hash:', tx.hash);

  // Wait for confirmation
  const receipt = await tx.wait();
  console.log('✅ Bridge initiated! Block:', receipt.blockNumber);

  // Get request ID from event
  const event = receipt.logs.find(log => {
    try {
      return bridge.interface.parseLog(log).name === 'BridgeToNetwork';
    } catch {
      return false;
    }
  });

  if (event) {
    const parsedEvent = bridge.interface.parseLog(event);
    console.log('Request ID:', parsedEvent.args.requestId);
  }
}

bridgeToBitcoin().catch(console.error);
`;

  log(exampleCode, 'cyan');
}

/**
 * Save integration guide
 */
function saveIntegrationGuide(deployment, apis) {
  const guide = {
    title: 'Nexus AGI Bridge Integration Guide',
    timestamp: new Date().toISOString(),
    recipient: RECIPIENT_ADDRESS,

    contracts: {
      testnetWBTC: deployment.contracts.testnetWBTC,
      bridgeToken: deployment.contracts.ethereumBridgeToken
    },

    bridgeOperations: [
      {
        name: 'Bridge to Bitcoin',
        function: 'bridgeToBitcoin(btcAddress, amount)',
        description: 'Burns tokens on Ethereum and initiates bridge to Bitcoin',
        parameters: {
          btcAddress: 'Bitcoin address (string)',
          amount: 'Amount in base units (uint256, 8 decimals)'
        }
      },
      {
        name: 'Bridge to Polygon',
        function: 'bridgeToPolygon(polygonAddress, amount)',
        description: 'Locks tokens on Ethereum for minting on Polygon',
        parameters: {
          polygonAddress: 'Polygon address (address)',
          amount: 'Amount in base units (uint256, 8 decimals)'
        }
      },
      {
        name: 'Bridge from Bitcoin',
        function: 'bridgeFromBitcoin(recipient, amount, btcTxId)',
        description: 'Mints tokens on Ethereum from Bitcoin transaction (Operator only)',
        parameters: {
          recipient: 'Ethereum recipient address (address)',
          amount: 'Amount in base units (uint256)',
          btcTxId: 'Bitcoin transaction ID (string)'
        }
      },
      {
        name: 'Bridge from Polygon',
        function: 'bridgeFromPolygon(recipient, amount, polygonTxHash)',
        description: 'Unlocks tokens on Ethereum from Polygon (Operator only)',
        parameters: {
          recipient: 'Ethereum recipient address (address)',
          amount: 'Amount in base units (uint256)',
          polygonTxHash: 'Polygon transaction hash (string)'
        }
      }
    ],

    nexusAGIAPIs: {
      directoryURL: `https://${NEXUS_AGI_DIRECTORY_URL}`,
      totalAPIs: apis.length,
      sampleAPIs: apis.slice(0, 20).map(api => ({
        id: api.id,
        name: api.name,
        endpoint: api.endpoint,
        capabilities: api.capabilities,
        status: api.status
      }))
    },

    quickStart: {
      addToMetaMask: [
        {
          contract: 'TestnetWBTC',
          address: deployment.contracts.testnetWBTC.address,
          symbol: 'tWBTC',
          decimals: 8
        },
        {
          contract: 'EthereumBridgeToken',
          address: deployment.contracts.ethereumBridgeToken.address,
          symbol: 'XCBT',
          decimals: 8
        }
      ],

      testBridge: [
        '1. Ensure you have Sepolia ETH for gas',
        '2. Add both tokens to MetaMask',
        '3. Use Hardhat console or write a script',
        '4. Call bridge functions with testnet addresses',
        '5. Monitor transactions on Sepolia Etherscan'
      ]
    }
  };

  const guideFile = 'bridge_integration_guide.json';
  fs.writeFileSync(guideFile, JSON.stringify(guide, null, 2));
  log(`\n💾 Integration guide saved to: ${guideFile}`, 'green');
}

/**
 * Main execution
 */
async function main() {
  try {
    header('🌉 NEXUS AGI TESTNET BRIDGE - INTERACTION DASHBOARD');

    log('🔄 Fetching Nexus AGI Directory...', 'cyan');
    const apis = await fetchNexusAGIDirectory();
    log(`✅ Loaded ${apis.length} APIs from Nexus AGI Directory\n`, 'green');

    displayBlockchainAPIs(apis);

    const deployment = displayBridgeInfo();

    if (deployment) {
      displayBridgeOperations(deployment);
      generateExampleCode(deployment);
      saveIntegrationGuide(deployment, apis);
    }

    header('✅ INTERACTION DASHBOARD COMPLETE');

    log('📚 Next Steps:', 'bright');
    log('   1. Review the integration guide: bridge_integration_guide.json', 'cyan');
    log('   2. Add tokens to MetaMask using addresses above', 'cyan');
    log('   3. Create custom interaction scripts based on examples', 'cyan');
    log('   4. Test bridge operations on Sepolia testnet', 'cyan');
    log('   5. Explore 133+ APIs from Nexus AGI Directory\n', 'cyan');

  } catch (error) {
    log(`\n❌ Error: ${error.message}`, 'red');
    console.error(error);
    process.exit(1);
  }
}

// Run if called directly
if (require.main === module) {
  main();
}

module.exports = { fetchNexusAGIDirectory, displayBridgeInfo };
