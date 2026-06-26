/**
 * Pure Web3.js Deployment Script (NO HARDHAT)
 *
 * This script deploys contracts using only web3.js and solc compiler
 * Works with any Node.js version
 *
 * Usage: node scripts/deploy_pure_web3.js
 */

const { Web3 } = require('web3');
const solc = require('solc');
const fs = require('fs');
const path = require('path');
require('dotenv').config();

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
 * Compile Solidity contract
 */
function compileContract(contractName) {
  log(`\n📝 Compiling ${contractName}...`, 'cyan');

  const contractPath = path.join(__dirname, '..', 'contracts', `${contractName}.sol`);
  const source = fs.readFileSync(contractPath, 'utf8');

  // Read OpenZeppelin contracts
  const openzeppelinBase = path.join(__dirname, '..', 'node_modules', '@openzeppelin', 'contracts');

  const input = {
    language: 'Solidity',
    sources: {
      [`${contractName}.sol`]: {
        content: source
      }
    },
    settings: {
      outputSelection: {
        '*': {
          '*': ['abi', 'evm.bytecode']
        }
      },
      optimizer: {
        enabled: true,
        runs: 200
      }
    }
  };

  // Import callback for OpenZeppelin
  function findImports(importPath) {
    try {
      if (importPath.startsWith('@openzeppelin/')) {
        const filePath = path.join(__dirname, '..', 'node_modules', importPath);
        const content = fs.readFileSync(filePath, 'utf8');
        return { contents: content };
      }
      return { error: 'File not found' };
    } catch (error) {
      return { error: error.message };
    }
  }

  const output = JSON.parse(solc.compile(JSON.stringify(input), { import: findImports }));

  if (output.errors) {
    output.errors.forEach((err) => {
      if (err.severity === 'error') {
        log(`❌ Error: ${err.message}`, 'red');
      } else {
        log(`⚠️  Warning: ${err.message}`, 'yellow');
      }
    });

    if (output.errors.some(err => err.severity === 'error')) {
      throw new Error('Compilation failed');
    }
  }

  const contract = output.contracts[`${contractName}.sol`][contractName];
  log(`✅ ${contractName} compiled successfully!`, 'green');

  return {
    abi: contract.abi,
    bytecode: contract.evm.bytecode.object
  };
}

/**
 * Deploy contract
 */
async function deployContract(web3, account, contractName, compiled, constructorArgs = []) {
  log(`\n🚀 Deploying ${contractName}...`, 'cyan');

  const contract = new web3.eth.Contract(compiled.abi);

  log(`   Constructor args: ${JSON.stringify(constructorArgs)}`, 'yellow');

  const deploy = contract.deploy({
    data: '0x' + compiled.bytecode,
    arguments: constructorArgs
  });

  const gas = await deploy.estimateGas({ from: account });
  log(`   Estimated gas: ${gas.toLocaleString()}`, 'yellow');

  const gasPrice = await web3.eth.getGasPrice();
  const gasCost = (Number(gas) * Number(gasPrice)) / 1e18;
  log(`   Estimated cost: ${gasCost.toFixed(6)} MATIC`, 'yellow');

  log(`   Sending transaction...`, 'cyan');

  const instance = await deploy.send({
    from: account,
    gas: Math.floor(gas * 1.2), // Add 20% buffer
    gasPrice: gasPrice
  });

  log(`   ✅ Deployed to: ${instance.options.address}`, 'green');

  return instance;
}

/**
 * Main deployment
 */
async function main() {
  header('🌐 PURE WEB3.JS DEPLOYMENT (NO HARDHAT)');

  // Check configuration
  if (!process.env.POLYGON_RPC_URL) {
    log('❌ POLYGON_RPC_URL not set in .env', 'red');
    process.exit(1);
  }

  if (!process.env.PRIVATE_KEY || process.env.PRIVATE_KEY.length !== 64) {
    log('❌ PRIVATE_KEY not set or invalid in .env', 'red');
    process.exit(1);
  }

  // Initialize Web3
  log('🔗 Connecting to Polygon...', 'cyan');
  const web3 = new Web3(process.env.POLYGON_RPC_URL);

  // Load account
  const account = web3.eth.accounts.privateKeyToAccount('0x' + process.env.PRIVATE_KEY);
  web3.eth.accounts.wallet.add(account);
  web3.eth.defaultAccount = account.address;

  log(`✅ Connected!`, 'green');
  log(`   Account: ${account.address}`, 'cyan');

  // Check balance
  const balance = await web3.eth.getBalance(account.address);
  const balanceMATIC = Number(balance) / 1e18;
  log(`   Balance: ${balanceMATIC.toFixed(6)} MATIC`, balanceMATIC > 0 ? 'green' : 'red');

  if (balanceMATIC === 0) {
    log('\n⚠️  WARNING: You have 0 MATIC. Deployment will fail!', 'yellow');
    log('   Get MATIC and try again.', 'yellow');
    return;
  }

  // Get network info
  const chainId = await web3.eth.getChainId();
  const blockNumber = await web3.eth.getBlockNumber();
  log(`   Chain ID: ${chainId}`, 'cyan');
  log(`   Block: ${blockNumber.toLocaleString()}`, 'cyan');

  // Compile contracts
  header('📝 COMPILING CONTRACTS');

  const testnetWBTCCompiled = compileContract('TestnetWBTC');
  const bridgeTokenCompiled = compileContract('EthereumBridgeToken');

  // Deploy contracts
  header('🚀 DEPLOYING CONTRACTS');

  const initialSupply = process.env.INITIAL_WBTC_SUPPLY || '100';
  const bridgeSupply = process.env.INITIAL_BRIDGE_SUPPLY || '1000';

  log('\n1️⃣  Deploying TestnetWBTC...', 'bright');
  const wbtc = await deployContract(
    web3,
    account.address,
    'TestnetWBTC',
    testnetWBTCCompiled,
    [initialSupply]
  );

  log('\n2️⃣  Deploying EthereumBridgeToken...', 'bright');
  const bridgeToken = await deployContract(
    web3,
    account.address,
    'EthereumBridgeToken',
    bridgeTokenCompiled,
    [bridgeSupply]
  );

  // Deployment summary
  header('✅ DEPLOYMENT COMPLETE!');

  const deploymentInfo = {
    network: chainId === 137n ? 'Polygon Mainnet' : chainId === 80002n ? 'Polygon Amoy' : `Chain ${chainId}`,
    chainId: Number(chainId),
    deployer: account.address,
    timestamp: new Date().toISOString(),
    contracts: {
      TestnetWBTC: {
        address: wbtc.options.address,
        initialSupply: initialSupply,
        explorer: `https://polygonscan.com/address/${wbtc.options.address}`
      },
      EthereumBridgeToken: {
        address: bridgeToken.options.address,
        initialSupply: bridgeSupply,
        explorer: `https://polygonscan.com/address/${bridgeToken.options.address}`
      }
    }
  };

  console.log(JSON.stringify(deploymentInfo, null, 2));

  // Save to file
  const outputPath = path.join(__dirname, '..', 'deployment_info.json');
  fs.writeFileSync(outputPath, JSON.stringify(deploymentInfo, null, 2));
  log(`\n💾 Deployment info saved to: ${outputPath}`, 'green');

  // Next steps
  log('\n📋 NEXT STEPS:\n', 'bright');
  log(`   1. Verify on PolygonScan:`, 'cyan');
  log(`      TestnetWBTC: ${deploymentInfo.contracts.TestnetWBTC.explorer}`, 'yellow');
  log(`      EthereumBridgeToken: ${deploymentInfo.contracts.EthereumBridgeToken.explorer}`, 'yellow');
  log(`\n   2. Mint tokens to recipient:`, 'cyan');
  log(`      node scripts/interact_bridge.js`, 'yellow');
  log(`\n   3. Set up bridge operator:`, 'cyan');
  log(`      Update .env with contract addresses`, 'yellow');

  log('\n🎉 Deployment successful! No Hardhat required!\n', 'green');
}

// Run deployment
main().catch((error) => {
  log('\n❌ Deployment failed:', 'red');
  console.error(error);
  process.exit(1);
});
