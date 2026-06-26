/**
 * Ethers.js Deployment Script (NO HARDHAT)
 *
 * This script deploys contracts using ethers.js and solc compiler
 * More stable than web3.js v4
 *
 * Usage: node scripts/deploy_with_ethers.js
 */

const { ethers } = require('ethers');
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
async function deployContract(wallet, contractName, compiled, constructorArgs = []) {
  log(`\n🚀 Deploying ${contractName}...`, 'cyan');
  log(`   Constructor args: ${JSON.stringify(constructorArgs)}`, 'yellow');

  const factory = new ethers.ContractFactory(compiled.abi, compiled.bytecode, wallet);

  const contract = await factory.deploy(...constructorArgs);

  log(`   Transaction hash: ${contract.deploymentTransaction().hash}`, 'yellow');
  log(`   Waiting for confirmation...`, 'cyan');

  await contract.waitForDeployment();

  const address = await contract.getAddress();
  log(`   ✅ Deployed to: ${address}`, 'green');

  return contract;
}

/**
 * Main deployment
 */
async function main() {
  header('🌐 ETHERS.JS DEPLOYMENT (NO HARDHAT)');

  // Check configuration
  if (!process.env.POLYGON_RPC_URL) {
    log('❌ POLYGON_RPC_URL not set in .env', 'red');
    process.exit(1);
  }

  if (!process.env.PRIVATE_KEY || process.env.PRIVATE_KEY.length !== 64) {
    log('❌ PRIVATE_KEY not set or invalid in .env', 'red');
    process.exit(1);
  }

  // Initialize provider and wallet
  log('🔗 Connecting to Polygon...', 'cyan');
  const provider = new ethers.JsonRpcProvider(process.env.POLYGON_RPC_URL);
  const wallet = new ethers.Wallet('0x' + process.env.PRIVATE_KEY, provider);

  log(`✅ Connected!`, 'green');
  log(`   Account: ${wallet.address}`, 'cyan');

  // Check balance
  const balance = await provider.getBalance(wallet.address);
  const balanceMATIC = Number(ethers.formatEther(balance));
  log(`   Balance: ${balanceMATIC.toFixed(6)} MATIC`, balanceMATIC > 0 ? 'green' : 'red');

  if (balanceMATIC === 0) {
    log('\n⚠️  WARNING: You have 0 MATIC. Deployment will fail!', 'yellow');
    log('   Get MATIC and try again.', 'yellow');
    return;
  }

  // Get network info
  const network = await provider.getNetwork();
  const blockNumber = await provider.getBlockNumber();
  log(`   Chain ID: ${network.chainId}`, 'cyan');
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
    wallet,
    'TestnetWBTC',
    testnetWBTCCompiled,
    [initialSupply]
  );

  log('\n2️⃣  Deploying EthereumBridgeToken...', 'bright');
  const bridgeToken = await deployContract(
    wallet,
    'EthereumBridgeToken',
    bridgeTokenCompiled,
    [bridgeSupply]
  );

  // Deployment summary
  header('✅ DEPLOYMENT COMPLETE!');

  const wbtcAddress = await wbtc.getAddress();
  const bridgeAddress = await bridgeToken.getAddress();

  const deploymentInfo = {
    network: Number(network.chainId) === 137 ? 'Polygon Mainnet' : Number(network.chainId) === 80002 ? 'Polygon Amoy' : `Chain ${network.chainId}`,
    chainId: Number(network.chainId),
    deployer: wallet.address,
    timestamp: new Date().toISOString(),
    contracts: {
      TestnetWBTC: {
        address: wbtcAddress,
        initialSupply: initialSupply,
        explorer: `https://polygonscan.com/address/${wbtcAddress}`
      },
      EthereumBridgeToken: {
        address: bridgeAddress,
        initialSupply: bridgeSupply,
        explorer: `https://polygonscan.com/address/${bridgeAddress}`
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
  log(`\n   2. Your contracts are live on Polygon!`, 'green');
  log(`\n   3. Contract Addresses:`, 'cyan');
  log(`      TestnetWBTC: ${wbtcAddress}`, 'yellow');
  log(`      EthereumBridgeToken: ${bridgeAddress}`, 'yellow');

  log('\n🎉 Deployment successful! No Hardhat required!\n', 'green');
}

// Run deployment
main().catch((error) => {
  log('\n❌ Deployment failed:', 'red');
  console.error(error);
  process.exit(1);
});
