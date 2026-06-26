/**
 * Deploy WTBTC Bridge to Ethereum Mainnet
 *
 * This script:
 * 1. Deploys WTBTCBridge contract to Ethereum Mainnet
 * 2. Mints initial tokens to your wallet
 * 3. Saves deployment info for verification
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
          '*': ['abi', 'evm.bytecode', 'metadata']
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
    bytecode: contract.evm.bytecode.object,
    metadata: contract.metadata
  };
}

/**
 * Deploy contract
 */
async function deployContract(wallet, contractName, compiled, constructorArgs = []) {
  log(`\n🚀 Deploying ${contractName}...`, 'cyan');

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
  header('🌉 DEPLOY WTBTC BRIDGE TO ETHEREUM MAINNET');

  // Check configuration
  const ETHEREUM_RPC = process.env.ETHEREUM_RPC_URL || `https://mainnet.infura.io/v3/${process.env.INFURA_API_KEY || '5f5c1ddd0f2b469f83dc4b6a1cfa4057'}`;

  if (!process.env.PRIVATE_KEY || process.env.PRIVATE_KEY.length !== 64) {
    log('❌ PRIVATE_KEY not set or invalid in .env', 'red');
    process.exit(1);
  }

  // Initialize provider and wallet
  log('🔗 Connecting to Ethereum Mainnet...', 'cyan');
  const provider = new ethers.JsonRpcProvider(ETHEREUM_RPC);
  const wallet = new ethers.Wallet('0x' + process.env.PRIVATE_KEY, provider);

  log(`✅ Connected!`, 'green');
  log(`   Account: ${wallet.address}`, 'cyan');

  // Check balance
  const balance = await provider.getBalance(wallet.address);
  const balanceETH = Number(ethers.formatEther(balance));
  log(`   Balance: ${balanceETH.toFixed(6)} ETH`, balanceETH > 0 ? 'green' : 'red');

  if (balanceETH === 0) {
    log('\n⚠️  WARNING: You have 0 ETH. Deployment will fail!', 'yellow');
    log('   You need ETH on Ethereum Mainnet for gas fees.', 'yellow');
    log('   Gas estimate: ~0.003-0.005 ETH ($10-20 USD)', 'yellow');
    return;
  }

  // Get network info
  const network = await provider.getNetwork();
  const blockNumber = await provider.getBlockNumber();
  log(`   Chain ID: ${network.chainId}`, 'cyan');
  log(`   Block: ${blockNumber.toLocaleString()}`, 'cyan');

  if (Number(network.chainId) !== 1) {
    log('\n⚠️  WARNING: Not on Ethereum Mainnet!', 'yellow');
    log(`   Current Chain ID: ${network.chainId}`, 'yellow');
    log('   Expected Chain ID: 1 (Ethereum Mainnet)', 'yellow');
    log('\n   Proceeding anyway...', 'cyan');
  }

  // Compile contract
  header('📝 COMPILING BRIDGE CONTRACT');

  const bridgeCompiled = compileContract('WTBTCBridge');

  // Deploy contract
  header('🚀 DEPLOYING TO ETHEREUM MAINNET');

  log('\n🌉 Deploying WTBTCBridge...', 'bright');
  const bridge = await deployContract(
    wallet,
    'WTBTCBridge',
    bridgeCompiled,
    [] // No constructor args
  );

  const bridgeAddress = await bridge.getAddress();

  // Mint tokens to user
  header('🪙 MINTING TOKENS');

  log('Minting 10,000 WTBTC to your wallet...', 'cyan');
  const mintAmount = ethers.parseUnits('10000', 18);
  const mintTx = await bridge.mint(wallet.address, mintAmount);
  log(`Transaction hash: ${mintTx.hash}`, 'yellow');
  await mintTx.wait();
  log('✅ Tokens minted!', 'green');

  // Save deployment info
  header('✅ DEPLOYMENT COMPLETE!');

  const deploymentInfo = {
    network: 'Ethereum Mainnet',
    chainId: Number(network.chainId),
    deployer: wallet.address,
    timestamp: new Date().toISOString(),
    baseSepolia: {
      originalContract: '0xE274570e000C32F5Cb2BC7c476D3BDC77Ed74dD5',
      network: 'Base Sepolia',
      chainId: 84532
    },
    contracts: {
      WTBTCBridge: {
        address: bridgeAddress,
        initialSupply: '1000000',
        mintedToDeployer: '10000',
        explorer: `https://etherscan.io/address/${bridgeAddress}`,
        token: {
          name: 'Wrapped Testnet Bitcoin',
          symbol: 'WTBTC',
          decimals: 18
        }
      }
    }
  };

  console.log(JSON.stringify(deploymentInfo, null, 2));

  // Save to file
  const outputPath = path.join(__dirname, '..', 'ethereum_bridge_deployment.json');
  fs.writeFileSync(outputPath, JSON.stringify(deploymentInfo, null, 2));
  log(`\n💾 Deployment info saved to: ethereum_bridge_deployment.json`, 'green');

  // Save contract source for verification
  const verificationPath = path.join(__dirname, '..', 'bridge_verification.json');
  const verificationData = {
    contractAddress: bridgeAddress,
    contractName: 'WTBTCBridge',
    compilerVersion: 'v0.8.20+commit.a1b79de6',
    optimization: true,
    runs: 200,
    constructorArguments: '',
    sourceCode: fs.readFileSync(path.join(__dirname, '..', 'contracts', 'WTBTCBridge.sol'), 'utf8')
  };
  fs.writeFileSync(verificationPath, JSON.stringify(verificationData, null, 2));

  // Next steps
  log('\n📋 NEXT STEPS:\n', 'bright');
  log(`   1. 🔍 Verify on Etherscan:`, 'cyan');
  log(`      https://etherscan.io/address/${bridgeAddress}#code`, 'yellow');
  log(`      Use bridge_verification.json for manual verification`, 'yellow');

  log(`\n   2. 🌉 Bridge your tokens:`, 'cyan');
  log(`      Base Sepolia TX: 0xab880c7a9bc9a6ef6110e093c2632dcc6144ccf53f86f0059542a28a4a0d78cc`, 'yellow');
  log(`      Run: node scripts/bridge_from_base.js`, 'yellow');

  log(`\n   3. 💧 Add liquidity on Uniswap:`, 'cyan');
  log(`      https://app.uniswap.org/add/v2`, 'yellow');

  log(`\n   4. 📊 Track your token:`, 'cyan');
  log(`      Add to MetaMask: ${bridgeAddress}`, 'yellow');

  log('\n🎉 Bridge deployed successfully!\n', 'green');
  log('✨ Your WTBTC can now move between Base Sepolia and Ethereum Mainnet! ✨\n', 'magenta');
}

// Run deployment
main().catch((error) => {
  log('\n❌ Deployment failed:', 'red');
  console.error(error);
  process.exit(1);
});
