/**
 * Deploy TBTC Token to Base Sepolia
 *
 * This script:
 * 1. Deploys TBTC contract with 1 million supply
 * 2. Transfers tokens to recipient address
 * 3. Sets up bridge operator
 * 4. Saves deployment info for verification
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
  header('🪙 DEPLOY TBTC TO BASE SEPOLIA');

  // Check configuration
  const BASE_SEPOLIA_RPC = process.env.BASE_SEPOLIA_RPC_URL || 'https://sepolia.base.org';
  const PRIVATE_KEY = process.env.BASE_SEPOLIA_PRIVATE_KEY || process.env.PRIVATE_KEY;
  const RECIPIENT = process.env.BASE_SEPOLIA_RECIPIENT || process.env.RECIPIENT_ADDRESS;

  if (!PRIVATE_KEY || PRIVATE_KEY.length !== 64) {
    log('❌ Private key not set or invalid in .env', 'red');
    log('   Set BASE_SEPOLIA_PRIVATE_KEY in .env file', 'yellow');
    process.exit(1);
  }

  if (!RECIPIENT) {
    log('❌ Recipient address not set in .env', 'red');
    log('   Set BASE_SEPOLIA_RECIPIENT in .env file', 'yellow');
    process.exit(1);
  }

  // Initialize provider and wallet
  log('🔗 Connecting to Base Sepolia...', 'cyan');
  const provider = new ethers.JsonRpcProvider(BASE_SEPOLIA_RPC);
  const wallet = new ethers.Wallet('0x' + PRIVATE_KEY, provider);

  log(`✅ Connected!`, 'green');
  log(`   Deployer: ${wallet.address}`, 'cyan');
  log(`   Recipient: ${RECIPIENT}`, 'cyan');

  // Check balance
  const balance = await provider.getBalance(wallet.address);
  const balanceETH = Number(ethers.formatEther(balance));
  log(`   Balance: ${balanceETH.toFixed(6)} ETH`, balanceETH > 0 ? 'green' : 'red');

  if (balanceETH === 0) {
    log('\n⚠️  WARNING: You have 0 ETH. Deployment will fail!', 'yellow');
    log('   You need Base Sepolia ETH for gas fees.', 'yellow');
    log('   Get from: https://www.alchemy.com/faucets/base-sepolia', 'yellow');
    return;
  }

  // Get network info
  const network = await provider.getNetwork();
  const blockNumber = await provider.getBlockNumber();
  log(`   Chain ID: ${network.chainId}`, 'cyan');
  log(`   Block: ${blockNumber.toLocaleString()}`, 'cyan');

  if (Number(network.chainId) !== 84532) {
    log('\n⚠️  WARNING: Not on Base Sepolia!', 'yellow');
    log(`   Current Chain ID: ${network.chainId}`, 'yellow');
    log('   Expected Chain ID: 84532 (Base Sepolia)', 'yellow');
    log('\n   Proceeding anyway...', 'cyan');
  }

  // Compile contract
  header('📝 COMPILING TBTC CONTRACT');

  const tbtcCompiled = compileContract('TBTC');

  // Deploy contract
  header('🚀 DEPLOYING TO BASE SEPOLIA');

  log('\n🪙 Deploying TBTC (1 Million Supply)...', 'bright');
  const tbtc = await deployContract(
    wallet,
    'TBTC',
    tbtcCompiled,
    [] // Constructor mints to deployer
  );

  const tbtcAddress = await tbtc.getAddress();

  // Transfer tokens to recipient
  header('💸 TRANSFERRING TOKENS');

  const transferAmount = ethers.parseUnits('1000000', 18); // All 1 million tokens
  log(`Transferring ${ethers.formatEther(transferAmount)} TBTC to ${RECIPIENT}...`, 'cyan');

  const transferTx = await tbtc.transfer(RECIPIENT, transferAmount);
  log(`Transaction hash: ${transferTx.hash}`, 'yellow');
  await transferTx.wait();
  log('✅ Tokens transferred!', 'green');

  // Check balances
  const recipientBalance = await tbtc.balanceOf(RECIPIENT);
  log(`\nRecipient balance: ${ethers.formatEther(recipientBalance)} TBTC`, 'green');

  // Save deployment info
  header('✅ DEPLOYMENT COMPLETE!');

  const deploymentInfo = {
    network: 'Base Sepolia',
    chainId: Number(network.chainId),
    deployer: wallet.address,
    recipient: RECIPIENT,
    timestamp: new Date().toISOString(),
    contracts: {
      TBTC: {
        address: tbtcAddress,
        totalSupply: '1000000',
        recipientBalance: ethers.formatEther(recipientBalance),
        explorer: `https://sepolia.basescan.org/address/${tbtcAddress}`,
        token: {
          name: 'Testnet Bitcoin',
          symbol: 'TBTC',
          decimals: 18
        },
        features: {
          burnMint: true,
          bridge: true,
          pausable: true,
          maxSupply: '1000000'
        }
      }
    }
  };

  console.log(JSON.stringify(deploymentInfo, null, 2));

  // Save to file
  const outputPath = path.join(__dirname, '..', 'tbtc_base_sepolia_deployment.json');
  fs.writeFileSync(outputPath, JSON.stringify(deploymentInfo, null, 2));
  log(`\n💾 Deployment info saved to: tbtc_base_sepolia_deployment.json`, 'green');

  // Save contract source for verification
  const verificationPath = path.join(__dirname, '..', 'tbtc_verification.json');
  const verificationData = {
    contractAddress: tbtcAddress,
    contractName: 'TBTC',
    compilerVersion: 'v0.8.20+commit.a1b79de6',
    optimization: true,
    runs: 200,
    constructorArguments: '',
    sourceCode: fs.readFileSync(path.join(__dirname, '..', 'contracts', 'TBTC.sol'), 'utf8')
  };
  fs.writeFileSync(verificationPath, JSON.stringify(verificationData, null, 2));

  // Next steps
  log('\n📋 NEXT STEPS:\n', 'bright');
  log(`   1. 🔍 Verify on BaseScan:`, 'cyan');
  log(`      https://sepolia.basescan.org/address/${tbtcAddress}#code`, 'yellow');
  log(`      Use tbtc_verification.json for manual verification`, 'yellow');

  log(`\n   2. 🌉 Bridge Bitcoin to TBTC:`, 'cyan');
  log(`      Run: node scripts/bridge_btc_to_tbtc.js`, 'yellow');

  log(`\n   3. 💧 Add liquidity on Uniswap Base:`, 'cyan');
  log(`      https://app.uniswap.org/#/add/v2`, 'yellow');

  log(`\n   4. 📊 Add to MetaMask:`, 'cyan');
  log(`      Token Address: ${tbtcAddress}`, 'yellow');
  log(`      Network: Base Sepolia`, 'yellow');

  log('\n🎉 TBTC launched successfully!\n', 'green');
  log('✨ 1 MILLION TBTC ready for bridging! ✨\n', 'magenta');
}

// Run deployment
main().catch((error) => {
  log('\n❌ Deployment failed:', 'red');
  console.error(error);
  process.exit(1);
});
