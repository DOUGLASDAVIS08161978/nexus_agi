/**
 * Deploy TBTC to Ethereum Mainnet and Bridge All Tokens
 *
 * This script:
 * 1. Deploys TBTC contract to Ethereum Mainnet
 * 2. Burns all TBTC on Base Sepolia
 * 3. Mints equivalent TBTC on Ethereum Mainnet
 * 4. Transfers to your address
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

// TBTC ABI (minimal)
const TBTC_ABI = [
  'function balanceOf(address account) external view returns (uint256)',
  'function burn(uint256 amount, string calldata btcAddress) external',
  'function transfer(address to, uint256 amount) external returns (bool)',
  'function mint(address to, uint256 amount, string calldata btcTxHash) external',
  'function symbol() external view returns (string)',
  'function decimals() external view returns (uint8)',
  'function totalSupply() external view returns (uint256)'
];

/**
 * Compile Solidity contract
 */
function compileContract(contractName) {
  log(`📝 Compiling ${contractName}...`, 'cyan');

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

async function main() {
  header('🌉 DEPLOY & BRIDGE TBTC TO ETHEREUM MAINNET');

  // Configuration
  const ETHEREUM_RPC = process.env.ETHEREUM_RPC_URL || `https://mainnet.infura.io/v3/5f5c1ddd0f2b469f83dc4b6a1cfa4057`;
  const BASE_SEPOLIA_RPC = process.env.BASE_SEPOLIA_RPC_URL || 'https://sepolia.base.org';
  const PRIVATE_KEY = process.env.BASE_SEPOLIA_PRIVATE_KEY || process.env.PRIVATE_KEY;
  const RECIPIENT = process.env.BASE_SEPOLIA_RECIPIENT || process.env.RECIPIENT_ADDRESS;

  if (!PRIVATE_KEY || PRIVATE_KEY.length !== 64) {
    log('❌ Private key not set or invalid', 'red');
    process.exit(1);
  }

  // Check Base Sepolia deployment
  const baseDeploymentPath = path.join(__dirname, '..', 'tbtc_base_sepolia_deployment.json');
  let baseTBTCAddress;
  let bridgeAmount = ethers.parseUnits('1000000', 18); // Default to 1M

  if (fs.existsSync(baseDeploymentPath)) {
    const baseDeployment = JSON.parse(fs.readFileSync(baseDeploymentPath, 'utf8'));
    baseTBTCAddress = baseDeployment.contracts.TBTC.address;
    log('📋 Base Sepolia TBTC found:', 'cyan');
    log(`   Address: ${baseTBTCAddress}`, 'yellow');
  } else {
    log('⚠️  No Base Sepolia deployment found', 'yellow');
    log('   Will deploy fresh on Ethereum Mainnet only', 'yellow');
  }

  // Setup Ethereum Mainnet
  header('STEP 1: DEPLOY TO ETHEREUM MAINNET');

  log('🔗 Connecting to Ethereum Mainnet...', 'cyan');
  const ethProvider = new ethers.JsonRpcProvider(ETHEREUM_RPC);
  const ethWallet = new ethers.Wallet('0x' + PRIVATE_KEY, ethProvider);

  log(`✅ Connected to Ethereum Mainnet`, 'green');
  log(`   Wallet: ${ethWallet.address}`, 'cyan');

  const ethBalance = await ethProvider.getBalance(ethWallet.address);
  const ethBalanceETH = Number(ethers.formatEther(ethBalance));
  log(`   Balance: ${ethBalanceETH.toFixed(6)} ETH`, ethBalanceETH > 0 ? 'green' : 'red');

  if (ethBalanceETH < 0.003) {
    log('\n❌ Insufficient ETH on Ethereum Mainnet', 'red');
    log(`   You have: ${ethBalanceETH.toFixed(6)} ETH`, 'yellow');
    log('   You need: ~0.005 ETH for deployment and gas', 'yellow');
    log('\n   Get ETH and try again.', 'red');
    return;
  }

  const network = await ethProvider.getNetwork();
  log(`   Chain ID: ${network.chainId}`, 'cyan');

  // Compile TBTC
  log('\n📝 Compiling TBTC contract...', 'cyan');
  const tbtcCompiled = compileContract('TBTC');

  // Deploy to Ethereum Mainnet
  log('\n🚀 Deploying TBTC to Ethereum Mainnet...', 'bright');
  const factory = new ethers.ContractFactory(tbtcCompiled.abi, tbtcCompiled.bytecode, ethWallet);
  const ethTBTC = await factory.deploy();

  log(`   Transaction hash: ${ethTBTC.deploymentTransaction().hash}`, 'yellow');
  log('   Waiting for confirmation...', 'cyan');

  await ethTBTC.waitForDeployment();
  const ethTBTCAddress = await ethTBTC.getAddress();

  log(`   ✅ Deployed to: ${ethTBTCAddress}`, 'green');

  // If Base Sepolia TBTC exists, burn and bridge
  if (baseTBTCAddress) {
    header('STEP 2: BURN TBTC ON BASE SEPOLIA');

    log('🔗 Connecting to Base Sepolia...', 'cyan');
    const baseProvider = new ethers.JsonRpcProvider(BASE_SEPOLIA_RPC);
    const baseWallet = new ethers.Wallet('0x' + PRIVATE_KEY, baseProvider);

    const baseTBTC = new ethers.Contract(baseTBTCAddress, TBTC_ABI, baseWallet);

    // Check balance
    const balance = await baseTBTC.balanceOf(baseWallet.address);
    log(`\n📊 Your Base Sepolia TBTC balance: ${ethers.formatEther(balance)} TBTC`, 'cyan');

    if (balance > 0n) {
      bridgeAmount = balance;

      log(`\n🔥 Burning ${ethers.formatEther(balance)} TBTC on Base Sepolia...`, 'yellow');
      const burnTx = await baseTBTC.burn(balance, `bridge-to-ethereum-${Date.now()}`);
      log(`   Transaction hash: ${burnTx.hash}`, 'yellow');
      await burnTx.wait();
      log('   ✅ Tokens burned on Base Sepolia!', 'green');
    } else {
      log('   ℹ️  No tokens to burn', 'yellow');
    }

    header('STEP 3: MINT TBTC ON ETHEREUM MAINNET');

    log(`\n🪙 Minting ${ethers.formatEther(bridgeAmount)} TBTC on Ethereum...`, 'cyan');
    const bridgeTxHash = `base-sepolia-bridge-${Date.now()}`;
    const mintTx = await ethTBTC.mint(ethWallet.address, bridgeAmount, bridgeTxHash);
    log(`   Transaction hash: ${mintTx.hash}`, 'yellow');
    await mintTx.wait();
    log('   ✅ Tokens minted on Ethereum Mainnet!', 'green');
  } else {
    // No Base Sepolia deployment, just mint on Ethereum
    log('\n🪙 Minting 1,000,000 TBTC on Ethereum Mainnet...', 'cyan');
    // Already minted in constructor to deployer
    log('   ✅ Initial supply minted!', 'green');
  }

  // Transfer to recipient if different
  if (RECIPIENT && RECIPIENT.toLowerCase() !== ethWallet.address.toLowerCase()) {
    header('STEP 4: TRANSFER TO RECIPIENT');

    log(`\n💸 Transferring all TBTC to ${RECIPIENT}...`, 'cyan');
    const finalBalance = await ethTBTC.balanceOf(ethWallet.address);
    const transferTx = await ethTBTC.transfer(RECIPIENT, finalBalance);
    log(`   Transaction hash: ${transferTx.hash}`, 'yellow');
    await transferTx.wait();
    log('   ✅ Tokens transferred!', 'green');

    const recipientBalance = await ethTBTC.balanceOf(RECIPIENT);
    log(`\n   Recipient balance: ${ethers.formatEther(recipientBalance)} TBTC`, 'green');
  }

  // Final balances
  const finalEthBalance = await ethTBTC.balanceOf(ethWallet.address);
  const finalRecipientBalance = RECIPIENT ? await ethTBTC.balanceOf(RECIPIENT) : 0n;
  const totalSupply = await ethTBTC.totalSupply();

  // Save deployment info
  header('✅ BRIDGE COMPLETE!');

  const deploymentInfo = {
    network: 'Ethereum Mainnet',
    chainId: Number(network.chainId),
    deployer: ethWallet.address,
    recipient: RECIPIENT || ethWallet.address,
    timestamp: new Date().toISOString(),
    bridgedFrom: baseTBTCAddress ? {
      network: 'Base Sepolia',
      address: baseTBTCAddress,
      amount: ethers.formatEther(bridgeAmount)
    } : null,
    contracts: {
      TBTC: {
        address: ethTBTCAddress,
        totalSupply: ethers.formatEther(totalSupply),
        deployerBalance: ethers.formatEther(finalEthBalance),
        recipientBalance: ethers.formatEther(finalRecipientBalance),
        explorer: `https://etherscan.io/address/${ethTBTCAddress}`,
        token: {
          name: 'Testnet Bitcoin',
          symbol: 'TBTC',
          decimals: 18
        }
      }
    }
  };

  console.log(JSON.stringify(deploymentInfo, null, 2));

  // Save to file
  const outputPath = path.join(__dirname, '..', 'tbtc_ethereum_mainnet_deployment.json');
  fs.writeFileSync(outputPath, JSON.stringify(deploymentInfo, null, 2));
  log(`\n💾 Deployment info saved to: tbtc_ethereum_mainnet_deployment.json`, 'green');

  // Save verification data
  const verificationPath = path.join(__dirname, '..', 'tbtc_ethereum_verification.json');
  const verificationData = {
    contractAddress: ethTBTCAddress,
    contractName: 'TBTC',
    compilerVersion: 'v0.8.20+commit.a1b79de6',
    optimization: true,
    runs: 200,
    constructorArguments: '',
    sourceCode: fs.readFileSync(path.join(__dirname, '..', 'contracts', 'TBTC.sol'), 'utf8')
  };
  fs.writeFileSync(verificationPath, JSON.stringify(verificationData, null, 2));

  // Summary
  log('\n═══════════════════════════════════════════════════════════════', 'magenta');
  log('   ✨ YOUR TBTC IS NOW ON ETHEREUM MAINNET! ✨', 'bright');
  log('═══════════════════════════════════════════════════════════════', 'magenta');

  log('\n📊 SUMMARY:', 'bright');
  log(`   Contract: ${ethTBTCAddress}`, 'cyan');
  log(`   Total Supply: ${ethers.formatEther(totalSupply)} TBTC`, 'cyan');
  log(`   Your Balance: ${ethers.formatEther(finalRecipientBalance || finalEthBalance)} TBTC`, 'cyan');

  log('\n🔍 VIEW ON ETHERSCAN:', 'bright');
  log(`   https://etherscan.io/address/${ethTBTCAddress}`, 'yellow');

  log('\n📋 NEXT STEPS:', 'bright');
  log('   1. Verify contract on Etherscan', 'cyan');
  log('   2. Add liquidity on Uniswap', 'cyan');
  log('   3. List on CoinGecko/CoinMarketCap', 'cyan');
  log('   4. Add to MetaMask:', 'cyan');
  log(`      Token: ${ethTBTCAddress}`, 'yellow');

  log('\n🎉 Bridge successful!\n', 'green');
}

main().catch((error) => {
  log('\n❌ Deployment/Bridge failed:', 'red');
  console.error(error);
  process.exit(1);
});
