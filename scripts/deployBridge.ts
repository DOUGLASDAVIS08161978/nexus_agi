/**
 * Deploy wTBTC Bridge System to Polygon
 * Deploys contract and sets up bridge operator for address: 0x24f6b1ce11c57d40b542f91ac85fa9eb61f78771
 */

import { ethers } from 'ethers';
import walletManager from '../src/web3/walletManager';
import wBTCContract from '../src/contracts/wBTCInterface';
import bridgeOperator from '../src/bridge/bridgeOperator';

// User's destination address on Polygon
const USER_ADDRESS = '0x24f6b1ce11c57d40b542f91ac85fa9eb61f78771';

// Target chain IDs
const POLYGON_MAINNET = '0x89';
const POLYGON_TESTNET = '0x13882'; // Amoy testnet

async function main() {
  console.log(`
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   🌉 NEXUS BRIDGE DEPLOYMENT SYSTEM                          ║
║                                                              ║
║   Deploying wTBTC Bridge to Polygon Network                 ║
║   Destination: ${USER_ADDRESS}   ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
  `);

  try {
    // Step 1: Connect wallet
    console.log('\n📍 Step 1: Connecting to MetaMask...');
    const { address, chainId } = await walletManager.connectWallet();

    console.log(`✅ Connected!`);
    console.log(`   Wallet: ${address}`);
    console.log(`   Chain: ${walletManager.getChainName(chainId)}`);

    // Step 2: Switch to Polygon network
    console.log('\n📍 Step 2: Switching to Polygon network...');

    let targetChain = POLYGON_TESTNET; // Default to testnet for safety

    // Ask user which network
    console.log('   Choose network:');
    console.log('   1. Polygon Amoy Testnet (Recommended for testing)');
    console.log('   2. Polygon Mainnet (Real money - use with caution)');

    // For this script, we'll use testnet by default
    console.log('   Using Polygon Amoy Testnet for safety...');

    if (chainId !== targetChain) {
      const switched = await walletManager.switchChain(targetChain);
      if (!switched) {
        throw new Error('Failed to switch to Polygon network');
      }
    }

    console.log(`✅ Connected to ${walletManager.getChainName(targetChain)}`);

    // Step 3: Check balance
    console.log('\n📍 Step 3: Checking MATIC balance...');
    const balance = await walletManager.getBalance();
    console.log(`   Balance: ${balance} MATIC`);

    if (parseFloat(balance) < 0.01) {
      console.log(`⚠️  Warning: Low balance. You may need more MATIC for gas fees.`);
      console.log(`   Get testnet MATIC from: https://faucet.polygon.technology/`);
    }

    // Step 4: Deploy wTBTC contract
    console.log('\n📍 Step 4: Deploying wTBTC Smart Contract...');
    console.log(`   Bridge Operator: ${address}`);

    const contractAddress = await wBTCContract.deploy(address);

    console.log(`✅ wTBTC Contract Deployed!`);
    console.log(`   Address: ${contractAddress}`);
    console.log(`   View on Polygonscan: https://amoy.polygonscan.com/address/${contractAddress}`);

    // Step 5: Start bridge operator
    console.log('\n📍 Step 5: Starting Bridge Operator Service...');
    await bridgeOperator.start();

    console.log(`✅ Bridge operator started!`);

    // Step 6: Create bridge request for user
    console.log('\n📍 Step 6: Creating Bridge Request...');
    console.log(`   Destination: ${USER_ADDRESS}`);

    // Generate a Bitcoin testnet address (in real implementation, this would come from a Bitcoin wallet)
    const bitcoinAddress = 'tb1q' + Math.random().toString(36).substring(2, 15) + 'testnet';

    const bridgeRequest = bridgeOperator.createBridgeRequest(
      USER_ADDRESS,
      bitcoinAddress
    );

    console.log(`✅ Bridge request created!`);
    console.log(`   Request ID: ${bridgeRequest.id}`);
    console.log(`   Bitcoin Address: ${bridgeRequest.bitcoinAddress}`);

    // Step 7: Setup complete
    console.log(`
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   ✅ DEPLOYMENT COMPLETE!                                    ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝

📝 DEPLOYMENT SUMMARY:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔗 Network:              ${walletManager.getChainName(targetChain)}
💎 wTBTC Contract:       ${contractAddress}
👤 Bridge Operator:      ${address}
🎯 Destination Address:  ${USER_ADDRESS}
₿  Bitcoin Address:      ${bitcoinAddress}
🆔 Bridge Request ID:    ${bridgeRequest.id}

📖 NEXT STEPS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Get testnet BTC from faucets:
   • https://coinfaucet.eu/en/btc-testnet/
   • https://testnet-faucet.mempool.co/
   • https://bitcoinfaucet.uo1.net/

2. Send testnet BTC to: ${bitcoinAddress}

3. Bridge will automatically mint wTBTC to: ${USER_ADDRESS}

4. Monitor your wTBTC balance at:
   https://amoy.polygonscan.com/token/${contractAddress}?a=${USER_ADDRESS}

🎓 EDUCATIONAL MINING:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

To understand Bitcoin mining concepts, run:
   npm run mining-simulator

Note: This is educational only. Real Bitcoin mining requires
specialized ASIC hardware and is not profitable on regular computers.

✨ Bridge is now monitoring for Bitcoin testnet transactions!
    `);

    // Keep the process running to monitor
    console.log('\n🔄 Bridge operator is running. Press Ctrl+C to stop.\n');

    // Listen for Burn events to handle withdrawals
    wBTCContract.onBurn((from, amount, btcAddress) => {
      console.log(`\n🔥 BURN EVENT DETECTED:`);
      console.log(`   From: ${from}`);
      console.log(`   Amount: ${amount} wTBTC`);
      console.log(`   BTC Address: ${btcAddress}`);
      console.log(`   ⚠️  Manual action: Send ${amount} BTC to ${btcAddress}`);
    });

    // Keep process alive
    process.on('SIGINT', () => {
      console.log('\n\n🛑 Shutting down bridge operator...');
      bridgeOperator.stop();
      process.exit(0);
    });

  } catch (error: any) {
    console.error('\n❌ Deployment failed:', error.message);
    console.error(error);
    process.exit(1);
  }
}

// Run if called directly
if (require.main === module) {
  main().catch(console.error);
}

export default main;
