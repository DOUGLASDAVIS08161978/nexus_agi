/**
 * Automated Bridge System
 * Complete end-to-end bridge automation for Bitcoin Testnet -> Polygon
 * Destination: 0x24f6b1ce11c57d40b542f91ac85fa9eb61f78771
 */

import walletManager from '../src/web3/walletManager';
import wBTCContract from '../src/contracts/wBTCInterface';
import bridgeOperator from '../src/bridge/bridgeOperator';
import bitcoinMonitor from '../src/bridge/bitcoinMonitor';

const USER_ADDRESS = '0x24f6b1ce11c57d40b542f91ac85fa9eb61f78771';
const POLYGON_TESTNET = '0x13882';

async function main() {
  console.log(`
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   🤖 NEXUS AUTOMATED BRIDGE SYSTEM                           ║
║                                                              ║
║   Bitcoin Testnet ₿  →  Polygon wTBTC 🟣                    ║
║   Fully Automated Token Bridging                            ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝

🎯 Destination Address: ${USER_ADDRESS}
  `);

  try {
    // Step 1: Initialize
    console.log('\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    console.log('📍 STEP 1: INITIALIZING BRIDGE SYSTEM');
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n');

    // Connect wallet
    console.log('🔌 Connecting to MetaMask...');
    const { address, chainId } = await walletManager.connectWallet();
    console.log(`✅ Connected: ${address}`);

    // Switch to Polygon
    console.log('\n🔄 Switching to Polygon Amoy Testnet...');
    if (chainId !== POLYGON_TESTNET) {
      await walletManager.switchChain(POLYGON_TESTNET);
    }
    console.log('✅ Connected to Polygon Amoy Testnet');

    // Check balance
    const balance = await walletManager.getBalance();
    console.log(`💰 Balance: ${balance} MATIC`);

    if (parseFloat(balance) < 0.01) {
      console.log('\n⚠️  LOW BALANCE WARNING');
      console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
      console.log('You need MATIC for gas fees. Get testnet MATIC from:');
      console.log('https://faucet.polygon.technology/');
      console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n');
      process.exit(1);
    }

    // Step 2: Deploy Contract (if not already deployed)
    console.log('\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    console.log('📍 STEP 2: DEPLOYING wTBTC CONTRACT');
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n');

    console.log('🚀 Deploying wTBTC smart contract...');
    const contractAddress = await wBTCContract.deploy(address);

    console.log(`✅ wTBTC Deployed Successfully!`);
    console.log(`   Contract: ${contractAddress}`);
    console.log(`   Operator: ${address}`);
    console.log(`   Explorer: https://amoy.polygonscan.com/address/${contractAddress}`);

    // Step 3: Setup Bridge
    console.log('\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    console.log('📍 STEP 3: CONFIGURING BRIDGE OPERATOR');
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n');

    console.log('🌉 Starting bridge operator...');
    await bridgeOperator.start();
    console.log('✅ Bridge operator active');

    // Create bridge request for user
    const bitcoinAddress = generateTestnetAddress();
    const bridgeRequest = bridgeOperator.createBridgeRequest(
      USER_ADDRESS,
      bitcoinAddress
    );

    console.log(`\n📋 Bridge Request Created:`);
    console.log(`   Request ID: ${bridgeRequest.id}`);
    console.log(`   Bitcoin Address: ${bitcoinAddress}`);
    console.log(`   Destination: ${USER_ADDRESS}`);

    // Step 4: Get Testnet BTC
    console.log('\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    console.log('📍 STEP 4: GET BITCOIN TESTNET COINS');
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n');

    console.log('🪙 Visit a Bitcoin testnet faucet to get free testnet BTC:');
    console.log('\n   1. https://coinfaucet.eu/en/btc-testnet/');
    console.log('   2. https://testnet-faucet.mempool.co/');
    console.log('   3. https://bitcoinfaucet.uo1.net/');
    console.log(`\n   Send testnet BTC to: ${bitcoinAddress}`);

    // Step 5: Monitor
    console.log('\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    console.log('📍 STEP 5: MONITORING FOR DEPOSITS');
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n');

    console.log('👁️  Bridge is now monitoring Bitcoin testnet...');
    console.log('⏳ Waiting for Bitcoin deposits...\n');

    console.log('When testnet BTC is received:');
    console.log('  1. ✓ Bridge detects the transaction');
    console.log('  2. ✓ Waits for 3 confirmations (~30 minutes)');
    console.log('  3. ✓ Automatically mints wTBTC on Polygon');
    console.log('  4. ✓ Sends to your address: ' + USER_ADDRESS.substring(0, 20) + '...');

    // Listen for events
    wBTCContract.onMint((to, amount, btcTxId) => {
      console.log('\n🎉 SUCCESS! wTBTC MINTED!');
      console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
      console.log(`   To: ${to}`);
      console.log(`   Amount: ${amount} wTBTC`);
      console.log(`   Bitcoin TX: ${btcTxId}`);
      console.log(`   View on Polygonscan: https://amoy.polygonscan.com/token/${contractAddress}?a=${to}`);
      console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n');
    });

    wBTCContract.onBurn((from, amount, btcAddress) => {
      console.log('\n🔥 BURN EVENT DETECTED');
      console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
      console.log(`   From: ${from}`);
      console.log(`   Amount: ${amount} wTBTC`);
      console.log(`   Bitcoin Address: ${btcAddress}`);
      console.log(`   ⚠️  Manual step: Send ${amount} testnet BTC to ${btcAddress}`);
      console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n');
    });

    // Status updates every 30 seconds
    setInterval(() => {
      const status = bridgeOperator.getStatus();
      console.log(`\n📊 Bridge Status: ${new Date().toLocaleTimeString()}`);
      console.log(`   Total Requests: ${status.totalRequests}`);
      console.log(`   Pending: ${status.pendingRequests}`);
      console.log(`   Completed: ${status.mintedRequests}`);
    }, 30000);

    console.log('\n✨ Bridge is fully operational!');
    console.log('🔄 Monitoring in progress... (Press Ctrl+C to stop)\n');

    // Graceful shutdown
    process.on('SIGINT', () => {
      console.log('\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
      console.log('🛑 Shutting down bridge system...');
      bridgeOperator.stop();

      const finalStatus = bridgeOperator.getStatus();
      console.log('\n📊 FINAL STATISTICS:');
      console.log(`   Total Requests: ${finalStatus.totalRequests}`);
      console.log(`   Successfully Minted: ${finalStatus.mintedRequests}`);
      console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
      console.log('\n✨ Thank you for using Nexus Bridge! ✨\n');

      process.exit(0);
    });

  } catch (error: any) {
    console.error('\n❌ ERROR:', error.message);
    console.error(error);
    process.exit(1);
  }
}

/**
 * Generate a mock Bitcoin testnet address for demonstration
 * In production, this would come from an actual Bitcoin wallet
 */
function generateTestnetAddress(): string {
  // Testnet addresses start with 'tb1' (bech32) or 'm'/'n' (legacy)
  const chars = 'qpzry9x8gf2tvdw0s3jn54khce6mua7l';
  let address = 'tb1q';

  for (let i = 0; i < 39; i++) {
    address += chars[Math.floor(Math.random() * chars.length)];
  }

  return address;
}

// Run
if (require.main === module) {
  main().catch(console.error);
}

export default main;
