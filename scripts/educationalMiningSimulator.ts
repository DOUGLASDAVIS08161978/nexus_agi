/**
 * Educational Bitcoin Mining Simulator
 * Demonstrates Bitcoin mining concepts and proof-of-work
 *
 * ⚠️  EDUCATIONAL PURPOSE ONLY
 * This simulates mining concepts but does NOT mine real Bitcoin.
 * Real Bitcoin mining requires specialized ASIC hardware.
 */

import crypto from 'crypto';

interface Block {
  height: number;
  previousHash: string;
  timestamp: number;
  nonce: number;
  difficulty: number;
  hash: string;
  reward: number;
  transactions: number;
}

class EducationalMiningSimulator {
  private blockchain: Block[] = [];
  private totalRewards = 0;
  private startTime = Date.now();
  private difficulty = 4; // Number of leading zeros
  private readonly USER_ADDRESS = '0x24f6b1ce11c57d40b542f91ac85fa9eb61f78771';

  constructor() {
    console.log(`
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   🎓 EDUCATIONAL BITCOIN MINING SIMULATOR                    ║
║                                                              ║
║   Learn how Bitcoin proof-of-work mining works!             ║
║   This is for EDUCATIONAL PURPOSES ONLY                     ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝

📚 WHAT YOU'LL LEARN:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Proof-of-Work (PoW) - Finding a hash with specific properties
2. Nonce - The number miners increment to find valid hashes
3. Difficulty - How hard it is to find a valid block
4. Block Rewards - How miners are compensated
5. Blockchain Structure - How blocks link together

⚠️  IMPORTANT NOTICE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

This simulator demonstrates mining CONCEPTS but does NOT:
• Mine real Bitcoin (mainnet or testnet)
• Connect to the Bitcoin network
• Generate actual cryptocurrency value
• Use specialized mining hardware (ASICs)

Real Bitcoin mining requires:
• ASIC hardware ($1,000-$10,000+)
• Significant electricity costs
• Connection to a mining pool
• Competitive hash rates (EH/s, not H/s)

For actual testnet BTC, use faucets:
• https://coinfaucet.eu/en/btc-testnet/
• https://testnet-faucet.mempool.co/
• https://bitcoinfaucet.uo1.net/

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Starting simulation...
    `);
  }

  /**
   * Initialize genesis block
   */
  private initializeGenesisBlock(): void {
    const genesis: Block = {
      height: 0,
      previousHash: '0'.repeat(64),
      timestamp: Date.now(),
      nonce: 0,
      difficulty: 0,
      hash: '0'.repeat(64),
      reward: 50.0,
      transactions: 0
    };

    this.blockchain.push(genesis);
    console.log('🔗 Genesis block initialized\n');
  }

  /**
   * Mine a single block (educational demonstration)
   */
  private mineBlock(previousBlock: Block): Block {
    const block: Block = {
      height: previousBlock.height + 1,
      previousHash: previousBlock.hash,
      timestamp: Date.now(),
      nonce: 0,
      difficulty: this.difficulty,
      hash: '',
      reward: 6.25, // Current Bitcoin block reward (post-halving)
      transactions: Math.floor(Math.random() * 2000) + 500
    };

    const target = '0'.repeat(this.difficulty);
    const startTime = Date.now();

    console.log(`⛏️  Mining Block #${block.height}...`);
    console.log(`   Difficulty: ${this.difficulty} (need ${this.difficulty} leading zeros)`);
    console.log(`   Target: ${target}...`);

    // Proof-of-work: Find a nonce that produces a hash with required leading zeros
    while (true) {
      const data = `${block.height}${block.previousHash}${block.timestamp}${block.nonce}${block.transactions}`;
      const hash = crypto.createHash('sha256').update(data).digest('hex');

      if (hash.startsWith(target)) {
        block.hash = hash;
        break;
      }

      block.nonce++;

      // In real mining, you'd continue indefinitely. We limit for demo purposes.
      if (block.nonce > 10000000) {
        block.hash = hash;
        console.log('   ⚠️  Demo limit reached, accepting hash...');
        break;
      }

      // Progress indicator
      if (block.nonce % 100000 === 0) {
        process.stdout.write(`   Trying nonce: ${block.nonce.toLocaleString()}\r`);
      }
    }

    const miningTime = ((Date.now() - startTime) / 1000).toFixed(2);
    const hashRate = (block.nonce / parseFloat(miningTime)).toFixed(0);

    console.log(`\n✅ Block #${block.height} mined!`);
    console.log(`   Hash: ${block.hash.substring(0, 32)}...`);
    console.log(`   Nonce: ${block.nonce.toLocaleString()}`);
    console.log(`   Time: ${miningTime}s`);
    console.log(`   Hash Rate: ${parseInt(hashRate).toLocaleString()} H/s`);
    console.log(`   Reward: ${block.reward} BTC`);
    console.log(`   Transactions: ${block.transactions}`);

    // Educational comparison
    console.log(`\n   📊 REALITY CHECK:`);
    console.log(`   • Your simulated rate: ${parseInt(hashRate).toLocaleString()} H/s`);
    console.log(`   • Bitcoin network rate: ~400,000,000,000,000,000 H/s (400 EH/s)`);
    console.log(`   • You'd need ${(400000000000000000 / parseInt(hashRate)).toExponential(2)} more power!\n`);

    return block;
  }

  /**
   * Simulate mining session
   */
  public async simulateMining(numberOfBlocks: number = 5): Promise<void> {
    this.initializeGenesisBlock();

    console.log(`🎯 Mining ${numberOfBlocks} blocks to demonstrate concepts...\n`);

    for (let i = 0; i < numberOfBlocks; i++) {
      const previousBlock = this.blockchain[this.blockchain.length - 1];
      const newBlock = this.mineBlock(previousBlock);

      this.blockchain.push(newBlock);
      this.totalRewards += newBlock.reward;

      // Increase difficulty every 2 blocks (in real Bitcoin, it's every 2016 blocks)
      if ((i + 1) % 2 === 0 && this.difficulty < 6) {
        this.difficulty++;
        console.log(`📈 Difficulty increased to ${this.difficulty}\n`);
      }

      // Small delay between blocks for readability
      await new Promise(resolve => setTimeout(resolve, 1000));
    }

    this.printSummary();
    this.explainBridging();
  }

  /**
   * Print mining summary
   */
  private printSummary(): void {
    const runtime = ((Date.now() - this.startTime) / 1000).toFixed(2);

    console.log(`
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   📊 MINING SIMULATION COMPLETE                              ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝

📈 STATISTICS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⛓️  Blocks Mined:        ${this.blockchain.length - 1}
💰 Total Rewards:        ${this.totalRewards} BTC (simulated)
⏱️  Runtime:             ${runtime} seconds
🎯 Final Difficulty:     ${this.difficulty}
📦 Total Transactions:   ${this.blockchain.reduce((sum, b) => sum + b.transactions, 0).toLocaleString()}

🔗 BLOCKCHAIN:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
`);

    // Show last 5 blocks
    const recentBlocks = this.blockchain.slice(-5);
    recentBlocks.forEach(block => {
      console.log(`   Block #${block.height.toString().padStart(3, ' ')}: ${block.hash.substring(0, 24)}... (${block.reward} BTC)`);
    });

    console.log(`
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
`);
  }

  /**
   * Explain bridging process
   */
  private explainBridging(): void {
    console.log(`
🌉 HOW TO BRIDGE REAL TESTNET BTC TO POLYGON:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Since actual mining is impractical, here's how to get REAL testnet BTC
and bridge it to your Polygon address:

STEP 1: Get Testnet Bitcoin 🪙
   Visit a Bitcoin testnet faucet:
   • https://coinfaucet.eu/en/btc-testnet/
   • https://testnet-faucet.mempool.co/
   • https://bitcoinfaucet.uo1.net/

   Request testnet BTC to a Bitcoin testnet wallet

STEP 2: Deploy the Bridge 🌉
   Run: npm run deploy-bridge

   This will:
   • Deploy the wTBTC contract to Polygon
   • Set up automated monitoring
   • Give you a Bitcoin deposit address

STEP 3: Send Testnet BTC ₿
   Send your testnet BTC to the provided address
   The bridge monitors the Bitcoin testnet blockchain

STEP 4: Automatic Minting 🪙
   Once your BTC deposit has 3 confirmations:
   • Bridge automatically detects it
   • Mints equivalent wTBTC tokens
   • Sends to your Polygon address: ${this.USER_ADDRESS}

STEP 5: Check Your Balance 💰
   View your wTBTC tokens on Polygon:
   • MetaMask (add custom token)
   • Polygonscan explorer
   • Your balance will show the bridged amount

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💡 KEY CONCEPTS YOU LEARNED:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✓ Proof-of-Work: Finding hashes with specific properties
✓ Nonce: The variable miners increment to find valid blocks
✓ Difficulty: Auto-adjusts to maintain ~10 min block time
✓ Mining Rewards: Compensation for securing the network
✓ Hash Rate: Computational power (real network: 400 EH/s!)
✓ Blockchain: Linked blocks through cryptographic hashes
✓ Bridging: Moving value between different blockchains

🎓 EDUCATIONAL VALUE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

You now understand WHY mining is:
• Computationally expensive
• Requires specialized hardware
• Creates decentralized consensus
• Secures the Bitcoin network
• Distributes new coins fairly

This knowledge helps you understand cryptocurrency fundamentals
without the cost and complexity of real mining!

✨ Ready to deploy the actual bridge? Run: npm run deploy-bridge
    `);
  }
}

// Run simulator
async function main() {
  const simulator = new EducationalMiningSimulator();
  await simulator.simulateMining(5); // Mine 5 blocks for demonstration
}

if (require.main === module) {
  main().catch(console.error);
}

export default EducationalMiningSimulator;
