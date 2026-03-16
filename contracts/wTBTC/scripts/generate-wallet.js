const { ethers } = require("hardhat");

async function main() {
  console.log("=".repeat(60));
  console.log("🔐 GENERATING TEST WALLET FOR DEPLOYMENT");
  console.log("=".repeat(60));
  console.log();

  // Generate a random wallet
  const wallet = ethers.Wallet.createRandom();

  console.log("📋 Wallet Details:");
  console.log("-".repeat(60));
  console.log("Address:", wallet.address);
  console.log("Private Key:", wallet.privateKey);
  console.log("Private Key (without 0x):", wallet.privateKey.substring(2));
  console.log();

  console.log("⚠️  IMPORTANT NOTES:");
  console.log("-".repeat(60));
  console.log("1. This is a TEST wallet for demonstration");
  console.log("2. NEVER use this for real funds");
  console.log("3. For production, use your own secure wallet");
  console.log("4. Keep your private key SAFE and SECRET");
  console.log();

  console.log("💰 To Deploy:");
  console.log("-".repeat(60));
  console.log("1. Get Sepolia ETH from faucet:");
  console.log("   https://sepoliafaucet.com/");
  console.log();
  console.log("2. Send 0.02 Sepolia ETH to:", wallet.address);
  console.log();
  console.log("3. Add to .env file:");
  console.log(`   PRIVATE_KEY=${wallet.privateKey.substring(2)}`);
  console.log(`   BRIDGE_OPERATOR_ADDRESS=${wallet.address}`);
  console.log();
  console.log("4. Deploy:");
  console.log("   npm run deploy:sepolia");
  console.log();

  console.log("=".repeat(60));
}

main().catch(console.error);
