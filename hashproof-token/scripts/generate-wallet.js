const { ethers } = require("hardhat");

async function main() {
  console.log("\n" + "=".repeat(80));
  console.log("ETHEREUM WALLET GENERATOR");
  console.log("=".repeat(80) + "\n");

  // Generate a random wallet
  const wallet = ethers.Wallet.createRandom();

  console.log("🔐 NEW WALLET CREATED:");
  console.log("   Address:     ", wallet.address);
  console.log("   Private Key: ", wallet.privateKey);
  console.log("   Mnemonic:    ", wallet.mnemonic.phrase);

  console.log("\n⚠️  SECURITY WARNINGS:");
  console.log("   - NEVER share your private key or mnemonic");
  console.log("   - Store them in a secure location");
  console.log("   - This is for TESTNET use only");

  console.log("\n💰 GET FREE SEPOLIA ETH:");
  console.log("   1. Copy your address above");
  console.log("   2. Visit these faucets:");
  console.log("      → https://sepoliafaucet.com");
  console.log("      → https://www.alchemy.com/faucets/ethereum-sepolia");
  console.log("      → https://faucet.quicknode.com/ethereum/sepolia");
  console.log("   3. Paste your address and request test ETH");
  console.log("   4. Wait 30-60 seconds for the ETH to arrive");

  console.log("\n📝 TO USE THIS WALLET FOR DEPLOYMENT:");
  console.log("   1. Create a .env file in this directory");
  console.log("   2. Add this line:");
  console.log(`      PRIVATE_KEY=${wallet.privateKey}`);
  console.log("   3. Run: npx hardhat run scripts/deploy-sepolia.js --network sepolia");

  console.log("\n" + "=".repeat(80) + "\n");
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });
