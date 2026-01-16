/**
 * Deployment script for TestnetWBTC token
 * Run with: npx hardhat run scripts/deploy_token.js --network sepolia
 */

const hre = require("hardhat");

async function main() {
  console.log("\n╔═══════════════════════════════════════════════════════════════╗");
  console.log("║                                                               ║");
  console.log("║              TESTNET WBTC TOKEN DEPLOYMENT                    ║");
  console.log("║                                                               ║");
  console.log("╚═══════════════════════════════════════════════════════════════╝\n");

  // Get deployer account
  const [deployer] = await hre.ethers.getSigners();
  console.log("🔑 Deploying with account:", deployer.address);

  // Check balance
  const balance = await hre.ethers.provider.getBalance(deployer.address);
  console.log("💰 Account balance:", hre.ethers.formatEther(balance), "ETH");

  if (balance === 0n) {
    console.log("\n❌ ERROR: Insufficient balance!");
    console.log("Get Sepolia ETH from:");
    console.log("   • https://www.alchemy.com/faucets/ethereum-sepolia");
    console.log("   • https://sepoliafaucet.com");
    return;
  }

  console.log("\n📋 Token Configuration:");
  console.log("   Name: Testnet Wrapped Bitcoin");
  console.log("   Symbol: tWBTC");
  console.log("   Decimals: 8");
  console.log("   Initial Supply: 5 tWBTC");

  // Deploy contract
  console.log("\n🚀 Deploying TestnetWBTC contract...");

  const TestnetWBTC = await hre.ethers.getContractFactory("TestnetWBTC");
  const token = await TestnetWBTC.deploy(5); // 5 tokens initial supply

  await token.waitForDeployment();

  const contractAddress = await token.getAddress();

  console.log("\n✅ Deployment Successful!");
  console.log("═══════════════════════════════════════════════════════════════");
  console.log("\n📍 Contract Address:", contractAddress);
  console.log("\n🎯 ADD TO METAMASK:");
  console.log("   Contract Address:", contractAddress);
  console.log("   Token Symbol: tWBTC");
  console.log("   Decimals: 8");

  // Verify token info
  console.log("\n🔍 Verifying deployment...");
  const name = await token.name();
  const symbol = await token.symbol();
  const decimals = await token.decimals();
  const totalSupply = await token.totalSupply();
  const owner = await token.owner();
  const deployerBalance = await token.balanceOf(deployer.address);

  console.log("\n📊 Token Information:");
  console.log("   Name:", name);
  console.log("   Symbol:", symbol);
  console.log("   Decimals:", decimals);
  console.log("   Total Supply:", hre.ethers.formatUnits(totalSupply, decimals), symbol);
  console.log("   Owner:", owner);
  console.log("   Your Balance:", hre.ethers.formatUnits(deployerBalance, decimals), symbol);

  console.log("\n🔗 Useful Links:");
  console.log("   Sepolia Etherscan:", `https://sepolia.etherscan.io/address/${contractAddress}`);
  console.log("   Your Wallet:", `https://sepolia.etherscan.io/address/${deployer.address}`);

  console.log("\n💸 To Send Tokens:");
  console.log("   1. Add token to MetaMask using contract address above");
  console.log("   2. Click on tWBTC in MetaMask");
  console.log("   3. Click 'Send'");
  console.log("   4. Paste recipient address");
  console.log("   5. Enter amount and confirm");

  console.log("\n🪙 To Mint More Tokens:");
  console.log("   Use the mint function with recipient address and amount");
  console.log("   Example: mint('0x...', 500000000) for 5 tokens");

  console.log("\n═══════════════════════════════════════════════════════════════");
  console.log("✅ DEPLOYMENT COMPLETE");
  console.log("═══════════════════════════════════════════════════════════════\n");

  // Save deployment info
  const deploymentInfo = {
    network: hre.network.name,
    contractAddress: contractAddress,
    deployer: deployer.address,
    timestamp: new Date().toISOString(),
    tokenInfo: {
      name: name,
      symbol: symbol,
      decimals: decimals.toString(),
      totalSupply: hre.ethers.formatUnits(totalSupply, decimals)
    }
  };

  console.log("\n💾 Deployment Info (save this):");
  console.log(JSON.stringify(deploymentInfo, null, 2));
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error("\n❌ Deployment Failed!");
    console.error(error);
    process.exit(1);
  });
