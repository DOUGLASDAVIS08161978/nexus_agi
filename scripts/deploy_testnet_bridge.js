/**
 * Comprehensive Testnet Bridge Deployment Script
 * Deploys TestnetWBTC and EthereumBridgeToken to Sepolia
 * Mints tokens and transfers to recipient
 * Sets up bridge operator permissions
 *
 * Run with: npx hardhat run scripts/deploy_testnet_bridge.js --network sepolia
 */

const hre = require("hardhat");
require("dotenv").config();

// Configuration from environment variables
const RECIPIENT_ADDRESS = process.env.RECIPIENT_ADDRESS || "0x24f6b1ce11c57d40b542f91ac85fa9eb61f78771";
const INITIAL_WBTC_SUPPLY = process.env.INITIAL_WBTC_SUPPLY || "100";
const INITIAL_BRIDGE_SUPPLY = process.env.INITIAL_BRIDGE_SUPPLY || "1000";

async function main() {
  console.log("\n" + "═".repeat(80));
  console.log("║" + " ".repeat(78) + "║");
  console.log("║" + "     NEXUS AGI - TESTNET BRIDGE DEPLOYMENT TO SEPOLIA".padEnd(78) + "║");
  console.log("║" + " ".repeat(78) + "║");
  console.log("═".repeat(80) + "\n");

  // Get deployer account
  const [deployer] = await hre.ethers.getSigners();
  console.log("🔑 Deploying with account:", deployer.address);

  // Check balance
  const balance = await hre.ethers.provider.getBalance(deployer.address);
  console.log("💰 Account balance:", hre.ethers.formatEther(balance), "ETH");

  if (balance === 0n) {
    console.log("\n❌ ERROR: Insufficient balance!");
    console.log("\n📌 Get Sepolia ETH from:");
    console.log("   • https://www.alchemy.com/faucets/ethereum-sepolia");
    console.log("   • https://sepoliafaucet.com");
    console.log("   • https://sepolia-faucet.pk910.de");
    console.log("\n⚠️  You need Sepolia ETH to pay for gas fees.");
    console.log("    Request ~0.5 ETH from the faucets above.\n");
    return;
  }

  console.log("\n" + "─".repeat(80));
  console.log("  PHASE 1: DEPLOYING TESTNET WBTC CONTRACT");
  console.log("─".repeat(80) + "\n");

  console.log("📋 TestnetWBTC Configuration:");
  console.log("   Name: Testnet Wrapped Bitcoin");
  console.log("   Symbol: tWBTC");
  console.log("   Decimals: 8");
  console.log("   Initial Supply:", INITIAL_WBTC_SUPPLY, "tWBTC");
  console.log("   Recipient:", RECIPIENT_ADDRESS);

  // Deploy TestnetWBTC
  console.log("\n🚀 Deploying TestnetWBTC...");
  const TestnetWBTC = await hre.ethers.getContractFactory("TestnetWBTC");
  const wbtc = await TestnetWBTC.deploy(INITIAL_WBTC_SUPPLY);
  await wbtc.waitForDeployment();
  const wbtcAddress = await wbtc.getAddress();

  console.log("✅ TestnetWBTC deployed to:", wbtcAddress);

  // Verify TestnetWBTC deployment
  console.log("\n🔍 Verifying TestnetWBTC deployment...");
  const wbtcName = await wbtc.name();
  const wbtcSymbol = await wbtc.symbol();
  const wbtcDecimals = await wbtc.decimals();
  const wbtcTotalSupply = await wbtc.totalSupply();

  console.log("   Name:", wbtcName);
  console.log("   Symbol:", wbtcSymbol);
  console.log("   Decimals:", wbtcDecimals);
  console.log("   Total Supply:", hre.ethers.formatUnits(wbtcTotalSupply, wbtcDecimals), wbtcSymbol);

  // Mint additional tokens to recipient
  console.log("\n💸 Minting 10 tWBTC to recipient:", RECIPIENT_ADDRESS);
  const mintAmount = 10n * (10n ** 8n); // 10 tWBTC with 8 decimals
  const mintTx = await wbtc.mint(RECIPIENT_ADDRESS, mintAmount);
  await mintTx.wait();
  console.log("✅ Minted successfully! Tx:", mintTx.hash);

  const recipientBalance = await wbtc.balanceOf(RECIPIENT_ADDRESS);
  console.log("   Recipient balance:", hre.ethers.formatUnits(recipientBalance, 8), "tWBTC");

  console.log("\n" + "─".repeat(80));
  console.log("  PHASE 2: DEPLOYING ETHEREUM BRIDGE TOKEN CONTRACT");
  console.log("─".repeat(80) + "\n");

  console.log("📋 EthereumBridgeToken Configuration:");
  console.log("   Name: Cross-Chain Bridge Token");
  console.log("   Symbol: XCBT");
  console.log("   Decimals: 8");
  console.log("   Initial Supply:", INITIAL_BRIDGE_SUPPLY, "XCBT");

  // Deploy EthereumBridgeToken
  console.log("\n🚀 Deploying EthereumBridgeToken...");
  const EthereumBridgeToken = await hre.ethers.getContractFactory("EthereumBridgeToken");
  const bridgeToken = await EthereumBridgeToken.deploy(INITIAL_BRIDGE_SUPPLY);
  await bridgeToken.waitForDeployment();
  const bridgeTokenAddress = await bridgeToken.getAddress();

  console.log("✅ EthereumBridgeToken deployed to:", bridgeTokenAddress);

  // Verify bridge token deployment
  console.log("\n🔍 Verifying EthereumBridgeToken deployment...");
  const bridgeName = await bridgeToken.name();
  const bridgeSymbol = await bridgeToken.symbol();
  const bridgeDecimals = await bridgeToken.decimals();
  const bridgeTotalSupply = await bridgeToken.totalSupply();

  console.log("   Name:", bridgeName);
  console.log("   Symbol:", bridgeSymbol);
  console.log("   Decimals:", bridgeDecimals);
  console.log("   Total Supply:", hre.ethers.formatUnits(bridgeTotalSupply, bridgeDecimals), bridgeSymbol);

  console.log("\n" + "─".repeat(80));
  console.log("  PHASE 3: CONFIGURING BRIDGE OPERATOR");
  console.log("─".repeat(80) + "\n");

  // Add bridge operator
  console.log("🔧 Adding bridge operator:", RECIPIENT_ADDRESS);
  const addOperatorTx = await bridgeToken.addBridgeOperator(RECIPIENT_ADDRESS);
  await addOperatorTx.wait();
  console.log("✅ Bridge operator added! Tx:", addOperatorTx.hash);

  // Verify operator was added
  const isOperator = await bridgeToken.bridgeOperators(RECIPIENT_ADDRESS);
  console.log("   Operator status:", isOperator ? "✅ Active" : "❌ Not active");

  // Mint bridge tokens to recipient
  console.log("\n💸 Minting 50 XCBT to recipient:", RECIPIENT_ADDRESS);
  const bridgeMintAmount = 50n * (10n ** 8n); // 50 XCBT with 8 decimals
  const bridgeMintTx = await bridgeToken.mint(RECIPIENT_ADDRESS, bridgeMintAmount);
  await bridgeMintTx.wait();
  console.log("✅ Minted successfully! Tx:", bridgeMintTx.hash);

  const recipientBridgeBalance = await bridgeToken.balanceOf(RECIPIENT_ADDRESS);
  console.log("   Recipient balance:", hre.ethers.formatUnits(recipientBridgeBalance, 8), "XCBT");

  console.log("\n" + "═".repeat(80));
  console.log("║" + " ".repeat(78) + "║");
  console.log("║" + "                    ✅ DEPLOYMENT SUCCESSFUL!".padEnd(78) + "║");
  console.log("║" + " ".repeat(78) + "║");
  console.log("═".repeat(80) + "\n");

  console.log("📝 DEPLOYMENT SUMMARY\n");
  console.log("🪙 TestnetWBTC (tWBTC):");
  console.log("   Contract Address:", wbtcAddress);
  console.log("   Etherscan:", `https://sepolia.etherscan.io/address/${wbtcAddress}`);
  console.log("   Your Balance:", hre.ethers.formatUnits(recipientBalance, 8), "tWBTC");

  console.log("\n🌉 EthereumBridgeToken (XCBT):");
  console.log("   Contract Address:", bridgeTokenAddress);
  console.log("   Etherscan:", `https://sepolia.etherscan.io/address/${bridgeTokenAddress}`);
  console.log("   Your Balance:", hre.ethers.formatUnits(recipientBridgeBalance, 8), "XCBT");

  console.log("\n👤 Your Wallet:");
  console.log("   Address:", RECIPIENT_ADDRESS);
  console.log("   Etherscan:", `https://sepolia.etherscan.io/address/${RECIPIENT_ADDRESS}`);

  console.log("\n🎯 ADD TO METAMASK:\n");
  console.log("   1. TestnetWBTC (tWBTC)");
  console.log("      Contract Address:", wbtcAddress);
  console.log("      Symbol: tWBTC");
  console.log("      Decimals: 8\n");

  console.log("   2. EthereumBridgeToken (XCBT)");
  console.log("      Contract Address:", bridgeTokenAddress);
  console.log("      Symbol: XCBT");
  console.log("      Decimals: 8\n");

  console.log("🌉 BRIDGE FUNCTIONS AVAILABLE:\n");
  console.log("   • bridgeToBitcoin(btcAddress, amount) - Bridge XCBT to Bitcoin");
  console.log("   • bridgeToPolygon(polygonAddress, amount) - Bridge XCBT to Polygon");
  console.log("   • bridgeFromBitcoin(recipient, amount, btcTxId) - Receive from Bitcoin");
  console.log("   • bridgeFromPolygon(recipient, amount, polygonTxHash) - Receive from Polygon");

  console.log("\n📚 NEXT STEPS:\n");
  console.log("   1. Add both tokens to MetaMask using the contract addresses above");
  console.log("   2. Verify tokens appear in your wallet");
  console.log("   3. Use the bridge interaction script to test bridging");
  console.log("   4. Run: node scripts/interact_bridge.js");

  // Save deployment info to file
  const deploymentInfo = {
    network: hre.network.name,
    timestamp: new Date().toISOString(),
    deployer: deployer.address,
    recipient: RECIPIENT_ADDRESS,
    contracts: {
      testnetWBTC: {
        address: wbtcAddress,
        name: wbtcName,
        symbol: wbtcSymbol,
        decimals: wbtcDecimals.toString(),
        totalSupply: hre.ethers.formatUnits(wbtcTotalSupply, wbtcDecimals),
        recipientBalance: hre.ethers.formatUnits(recipientBalance, 8),
        etherscan: `https://sepolia.etherscan.io/address/${wbtcAddress}`
      },
      ethereumBridgeToken: {
        address: bridgeTokenAddress,
        name: bridgeName,
        symbol: bridgeSymbol,
        decimals: bridgeDecimals.toString(),
        totalSupply: hre.ethers.formatUnits(bridgeTotalSupply, bridgeDecimals),
        recipientBalance: hre.ethers.formatUnits(recipientBridgeBalance, 8),
        bridgeOperator: RECIPIENT_ADDRESS,
        etherscan: `https://sepolia.etherscan.io/address/${bridgeTokenAddress}`
      }
    },
    metamaskImport: {
      testnetWBTC: {
        address: wbtcAddress,
        symbol: "tWBTC",
        decimals: 8
      },
      bridgeToken: {
        address: bridgeTokenAddress,
        symbol: "XCBT",
        decimals: 8
      }
    }
  };

  const fs = require('fs');
  const deploymentFile = `deployment_${hre.network.name}_${Date.now()}.json`;
  fs.writeFileSync(deploymentFile, JSON.stringify(deploymentInfo, null, 2));
  console.log("\n💾 Deployment info saved to:", deploymentFile);

  console.log("\n" + "═".repeat(80) + "\n");
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error("\n" + "═".repeat(80));
    console.error("║  ❌ DEPLOYMENT FAILED!");
    console.error("═".repeat(80) + "\n");
    console.error(error);
    process.exit(1);
  });
