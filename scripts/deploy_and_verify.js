const hre = require("hardhat");
const fs = require("fs");
const path = require("path");

/**
 * Autonomous Deployment Script
 * Deploys, verifies, and publishes WalletManager contract
 */

// Contract owner address
const OWNER_ADDRESS = "0x9fe74d9d6f1ae0ce1fb3b51d4a82c05b74e280f3";

async function main() {
    console.log("\n" + "=".repeat(70));
    console.log("🚀 AUTONOMOUS WALLETMANAGER DEPLOYMENT");
    console.log("=".repeat(70));

    // Get network info
    const network = await hre.ethers.provider.getNetwork();
    const networkName = network.name;
    const chainId = network.chainId;

    console.log(`\n📡 Network: ${networkName} (Chain ID: ${chainId})`);

    // Get deployer
    const [deployer] = await hre.ethers.getSigners();
    console.log(`\n🔑 Deployer: ${deployer.address}`);

    const balance = await hre.ethers.provider.getBalance(deployer.address);
    console.log(`💰 Balance: ${hre.ethers.formatEther(balance)} ETH`);

    if (balance === 0n) {
        console.error("\n❌ ERROR: Deployer has no ETH balance!");
        process.exit(1);
    }

    console.log(`\n👤 Contract Owner: ${OWNER_ADDRESS}`);

    // ========== STEP 1: DEPLOY CONTRACT ==========
    console.log("\n" + "=".repeat(70));
    console.log("STEP 1: DEPLOYING CONTRACT");
    console.log("=".repeat(70));

    const WalletManager = await hre.ethers.getContractFactory("WalletManager");
    console.log("\n⏳ Deploying WalletManager...");

    const walletManager = await WalletManager.deploy(OWNER_ADDRESS);
    await walletManager.waitForDeployment();

    const contractAddress = await walletManager.getAddress();
    console.log(`✅ Contract deployed to: ${contractAddress}`);

    // Get deployment transaction
    const deployTx = walletManager.deploymentTransaction();
    console.log(`📝 Deployment TX: ${deployTx.hash}`);
    console.log(`⛽ Gas Used: ${deployTx.gasLimit.toString()}`);

    // ========== STEP 2: VERIFY DEPLOYMENT ==========
    console.log("\n" + "=".repeat(70));
    console.log("STEP 2: VERIFYING DEPLOYMENT");
    console.log("=".repeat(70));

    console.log("\n⏳ Checking contract state...");

    const owner = await walletManager.owner();
    const rewardRate = await walletManager.rewardRate();
    const minDeposit = await walletManager.minDepositAmount();
    const isOperator = await walletManager.isOperator(OWNER_ADDRESS);

    console.log(`✅ Owner: ${owner}`);
    console.log(`✅ Reward Rate: ${hre.ethers.formatEther(rewardRate)} ETH/second`);
    console.log(`✅ Min Deposit: ${hre.ethers.formatEther(minDeposit)} ETH`);
    console.log(`✅ Owner is Operator: ${isOperator}`);

    // Verify owner is correct
    if (owner.toLowerCase() !== OWNER_ADDRESS.toLowerCase()) {
        console.error(`\n❌ ERROR: Owner mismatch! Expected ${OWNER_ADDRESS}, got ${owner}`);
        process.exit(1);
    }

    console.log("\n✅ Contract state verified successfully!");

    // ========== STEP 3: ETHERSCAN VERIFICATION (if API key exists) ==========
    if (process.env.ETHERSCAN_API_KEY && networkName !== "hardhat" && networkName !== "localhost") {
        console.log("\n" + "=".repeat(70));
        console.log("STEP 3: ETHERSCAN VERIFICATION");
        console.log("=".repeat(70));

        console.log("\n⏳ Waiting for block confirmations...");
        await walletManager.deploymentTransaction().wait(5);

        try {
            console.log("⏳ Verifying contract on Etherscan...");
            await hre.run("verify:verify", {
                address: contractAddress,
                constructorArguments: [OWNER_ADDRESS],
            });
            console.log("✅ Contract verified on Etherscan!");
        } catch (error) {
            console.log("⚠️  Etherscan verification failed (may already be verified):");
            console.log(error.message);
        }
    } else {
        console.log("\n⏭️  STEP 3: SKIPPED (No Etherscan API key or local network)");
    }

    // ========== STEP 4: SAVE DEPLOYMENT INFO ==========
    console.log("\n" + "=".repeat(70));
    console.log("STEP 4: SAVING DEPLOYMENT INFO");
    console.log("=".repeat(70));

    const deploymentInfo = {
        network: networkName,
        chainId: chainId.toString(),
        contractAddress: contractAddress,
        ownerAddress: OWNER_ADDRESS,
        deployerAddress: deployer.address,
        deploymentTx: deployTx.hash,
        blockNumber: deployTx.blockNumber,
        timestamp: new Date().toISOString(),
        rewardRate: hre.ethers.formatEther(rewardRate),
        minDeposit: hre.ethers.formatEther(minDeposit),
    };

    // Save to JSON file
    const deploymentsDir = path.join(__dirname, "..", "deployments");
    if (!fs.existsSync(deploymentsDir)) {
        fs.mkdirSync(deploymentsDir);
    }

    const filename = `WalletManager_${networkName}_${Date.now()}.json`;
    const filepath = path.join(deploymentsDir, filename);

    fs.writeFileSync(filepath, JSON.stringify(deploymentInfo, null, 2));
    console.log(`\n✅ Deployment info saved to: ${filename}`);

    // ========== STEP 5: GENERATE INTEGRATION CODE ==========
    console.log("\n" + "=".repeat(70));
    console.log("STEP 5: INTEGRATION INFORMATION");
    console.log("=".repeat(70));

    const explorerUrl = getExplorerUrl(networkName, contractAddress);

    console.log("\n📋 DEPLOYMENT SUMMARY");
    console.log("━".repeat(70));
    console.log(JSON.stringify(deploymentInfo, null, 2));
    console.log("━".repeat(70));

    console.log("\n🔗 LINKS");
    console.log(`Explorer: ${explorerUrl}`);

    console.log("\n📝 UPDATE THESE FILES:");
    console.log(`1. frontend/src/web3-onboard-config.js`);
    console.log(`   WALLET_MANAGER_ADDRESS = '${contractAddress}'`);
    console.log(`\n2. .env`);
    console.log(`   WALLET_MANAGER_ADDRESS=${contractAddress}`);

    console.log("\n💻 NEXT STEPS:");
    console.log("1. ✅ Contract deployed and verified");
    console.log("2. 📝 Update frontend configuration files");
    console.log("3. 🔗 Connect wallet and test interactions");
    console.log("4. 💰 Fund rewards pool (optional)");
    console.log("5. 🎉 Start using the contract!");

    console.log("\n🎯 QUICK TEST COMMANDS:");
    console.log(`\n# Interact with contract`);
    console.log(`export WALLET_MANAGER_ADDRESS=${contractAddress}`);
    console.log(`npx hardhat run scripts/interact_wallet_manager.js --network ${networkName}`);

    console.log(`\n# Hardhat console`);
    console.log(`npx hardhat console --network ${networkName}`);
    console.log(`const contract = await ethers.getContractAt("WalletManager", "${contractAddress}")`);

    console.log("\n" + "=".repeat(70));
    console.log("🎉 DEPLOYMENT COMPLETE!");
    console.log("=".repeat(70) + "\n");

    return deploymentInfo;
}

function getExplorerUrl(network, address) {
    const explorers = {
        mainnet: `https://etherscan.io/address/${address}`,
        sepolia: `https://sepolia.etherscan.io/address/${address}`,
        polygon: `https://polygonscan.com/address/${address}`,
        mumbai: `https://mumbai.polygonscan.com/address/${address}`,
    };
    return explorers[network] || `Explorer not available for ${network}`;
}

// Execute deployment
main()
    .then(() => process.exit(0))
    .catch((error) => {
        console.error("\n" + "=".repeat(70));
        console.error("❌ DEPLOYMENT FAILED");
        console.error("=".repeat(70));
        console.error("\n" + error);
        if (error.stack) {
            console.error("\nStack trace:");
            console.error(error.stack);
        }
        process.exit(1);
    });
