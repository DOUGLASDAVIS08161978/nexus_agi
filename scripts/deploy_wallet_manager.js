const hre = require("hardhat");

/**
 * Deploy WalletManager Contract
 *
 * This script deploys the WalletManager contract with a specified owner address
 */

// Target wallet address to set as owner
const OWNER_ADDRESS = "0x24f6b1ce11c57d40b542f91ac85fa9eb61f78771";

async function main() {
    console.log("=".repeat(60));
    console.log("Deploying WalletManager Contract");
    console.log("=".repeat(60));

    // Get network info
    const network = await hre.ethers.provider.getNetwork();
    console.log(`\nNetwork: ${network.name} (Chain ID: ${network.chainId})`);

    // Get deployer
    const [deployer] = await hre.ethers.getSigners();
    console.log(`Deployer: ${deployer.address}`);

    const balance = await hre.ethers.provider.getBalance(deployer.address);
    console.log(`Deployer Balance: ${hre.ethers.formatEther(balance)} ETH`);

    console.log(`\nContract Owner will be: ${OWNER_ADDRESS}`);

    // Deploy WalletManager
    console.log("\n📝 Deploying WalletManager...");
    const WalletManager = await hre.ethers.getContractFactory("WalletManager");
    const walletManager = await WalletManager.deploy(OWNER_ADDRESS);

    await walletManager.waitForDeployment();
    const contractAddress = await walletManager.getAddress();

    console.log(`✅ WalletManager deployed to: ${contractAddress}`);

    // Verify deployment
    console.log("\n📋 Verifying deployment...");
    const owner = await walletManager.owner();
    const rewardRate = await walletManager.rewardRate();
    const minDeposit = await walletManager.minDepositAmount();

    console.log(`   Owner: ${owner}`);
    console.log(`   Reward Rate: ${rewardRate.toString()} wei/sec`);
    console.log(`   Min Deposit: ${hre.ethers.formatEther(minDeposit)} ETH`);

    // Check if deployer is operator
    const isOperator = await walletManager.isOperator(OWNER_ADDRESS);
    console.log(`   ${OWNER_ADDRESS} is operator: ${isOperator}`);

    // Summary
    console.log("\n" + "=".repeat(60));
    console.log("✅ DEPLOYMENT COMPLETE");
    console.log("=".repeat(60));
    console.log(`\nContract Address: ${contractAddress}`);
    console.log(`Owner Address: ${owner}`);
    console.log(`Network: ${network.name}`);

    // Save deployment info
    const deploymentInfo = {
        network: network.name,
        chainId: network.chainId.toString(),
        contractAddress: contractAddress,
        ownerAddress: owner,
        deployerAddress: deployer.address,
        timestamp: new Date().toISOString(),
        blockNumber: await hre.ethers.provider.getBlockNumber()
    };

    console.log("\n📄 Deployment Info:");
    console.log(JSON.stringify(deploymentInfo, null, 2));

    // Instructions
    console.log("\n" + "=".repeat(60));
    console.log("📚 NEXT STEPS");
    console.log("=".repeat(60));
    console.log("\n1. Save the contract address for your records");
    console.log("2. Connect your wallet to interact with the contract");
    console.log("3. Use depositETH() to deposit funds");
    console.log("4. Use withdrawETH() to withdraw funds");
    console.log("5. Use claimRewards() to claim accumulated rewards");

    console.log("\n💡 Example interactions:");
    console.log(`   - Deposit: contract.depositETH({ value: ethers.parseEther("0.1") })`);
    console.log(`   - Withdraw: contract.withdrawETH(ethers.parseEther("0.05"))`);
    console.log(`   - Check Balance: contract.getETHBalance("${OWNER_ADDRESS}")`);
    console.log(`   - Claim Rewards: contract.claimRewards()`);

    return deploymentInfo;
}

// Execute deployment
main()
    .then(() => process.exit(0))
    .catch((error) => {
        console.error("\n❌ Deployment failed:");
        console.error(error);
        process.exit(1);
    });
