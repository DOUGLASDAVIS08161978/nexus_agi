const hre = require("hardhat");

/**
 * Interact with WalletManager Contract
 *
 * This script provides interactive functions to work with the deployed WalletManager contract
 */

// Update this with your deployed contract address
const CONTRACT_ADDRESS = process.env.WALLET_MANAGER_ADDRESS || "0x798d8B4D8677c4cf3Bc9B0B38ba8dfe318DE60E4";

// Your wallet address
const YOUR_WALLET = "0x24f6b1ce11c57d40b542f91ac85fa9eb61f78771";

async function main() {
    console.log("=".repeat(60));
    console.log("WalletManager Contract Interaction");
    console.log("=".repeat(60));

    // Check if contract address is set
    if (CONTRACT_ADDRESS === "YOUR_CONTRACT_ADDRESS_HERE") {
        console.error("\n❌ Please set CONTRACT_ADDRESS in this script or WALLET_MANAGER_ADDRESS in .env");
        console.log("\nTo deploy first, run:");
        console.log("  npx hardhat run scripts/deploy_wallet_manager.js --network <network>");
        process.exit(1);
    }

    // Get signer
    const [signer] = await hre.ethers.getSigners();
    console.log(`\n🔑 Using account: ${signer.address}`);

    const balance = await hre.ethers.provider.getBalance(signer.address);
    console.log(`💰 Account Balance: ${hre.ethers.formatEther(balance)} ETH`);

    // Get contract
    console.log(`\n📄 Contract Address: ${CONTRACT_ADDRESS}`);
    const WalletManager = await hre.ethers.getContractFactory("WalletManager");
    const contract = WalletManager.attach(CONTRACT_ADDRESS);

    // Get contract info
    console.log("\n" + "=".repeat(60));
    console.log("📊 CONTRACT INFORMATION");
    console.log("=".repeat(60));

    try {
        const owner = await contract.owner();
        const rewardRate = await contract.rewardRate();
        const minDeposit = await contract.minDepositAmount();
        const totalDeposited = await contract.totalDeposited();
        const contractBalance = await contract.getContractBalance();

        console.log(`Owner: ${owner}`);
        console.log(`Reward Rate: ${hre.ethers.formatEther(rewardRate)} ETH/second`);
        console.log(`Min Deposit: ${hre.ethers.formatEther(minDeposit)} ETH`);
        console.log(`Total Deposited: ${hre.ethers.formatEther(totalDeposited)} ETH`);
        console.log(`Contract Balance: ${hre.ethers.formatEther(contractBalance)} ETH`);
    } catch (error) {
        console.error("❌ Failed to get contract info:", error.message);
        process.exit(1);
    }

    // Get user info
    console.log("\n" + "=".repeat(60));
    console.log("👤 YOUR WALLET INFORMATION");
    console.log("=".repeat(60));

    const userETHBalance = await contract.getETHBalance(signer.address);
    const pendingRewards = await contract.getPendingRewards(signer.address);
    const isOperator = await contract.isOperator(signer.address);

    console.log(`Address: ${signer.address}`);
    console.log(`Deposited in Contract: ${hre.ethers.formatEther(userETHBalance)} ETH`);
    console.log(`Pending Rewards: ${hre.ethers.formatEther(pendingRewards)} ETH`);
    console.log(`Is Operator: ${isOperator}`);

    // Interactive menu
    console.log("\n" + "=".repeat(60));
    console.log("🎮 AVAILABLE ACTIONS");
    console.log("=".repeat(60));
    console.log("\nTo perform actions, modify this script and uncomment the desired action:");
    console.log("\n1. Deposit ETH");
    console.log("2. Withdraw ETH");
    console.log("3. Claim Rewards");
    console.log("4. Check Balance");
    console.log("5. Fund Rewards (owner only)");

    // ====== UNCOMMENT ACTIONS BELOW AS NEEDED ======

    // ACTION 1: Deposit ETH
    // console.log("\n💰 Depositing 0.01 ETH...");
    // const depositTx = await contract.depositETH({
    //     value: hre.ethers.parseEther("0.01")
    // });
    // await depositTx.wait();
    // console.log("✅ Deposit successful!");
    // console.log(`Transaction: ${depositTx.hash}`);

    // ACTION 2: Withdraw ETH
    // console.log("\n💸 Withdrawing 0.005 ETH...");
    // const withdrawTx = await contract.withdrawETH(
    //     hre.ethers.parseEther("0.005")
    // );
    // await withdrawTx.wait();
    // console.log("✅ Withdrawal successful!");
    // console.log(`Transaction: ${withdrawTx.hash}`);

    // ACTION 3: Claim Rewards
    // console.log("\n🎁 Claiming rewards...");
    // const claimTx = await contract.claimRewards();
    // await claimTx.wait();
    // console.log("✅ Rewards claimed!");
    // console.log(`Transaction: ${claimTx.hash}`);

    // ACTION 4: Check Updated Balance
    // const newBalance = await contract.getETHBalance(signer.address);
    // const newRewards = await contract.getPendingRewards(signer.address);
    // console.log(`\nUpdated Balance: ${hre.ethers.formatEther(newBalance)} ETH`);
    // console.log(`Updated Rewards: ${hre.ethers.formatEther(newRewards)} ETH`);

    // ACTION 5: Fund Rewards (Owner only)
    // if (signer.address.toLowerCase() === owner.toLowerCase()) {
    //     console.log("\n💎 Funding rewards pool with 0.1 ETH...");
    //     const fundTx = await contract.fundRewards({
    //         value: hre.ethers.parseEther("0.1")
    //     });
    //     await fundTx.wait();
    //     console.log("✅ Rewards funded!");
    //     console.log(`Transaction: ${fundTx.hash}`);
    // }

    console.log("\n" + "=".repeat(60));
    console.log("✅ Script completed successfully");
    console.log("=".repeat(60));
}

// Helper function for deposits
async function depositETH(amount) {
    const [signer] = await hre.ethers.getSigners();
    const contract = await hre.ethers.getContractAt("WalletManager", CONTRACT_ADDRESS);

    console.log(`💰 Depositing ${amount} ETH...`);
    const tx = await contract.depositETH({
        value: hre.ethers.parseEther(amount)
    });
    await tx.wait();
    console.log(`✅ Deposited ${amount} ETH`);
    console.log(`Transaction: ${tx.hash}`);
}

// Helper function for withdrawals
async function withdrawETH(amount) {
    const [signer] = await hre.ethers.getSigners();
    const contract = await hre.ethers.getContractAt("WalletManager", CONTRACT_ADDRESS);

    console.log(`💸 Withdrawing ${amount} ETH...`);
    const tx = await contract.withdrawETH(hre.ethers.parseEther(amount));
    await tx.wait();
    console.log(`✅ Withdrawn ${amount} ETH`);
    console.log(`Transaction: ${tx.hash}`);
}

// Helper function for claiming rewards
async function claimRewards() {
    const [signer] = await hre.ethers.getSigners();
    const contract = await hre.ethers.getContractAt("WalletManager", CONTRACT_ADDRESS);

    console.log("🎁 Claiming rewards...");
    const tx = await contract.claimRewards();
    await tx.wait();
    console.log("✅ Rewards claimed!");
    console.log(`Transaction: ${tx.hash}`);
}

// Run main function
main()
    .then(() => process.exit(0))
    .catch((error) => {
        console.error("\n❌ Error:");
        console.error(error);
        process.exit(1);
    });

// Export helper functions for use in other scripts
module.exports = {
    depositETH,
    withdrawETH,
    claimRewards
};
