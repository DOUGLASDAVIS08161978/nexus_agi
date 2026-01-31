#!/usr/bin/env node
/**
 * SIMPLE DEPLOYMENT SCRIPT - NO HARDHAT!
 * Just ethers.js + your wallet
 */

const { ethers } = require('ethers');
const fs = require('fs');
const path = require('path');
require('dotenv').config();

// Load compiled contract
const contractData = JSON.parse(
    fs.readFileSync(path.join(__dirname, 'build', 'WalletManager.json'), 'utf8')
);

const OWNER_ADDRESS = "0x9fe74d9d6f1ae0ce1fb3b51d4a82c05b74e280f3";

async function deploy(network = 'mainnet') {
    console.log("\n" + "=".repeat(60));
    console.log("🚀 DEPLOYING WALLETMANAGER - SIMPLE & FAST!");
    console.log("=".repeat(60));

    // Network configs
    const networks = {
        mainnet: {
            name: 'Ethereum Mainnet',
            rpc: process.env.MAINNET_RPC_URL,
            explorer: 'https://etherscan.io'
        },
        sepolia: {
            name: 'Sepolia Testnet',
            rpc: process.env.SEPOLIA_RPC_URL,
            explorer: 'https://sepolia.etherscan.io'
        }
    };

    const config = networks[network];
    console.log(`\n📡 Network: ${config.name}`);

    // Connect
    const provider = new ethers.JsonRpcProvider(config.rpc);
    const wallet = new ethers.Wallet(process.env.PRIVATE_KEY, provider);

    console.log(`\n🔑 Deployer: ${wallet.address}`);

    // Check balance
    const balance = await provider.getBalance(wallet.address);
    console.log(`💰 Balance: ${ethers.formatEther(balance)} ETH`);

    if (balance === 0n) {
        console.error("\n❌ No ETH! Send some to:", wallet.address);
        process.exit(1);
    }

    console.log(`👤 Owner: ${OWNER_ADDRESS}`);

    // Deploy
    console.log("\n⏳ Deploying...");
    const factory = new ethers.ContractFactory(
        contractData.abi,
        contractData.bytecode,
        wallet
    );

    const contract = await factory.deploy(OWNER_ADDRESS);
    console.log(`📝 TX: ${contract.deploymentTransaction().hash}`);

    await contract.waitForDeployment();
    const address = await contract.getAddress();

    console.log(`\n✅ DEPLOYED: ${address}`);

    // Verify
    const owner = await contract.owner();
    const rewardRate = await contract.rewardRate();
    const minDeposit = await contract.minDepositAmount();

    console.log(`\n📋 Contract Info:`);
    console.log(`   Owner: ${owner}`);
    console.log(`   Reward Rate: ${ethers.formatEther(rewardRate)} ETH/sec`);
    console.log(`   Min Deposit: ${ethers.formatEther(minDeposit)} ETH`);

    // Save
    const info = {
        network: config.name,
        address: address,
        owner: OWNER_ADDRESS,
        deployer: wallet.address,
        tx: contract.deploymentTransaction().hash,
        timestamp: new Date().toISOString(),
        explorer: `${config.explorer}/address/${address}`
    };

    fs.writeFileSync(
        `deployment_${network}.json`,
        JSON.stringify(info, null, 2)
    );

    console.log(`\n🔗 Explorer: ${info.explorer}`);
    console.log(`\n✅ Done! Contract ready to use!`);
    console.log("=".repeat(60) + "\n");

    return info;
}

// Run
const network = process.argv[2] || 'mainnet';
deploy(network).catch(err => {
    console.error("\n❌ Failed:", err.message);
    process.exit(1);
});
