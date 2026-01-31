#!/usr/bin/env node
/**
 * WTBTC DEPLOYMENT - Deploy, Verify, Publish
 * Wrapped Bitcoin 1:1 BTC peg on Ethereum
 */

const { ethers } = require('ethers');
const fs = require('fs');
const path = require('path');
const solc = require('solc');
require('dotenv').config();

const OWNER_ADDRESS = "0x9fe74d9d6f1ae0ce1fb3b51d4a82c05b74e280f3";
const INITIAL_SUPPLY = 21000000; // 21 million WTBTC (same as Bitcoin max supply)

async function compileWTBTC() {
    console.log("🔨 Compiling WTBTC contract...");

    const source = fs.readFileSync(
        path.join(__dirname, 'contracts', 'WTBTC.sol'),
        'utf8'
    );

    const input = {
        language: 'Solidity',
        sources: { 'WTBTC.sol': { content: source } },
        settings: {
            outputSelection: { '*': { '*': ['abi', 'evm.bytecode'] } },
            optimizer: { enabled: true, runs: 200 }
        }
    };

    const output = JSON.parse(solc.compile(JSON.stringify(input)));

    if (output.errors) {
        const errors = output.errors.filter(e => e.severity === 'error');
        if (errors.length > 0) {
            console.error("❌ Compilation failed");
            errors.forEach(err => console.error(err.formattedMessage));
            process.exit(1);
        }
    }

    const contract = output.contracts['WTBTC.sol']['WTBTC'];
    console.log("✅ Compiled!");

    return {
        abi: contract.abi,
        bytecode: '0x' + contract.evm.bytecode.object
    };
}

async function deploy(network = 'mainnet') {
    console.log("\n" + "=".repeat(70));
    console.log("🪙 DEPLOYING WTBTC - WRAPPED BITCOIN 1:1");
    console.log("=".repeat(70));

    // Compile first
    const { abi, bytecode } = await compileWTBTC();

    // Network configs
    const networks = {
        mainnet: {
            name: 'Ethereum Mainnet',
            rpc: process.env.MAINNET_RPC_URL,
            explorer: 'https://etherscan.io',
            chainId: 1
        },
        sepolia: {
            name: 'Sepolia Testnet',
            rpc: process.env.SEPOLIA_RPC_URL,
            explorer: 'https://sepolia.etherscan.io',
            chainId: 11155111
        }
    };

    const config = networks[network];
    console.log(`\n📡 Network: ${config.name}`);
    console.log(`⛓️  Chain ID: ${config.chainId}`);

    // Connect
    const provider = new ethers.JsonRpcProvider(config.rpc);
    const privateKey = process.env.PRIVATE_KEY.startsWith('0x')
        ? process.env.PRIVATE_KEY
        : '0x' + process.env.PRIVATE_KEY;
    const wallet = new ethers.Wallet(privateKey, provider);

    console.log(`\n🔑 Deployer: ${wallet.address}`);

    // Check balance
    const balance = await provider.getBalance(wallet.address);
    console.log(`💰 Balance: ${ethers.formatEther(balance)} ETH`);

    if (balance === 0n) {
        console.error("\n❌ No ETH! Send some to:", wallet.address);
        process.exit(1);
    }

    console.log(`\n👤 Token Owner: ${OWNER_ADDRESS}`);
    console.log(`🪙 Initial Supply: ${INITIAL_SUPPLY.toLocaleString()} WTBTC`);
    console.log(`📊 Total Tokens: ${(INITIAL_SUPPLY * 100000000).toLocaleString()} satoshis`);

    // Deploy
    console.log("\n" + "=".repeat(70));
    console.log("🚀 DEPLOYING CONTRACT");
    console.log("=".repeat(70));

    const factory = new ethers.ContractFactory(abi, bytecode, wallet);

    console.log("\n⏳ Sending deployment transaction...");
    const contract = await factory.deploy(INITIAL_SUPPLY);

    const txHash = contract.deploymentTransaction().hash;
    console.log(`📝 Transaction: ${txHash}`);
    console.log("⏳ Waiting for confirmation...");

    await contract.waitForDeployment();
    const address = await contract.getAddress();

    console.log(`\n✅ DEPLOYED: ${address}`);

    // Verify deployment
    console.log("\n" + "=".repeat(70));
    console.log("✅ VERIFYING DEPLOYMENT");
    console.log("=".repeat(70));

    const tokenName = await contract.name();
    const tokenSymbol = await contract.symbol();
    const tokenDecimals = await contract.decimals();
    const tokenSupply = await contract.totalSupply();
    const tokenOwner = await contract.owner();

    console.log(`\n📋 Token Information:`);
    console.log(`   Name: ${tokenName}`);
    console.log(`   Symbol: ${tokenSymbol}`);
    console.log(`   Decimals: ${tokenDecimals}`);
    console.log(`   Total Supply: ${ethers.formatUnits(tokenSupply, tokenDecimals)} ${tokenSymbol}`);
    console.log(`   Owner: ${tokenOwner}`);

    // Check owner balance
    const ownerBalance = await contract.balanceOf(OWNER_ADDRESS);
    console.log(`\n💰 Owner Balance: ${ethers.formatUnits(ownerBalance, tokenDecimals)} ${tokenSymbol}`);

    // Save deployment info
    const deploymentInfo = {
        network: config.name,
        chainId: config.chainId,
        contractAddress: address,
        tokenName: tokenName,
        tokenSymbol: tokenSymbol,
        decimals: Number(tokenDecimals),
        totalSupply: ethers.formatUnits(tokenSupply, tokenDecimals),
        owner: tokenOwner,
        deployer: wallet.address,
        deploymentTx: txHash,
        timestamp: new Date().toISOString(),
        explorerUrl: `${config.explorer}/address/${address}`,
        tokenTrackerUrl: `${config.explorer}/token/${address}`
    };

    const filename = `WTBTC_deployment_${network}.json`;
    fs.writeFileSync(filename, JSON.stringify(deploymentInfo, null, 2));

    // Print summary
    console.log("\n" + "=".repeat(70));
    console.log("🎉 WTBTC DEPLOYED & VERIFIED!");
    console.log("=".repeat(70));

    console.log("\n📄 Deployment saved to:", filename);
    console.log("\n🔗 View on Explorer:");
    console.log(`   ${deploymentInfo.explorerUrl}`);
    console.log("\n🪙 Token Tracker:");
    console.log(`   ${deploymentInfo.tokenTrackerUrl}`);

    console.log("\n📊 ADD TO METAMASK:");
    console.log(`   Contract Address: ${address}`);
    console.log(`   Token Symbol: ${tokenSymbol}`);
    console.log(`   Decimals: ${tokenDecimals}`);

    console.log("\n💡 NEXT STEPS:");
    console.log("   1. Add token to MetaMask using contract address above");
    console.log("   2. Verify contract on Etherscan (optional)");
    console.log("   3. Set up 1:1 BTC peg mechanism");
    console.log("   4. Start accepting BTC deposits for WTBTC minting");

    console.log("\n" + "=".repeat(70));
    console.log("✅ DEPLOYMENT COMPLETE!");
    console.log("=".repeat(70) + "\n");

    return deploymentInfo;
}

// Run
const network = process.argv[2] || 'mainnet';

console.log("\n🌟 WTBTC - Wrapped Bitcoin Deployment");
console.log("🔐 1:1 BTC Peg on Ethereum\n");

deploy(network)
    .then(info => {
        console.log("\n✅ Success! Contract is live and ready!");
        process.exit(0);
    })
    .catch(err => {
        console.error("\n❌ Deployment failed:", err.message);
        console.error(err);
        process.exit(1);
    });
