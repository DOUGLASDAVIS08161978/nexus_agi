/**
 * NEXUS AGI - DEPLOY ALL ERC-20 TOKENS
 * =====================================
 *
 * Deploys all 10 ERC-20 tokens to the specified network
 *
 * Usage:
 *   npx hardhat run scripts/deploy-all-tokens.js --network <network>
 *
 * Networks: sepolia, arbitrum, optimism, base, polygon, avalanche, bsc
 */

const hre = require("hardhat");
const fs = require('fs');

async function main() {
    const [deployer] = await hre.ethers.getSigners();

    console.log("=" .repeat(80));
    console.log("NEXUS AGI - DEPLOYING ALL ERC-20 TOKENS");
    console.log("=" .repeat(80));

    console.log("\n📝 Deployment Configuration:");
    console.log(`   Network: ${hre.network.name}`);
    console.log(`   Deployer: ${deployer.address}`);
    console.log(`   Balance: ${hre.ethers.formatEther(await hre.ethers.provider.getBalance(deployer.address))} ETH`);

    const deploymentResults = {
        network: hre.network.name,
        deployer: deployer.address,
        timestamp: new Date().toISOString(),
        tokens: {}
    };

    // Token 1: NexusToken (NEX)
    console.log("\n\n🚀 [1/10] Deploying NexusToken (NEX)...");
    const NexusToken = await hre.ethers.getContractFactory("NexusToken");
    const nexusToken = await NexusToken.deploy(deployer.address);
    await nexusToken.waitForDeployment();
    const nexusAddress = await nexusToken.getAddress();

    console.log(`   ✅ NexusToken deployed to: ${nexusAddress}`);
    deploymentResults.tokens.NexusToken = {
        address: nexusAddress,
        symbol: "NEX",
        name: "Nexus AGI Token",
        type: "Governance"
    };

    // Token 2: HashPowerToken (HASH)
    console.log("\n🚀 [2/10] Deploying HashPowerToken (HASH)...");
    const HashPowerToken = await hre.ethers.getContractFactory("HashPowerToken");
    const hashPowerToken = await HashPowerToken.deploy(deployer.address);
    await hashPowerToken.waitForDeployment();
    const hashPowerAddress = await hashPowerToken.getAddress();

    console.log(`   ✅ HashPowerToken deployed to: ${hashPowerAddress}`);
    deploymentResults.tokens.HashPowerToken = {
        address: hashPowerAddress,
        symbol: "HASH",
        name: "Hash Power Token",
        type: "Mining"
    };

    // Token 3: BridgeRewardToken (BRG)
    console.log("\n🚀 [3/10] Deploying BridgeRewardToken (BRG)...");
    const BridgeRewardToken = await hre.ethers.getContractFactory("BridgeRewardToken");
    const bridgeRewardToken = await BridgeRewardToken.deploy(deployer.address);
    await bridgeRewardToken.waitForDeployment();
    const bridgeRewardAddress = await bridgeRewardToken.getAddress();

    console.log(`   ✅ BridgeRewardToken deployed to: ${bridgeRewardAddress}`);
    deploymentResults.tokens.BridgeRewardToken = {
        address: bridgeRewardAddress,
        symbol: "BRG",
        name: "Bridge Reward Token",
        type: "Bridge Rewards"
    };

    // Token 4: ConsciousnessToken (MIND)
    console.log("\n🚀 [4/10] Deploying ConsciousnessToken (MIND)...");
    const ConsciousnessToken = await hre.ethers.getContractFactory("ConsciousnessToken");
    const consciousnessToken = await ConsciousnessToken.deploy(deployer.address);
    await consciousnessToken.waitForDeployment();
    const consciousnessAddress = await consciousnessToken.getAddress();

    console.log(`   ✅ ConsciousnessToken deployed to: ${consciousnessAddress}`);
    deploymentResults.tokens.ConsciousnessToken = {
        address: consciousnessAddress,
        symbol: "MIND",
        name: "Consciousness Token",
        type: "AI Metrics (Soul-Bound)"
    };

    // Token 5: StakingRewardToken (STAKE)
    console.log("\n🚀 [5/10] Deploying StakingRewardToken (STAKE)...");
    const StakingRewardToken = await hre.ethers.getContractFactory("StakingRewardToken");
    const stakingRewardToken = await StakingRewardToken.deploy(deployer.address);
    await stakingRewardToken.waitForDeployment();
    const stakingRewardAddress = await stakingRewardToken.getAddress();

    console.log(`   ✅ StakingRewardToken deployed to: ${stakingRewardAddress}`);
    deploymentResults.tokens.StakingRewardToken = {
        address: stakingRewardAddress,
        symbol: "STAKE",
        name: "Staking Reward Token",
        type: "Staking"
    };

    // Token 6: GovernanceVoteToken (VOTE)
    console.log("\n🚀 [6/10] Deploying GovernanceVoteToken (VOTE)...");
    const GovernanceVoteToken = await hre.ethers.getContractFactory("GovernanceVoteToken");
    const governanceVoteToken = await GovernanceVoteToken.deploy(deployer.address);
    await governanceVoteToken.waitForDeployment();
    const governanceVoteAddress = await governanceVoteToken.getAddress();

    console.log(`   ✅ GovernanceVoteToken deployed to: ${governanceVoteAddress}`);
    deploymentResults.tokens.GovernanceVoteToken = {
        address: governanceVoteAddress,
        symbol: "VOTE",
        name: "Governance Vote Token",
        type: "Voting Rights"
    };

    // Token 7: LiquidityPoolToken (LP)
    console.log("\n🚀 [7/10] Deploying LiquidityPoolToken (LP)...");
    const LiquidityPoolToken = await hre.ethers.getContractFactory("LiquidityPoolToken");
    const liquidityPoolToken = await LiquidityPoolToken.deploy(deployer.address);
    await liquidityPoolToken.waitForDeployment();
    const liquidityPoolAddress = await liquidityPoolToken.getAddress();

    console.log(`   ✅ LiquidityPoolToken deployed to: ${liquidityPoolAddress}`);
    deploymentResults.tokens.LiquidityPoolToken = {
        address: liquidityPoolAddress,
        symbol: "LP",
        name: "Liquidity Pool Token",
        type: "AMM Liquidity"
    };

    // Token 8: NetworkFeeToken (FEE)
    console.log("\n🚀 [8/10] Deploying NetworkFeeToken (FEE)...");
    const NetworkFeeToken = await hre.ethers.getContractFactory("NetworkFeeToken");
    const networkFeeToken = await NetworkFeeToken.deploy(deployer.address);
    await networkFeeToken.waitForDeployment();
    const networkFeeAddress = await networkFeeToken.getAddress();

    console.log(`   ✅ NetworkFeeToken deployed to: ${networkFeeAddress}`);
    deploymentResults.tokens.NetworkFeeToken = {
        address: networkFeeAddress,
        symbol: "FEE",
        name: "Network Fee Token",
        type: "Gas/Fees"
    };

    // Token 9: OracleDataToken (DATA)
    console.log("\n🚀 [9/10] Deploying OracleDataToken (DATA)...");
    const OracleDataToken = await hre.ethers.getContractFactory("OracleDataToken");
    const oracleDataToken = await OracleDataToken.deploy(deployer.address);
    await oracleDataToken.waitForDeployment();
    const oracleDataAddress = await oracleDataToken.getAddress();

    console.log(`   ✅ OracleDataToken deployed to: ${oracleDataAddress}`);
    deploymentResults.tokens.OracleDataToken = {
        address: oracleDataAddress,
        symbol: "DATA",
        name: "Oracle Data Token",
        type: "Oracle Access"
    };

    // Token 10: YieldFarmingToken (YIELD)
    console.log("\n🚀 [10/10] Deploying YieldFarmingToken (YIELD)...");
    const YieldFarmingToken = await hre.ethers.getContractFactory("YieldFarmingToken");
    const yieldFarmingToken = await YieldFarmingToken.deploy(deployer.address);
    await yieldFarmingToken.waitForDeployment();
    const yieldFarmingAddress = await yieldFarmingToken.getAddress();

    console.log(`   ✅ YieldFarmingToken deployed to: ${yieldFarmingAddress}`);
    deploymentResults.tokens.YieldFarmingToken = {
        address: yieldFarmingAddress,
        symbol: "YIELD",
        name: "Yield Farming Token",
        type: "Yield Farming"
    };

    // Save deployment results
    const resultsFile = `deployments_${hre.network.name}_${Date.now()}.json`;
    fs.writeFileSync(resultsFile, JSON.stringify(deploymentResults, null, 2));

    // Display summary
    console.log("\n\n" + "=".repeat(80));
    console.log("✨ DEPLOYMENT COMPLETE!");
    console.log("=".repeat(80));

    console.log("\n📊 Deployment Summary:");
    console.log(`   Network: ${hre.network.name}`);
    console.log(`   Total Tokens Deployed: 10`);
    console.log(`   Results saved to: ${resultsFile}`);

    console.log("\n📝 Contract Addresses:");
    for (const [name, info] of Object.entries(deploymentResults.tokens)) {
        console.log(`   ${info.symbol.padEnd(6)} | ${info.address} | ${info.type}`);
    }

    console.log("\n🔗 Block Explorer Verification:");
    console.log("   Run the following commands to verify contracts:\n");

    const explorers = {
        sepolia: "https://sepolia.etherscan.io",
        arbitrum: "https://arbiscan.io",
        optimism: "https://optimistic.etherscan.io",
        base: "https://basescan.org",
        polygon: "https://polygonscan.com",
        avalanche: "https://snowtrace.io",
        bsc: "https://bscscan.com"
    };

    const explorerUrl = explorers[hre.network.name] || "https://etherscan.io";

    for (const [name, info] of Object.entries(deploymentResults.tokens)) {
        console.log(`   ${explorerUrl}/address/${info.address}`);
    }

    console.log("\n" + "=".repeat(80));
}

main()
    .then(() => process.exit(0))
    .catch((error) => {
        console.error(error);
        process.exit(1);
    });
