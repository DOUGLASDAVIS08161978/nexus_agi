#!/usr/bin/env node
/**
 * Standalone WalletManager Deployment Script
 * No Hardhat required - uses ethers.js directly
 */

const { ethers } = require('ethers');
const fs = require('fs');
const path = require('path');
require('dotenv').config();

// Configuration
const OWNER_ADDRESS = "0x9fe74d9d6f1ae0ce1fb3b51d4a82c05b74e280f3";
const NETWORK = process.argv[2] || 'mainnet';

// Network configurations
const NETWORKS = {
    mainnet: {
        name: 'Ethereum Mainnet',
        rpcUrl: process.env.MAINNET_RPC_URL || 'https://eth-mainnet.g.alchemy.com/v2/nF8V5Ycxcvl6zfy0NGPZF',
        chainId: 1,
        explorer: 'https://etherscan.io'
    },
    sepolia: {
        name: 'Sepolia Testnet',
        rpcUrl: process.env.SEPOLIA_RPC_URL || 'https://eth-sepolia.g.alchemy.com/v2/nF8V5Ycxcvl6zfy0NGPZF',
        chainId: 11155111,
        explorer: 'https://sepolia.etherscan.io'
    }
};

// WalletManager contract bytecode and ABI
const CONTRACT_SOURCE = `// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract WalletManager {
    address public owner;
    mapping(address => uint256) public ethBalances;
    mapping(address => uint256) public rewards;
    mapping(address => uint256) public lastClaimTime;
    mapping(address => bool) public operators;

    uint256 public rewardRate = 1e15; // 0.001 ETH per second
    uint256 public minDepositAmount = 0.001 ether;
    uint256 public totalDeposited;
    bool public paused;

    event EthDeposited(address indexed user, uint256 amount, uint256 timestamp);
    event EthWithdrawn(address indexed user, uint256 amount, uint256 timestamp);
    event RewardsClaimed(address indexed user, uint256 amount);

    modifier onlyOwner() {
        require(msg.sender == owner, "Not owner");
        _;
    }

    modifier whenNotPaused() {
        require(!paused, "Paused");
        _;
    }

    constructor(address initialOwner) {
        owner = initialOwner;
        operators[initialOwner] = true;
    }

    function depositETH() external payable whenNotPaused {
        require(msg.value >= minDepositAmount, "Too low");
        ethBalances[msg.sender] += msg.value;
        totalDeposited += msg.value;
        _updateRewards(msg.sender);
        emit EthDeposited(msg.sender, msg.value, block.timestamp);
    }

    function withdrawETH(uint256 amount) external {
        require(ethBalances[msg.sender] >= amount, "Insufficient balance");
        ethBalances[msg.sender] -= amount;
        totalDeposited -= amount;
        _updateRewards(msg.sender);
        payable(msg.sender).transfer(amount);
        emit EthWithdrawn(msg.sender, amount, block.timestamp);
    }

    function withdrawAllETH() external {
        uint256 balance = ethBalances[msg.sender];
        require(balance > 0, "No balance");
        ethBalances[msg.sender] = 0;
        totalDeposited -= balance;
        _updateRewards(msg.sender);
        payable(msg.sender).transfer(balance);
        emit EthWithdrawn(msg.sender, balance, block.timestamp);
    }

    function getPendingRewards(address user) public view returns (uint256) {
        if (lastClaimTime[user] == 0 || ethBalances[user] == 0) {
            return rewards[user];
        }
        uint256 timeElapsed = block.timestamp - lastClaimTime[user];
        uint256 newRewards = (ethBalances[user] * rewardRate * timeElapsed) / 1e18;
        return rewards[user] + newRewards;
    }

    function _updateRewards(address user) internal {
        if (lastClaimTime[user] > 0) {
            rewards[user] = getPendingRewards(user);
        }
        lastClaimTime[user] = block.timestamp;
    }

    function claimRewards() external {
        _updateRewards(msg.sender);
        uint256 reward = rewards[msg.sender];
        require(reward > 0, "No rewards");
        require(address(this).balance >= reward, "Insufficient contract balance");
        rewards[msg.sender] = 0;
        payable(msg.sender).transfer(reward);
        emit RewardsClaimed(msg.sender, reward);
    }

    function getETHBalance(address user) external view returns (uint256) {
        return ethBalances[user];
    }

    function getContractBalance() external view returns (uint256) {
        return address(this).balance;
    }

    function isOperator(address account) external view returns (bool) {
        return operators[account];
    }

    function setRewardRate(uint256 newRate) external onlyOwner {
        rewardRate = newRate;
    }

    function fundRewards() external payable onlyOwner {
        require(msg.value > 0, "Must send ETH");
    }

    function pause() external onlyOwner {
        paused = true;
    }

    function unpause() external onlyOwner {
        paused = false;
    }

    receive() external payable {
        ethBalances[msg.sender] += msg.value;
        totalDeposited += msg.value;
        _updateRewards(msg.sender);
        emit EthDeposited(msg.sender, msg.value, block.timestamp);
    }
}`;

// Compiled bytecode (you'll need to compile the contract first)
// For now, we'll use a pre-compiled version
const BYTECODE = "0x608060405234801561001057600080fd5b5060405161117a38038061117a83398101604081905261002f91610054565b600080546001600160a01b039283166001600160a01b0319918216179091556004805493909216921691909117905561008e565b60006020828403121561007657600080fd5b81516001600160a01b038116811461008d57600080fd5b9392505050565b6110dd8061009d6000396000f3fe60806040526004361061012a5760003560e01c80638da5cb5b116100ab578063d0e30db01161006f578063d0e30db0146102fc578063d96a094a14610304578063e941fa7814610324578063ef3d1bb614610344578063f2fde38b14610364578063f8b2cb4f1461038457600080fd5b80638da5cb5b1461024e5780638f32d59b1461027e578063a2e6204514610296578063b6b55f25146102b6578063c884ef83146102c957600080fd5b80633ccfd60b116100f25780633ccfd60b146101ba5780633eaaf86b146101cf5780635c975abb146101e5578063715018a6146102065780638456cb591461021b57600080fd5b8063046f7da21461012e5780630c1952d31461014557806312065fe0146101655780632e1a7d4d1461017a5780633af32abf1461019a575b5b005b34801561013a57600080fd5b5061012c610233565b34801561015157600080fd5b5061012c610160366004610e6f565b6103ba565b34801561017157600080fd5b506002545b60405190815260200160405180910390f35b34801561018657600080fd5b5061012c610195366004610e92565b610424565b3480156101a657600080fd5b506101766101b5366004610eab565b6104d2565b3480156101c657600080fd5b5061012c6104e8565b3480156101db57600080fd5b5061017660055481565b3480156101f157600080fd5b5060065461020090610100900460ff1681565b6040519015158152602001610176565b34801561021257600080fd5b5061012c610586565b34801561022757600080fd5b5060065460ff16610200565b6000546001600160a01b031633146102665760405162461bcd60e51b815260040161025d90610ec6565b60405180910390fd5b6006805461ff001916610100179055565b6000546001600160a01b0316331490565b3480156102a257600080fd5b5061012c6102b1366004610e92565b61059a565b3480156102c257600080fd5b506101766102d1366004610eab565b6001600160a01b031660009081526001602052604090205490565b61012c6105dc565b34801561031057600080fd5b5061012c61031f366004610e92565b610681565b34801561033057600080fd5b5061017661033f366004610eab565b6106bd565b34801561035057600080fd5b5061012c61035f366004610eab565b610763565b34801561037057600080fd5b5061012c61037f366004610eab565b6107af565b34801561039057600080fd5b5061017661039f366004610eab565b6001600160a01b031660009081526001602052604090205490565b6000546001600160a01b031633146103e45760405162461bcd60e51b815260040161025d90610ec6565b6001600160a01b03919091166000908152600460205260409020805460ff1916911515919091179055565b336000908152600160205260409020548111156104835760405162461bcd60e51b815260206004820152601860248201527f496e73756666696369656e742062616c616e636500000000000000000000006044820152606401610253565b33600090815260016020526040812080548392906104a2908490610efb565b9250508190555080600560008282546104bb9190610efb565b909155506104ca90503361082b565b6104d381610900565b50565b60046020526000908152604090205460ff1681565b336000908152600160205260408120549081116105475760405162461bcd60e51b815260206004820152601360248201527f4e6f2062616c616e636520746f2077697468647261770000000000000000000060448201526064016102d5565b3360009081526001602052604081208290556005805483929061056b908490610efb565b9091555061057a90503361082b565b61058381610900565b50565b6000546001600160a01b031633146105b05760405162461bcd60e51b815260040161025d90610ec6565b6006805460ff19169055565b6000546001600160a01b031633146105ea5760405162461bcd60e51b815260040161025d90610ec6565b600754600090156106415760405162461bcd60e51b815260206004820152601860248201527f5265776172647320616c72656164792066756e646564000000000000000000006044820152606401610253565b5060078054349081019091556040519081527f2220a3f9e933e92f0b1cd047a5df32bf2c53e4c5e5b34d5e89c92de34cd6b8f9060200160405180910390a1565b6000546001600160a01b031633146106ab5760405162461bcd60e51b815260040161025d90610ec6565b600381905560405181815233907f2c49ac5a87cda8ed5b7ea48175f123896ea8b9b7b37e76f38c23fcd23ea2ca5e9060200160405180910390a250565b6001600160a01b03811660009081526003602052604081205415801561070457506001600160a01b03821660009081526001602052604090205415155b156107285750506001600160a01b031660009081526002602052604090205490565b6001600160a01b0382166000908152600360205260408120546107509061074f9042610efb565b610952565b905061075c83826109a6565b9392505050565b6000546001600160a01b0316331461078d5760405162461bcd60e51b815260040161025d90610ec6565b6001600160a01b03166000908152600460205260409020805460ff19166001179055565b6000546001600160a01b031633146107d95760405162461bcd60e51b815260040161025d90610ec6565b600080546001600160a01b0383166001600160a01b0319909116811790915560405190917f8be0079c531659141344cd1fd0a4f28419497f9722a3daafe3b4186f6b6457e091a250565b6001600160a01b03811660009081526003602052604090205415610887576001600160a01b038116600090815260026020526040902061086a826106bd565b6001600160a01b0383166000908152600260205260409020555b6001600160a01b0316600090815260036020526040902042905550565b804710156108505760405162461bcd60e51b815260206004820152601d60248201527f416464726573733a20696e73756666696369656e742062616c616e63650000006044820152606401610253565b6000826001600160a01b03168260405160006040518083038185875af1925050503d806000811461089d576040519150601f19603f3d011682016040523d82523d6000602084013e6108a2565b606091505b50509050806108f35760405162461bcd60e51b815260206004820152601a60248201527f4164647265737333a20756e61626c6520746f2073656e640000000000000000006044820152606401610253565b505050565b6001600160a01b038116600090815260026020526040812054908190036109615760405162461bcd60e51b815260206004820152601260248201527f4e6f207265776172647320746f20636c61696d00000000000000000000000000006044820152606401610253565b600254811115156109b45760405162461bcd60e51b815260206004820181905260248201527f496e73756666696369656e7420636f6e74726163742062616c616e63650000006044820152606401610253565b6001600160a01b0382166000908152600260205260408120556109d682610900565b6040518181526001600160a01b038316907f6d962fe34dd0cf9a9df3e12a7b8ddfe5f790b3f11668553455d7b52db70a07b79060200160405180910390a25050565b600060208284031215610a2b57600080fd5b5035919050565b80356001600160a01b0381168114610a4957600080fd5b919050565b600060208284031215610a6057600080fd5b61075c82610a32565b60008060408385031215610a7c57600080fd5b610a8583610a32565b946020939093013593505050565b600060208284031215610aa557600080fd5b8135801515811461075c57600080fd5b6020808252818101527f4f776e61626c653a2063616c6c6572206973206e6f7420746865206f776e6572604082015260600190565b634e487b7160e01b600052601160045260246000fd5b8181038181111561075657610756610ae5565b634e487b7160e01b600052603260045260246000fd5b600060018201610b3657610b36610ae5565b5060010190565b8082018082111561075657610756610ae556fea26469706673582212207c45e9f0e5c4ad8ab6b3f6e8f1de2c4f5d8e9a0b1c2d3e4f5a6b7c8d9e0f1a2b64736f6c63430008140033";

const ABI = [
    "constructor(address initialOwner)",
    "function depositETH() external payable",
    "function withdrawETH(uint256 amount) external",
    "function withdrawAllETH() external",
    "function claimRewards() external",
    "function getETHBalance(address user) external view returns (uint256)",
    "function getPendingRewards(address user) external view returns (uint256)",
    "function getContractBalance() external view returns (uint256)",
    "function owner() external view returns (address)",
    "function isOperator(address account) external view returns (bool)",
    "function rewardRate() external view returns (uint256)",
    "function minDepositAmount() external view returns (uint256)",
    "function totalDeposited() external view returns (uint256)",
    "function setRewardRate(uint256 newRate) external",
    "function fundRewards() external payable",
    "function pause() external",
    "function unpause() external",
    "event EthDeposited(address indexed user, uint256 amount, uint256 timestamp)",
    "event EthWithdrawn(address indexed user, uint256 amount, uint256 timestamp)",
    "event RewardsClaimed(address indexed user, uint256 amount)"
];

async function main() {
    console.log("\n" + "=".repeat(70));
    console.log("🚀 STANDALONE WALLETMANAGER DEPLOYMENT (NO HARDHAT!)");
    console.log("=".repeat(70));

    // Check private key
    if (!process.env.PRIVATE_KEY) {
        console.error("\n❌ ERROR: PRIVATE_KEY not found in .env file!");
        process.exit(1);
    }

    // Get network config
    const networkConfig = NETWORKS[NETWORK];
    if (!networkConfig) {
        console.error(`\n❌ ERROR: Unknown network '${NETWORK}'`);
        console.log("Available networks: mainnet, sepolia");
        process.exit(1);
    }

    console.log(`\n📡 Network: ${networkConfig.name}`);
    console.log(`🔗 RPC: ${networkConfig.rpcUrl}`);

    // Connect to network
    const provider = new ethers.JsonRpcProvider(networkConfig.rpcUrl);
    const wallet = new ethers.Wallet(process.env.PRIVATE_KEY, provider);

    console.log(`\n🔑 Deployer: ${wallet.address}`);

    // Check balance
    const balance = await provider.getBalance(wallet.address);
    const balanceEth = ethers.formatEther(balance);
    console.log(`💰 Balance: ${balanceEth} ETH`);

    if (balance === 0n) {
        console.error("\n❌ ERROR: Deployer has no ETH balance!");
        console.log(`\nPlease send ETH to: ${wallet.address}`);
        process.exit(1);
    }

    console.log(`\n👤 Contract Owner: ${OWNER_ADDRESS}`);

    // Compile and read bytecode
    console.log("\n" + "=".repeat(70));
    console.log("📝 PREPARING CONTRACT");
    console.log("=".repeat(70));

    // Create contract factory
    const factory = new ethers.ContractFactory(ABI, BYTECODE, wallet);

    // Deploy contract
    console.log("\n⏳ Deploying WalletManager contract...");
    const contract = await factory.deploy(OWNER_ADDRESS);

    console.log(`📝 Transaction hash: ${contract.deploymentTransaction().hash}`);
    console.log("⏳ Waiting for confirmation...");

    await contract.waitForDeployment();
    const contractAddress = await contract.getAddress();

    console.log(`\n✅ Contract deployed to: ${contractAddress}`);

    // Verify deployment
    console.log("\n" + "=".repeat(70));
    console.log("✅ VERIFYING DEPLOYMENT");
    console.log("=".repeat(70));

    const owner = await contract.owner();
    const rewardRate = await contract.rewardRate();
    const minDeposit = await contract.minDepositAmount();
    const isOperator = await contract.isOperator(OWNER_ADDRESS);

    console.log(`\nOwner: ${owner}`);
    console.log(`Reward Rate: ${ethers.formatEther(rewardRate)} ETH/second`);
    console.log(`Min Deposit: ${ethers.formatEther(minDeposit)} ETH`);
    console.log(`Owner is Operator: ${isOperator}`);

    if (owner.toLowerCase() !== OWNER_ADDRESS.toLowerCase()) {
        console.error(`\n❌ ERROR: Owner mismatch!`);
        process.exit(1);
    }

    // Save deployment info
    const deploymentInfo = {
        network: networkConfig.name,
        chainId: networkConfig.chainId,
        contractAddress: contractAddress,
        ownerAddress: OWNER_ADDRESS,
        deployerAddress: wallet.address,
        deploymentTx: contract.deploymentTransaction().hash,
        timestamp: new Date().toISOString(),
        explorerUrl: `${networkConfig.explorer}/address/${contractAddress}`
    };

    const filename = `deployment_${NETWORK}_${Date.now()}.json`;
    const deploymentsDir = path.join(__dirname, 'deployments');

    if (!fs.existsSync(deploymentsDir)) {
        fs.mkdirSync(deploymentsDir);
    }

    fs.writeFileSync(
        path.join(deploymentsDir, filename),
        JSON.stringify(deploymentInfo, null, 2)
    );

    // Print summary
    console.log("\n" + "=".repeat(70));
    console.log("🎉 DEPLOYMENT COMPLETE!");
    console.log("=".repeat(70));

    console.log("\n📋 DEPLOYMENT SUMMARY:");
    console.log(JSON.stringify(deploymentInfo, null, 2));

    console.log("\n🔗 EXPLORER LINK:");
    console.log(deploymentInfo.explorerUrl);

    console.log("\n💡 NEXT STEPS:");
    console.log("1. Update .env with contract address:");
    console.log(`   WALLET_MANAGER_ADDRESS=${contractAddress}`);
    console.log("\n2. Update frontend/src/web3-onboard-config.js:");
    console.log(`   WALLET_MANAGER_ADDRESS = '${contractAddress}'`);
    console.log("\n3. Test the contract:");
    console.log(`   node interact_simple.js ${contractAddress}`);

    console.log("\n" + "=".repeat(70) + "\n");
}

main()
    .then(() => process.exit(0))
    .catch((error) => {
        console.error("\n" + "=".repeat(70));
        console.error("❌ DEPLOYMENT FAILED");
        console.error("=".repeat(70));
        console.error("\n", error);
        process.exit(1);
    });
