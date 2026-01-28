/**
 * WalletManager Contract Interaction Module
 *
 * This module provides functions to interact with the WalletManager smart contract
 */

import { ethers } from 'ethers';
import { WALLET_MANAGER_ADDRESS } from './web3-onboard-config.js';

// WalletManager Contract ABI (add your compiled ABI here)
const WALLET_MANAGER_ABI = [
    // Read functions
    "function getETHBalance(address user) view returns (uint256)",
    "function getTokenBalance(address user, address token) view returns (uint256)",
    "function getPendingRewards(address user) view returns (uint256)",
    "function getContractBalance() view returns (uint256)",
    "function ethBalances(address) view returns (uint256)",
    "function rewards(address) view returns (uint256)",
    "function rewardRate() view returns (uint256)",
    "function minDepositAmount() view returns (uint256)",
    "function totalDeposited() view returns (uint256)",
    "function owner() view returns (address)",
    "function isOperator(address) view returns (bool)",

    // Write functions
    "function depositETH() payable",
    "function withdrawETH(uint256 amount)",
    "function withdrawAllETH()",
    "function depositToken(address token, uint256 amount)",
    "function withdrawToken(address token, uint256 amount)",
    "function withdrawAllTokens(address token)",
    "function claimRewards()",

    // Events
    "event EthDeposited(address indexed user, uint256 amount, uint256 timestamp)",
    "event EthWithdrawn(address indexed user, uint256 amount, uint256 timestamp)",
    "event TokenDeposited(address indexed user, address indexed token, uint256 amount)",
    "event TokenWithdrawn(address indexed user, address indexed token, uint256 amount)",
    "event RewardsClaimed(address indexed user, uint256 amount)"
];

/**
 * Get contract instance
 */
function getContract(provider, signer = null) {
    if (signer) {
        return new ethers.Contract(WALLET_MANAGER_ADDRESS, WALLET_MANAGER_ABI, signer);
    }
    return new ethers.Contract(WALLET_MANAGER_ADDRESS, WALLET_MANAGER_ABI, provider);
}

/**
 * Deposit ETH into the contract
 */
export async function depositETH(provider, amount) {
    try {
        const signer = await provider.getSigner();
        const contract = getContract(provider, signer);

        const tx = await contract.depositETH({
            value: ethers.parseEther(amount.toString())
        });

        console.log('Deposit transaction sent:', tx.hash);
        const receipt = await tx.wait();
        console.log('Deposit confirmed:', receipt);

        return { success: true, txHash: tx.hash, receipt };
    } catch (error) {
        console.error('Deposit failed:', error);
        return { success: false, error: error.message };
    }
}

/**
 * Withdraw ETH from the contract
 */
export async function withdrawETH(provider, amount) {
    try {
        const signer = await provider.getSigner();
        const contract = getContract(provider, signer);

        const tx = await contract.withdrawETH(ethers.parseEther(amount.toString()));

        console.log('Withdrawal transaction sent:', tx.hash);
        const receipt = await tx.wait();
        console.log('Withdrawal confirmed:', receipt);

        return { success: true, txHash: tx.hash, receipt };
    } catch (error) {
        console.error('Withdrawal failed:', error);
        return { success: false, error: error.message };
    }
}

/**
 * Withdraw all ETH
 */
export async function withdrawAllETH(provider) {
    try {
        const signer = await provider.getSigner();
        const contract = getContract(provider, signer);

        const tx = await contract.withdrawAllETH();

        console.log('Withdraw all transaction sent:', tx.hash);
        const receipt = await tx.wait();
        console.log('Withdraw all confirmed:', receipt);

        return { success: true, txHash: tx.hash, receipt };
    } catch (error) {
        console.error('Withdraw all failed:', error);
        return { success: false, error: error.message };
    }
}

/**
 * Claim accumulated rewards
 */
export async function claimRewards(provider) {
    try {
        const signer = await provider.getSigner();
        const contract = getContract(provider, signer);

        const tx = await contract.claimRewards();

        console.log('Claim rewards transaction sent:', tx.hash);
        const receipt = await tx.wait();
        console.log('Claim rewards confirmed:', receipt);

        return { success: true, txHash: tx.hash, receipt };
    } catch (error) {
        console.error('Claim rewards failed:', error);
        return { success: false, error: error.message };
    }
}

/**
 * Get user's ETH balance in the contract
 */
export async function getUserETHBalance(provider, userAddress) {
    try {
        const contract = getContract(provider);
        const balance = await contract.getETHBalance(userAddress);
        return ethers.formatEther(balance);
    } catch (error) {
        console.error('Failed to get ETH balance:', error);
        return '0';
    }
}

/**
 * Get user's pending rewards
 */
export async function getPendingRewards(provider, userAddress) {
    try {
        const contract = getContract(provider);
        const rewards = await contract.getPendingRewards(userAddress);
        return ethers.formatEther(rewards);
    } catch (error) {
        console.error('Failed to get pending rewards:', error);
        return '0';
    }
}

/**
 * Get contract information
 */
export async function getContractInfo(provider) {
    try {
        const contract = getContract(provider);

        const [owner, rewardRate, minDeposit, totalDeposited, contractBalance] = await Promise.all([
            contract.owner(),
            contract.rewardRate(),
            contract.minDepositAmount(),
            contract.totalDeposited(),
            contract.getContractBalance()
        ]);

        return {
            owner,
            rewardRate: ethers.formatEther(rewardRate),
            minDeposit: ethers.formatEther(minDeposit),
            totalDeposited: ethers.formatEther(totalDeposited),
            contractBalance: ethers.formatEther(contractBalance)
        };
    } catch (error) {
        console.error('Failed to get contract info:', error);
        return null;
    }
}

/**
 * Listen to contract events
 */
export function subscribeToEvents(provider, eventName, callback) {
    const contract = getContract(provider);

    const filter = contract.filters[eventName]();
    contract.on(filter, callback);

    // Return unsubscribe function
    return () => {
        contract.off(filter, callback);
    };
}

/**
 * Deposit ERC20 tokens
 */
export async function depositToken(provider, tokenAddress, amount) {
    try {
        const signer = await provider.getSigner();
        const contract = getContract(provider, signer);

        // First, approve the contract to spend tokens
        const tokenContract = new ethers.Contract(
            tokenAddress,
            ['function approve(address spender, uint256 amount) returns (bool)'],
            signer
        );

        const approveTx = await tokenContract.approve(
            WALLET_MANAGER_ADDRESS,
            ethers.parseEther(amount.toString())
        );
        await approveTx.wait();

        // Then deposit
        const tx = await contract.depositToken(
            tokenAddress,
            ethers.parseEther(amount.toString())
        );

        console.log('Token deposit transaction sent:', tx.hash);
        const receipt = await tx.wait();
        console.log('Token deposit confirmed:', receipt);

        return { success: true, txHash: tx.hash, receipt };
    } catch (error) {
        console.error('Token deposit failed:', error);
        return { success: false, error: error.message };
    }
}

/**
 * Withdraw ERC20 tokens
 */
export async function withdrawToken(provider, tokenAddress, amount) {
    try {
        const signer = await provider.getSigner();
        const contract = getContract(provider, signer);

        const tx = await contract.withdrawToken(
            tokenAddress,
            ethers.parseEther(amount.toString())
        );

        console.log('Token withdrawal transaction sent:', tx.hash);
        const receipt = await tx.wait();
        console.log('Token withdrawal confirmed:', receipt);

        return { success: true, txHash: tx.hash, receipt };
    } catch (error) {
        console.error('Token withdrawal failed:', error);
        return { success: false, error: error.message };
    }
}

export { WALLET_MANAGER_ADDRESS };
