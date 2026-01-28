/**
 * Web3-Onboard Configuration
 *
 * This module configures Web3-Onboard for wallet connections
 * including MetaMask and other injected wallet providers
 */

import Onboard from '@web3-onboard/core';
import injectedModule from '@web3-onboard/injected-wallets';

// Your deployed contract address (update after deployment)
const WALLET_MANAGER_ADDRESS = 'YOUR_CONTRACT_ADDRESS_HERE';

// Supported chains configuration
const chains = [
    {
        id: '0x1', // Ethereum Mainnet
        token: 'ETH',
        label: 'Ethereum Mainnet',
        rpcUrl: 'https://ethereum-rpc.publicnode.com'
    },
    {
        id: '0xaa36a7', // Sepolia Testnet
        token: 'SepoliaETH',
        label: 'Sepolia Testnet',
        rpcUrl: 'https://ethereum-sepolia-rpc.publicnode.com'
    },
    {
        id: '0x89', // Polygon Mainnet
        token: 'MATIC',
        label: 'Polygon Mainnet',
        rpcUrl: 'https://polygon-rpc.com'
    },
    {
        id: '0x13881', // Mumbai Testnet
        token: 'MATIC',
        label: 'Mumbai Testnet',
        rpcUrl: 'https://rpc-mumbai.maticvigil.com'
    },
    {
        id: '0x7A69', // Hardhat Local
        token: 'ETH',
        label: 'Hardhat Local',
        rpcUrl: 'http://127.0.0.1:8545'
    }
];

// Wallet modules configuration
const wallets = [
    injectedModule() // Supports MetaMask, Coinbase Wallet, etc.
];

// App metadata
const appMetadata = {
    name: 'Nexus AGI Wallet Manager',
    icon: '<svg><!-- Your app icon SVG --></svg>',
    description: 'Connect your wallet to interact with Nexus AGI smart contracts',
    recommendedInjectedWallets: [
        { name: 'MetaMask', url: 'https://metamask.io' },
        { name: 'Coinbase', url: 'https://wallet.coinbase.com/' }
    ]
};

// Initialize Web3-Onboard
const onboard = Onboard({
    wallets,
    chains,
    appMetadata,
    accountCenter: {
        desktop: {
            enabled: true,
            position: 'topRight'
        },
        mobile: {
            enabled: true,
            position: 'topRight'
        }
    },
    connect: {
        autoConnectLastWallet: true
    },
    notify: {
        enabled: true,
        position: 'topRight'
    }
});

/**
 * Connect wallet and return provider
 */
export async function connectWallet() {
    try {
        const wallets = await onboard.connectWallet();

        if (wallets[0]) {
            const { accounts, chains, provider } = wallets[0];

            console.log('Wallet connected:', {
                address: accounts[0].address,
                chainId: chains[0].id,
                balance: accounts[0].balance
            });

            return {
                provider,
                address: accounts[0].address,
                chainId: chains[0].id
            };
        }

        throw new Error('No wallet connected');
    } catch (error) {
        console.error('Failed to connect wallet:', error);
        throw error;
    }
}

/**
 * Disconnect wallet
 */
export async function disconnectWallet() {
    const [primaryWallet] = onboard.state.get().wallets;
    if (primaryWallet) {
        await onboard.disconnectWallet({ label: primaryWallet.label });
    }
}

/**
 * Get current wallet state
 */
export function getWalletState() {
    const state = onboard.state.get();
    return state.wallets[0] || null;
}

/**
 * Switch to a different chain
 */
export async function switchChain(chainId) {
    try {
        await onboard.setChain({ chainId });
        return true;
    } catch (error) {
        console.error('Failed to switch chain:', error);
        return false;
    }
}

/**
 * Subscribe to wallet state changes
 */
export function subscribeToWallet(callback) {
    const wallets$ = onboard.state.select('wallets');
    const subscription = wallets$.subscribe(callback);
    return subscription;
}

// Export the onboard instance
export default onboard;
export { WALLET_MANAGER_ADDRESS };
