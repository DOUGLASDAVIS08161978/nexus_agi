/**
 * Web3-Onboard Configuration for Multi-Chain DApp
 * Supports: Ethereum Mainnet, Sepolia, Polygon, and testnet interactions
 */

import { init } from '@web3-onboard/core';
import injectedModule from '@web3-onboard/injected-wallets';

// Initialize wallet modules
const injected = injectedModule();

// Chain configurations
export const chains = [
  {
    id: '0x1', // 1 in hex
    token: 'ETH',
    label: 'Ethereum Mainnet',
    rpcUrl: process.env.ETHEREUM_RPC_URL || 'https://eth.llamarpc.com'
  },
  {
    id: '0xaa36a7', // 11155111 in hex
    token: 'SepoliaETH',
    label: 'Sepolia Testnet',
    rpcUrl: process.env.SEPOLIA_RPC_URL || 'https://ethereum-sepolia-rpc.publicnode.com'
  },
  {
    id: '0x89', // 137 in hex
    token: 'MATIC',
    label: 'Polygon Mainnet',
    rpcUrl: process.env.POLYGON_RPC_URL || 'https://polygon-rpc.com'
  },
  {
    id: '0x13882', // 80002 in hex
    token: 'MATIC',
    label: 'Polygon Amoy Testnet',
    rpcUrl: process.env.POLYGON_TESTNET_RPC_URL || 'https://rpc-amoy.polygon.technology'
  }
];

// App metadata
const appMetadata = {
  name: 'Nexus AGI Bridge',
  icon: '<svg><!-- Your app icon SVG --></svg>',
  description: 'Bitcoin Testnet to Polygon Bridge',
  recommendedInjectedWallets: [
    { name: 'MetaMask', url: 'https://metamask.io' },
    { name: 'Coinbase', url: 'https://www.coinbase.com/wallet' }
  ]
};

// Initialize Web3-Onboard
export const web3Onboard = init({
  wallets: [injected],
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
  notify: {
    desktop: {
      enabled: true,
      position: 'topRight'
    },
    mobile: {
      enabled: true,
      position: 'topRight'
    }
  }
});

export default web3Onboard;
