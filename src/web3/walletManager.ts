/**
 * Wallet Manager
 * Handles wallet connections, chain switching, and account management
 */

import { web3Onboard, chains } from './walletConfig';
import { ethers } from 'ethers';

export class WalletManager {
  private static instance: WalletManager;
  private currentWallet: any = null;
  private provider: ethers.providers.Web3Provider | null = null;
  private signer: ethers.Signer | null = null;

  private constructor() {}

  static getInstance(): WalletManager {
    if (!WalletManager.instance) {
      WalletManager.instance = new WalletManager();
    }
    return WalletManager.instance;
  }

  /**
   * Connect wallet using Web3-Onboard
   */
  async connectWallet(): Promise<{
    address: string;
    chainId: string;
    provider: ethers.providers.Web3Provider;
  }> {
    try {
      const wallets = await web3Onboard.connectWallet();

      if (wallets.length === 0) {
        throw new Error('No wallet connected');
      }

      this.currentWallet = wallets[0];

      // Create ethers provider from wallet
      this.provider = new ethers.providers.Web3Provider(
        this.currentWallet.provider,
        'any'
      );

      this.signer = this.provider.getSigner();

      const address = this.currentWallet.accounts[0].address;
      const chainId = this.currentWallet.chains[0].id;

      console.log(`✅ Wallet connected: ${address} on chain ${chainId}`);

      return {
        address,
        chainId,
        provider: this.provider
      };
    } catch (error) {
      console.error('❌ Wallet connection failed:', error);
      throw error;
    }
  }

  /**
   * Switch to a specific chain
   */
  async switchChain(chainId: string): Promise<boolean> {
    try {
      if (!this.currentWallet) {
        throw new Error('No wallet connected');
      }

      const success = await web3Onboard.setChain({
        chainId,
        wallet: this.currentWallet.label
      });

      if (success) {
        console.log(`✅ Switched to chain ${chainId}`);
      }

      return success;
    } catch (error) {
      console.error(`❌ Failed to switch chain:`, error);
      throw error;
    }
  }

  /**
   * Get current wallet address
   */
  getAddress(): string | null {
    return this.currentWallet?.accounts[0]?.address || null;
  }

  /**
   * Get current chain ID
   */
  getChainId(): string | null {
    return this.currentWallet?.chains[0]?.id || null;
  }

  /**
   * Get ethers provider
   */
  getProvider(): ethers.providers.Web3Provider | null {
    return this.provider;
  }

  /**
   * Get ethers signer
   */
  getSigner(): ethers.Signer | null {
    return this.signer;
  }

  /**
   * Get wallet balance
   */
  async getBalance(address?: string): Promise<string> {
    if (!this.provider) {
      throw new Error('Provider not initialized');
    }

    const addr = address || this.getAddress();
    if (!addr) {
      throw new Error('No address available');
    }

    const balance = await this.provider.getBalance(addr);
    return ethers.utils.formatEther(balance);
  }

  /**
   * Disconnect wallet
   */
  async disconnect(): Promise<void> {
    if (this.currentWallet) {
      await web3Onboard.disconnectWallet({ label: this.currentWallet.label });
      this.currentWallet = null;
      this.provider = null;
      this.signer = null;
      console.log('✅ Wallet disconnected');
    }
  }

  /**
   * Check if wallet is connected
   */
  isConnected(): boolean {
    return this.currentWallet !== null;
  }

  /**
   * Subscribe to wallet state changes
   */
  subscribeToChanges(callback: (state: any) => void) {
    const state$ = web3Onboard.state.select();
    return state$.subscribe(callback);
  }

  /**
   * Get chain name from chain ID
   */
  getChainName(chainId: string): string {
    const chain = chains.find(c => c.id === chainId);
    return chain?.label || 'Unknown Chain';
  }

  /**
   * Check if current chain is supported
   */
  isSupportedChain(chainId: string): boolean {
    return chains.some(c => c.id === chainId);
  }
}

export default WalletManager.getInstance();
