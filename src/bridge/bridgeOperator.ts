/**
 * Bridge Operator Service
 * Coordinates Bitcoin testnet deposits and wTBTC minting on Polygon
 */

import bitcoinMonitor, { BitcoinTransaction } from './bitcoinMonitor';
import wBTCContract from '../contracts/wBTCInterface';
import walletManager from '../web3/walletManager';

export interface BridgeRequest {
  id: string;
  userAddress: string;
  bitcoinAddress: string;
  bitcoinTxId?: string;
  amount: number;
  status: 'pending' | 'confirmed' | 'minted' | 'failed';
  createdAt: number;
  mintedTxHash?: string;
  confirmations: number;
}

export class BridgeOperator {
  private static instance: BridgeOperator;
  private bridgeRequests: Map<string, BridgeRequest> = new Map();
  private addressToRequest: Map<string, string> = new Map(); // Bitcoin address -> request ID
  private isRunning = false;
  private readonly MIN_CONFIRMATIONS = 3;

  private constructor() {}

  static getInstance(): BridgeOperator {
    if (!BridgeOperator.instance) {
      BridgeOperator.instance = new BridgeOperator();
    }
    return BridgeOperator.instance;
  }

  /**
   * Create a new bridge request (user wants to deposit BTC and get wTBTC)
   */
  createBridgeRequest(
    userAddress: string,
    bitcoinAddress: string
  ): BridgeRequest {
    const requestId = this.generateRequestId();

    const request: BridgeRequest = {
      id: requestId,
      userAddress,
      bitcoinAddress,
      amount: 0,
      status: 'pending',
      createdAt: Date.now(),
      confirmations: 0
    };

    this.bridgeRequests.set(requestId, request);
    this.addressToRequest.set(bitcoinAddress, requestId);

    // Start watching this Bitcoin address
    bitcoinMonitor.watchAddress(bitcoinAddress);

    console.log(
      `🌉 Bridge request created:\n` +
      `   Request ID: ${requestId}\n` +
      `   User: ${userAddress}\n` +
      `   Bitcoin Address: ${bitcoinAddress}`
    );

    return request;
  }

  /**
   * Start the bridge operator service
   */
  async start(): Promise<void> {
    if (this.isRunning) {
      console.log('⚠️  Bridge operator already running');
      return;
    }

    // Verify we're connected to the right network
    const chainId = walletManager.getChainId();
    if (!chainId) {
      throw new Error('Wallet not connected');
    }

    console.log('🚀 Starting Bridge Operator Service...');
    console.log(`   Network: ${walletManager.getChainName(chainId)}`);
    console.log(`   Operator: ${walletManager.getAddress()}`);

    this.isRunning = true;

    // Start Bitcoin monitor
    await bitcoinMonitor.startMonitoring(
      this.handleNewBitcoinTransaction.bind(this)
    );

    console.log('✅ Bridge operator running');
  }

  /**
   * Stop the bridge operator service
   */
  stop(): void {
    bitcoinMonitor.stopMonitoring();
    this.isRunning = false;
    console.log('🛑 Bridge operator stopped');
  }

  /**
   * Handle new Bitcoin transaction detected by monitor
   */
  private async handleNewBitcoinTransaction(
    address: string,
    tx: BitcoinTransaction
  ): Promise<void> {
    const requestId = this.addressToRequest.get(address);

    if (!requestId) {
      console.log(`⚠️  Transaction for unwatched address: ${address}`);
      return;
    }

    const request = this.bridgeRequests.get(requestId);
    if (!request) {
      console.log(`⚠️  No bridge request found for ${requestId}`);
      return;
    }

    // Update request with transaction details
    request.bitcoinTxId = tx.txid;
    request.amount = tx.value;
    request.confirmations = tx.confirmations;

    if (tx.confirmations >= this.MIN_CONFIRMATIONS) {
      request.status = 'confirmed';
      console.log(`✅ Bitcoin transaction confirmed (${tx.confirmations} confirmations)`);

      // Mint wTBTC tokens
      await this.mintTokens(request);
    } else {
      console.log(
        `⏳ Waiting for confirmations: ${tx.confirmations}/${this.MIN_CONFIRMATIONS}`
      );
    }
  }

  /**
   * Mint wTBTC tokens on Polygon after Bitcoin deposit is confirmed
   */
  private async mintTokens(request: BridgeRequest): Promise<void> {
    try {
      console.log(
        `🪙 Minting ${request.amount} wTBTC for ${request.userAddress}...`
      );

      const tx = await wBTCContract.mint(
        request.userAddress,
        request.amount.toString(),
        request.bitcoinTxId!
      );

      request.status = 'minted';
      request.mintedTxHash = tx.hash;

      console.log(
        `✅ Successfully minted ${request.amount} wTBTC!\n` +
        `   Tx Hash: ${tx.hash}\n` +
        `   User: ${request.userAddress}`
      );

      // Stop watching this address
      bitcoinMonitor.unwatchAddress(request.bitcoinAddress);
    } catch (error: any) {
      console.error('❌ Failed to mint tokens:', error.message);
      request.status = 'failed';
    }
  }

  /**
   * Handle burn request (user wants to burn wTBTC and get BTC back)
   * This would listen to Burn events from the contract
   */
  async handleBurnRequest(
    from: string,
    amount: string,
    btcAddress: string
  ): Promise<void> {
    console.log(
      `🔥 Burn request received:\n` +
      `   From: ${from}\n` +
      `   Amount: ${amount} wTBTC\n` +
      `   BTC Address: ${btcAddress}`
    );

    // In a real implementation, this would:
    // 1. Verify the burn transaction on Polygon
    // 2. Send BTC from the bridge's Bitcoin wallet to the user's address
    // 3. Record the unlock transaction

    console.log(
      `⚠️  Note: Automatic BTC unlocking requires a Bitcoin wallet integration.\n` +
      `   Manual action required to send ${amount} BTC to ${btcAddress}`
    );
  }

  /**
   * Get bridge request by ID
   */
  getBridgeRequest(requestId: string): BridgeRequest | undefined {
    return this.bridgeRequests.get(requestId);
  }

  /**
   * Get all bridge requests
   */
  getAllBridgeRequests(): BridgeRequest[] {
    return Array.from(this.bridgeRequests.values());
  }

  /**
   * Get bridge requests for a user
   */
  getUserBridgeRequests(userAddress: string): BridgeRequest[] {
    return Array.from(this.bridgeRequests.values()).filter(
      req => req.userAddress.toLowerCase() === userAddress.toLowerCase()
    );
  }

  /**
   * Generate unique request ID
   */
  private generateRequestId(): string {
    return `BR-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  }

  /**
   * Get service status
   */
  getStatus(): {
    isRunning: boolean;
    totalRequests: number;
    pendingRequests: number;
    mintedRequests: number;
  } {
    const requests = Array.from(this.bridgeRequests.values());

    return {
      isRunning: this.isRunning,
      totalRequests: requests.length,
      pendingRequests: requests.filter(r => r.status === 'pending').length,
      mintedRequests: requests.filter(r => r.status === 'minted').length
    };
  }
}

export default BridgeOperator.getInstance();
