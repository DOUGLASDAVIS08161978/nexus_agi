/**
 * Bitcoin Testnet Monitor
 * Monitors Bitcoin testnet addresses for incoming transactions
 */

import axios from 'axios';

// Bitcoin testnet block explorer APIs
const BLOCKSTREAM_API = 'https://blockstream.info/testnet/api';
const MEMPOOL_API = 'https://mempool.space/testnet/api';

export interface BitcoinTransaction {
  txid: string;
  value: number; // in BTC
  confirmations: number;
  blockHeight: number;
  timestamp: number;
}

export class BitcoinMonitor {
  private watchedAddresses: Map<string, number> = new Map();
  private pollingInterval: NodeJS.Timeout | null = null;
  private isRunning = false;

  /**
   * Add address to watch list
   */
  watchAddress(address: string, lastCheckedHeight: number = 0): void {
    console.log(`👁️  Watching Bitcoin testnet address: ${address}`);
    this.watchedAddresses.set(address, lastCheckedHeight);
  }

  /**
   * Remove address from watch list
   */
  unwatchAddress(address: string): void {
    this.watchedAddresses.delete(address);
    console.log(`🚫 Stopped watching address: ${address}`);
  }

  /**
   * Get address transactions from Blockstream API
   */
  private async getAddressTransactions(address: string): Promise<any[]> {
    try {
      const response = await axios.get(
        `${BLOCKSTREAM_API}/address/${address}/txs`,
        { timeout: 10000 }
      );
      return response.data;
    } catch (error: any) {
      console.error(`❌ Error fetching transactions for ${address}:`, error.message);
      return [];
    }
  }

  /**
   * Get current block height
   */
  async getCurrentBlockHeight(): Promise<number> {
    try {
      const response = await axios.get(`${BLOCKSTREAM_API}/blocks/tip/height`, {
        timeout: 5000
      });
      return response.data;
    } catch (error) {
      console.error('❌ Error fetching block height:', error);
      return 0;
    }
  }

  /**
   * Parse transactions and filter new ones
   */
  private async checkAddressForNewTransactions(
    address: string,
    lastCheckedHeight: number
  ): Promise<BitcoinTransaction[]> {
    const txs = await this.getAddressTransactions(address);
    const newTransactions: BitcoinTransaction[] = [];

    for (const tx of txs) {
      // Check if this is a new transaction
      if (tx.status.block_height && tx.status.block_height > lastCheckedHeight) {
        // Find outputs to our address
        for (const vout of tx.vout) {
          if (vout.scriptpubkey_address === address) {
            newTransactions.push({
              txid: tx.txid,
              value: vout.value / 100000000, // Convert satoshis to BTC
              confirmations: tx.status.confirmations || 0,
              blockHeight: tx.status.block_height || 0,
              timestamp: tx.status.block_time || Date.now() / 1000
            });
          }
        }
      }
    }

    return newTransactions;
  }

  /**
   * Start monitoring (polls every 30 seconds)
   */
  async startMonitoring(
    onNewTransaction: (address: string, tx: BitcoinTransaction) => void
  ): Promise<void> {
    if (this.isRunning) {
      console.log('⚠️  Monitor already running');
      return;
    }

    this.isRunning = true;
    console.log('🚀 Starting Bitcoin testnet monitor...');

    const checkAddresses = async () => {
      const currentHeight = await this.getCurrentBlockHeight();

      for (const [address, lastHeight] of this.watchedAddresses.entries()) {
        const newTxs = await this.checkAddressForNewTransactions(
          address,
          lastHeight
        );

        for (const tx of newTxs) {
          console.log(
            `🔔 New transaction detected!\n` +
            `   Address: ${address}\n` +
            `   Amount: ${tx.value} BTC\n` +
            `   TxID: ${tx.txid}\n` +
            `   Confirmations: ${tx.confirmations}`
          );

          onNewTransaction(address, tx);
        }

        // Update last checked height
        if (newTxs.length > 0) {
          const maxHeight = Math.max(...newTxs.map(t => t.blockHeight));
          this.watchedAddresses.set(address, maxHeight);
        } else {
          this.watchedAddresses.set(address, currentHeight);
        }
      }
    };

    // Initial check
    await checkAddresses();

    // Poll every 30 seconds
    this.pollingInterval = setInterval(checkAddresses, 30000);
  }

  /**
   * Stop monitoring
   */
  stopMonitoring(): void {
    if (this.pollingInterval) {
      clearInterval(this.pollingInterval);
      this.pollingInterval = null;
    }
    this.isRunning = false;
    console.log('🛑 Bitcoin monitor stopped');
  }

  /**
   * Generate a new Bitcoin testnet address (requires external wallet)
   * This is a helper method that returns faucet URLs
   */
  static getBitcoinTestnetFaucets(): string[] {
    return [
      'https://coinfaucet.eu/en/btc-testnet/',
      'https://testnet-faucet.mempool.co/',
      'https://bitcoinfaucet.uo1.net/'
    ];
  }

  /**
   * Verify transaction has minimum confirmations
   */
  async verifyTransaction(
    txid: string,
    minConfirmations: number = 3
  ): Promise<boolean> {
    try {
      const response = await axios.get(`${BLOCKSTREAM_API}/tx/${txid}`, {
        timeout: 5000
      });

      const confirmations = response.data.status.confirmations || 0;

      return confirmations >= minConfirmations;
    } catch (error) {
      console.error(`❌ Error verifying transaction ${txid}:`, error);
      return false;
    }
  }

  /**
   * Get transaction details
   */
  async getTransactionDetails(txid: string): Promise<any> {
    try {
      const response = await axios.get(`${BLOCKSTREAM_API}/tx/${txid}`, {
        timeout: 5000
      });
      return response.data;
    } catch (error) {
      console.error(`❌ Error fetching transaction ${txid}:`, error);
      return null;
    }
  }
}

export default new BitcoinMonitor();
