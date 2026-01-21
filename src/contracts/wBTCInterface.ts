/**
 * Wrapped Testnet BTC Contract Interface
 * Provides methods to interact with the wTBTC smart contract
 */

import { ethers, Contract, ContractFactory } from 'ethers';
import walletManager from '../web3/walletManager';

// Contract ABI
export const wTBTC_ABI = [
  "constructor(address _initialOperator)",
  "function name() view returns (string)",
  "function symbol() view returns (string)",
  "function decimals() view returns (uint8)",
  "function totalSupply() view returns (uint256)",
  "function balanceOf(address) view returns (uint256)",
  "function allowance(address, address) view returns (uint256)",
  "function bridgeOperator() view returns (address)",
  "function paused() view returns (bool)",
  "function mint(address to, uint256 amount, string bitcoinTxId)",
  "function burn(uint256 amount, string bitcoinAddress)",
  "function approve(address spender, uint256 amount) returns (bool)",
  "function transfer(address to, uint256 amount) returns (bool)",
  "function transferFrom(address from, address to, uint256 amount) returns (bool)",
  "function changeBridgeOperator(address newOperator)",
  "function setPaused(bool _paused)",
  "event Transfer(address indexed from, address indexed to, uint256 value)",
  "event Approval(address indexed owner, address indexed spender, uint256 value)",
  "event Mint(address indexed to, uint256 amount, string bitcoinTxId)",
  "event Burn(address indexed from, uint256 amount, string bitcoinAddress)",
  "event BridgeOperatorChanged(address indexed oldOperator, address indexed newOperator)",
  "event Paused(bool status)"
];

// Contract bytecode (you'll need to compile the Solidity contract to get this)
export const wTBTC_BYTECODE = "0x608060405234801561001057600080fd5b5060405161114338038061114383398101604081905261002f91610054565b600080546001600160a01b0319166001600160a01b0392909216919091179055610084565b60006020828403121561006657600080fd5b81516001600160a01b038116811461007d57600080fd5b9392505050565b6110b0806100936000396000f3fe608060405234801561001057600080fd5b50600436106100f55760003560e01c8063715018a611610097578063a9059cbb11610066578063a9059cbb146101ff578063dd62ed3e14610212578063f2fde38b1461024b57600080fd5b8063715018a6146101b55780638da5cb5b146101bd57806395d89b41146101e4578063a457c2d7146101ec57600080fd5b8063313ce567116100d3578063313ce56714610156578063395093511461016557806342966c681461017857806370a082311461018d57600080fd5b806306fdde03146100fa578063095ea7b31461011857806318160ddd1461013b57806323b872dd1461014d575b600080fd5b61010261025e565b60405161010f91906111f565b60405180910390f35b61012b610126366004611269565b6102f0565b604051901515815260200161010f565b6001545b60405190815260200161010f565b61012b61015b366004611293565b61030a565b6040516012815260200161010f565b61012b610173366004611269565b61032e565b61018b6101863660046112cf565b610350565b005b61013f61019b3660046112e8565b6001600160a01b031660009081526002602052604090205490565b61018b61035d565b6000546001600160a01b03165b6040516001600160a01b03909116815260200161010f565b610102610371565b61012b6101fa366004611269565b610380565b61012b61020d366004611269565b610405565b61013f61022036600461130a565b6001600160a01b03918216600090815260036020908152604080832093909416825291909152205490565b61018b6102593660046112e8565b610413565b60606004805461026d9061133d565b80601f01602080910402602001604051908101604052809291908181526020018280546102999061133d565b80156102e65780601f106102bb576101008083540402835291602001916102e6565b820191906000526020600020905b8154815290600101906020018083116102c957829003601f168201915b5050505050905090565b6000336102fe818585610489565b60019150505b92915050565b60003361031885828561059e565b610323858585610618565b506001949350505050565b6000336102fe81858561034183836102"; // Truncated for brevity

export class WBTCContract {
  private contract: Contract | null = null;
  private address: string | null = null;

  /**
   * Deploy a new wTBTC contract
   */
  async deploy(bridgeOperatorAddress: string): Promise<string> {
    const signer = walletManager.getSigner();
    if (!signer) {
      throw new Error('No signer available. Connect wallet first.');
    }

    console.log(`🚀 Deploying wTBTC contract with operator: ${bridgeOperatorAddress}`);

    // Note: In production, you would compile the contract and use the bytecode
    // For now, we'll create a factory from ABI and bytecode
    const factory = new ContractFactory(wTBTC_ABI, wTBTC_BYTECODE, signer);

    const contract = await factory.deploy(bridgeOperatorAddress);
    await contract.deployed();

    this.contract = contract;
    this.address = contract.address;

    console.log(`✅ wTBTC deployed at: ${contract.address}`);

    return contract.address;
  }

  /**
   * Connect to an existing wTBTC contract
   */
  async connect(contractAddress: string): Promise<void> {
    const provider = walletManager.getProvider();
    if (!provider) {
      throw new Error('No provider available. Connect wallet first.');
    }

    this.address = contractAddress;
    this.contract = new Contract(contractAddress, wTBTC_ABI, provider);

    console.log(`✅ Connected to wTBTC at: ${contractAddress}`);
  }

  /**
   * Get contract with signer (for write operations)
   */
  private getContractWithSigner(): Contract {
    if (!this.contract) {
      throw new Error('Contract not initialized');
    }

    const signer = walletManager.getSigner();
    if (!signer) {
      throw new Error('No signer available');
    }

    return this.contract.connect(signer);
  }

  /**
   * Get token balance of an address
   */
  async balanceOf(address: string): Promise<string> {
    if (!this.contract) throw new Error('Contract not initialized');

    const balance = await this.contract.balanceOf(address);
    return ethers.utils.formatEther(balance);
  }

  /**
   * Get total supply
   */
  async totalSupply(): Promise<string> {
    if (!this.contract) throw new Error('Contract not initialized');

    const supply = await this.contract.totalSupply();
    return ethers.utils.formatEther(supply);
  }

  /**
   * Get bridge operator address
   */
  async getBridgeOperator(): Promise<string> {
    if (!this.contract) throw new Error('Contract not initialized');

    return await this.contract.bridgeOperator();
  }

  /**
   * Check if contract is paused
   */
  async isPaused(): Promise<boolean> {
    if (!this.contract) throw new Error('Contract not initialized');

    return await this.contract.paused();
  }

  /**
   * Mint tokens (bridge operator only)
   */
  async mint(
    toAddress: string,
    amount: string,
    bitcoinTxId: string
  ): Promise<ethers.ContractTransaction> {
    const contract = this.getContractWithSigner();

    const amountWei = ethers.utils.parseEther(amount);

    console.log(`🪙 Minting ${amount} wTBTC to ${toAddress}...`);

    const tx = await contract.mint(toAddress, amountWei, bitcoinTxId);
    await tx.wait();

    console.log(`✅ Minted ${amount} wTBTC. Tx: ${tx.hash}`);

    return tx;
  }

  /**
   * Burn tokens to unlock BTC
   */
  async burn(
    amount: string,
    bitcoinAddress: string
  ): Promise<ethers.ContractTransaction> {
    const contract = this.getContractWithSigner();

    const amountWei = ethers.utils.parseEther(amount);

    console.log(`🔥 Burning ${amount} wTBTC for BTC address ${bitcoinAddress}...`);

    const tx = await contract.burn(amountWei, bitcoinAddress);
    await tx.wait();

    console.log(`✅ Burned ${amount} wTBTC. Tx: ${tx.hash}`);

    return tx;
  }

  /**
   * Transfer tokens
   */
  async transfer(
    toAddress: string,
    amount: string
  ): Promise<ethers.ContractTransaction> {
    const contract = this.getContractWithSigner();

    const amountWei = ethers.utils.parseEther(amount);

    console.log(`💸 Transferring ${amount} wTBTC to ${toAddress}...`);

    const tx = await contract.transfer(toAddress, amountWei);
    await tx.wait();

    console.log(`✅ Transferred ${amount} wTBTC. Tx: ${tx.hash}`);

    return tx;
  }

  /**
   * Approve spender
   */
  async approve(
    spenderAddress: string,
    amount: string
  ): Promise<ethers.ContractTransaction> {
    const contract = this.getContractWithSigner();

    const amountWei = ethers.utils.parseEther(amount);

    console.log(`✅ Approving ${spenderAddress} to spend ${amount} wTBTC...`);

    const tx = await contract.approve(spenderAddress, amountWei);
    await tx.wait();

    console.log(`✅ Approval complete. Tx: ${tx.hash}`);

    return tx;
  }

  /**
   * Listen to Mint events
   */
  onMint(callback: (to: string, amount: string, btcTxId: string) => void): void {
    if (!this.contract) throw new Error('Contract not initialized');

    this.contract.on('Mint', (to, amount, btcTxId) => {
      callback(to, ethers.utils.formatEther(amount), btcTxId);
    });
  }

  /**
   * Listen to Burn events
   */
  onBurn(callback: (from: string, amount: string, btcAddress: string) => void): void {
    if (!this.contract) throw new Error('Contract not initialized');

    this.contract.on('Burn', (from, amount, btcAddress) => {
      callback(from, ethers.utils.formatEther(amount), btcAddress);
    });
  }

  /**
   * Get contract address
   */
  getAddress(): string | null {
    return this.address;
  }
}

export default new WBTCContract();
