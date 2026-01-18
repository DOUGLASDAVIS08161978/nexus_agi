"""
Ethereum Network Connector
Handles connection to Ethereum network, transaction creation, validation and broadcasting
"""

import json
import time
import requests
from web3 import Web3
from eth_account import Account
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
from decimal import Decimal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EthereumTransaction:
    """Ethereum transaction data structure"""
    from_address: str
    to_address: str
    amount: Decimal
    gas_price: int
    gas_limit: int
    nonce: int
    tx_hash: Optional[str] = None
    status: str = "pending"
    block_number: Optional[int] = None
    confirmations: int = 0


class EthereumNetworkConnector:
    """
    Ethereum Network Connector
    Connects to Ethereum network via multiple providers for redundancy
    Handles transaction creation, signing, validation and broadcasting
    """

    def __init__(self, network: str = "mainnet"):
        """
        Initialize Ethereum network connector

        Args:
            network: Supported networks:
                - Ethereum: 'mainnet', 'goerli', 'sepolia'
                - Arbitrum: 'arbitrum', 'arbitrum-sepolia'
                - Optimism: 'optimism', 'optimism-sepolia'
                - Base: 'base', 'base-sepolia'
                - Avalanche: 'avalanche', 'avalanche-fuji'
                - BNB Chain: 'bsc', 'bsc-testnet'
                - Or custom RPC URL starting with 'http'
        """
        self.network = network
        self.providers = self._initialize_providers()
        self.w3 = None
        self.connected = False
        self.current_provider_index = 0

        # Connect to network
        self._connect_to_network()

    def _initialize_providers(self) -> List[str]:
        """Initialize list of Ethereum RPC providers for redundancy"""
        providers = {
            "mainnet": [
                "https://eth.llamarpc.com",
                "https://rpc.ankr.com/eth",
                "https://ethereum.publicnode.com",
                "https://1rpc.io/eth",
                "https://eth.drpc.org"
            ],
            "goerli": [
                "https://goerli.infura.io/v3/9aa3d95b3bc440fa88ea12eaa4456161",
                "https://rpc.ankr.com/eth_goerli"
            ],
            "sepolia": [
                "https://rpc.sepolia.org",
                "https://rpc2.sepolia.org",
                "https://ethereum-sepolia.publicnode.com"
            ],
            # Layer 2 Networks
            "arbitrum": [
                "https://arb1.arbitrum.io/rpc",
                "https://arbitrum.llamarpc.com",
                "https://rpc.ankr.com/arbitrum",
                "https://arbitrum-one.publicnode.com"
            ],
            "arbitrum-sepolia": [
                "https://sepolia-rollup.arbitrum.io/rpc",
                "https://arbitrum-sepolia.publicnode.com"
            ],
            "optimism": [
                "https://mainnet.optimism.io",
                "https://optimism.llamarpc.com",
                "https://rpc.ankr.com/optimism",
                "https://optimism.publicnode.com"
            ],
            "optimism-sepolia": [
                "https://sepolia.optimism.io",
                "https://optimism-sepolia.publicnode.com"
            ],
            "base": [
                "https://mainnet.base.org",
                "https://base.llamarpc.com",
                "https://rpc.ankr.com/base",
                "https://base.publicnode.com"
            ],
            "base-sepolia": [
                "https://sepolia.base.org",
                "https://base-sepolia.publicnode.com"
            ],
            # Alternative L1 Chains
            "avalanche": [
                "https://api.avax.network/ext/bc/C/rpc",
                "https://avalanche.public-rpc.com",
                "https://rpc.ankr.com/avalanche",
                "https://avalanche-c-chain.publicnode.com"
            ],
            "avalanche-fuji": [
                "https://api.avax-test.network/ext/bc/C/rpc",
                "https://avalanche-fuji-c-chain.publicnode.com"
            ],
            "bsc": [
                "https://bsc-dataseed.binance.org",
                "https://bsc.publicnode.com",
                "https://rpc.ankr.com/bsc",
                "https://binance.llamarpc.com"
            ],
            "bsc-testnet": [
                "https://data-seed-prebsc-1-s1.binance.org:8545",
                "https://bsc-testnet.publicnode.com"
            ]
        }

        # If custom RPC URL provided
        if self.network.startswith("http"):
            return [self.network]

        return providers.get(self.network, providers["mainnet"])

    def _connect_to_network(self) -> bool:
        """Establish connection to Ethereum network with fallback providers"""
        for i, provider_url in enumerate(self.providers):
            try:
                logger.info(f"Attempting to connect to Ethereum via {provider_url}")
                self.w3 = Web3(Web3.HTTPProvider(
                    provider_url,
                    request_kwargs={'timeout': 30}
                ))

                if self.w3.is_connected():
                    self.connected = True
                    self.current_provider_index = i
                    latest_block = self.w3.eth.block_number
                    logger.info(f"✓ Connected to Ethereum {self.network}")
                    logger.info(f"✓ Latest block: {latest_block}")
                    logger.info(f"✓ Chain ID: {self.w3.eth.chain_id}")
                    return True

            except Exception as e:
                logger.warning(f"Failed to connect via {provider_url}: {e}")
                continue

        logger.error("Failed to connect to any Ethereum provider")
        self.connected = False
        return False

    def get_balance(self, address: str) -> Decimal:
        """
        Get ETH balance for an address

        Args:
            address: Ethereum address

        Returns:
            Balance in ETH
        """
        if not self.connected:
            raise Exception("Not connected to Ethereum network")

        try:
            balance_wei = self.w3.eth.get_balance(address)
            balance_eth = Web3.from_wei(balance_wei, 'ether')
            return Decimal(str(balance_eth))
        except Exception as e:
            logger.error(f"Error getting balance for {address}: {e}")
            raise

    def get_gas_price(self) -> int:
        """
        Get current gas price with buffer for fast confirmation

        Returns:
            Gas price in wei
        """
        if not self.connected:
            raise Exception("Not connected to Ethereum network")

        try:
            base_gas_price = self.w3.eth.gas_price
            # Add 20% buffer for faster confirmation
            buffered_gas_price = int(base_gas_price * 1.2)
            logger.info(f"Current gas price: {Web3.from_wei(buffered_gas_price, 'gwei')} Gwei")
            return buffered_gas_price
        except Exception as e:
            logger.error(f"Error getting gas price: {e}")
            # Fallback to 50 Gwei if unable to fetch
            return Web3.to_wei(50, 'gwei')

    def estimate_gas(self, transaction: Dict) -> int:
        """
        Estimate gas limit for transaction

        Args:
            transaction: Transaction parameters

        Returns:
            Estimated gas limit
        """
        if not self.connected:
            raise Exception("Not connected to Ethereum network")

        try:
            estimated_gas = self.w3.eth.estimate_gas(transaction)
            # Add 20% buffer
            buffered_gas = int(estimated_gas * 1.2)
            return buffered_gas
        except Exception as e:
            logger.warning(f"Error estimating gas: {e}")
            # Fallback to standard 21000 for simple transfers
            return 21000

    def create_transaction(
        self,
        to_address: str,
        amount_eth: Decimal,
        from_address: Optional[str] = None,
        gas_price: Optional[int] = None,
        gas_limit: Optional[int] = None
    ) -> EthereumTransaction:
        """
        Create Ethereum transaction

        Args:
            to_address: Recipient address
            amount_eth: Amount in ETH
            from_address: Sender address (optional)
            gas_price: Gas price in wei (optional, will auto-fetch)
            gas_limit: Gas limit (optional, will estimate)

        Returns:
            EthereumTransaction object
        """
        if not self.connected:
            raise Exception("Not connected to Ethereum network")

        # Validate addresses
        if not Web3.is_address(to_address):
            raise ValueError(f"Invalid recipient address: {to_address}")

        to_address = Web3.to_checksum_address(to_address)

        if from_address:
            if not Web3.is_address(from_address):
                raise ValueError(f"Invalid sender address: {from_address}")
            from_address = Web3.to_checksum_address(from_address)

        # Get gas price if not provided
        if gas_price is None:
            gas_price = self.get_gas_price()

        # Get nonce
        nonce = self.w3.eth.get_transaction_count(from_address) if from_address else 0

        # Estimate gas if not provided
        if gas_limit is None:
            gas_limit = 21000  # Standard ETH transfer

        transaction = EthereumTransaction(
            from_address=from_address or "",
            to_address=to_address,
            amount=amount_eth,
            gas_price=gas_price,
            gas_limit=gas_limit,
            nonce=nonce
        )

        logger.info(f"Created transaction: {amount_eth} ETH -> {to_address}")
        return transaction

    def sign_and_send_transaction(
        self,
        transaction: EthereumTransaction,
        private_key: str
    ) -> str:
        """
        Sign and broadcast transaction to Ethereum network

        Args:
            transaction: EthereumTransaction object
            private_key: Private key for signing (hex string)

        Returns:
            Transaction hash
        """
        if not self.connected:
            raise Exception("Not connected to Ethereum network")

        try:
            # Build transaction dict
            tx_dict = {
                'nonce': transaction.nonce,
                'to': transaction.to_address,
                'value': Web3.to_wei(transaction.amount, 'ether'),
                'gas': transaction.gas_limit,
                'gasPrice': transaction.gas_price,
                'chainId': self.w3.eth.chain_id
            }

            # Sign transaction
            signed_txn = self.w3.eth.account.sign_transaction(tx_dict, private_key)

            # Send transaction
            tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
            tx_hash_hex = tx_hash.hex()

            transaction.tx_hash = tx_hash_hex
            transaction.status = "broadcasted"

            logger.info(f"✓ Transaction broadcasted: {tx_hash_hex}")
            logger.info(f"  Amount: {transaction.amount} ETH")
            logger.info(f"  To: {transaction.to_address}")
            logger.info(f"  Gas Price: {Web3.from_wei(transaction.gas_price, 'gwei')} Gwei")

            return tx_hash_hex

        except Exception as e:
            logger.error(f"Error signing/sending transaction: {e}")
            transaction.status = "failed"
            raise

    def validate_transaction(self, tx_hash: str) -> Dict:
        """
        Validate and get transaction details

        Args:
            tx_hash: Transaction hash

        Returns:
            Transaction details dict
        """
        if not self.connected:
            raise Exception("Not connected to Ethereum network")

        try:
            tx = self.w3.eth.get_transaction(tx_hash)
            tx_receipt = None

            try:
                tx_receipt = self.w3.eth.get_transaction_receipt(tx_hash)
            except:
                pass  # Transaction not yet mined

            result = {
                "hash": tx_hash,
                "from": tx['from'],
                "to": tx['to'],
                "value": Web3.from_wei(tx['value'], 'ether'),
                "gas": tx['gas'],
                "gasPrice": Web3.from_wei(tx['gasPrice'], 'gwei'),
                "nonce": tx['nonce'],
                "blockNumber": tx.get('blockNumber'),
                "status": "pending"
            }

            if tx_receipt:
                result["status"] = "success" if tx_receipt['status'] == 1 else "failed"
                result["blockNumber"] = tx_receipt['blockNumber']
                result["gasUsed"] = tx_receipt['gasUsed']
                result["confirmations"] = self.w3.eth.block_number - tx_receipt['blockNumber']

            return result

        except Exception as e:
            logger.error(f"Error validating transaction {tx_hash}: {e}")
            raise

    def wait_for_confirmation(
        self,
        tx_hash: str,
        confirmations: int = 12,
        timeout: int = 600
    ) -> bool:
        """
        Wait for transaction confirmation

        Args:
            tx_hash: Transaction hash
            confirmations: Number of confirmations to wait for
            timeout: Timeout in seconds

        Returns:
            True if confirmed, False if timeout
        """
        if not self.connected:
            raise Exception("Not connected to Ethereum network")

        logger.info(f"Waiting for {confirmations} confirmations...")
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                tx_receipt = self.w3.eth.get_transaction_receipt(tx_hash)

                if tx_receipt:
                    current_block = self.w3.eth.block_number
                    tx_block = tx_receipt['blockNumber']
                    current_confirmations = current_block - tx_block

                    logger.info(f"Confirmations: {current_confirmations}/{confirmations}")

                    if current_confirmations >= confirmations:
                        if tx_receipt['status'] == 1:
                            logger.info(f"✓ Transaction confirmed in block {tx_block}")
                            return True
                        else:
                            logger.error("✗ Transaction failed")
                            return False

                time.sleep(15)  # Check every 15 seconds

            except Exception as e:
                logger.warning(f"Error checking confirmation: {e}")
                time.sleep(15)

        logger.warning(f"Timeout waiting for confirmation after {timeout}s")
        return False

    def get_network_stats(self) -> Dict:
        """Get current Ethereum network statistics"""
        if not self.connected:
            raise Exception("Not connected to Ethereum network")

        try:
            latest_block = self.w3.eth.block_number
            gas_price = self.w3.eth.gas_price

            return {
                "network": self.network,
                "connected": self.connected,
                "latest_block": latest_block,
                "gas_price_gwei": float(Web3.from_wei(gas_price, 'gwei')),
                "chain_id": self.w3.eth.chain_id,
                "provider": self.providers[self.current_provider_index]
            }
        except Exception as e:
            logger.error(f"Error getting network stats: {e}")
            raise


if __name__ == "__main__":
    # Test connection
    print("Testing Ethereum Network Connector...\n")

    connector = EthereumNetworkConnector(network="mainnet")

    if connector.connected:
        stats = connector.get_network_stats()
        print(f"Network Stats:")
        print(f"  Network: {stats['network']}")
        print(f"  Latest Block: {stats['latest_block']}")
        print(f"  Gas Price: {stats['gas_price_gwei']:.2f} Gwei")
        print(f"  Chain ID: {stats['chain_id']}")
        print(f"  Provider: {stats['provider']}")

        # Test balance check
        test_address = "0x12f4441ec006a6b00ae8bc8800c2443f9b0c775d"
        balance = connector.get_balance(test_address)
        print(f"\nBalance for {test_address}: {balance} ETH")
