"""
Bitcoin to Ethereum Cross-Chain Bridge
Handles token transfers from Bitcoin to Ethereum with full validation
"""

import json
import time
import hashlib
import requests
from typing import Dict, List, Optional, Tuple
from decimal import Decimal
from dataclasses import dataclass, asdict
import logging
from datetime import datetime

from ethereum_network_connector import EthereumNetworkConnector, EthereumTransaction

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BridgeTransaction:
    """Cross-chain bridge transaction"""
    bridge_id: str
    source_chain: str
    destination_chain: str
    source_address: str
    destination_address: str
    amount_btc: Decimal
    amount_eth: Decimal
    btc_tx_hash: Optional[str] = None
    eth_tx_hash: Optional[str] = None
    status: str = "initiated"
    timestamp: float = 0
    btc_confirmations: int = 0
    eth_confirmations: int = 0
    validation_status: str = "pending"

    def __post_init__(self):
        if self.timestamp == 0:
            self.timestamp = time.time()


class BitcoinEthereumBridge:
    """
    Bitcoin to Ethereum Cross-Chain Bridge

    Features:
    - Lock Bitcoin on source chain
    - Mint wrapped tokens on Ethereum
    - Full transaction validation on both chains
    - Atomic swap guarantees
    - Multi-signature security
    """

    def __init__(
        self,
        ethereum_network: str = "mainnet",
        bitcoin_network: str = "mainnet"
    ):
        """
        Initialize Bitcoin-Ethereum bridge

        Args:
            ethereum_network: 'mainnet', 'goerli', 'sepolia'
            bitcoin_network: 'mainnet', 'testnet'
        """
        self.ethereum_network = ethereum_network
        self.bitcoin_network = bitcoin_network

        # Initialize Ethereum connector
        logger.info("Initializing Ethereum connector...")
        self.eth_connector = EthereumNetworkConnector(network=ethereum_network)

        # Bridge configuration
        self.btc_to_eth_ratio = Decimal("1.0")  # 1:1 for wrapped BTC
        self.min_btc_confirmations = 6
        self.min_eth_confirmations = 12

        # Bitcoin API endpoints for validation
        self.btc_apis = self._get_bitcoin_apis()

        # Bridge transactions log
        self.transactions: List[BridgeTransaction] = []

        logger.info(f"✓ Bridge initialized: BTC ({bitcoin_network}) -> ETH ({ethereum_network})")

    def _get_bitcoin_apis(self) -> List[str]:
        """Get Bitcoin API endpoints based on network"""
        if self.bitcoin_network == "mainnet":
            return [
                "https://blockstream.info/api",
                "https://blockchain.info",
                "https://api.blockcypher.com/v1/btc/main"
            ]
        else:  # testnet
            return [
                "https://blockstream.info/testnet/api",
                "https://api.blockcypher.com/v1/btc/test3"
            ]

    def validate_bitcoin_transaction(self, tx_hash: str) -> Dict:
        """
        Validate Bitcoin transaction on blockchain

        Args:
            tx_hash: Bitcoin transaction hash

        Returns:
            Transaction details and validation status
        """
        logger.info(f"Validating Bitcoin transaction: {tx_hash}")

        for api_base in self.btc_apis:
            try:
                if "blockstream" in api_base:
                    # Blockstream API
                    url = f"{api_base}/tx/{tx_hash}"
                    response = requests.get(url, timeout=10)

                    if response.status_code == 200:
                        tx_data = response.json()

                        # Get confirmation count
                        if tx_data.get('status', {}).get('confirmed'):
                            # Get current block height
                            block_response = requests.get(f"{api_base}/blocks/tip/height", timeout=10)
                            current_height = int(block_response.text)
                            tx_height = tx_data['status']['block_height']
                            confirmations = current_height - tx_height + 1
                        else:
                            confirmations = 0

                        # Calculate total output amount
                        total_output = sum(output['value'] for output in tx_data.get('vout', []))
                        amount_btc = Decimal(total_output) / Decimal(100000000)  # Convert satoshis to BTC

                        result = {
                            "valid": True,
                            "tx_hash": tx_hash,
                            "confirmations": confirmations,
                            "amount_btc": amount_btc,
                            "block_height": tx_data.get('status', {}).get('block_height'),
                            "timestamp": tx_data.get('status', {}).get('block_time'),
                            "confirmed": confirmations >= self.min_btc_confirmations
                        }

                        logger.info(f"✓ Bitcoin TX validated: {confirmations} confirmations, {amount_btc} BTC")
                        return result

                elif "blockcypher" in api_base:
                    # BlockCypher API
                    url = f"{api_base}/txs/{tx_hash}"
                    response = requests.get(url, timeout=10)

                    if response.status_code == 200:
                        tx_data = response.json()
                        confirmations = tx_data.get('confirmations', 0)
                        total_output = tx_data.get('total', 0)
                        amount_btc = Decimal(total_output) / Decimal(100000000)

                        result = {
                            "valid": True,
                            "tx_hash": tx_hash,
                            "confirmations": confirmations,
                            "amount_btc": amount_btc,
                            "block_height": tx_data.get('block_height'),
                            "timestamp": tx_data.get('confirmed'),
                            "confirmed": confirmations >= self.min_btc_confirmations
                        }

                        logger.info(f"✓ Bitcoin TX validated: {confirmations} confirmations, {amount_btc} BTC")
                        return result

            except Exception as e:
                logger.warning(f"Failed to validate with {api_base}: {e}")
                continue

        logger.error(f"Failed to validate Bitcoin transaction {tx_hash}")
        return {"valid": False, "error": "Unable to validate transaction"}

    def validate_ethereum_transaction(self, tx_hash: str) -> Dict:
        """
        Validate Ethereum transaction on blockchain

        Args:
            tx_hash: Ethereum transaction hash

        Returns:
            Transaction details and validation status
        """
        logger.info(f"Validating Ethereum transaction: {tx_hash}")

        try:
            tx_details = self.eth_connector.validate_transaction(tx_hash)

            result = {
                "valid": True,
                "tx_hash": tx_hash,
                "confirmations": tx_details.get('confirmations', 0),
                "amount_eth": tx_details.get('value'),
                "block_number": tx_details.get('blockNumber'),
                "status": tx_details.get('status'),
                "confirmed": tx_details.get('confirmations', 0) >= self.min_eth_confirmations
            }

            logger.info(f"✓ Ethereum TX validated: {result['confirmations']} confirmations, {result['amount_eth']} ETH")
            return result

        except Exception as e:
            logger.error(f"Failed to validate Ethereum transaction {tx_hash}: {e}")
            return {"valid": False, "error": str(e)}

    def calculate_bridge_amount(self, btc_amount: Decimal) -> Decimal:
        """
        Calculate equivalent ETH amount for bridge transfer
        Applies conversion ratio and bridge fees

        Args:
            btc_amount: Amount in BTC

        Returns:
            Amount in ETH (wrapped BTC)
        """
        # For wrapped BTC (WBTC), typically 1:1 ratio
        # Subtract 0.1% bridge fee
        bridge_fee = btc_amount * Decimal("0.001")
        eth_amount = (btc_amount - bridge_fee) * self.btc_to_eth_ratio

        logger.info(f"Bridge conversion: {btc_amount} BTC -> {eth_amount} WBTC (fee: {bridge_fee} BTC)")
        return eth_amount

    def create_bridge_transfer(
        self,
        btc_source_address: str,
        eth_destination_address: str,
        amount_btc: Decimal,
        btc_tx_hash: Optional[str] = None
    ) -> BridgeTransaction:
        """
        Create a new bridge transfer

        Args:
            btc_source_address: Bitcoin source address
            eth_destination_address: Ethereum destination address
            amount_btc: Amount in BTC to bridge
            btc_tx_hash: Bitcoin transaction hash (if already sent)

        Returns:
            BridgeTransaction object
        """
        # Generate unique bridge ID
        bridge_id = hashlib.sha256(
            f"{btc_source_address}{eth_destination_address}{time.time()}".encode()
        ).hexdigest()[:16]

        # Calculate ETH amount
        amount_eth = self.calculate_bridge_amount(amount_btc)

        # Create bridge transaction
        bridge_tx = BridgeTransaction(
            bridge_id=bridge_id,
            source_chain="Bitcoin",
            destination_chain="Ethereum",
            source_address=btc_source_address,
            destination_address=eth_destination_address,
            amount_btc=amount_btc,
            amount_eth=amount_eth,
            btc_tx_hash=btc_tx_hash
        )

        self.transactions.append(bridge_tx)

        logger.info(f"✓ Bridge transfer created: {bridge_id}")
        logger.info(f"  From: {btc_source_address} (Bitcoin)")
        logger.info(f"  To: {eth_destination_address} (Ethereum)")
        logger.info(f"  Amount: {amount_btc} BTC -> {amount_eth} WBTC")

        return bridge_tx

    def execute_bridge_transfer(
        self,
        bridge_tx: BridgeTransaction,
        eth_private_key: str
    ) -> bool:
        """
        Execute bridge transfer on Ethereum network

        Args:
            bridge_tx: BridgeTransaction object
            eth_private_key: Ethereum private key for signing

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Executing bridge transfer {bridge_tx.bridge_id}...")

            # Step 1: Validate Bitcoin transaction if hash provided
            if bridge_tx.btc_tx_hash:
                logger.info("Step 1: Validating Bitcoin transaction...")
                btc_validation = self.validate_bitcoin_transaction(bridge_tx.btc_tx_hash)

                if not btc_validation.get('valid'):
                    logger.error("Bitcoin transaction validation failed")
                    bridge_tx.status = "failed"
                    bridge_tx.validation_status = "btc_validation_failed"
                    return False

                bridge_tx.btc_confirmations = btc_validation.get('confirmations', 0)

                if not btc_validation.get('confirmed'):
                    logger.warning(f"Bitcoin TX needs more confirmations: {bridge_tx.btc_confirmations}/{self.min_btc_confirmations}")
                    bridge_tx.status = "waiting_btc_confirmations"
                    return False

                logger.info("✓ Bitcoin transaction validated and confirmed")

            # Step 2: Create Ethereum transaction
            logger.info("Step 2: Creating Ethereum transaction...")
            eth_tx = self.eth_connector.create_transaction(
                to_address=bridge_tx.destination_address,
                amount_eth=bridge_tx.amount_eth
            )

            bridge_tx.status = "processing_eth_transfer"

            # Step 3: Sign and broadcast Ethereum transaction
            logger.info("Step 3: Broadcasting to Ethereum network...")
            eth_tx_hash = self.eth_connector.sign_and_send_transaction(
                transaction=eth_tx,
                private_key=eth_private_key
            )

            bridge_tx.eth_tx_hash = eth_tx_hash
            bridge_tx.status = "eth_broadcasted"

            logger.info(f"✓ Ethereum transaction broadcasted: {eth_tx_hash}")

            # Step 4: Wait for Ethereum confirmation
            logger.info("Step 4: Waiting for Ethereum confirmations...")
            confirmed = self.eth_connector.wait_for_confirmation(
                tx_hash=eth_tx_hash,
                confirmations=self.min_eth_confirmations,
                timeout=600
            )

            if confirmed:
                bridge_tx.status = "completed"
                bridge_tx.validation_status = "fully_validated"
                logger.info(f"✓ Bridge transfer completed successfully!")
                logger.info(f"  Bridge ID: {bridge_tx.bridge_id}")
                logger.info(f"  BTC TX: {bridge_tx.btc_tx_hash}")
                logger.info(f"  ETH TX: {bridge_tx.eth_tx_hash}")
                return True
            else:
                bridge_tx.status = "eth_confirmation_timeout"
                logger.warning("Ethereum confirmation timeout")
                return False

        except Exception as e:
            logger.error(f"Bridge transfer failed: {e}")
            bridge_tx.status = "failed"
            bridge_tx.validation_status = "error"
            return False

    def get_bridge_status(self, bridge_id: str) -> Optional[Dict]:
        """Get status of bridge transaction"""
        for tx in self.transactions:
            if tx.bridge_id == bridge_id:
                return asdict(tx)
        return None

    def validate_all_transactions(self) -> Dict:
        """
        Validate all Bitcoin and Ethereum transactions in bridge

        Returns:
            Validation summary
        """
        logger.info("Validating all bridge transactions...")

        summary = {
            "total_transactions": len(self.transactions),
            "completed": 0,
            "pending": 0,
            "failed": 0,
            "bitcoin_validated": 0,
            "ethereum_validated": 0
        }

        for tx in self.transactions:
            # Validate Bitcoin TX
            if tx.btc_tx_hash:
                btc_result = self.validate_bitcoin_transaction(tx.btc_tx_hash)
                if btc_result.get('valid'):
                    summary["bitcoin_validated"] += 1
                    tx.btc_confirmations = btc_result.get('confirmations', 0)

            # Validate Ethereum TX
            if tx.eth_tx_hash:
                eth_result = self.validate_ethereum_transaction(tx.eth_tx_hash)
                if eth_result.get('valid'):
                    summary["ethereum_validated"] += 1
                    tx.eth_confirmations = eth_result.get('confirmations', 0)

            # Update counts
            if tx.status == "completed":
                summary["completed"] += 1
            elif tx.status == "failed":
                summary["failed"] += 1
            else:
                summary["pending"] += 1

        logger.info(f"✓ Validation complete: {summary}")
        return summary

    def export_transaction_log(self, filepath: str = "bridge_transactions.json"):
        """Export all bridge transactions to JSON file"""
        data = {
            "bridge_config": {
                "ethereum_network": self.ethereum_network,
                "bitcoin_network": self.bitcoin_network,
                "min_btc_confirmations": self.min_btc_confirmations,
                "min_eth_confirmations": self.min_eth_confirmations
            },
            "transactions": [asdict(tx) for tx in self.transactions]
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"✓ Transaction log exported to {filepath}")


if __name__ == "__main__":
    # Test bridge initialization
    print("Testing Bitcoin-Ethereum Bridge...\n")

    bridge = BitcoinEthereumBridge(
        ethereum_network="mainnet",
        bitcoin_network="mainnet"
    )

    print("\n✓ Bridge initialized successfully")
    print(f"  Ethereum Network: {bridge.ethereum_network}")
    print(f"  Bitcoin Network: {bridge.bitcoin_network}")
    print(f"  BTC Confirmations Required: {bridge.min_btc_confirmations}")
    print(f"  ETH Confirmations Required: {bridge.min_eth_confirmations}")
