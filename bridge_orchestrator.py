#!/usr/bin/env python3
"""
Bitcoin to Ethereum Bridge Orchestrator
Main script to execute cross-chain token transfers with validation and broadcasting
"""

import sys
import os
import argparse
import json
from decimal import Decimal
from typing import Optional
import logging

from bitcoin_ethereum_bridge import BitcoinEthereumBridge, BridgeTransaction
from ethereum_network_connector import EthereumNetworkConnector
from smtp_notifier import SMTPNotifier

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BridgeOrchestrator:
    """
    Bridge Orchestrator
    Coordinates the entire bridge transfer process from Bitcoin to Ethereum
    """

    def __init__(
        self,
        ethereum_network: str = "mainnet",
        bitcoin_network: str = "mainnet",
        notification_email: Optional[str] = None
    ):
        """
        Initialize Bridge Orchestrator

        Args:
            ethereum_network: Ethereum network ('mainnet', 'goerli', 'sepolia')
            bitcoin_network: Bitcoin network ('mainnet', 'testnet')
            notification_email: Email for notifications
        """
        logger.info("="*60)
        logger.info("Bitcoin to Ethereum Bridge Orchestrator")
        logger.info("="*60)

        # Initialize bridge
        self.bridge = BitcoinEthereumBridge(
            ethereum_network=ethereum_network,
            bitcoin_network=bitcoin_network
        )

        # Initialize SMTP notifier
        self.notifier = SMTPNotifier()
        self.notification_email = notification_email

        logger.info("✓ Bridge Orchestrator initialized")

    def execute_transfer(
        self,
        btc_source_address: str,
        eth_destination_address: str,
        amount_btc: Decimal,
        eth_private_key: str,
        btc_tx_hash: Optional[str] = None
    ) -> bool:
        """
        Execute complete bridge transfer

        Args:
            btc_source_address: Bitcoin source address
            eth_destination_address: Ethereum destination address
            amount_btc: Amount in BTC
            eth_private_key: Ethereum private key for signing
            btc_tx_hash: Bitcoin transaction hash (optional)

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("\n" + "="*60)
            logger.info("INITIATING BRIDGE TRANSFER")
            logger.info("="*60)

            # Step 1: Create bridge transaction
            logger.info("\n[Step 1/5] Creating bridge transaction...")
            bridge_tx = self.bridge.create_bridge_transfer(
                btc_source_address=btc_source_address,
                eth_destination_address=eth_destination_address,
                amount_btc=amount_btc,
                btc_tx_hash=btc_tx_hash
            )

            # Send initiation notification
            if self.notification_email:
                self.notifier.notify_bridge_initiated(
                    recipient=self.notification_email,
                    bridge_id=bridge_tx.bridge_id,
                    source_address=btc_source_address,
                    destination_address=eth_destination_address,
                    amount_btc=float(amount_btc),
                    amount_eth=float(bridge_tx.amount_eth)
                )

            # Step 2: Validate Bitcoin transaction (if provided)
            if btc_tx_hash:
                logger.info("\n[Step 2/5] Validating Bitcoin transaction...")
                btc_validation = self.bridge.validate_bitcoin_transaction(btc_tx_hash)

                if not btc_validation.get('valid'):
                    logger.error("❌ Bitcoin transaction validation failed")
                    return False

                logger.info(f"✓ Bitcoin TX validated: {btc_validation['confirmations']} confirmations")

                # Send BTC confirmation notification
                if self.notification_email and btc_validation.get('confirmed'):
                    self.notifier.notify_btc_confirmed(
                        recipient=self.notification_email,
                        bridge_id=bridge_tx.bridge_id,
                        btc_tx_hash=btc_tx_hash,
                        confirmations=btc_validation['confirmations']
                    )

            # Step 3: Create Ethereum transaction
            logger.info("\n[Step 3/5] Creating Ethereum transaction...")
            eth_tx = self.bridge.eth_connector.create_transaction(
                to_address=eth_destination_address,
                amount_eth=bridge_tx.amount_eth
            )

            # Step 4: Sign and broadcast to Ethereum network
            logger.info("\n[Step 4/5] Broadcasting to Ethereum network...")
            logger.info(f"  Destination: {eth_destination_address}")
            logger.info(f"  Amount: {bridge_tx.amount_eth} WBTC")

            eth_tx_hash = self.bridge.eth_connector.sign_and_send_transaction(
                transaction=eth_tx,
                private_key=eth_private_key
            )

            bridge_tx.eth_tx_hash = eth_tx_hash
            bridge_tx.status = "eth_broadcasted"

            logger.info(f"✓ Transaction broadcasted to Ethereum!")
            logger.info(f"  TX Hash: {eth_tx_hash}")
            logger.info(f"  Etherscan: https://etherscan.io/tx/{eth_tx_hash}")

            # Send broadcast notification
            if self.notification_email:
                self.notifier.notify_eth_broadcasted(
                    recipient=self.notification_email,
                    bridge_id=bridge_tx.bridge_id,
                    eth_tx_hash=eth_tx_hash,
                    amount_eth=float(bridge_tx.amount_eth),
                    destination_address=eth_destination_address
                )

            # Step 5: Wait for Ethereum confirmation
            logger.info("\n[Step 5/5] Waiting for Ethereum confirmations...")
            confirmed = self.bridge.eth_connector.wait_for_confirmation(
                tx_hash=eth_tx_hash,
                confirmations=self.bridge.min_eth_confirmations,
                timeout=600
            )

            if confirmed:
                bridge_tx.status = "completed"
                bridge_tx.validation_status = "fully_validated"

                logger.info("\n" + "="*60)
                logger.info("✅ BRIDGE TRANSFER COMPLETED SUCCESSFULLY!")
                logger.info("="*60)
                logger.info(f"Bridge ID: {bridge_tx.bridge_id}")
                logger.info(f"BTC TX: {btc_tx_hash or 'N/A'}")
                logger.info(f"ETH TX: {eth_tx_hash}")
                logger.info(f"Amount: {amount_btc} BTC -> {bridge_tx.amount_eth} WBTC")
                logger.info(f"Destination: {eth_destination_address}")
                logger.info("="*60)

                # Send completion notification
                if self.notification_email:
                    self.notifier.notify_transfer_completed(
                        recipient=self.notification_email,
                        bridge_id=bridge_tx.bridge_id,
                        btc_tx_hash=btc_tx_hash or "N/A",
                        eth_tx_hash=eth_tx_hash,
                        amount_btc=float(amount_btc),
                        amount_eth=float(bridge_tx.amount_eth),
                        destination_address=eth_destination_address
                    )

                # Export transaction log
                self.bridge.export_transaction_log("bridge_transactions.json")

                return True
            else:
                logger.warning("⚠️ Ethereum confirmation timeout")
                bridge_tx.status = "eth_confirmation_timeout"
                return False

        except Exception as e:
            logger.error(f"❌ Bridge transfer failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def validate_all_and_notify(self) -> dict:
        """
        Validate all transactions and send notification

        Returns:
            Validation summary
        """
        logger.info("\n" + "="*60)
        logger.info("VALIDATING ALL TRANSACTIONS")
        logger.info("="*60)

        summary = self.bridge.validate_all_transactions()

        logger.info("\nValidation Summary:")
        logger.info(f"  Total Transactions: {summary['total_transactions']}")
        logger.info(f"  ✅ Completed: {summary['completed']}")
        logger.info(f"  ⏳ Pending: {summary['pending']}")
        logger.info(f"  ❌ Failed: {summary['failed']}")
        logger.info(f"  🔗 Bitcoin Validated: {summary['bitcoin_validated']}")
        logger.info(f"  🔗 Ethereum Validated: {summary['ethereum_validated']}")

        # Send validation notification
        if self.notification_email:
            self.notifier.notify_validation_complete(
                recipient=self.notification_email,
                validation_summary=summary
            )

        return summary

    def check_balances(self, addresses: dict) -> dict:
        """
        Check balances on both networks

        Args:
            addresses: Dict with 'bitcoin' and 'ethereum' addresses

        Returns:
            Balance information
        """
        balances = {}

        # Check Ethereum balance
        if 'ethereum' in addresses:
            try:
                eth_balance = self.bridge.eth_connector.get_balance(addresses['ethereum'])
                balances['ethereum'] = {
                    'address': addresses['ethereum'],
                    'balance': float(eth_balance),
                    'currency': 'ETH'
                }
                logger.info(f"Ethereum Balance ({addresses['ethereum']}): {eth_balance} ETH")
            except Exception as e:
                logger.error(f"Error checking Ethereum balance: {e}")

        return balances


def main():
    """Main entry point for bridge orchestrator"""
    parser = argparse.ArgumentParser(
        description='Bitcoin to Ethereum Bridge Orchestrator'
    )

    parser.add_argument(
        '--btc-address',
        type=str,
        help='Bitcoin source address'
    )

    parser.add_argument(
        '--eth-address',
        type=str,
        required=True,
        help='Ethereum destination address'
    )

    parser.add_argument(
        '--amount',
        type=str,
        help='Amount in BTC to transfer'
    )

    parser.add_argument(
        '--eth-private-key',
        type=str,
        help='Ethereum private key (or set ETH_PRIVATE_KEY env var)'
    )

    parser.add_argument(
        '--btc-tx-hash',
        type=str,
        help='Bitcoin transaction hash (if already sent)'
    )

    parser.add_argument(
        '--email',
        type=str,
        help='Email for notifications'
    )

    parser.add_argument(
        '--network',
        type=str,
        default='mainnet',
        choices=['mainnet', 'testnet', 'goerli', 'sepolia'],
        help='Network to use (default: mainnet)'
    )

    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate existing transactions'
    )

    parser.add_argument(
        '--check-balance',
        action='store_true',
        help='Check balance of Ethereum address'
    )

    args = parser.parse_args()

    # Get private key from env if not provided
    eth_private_key = args.eth_private_key or os.getenv('ETH_PRIVATE_KEY')

    # Initialize orchestrator
    orchestrator = BridgeOrchestrator(
        ethereum_network=args.network,
        bitcoin_network=args.network if args.network in ['mainnet', 'testnet'] else 'mainnet',
        notification_email=args.email
    )

    # Check balance mode
    if args.check_balance:
        orchestrator.check_balances({'ethereum': args.eth_address})
        return 0

    # Validate only mode
    if args.validate_only:
        orchestrator.validate_all_and_notify()
        return 0

    # Transfer mode - requires all parameters
    if not args.amount or not eth_private_key:
        logger.error("❌ Transfer mode requires --amount and --eth-private-key (or ETH_PRIVATE_KEY env var)")
        parser.print_help()
        return 1

    # Execute transfer
    success = orchestrator.execute_transfer(
        btc_source_address=args.btc_address or "bc1qfzhx87ckhn4tnkswhsth56h0gm5we4hdq5wass",
        eth_destination_address=args.eth_address,
        amount_btc=Decimal(args.amount),
        eth_private_key=eth_private_key,
        btc_tx_hash=args.btc_tx_hash
    )

    if success:
        logger.info("\n✅ Bridge operation completed successfully!")
        return 0
    else:
        logger.error("\n❌ Bridge operation failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
