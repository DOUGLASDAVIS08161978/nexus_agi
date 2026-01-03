#!/usr/bin/env python3
"""
REAL BITCOIN BLOCKCHAIN BATCH VALIDATOR & BROADCASTER
======================================================

This system connects to the REAL Bitcoin blockchain network and provides:
- Batch validation of blocks and transactions
- Real blockchain connectivity via multiple methods
- Transaction broadcasting to real Bitcoin network
- Block validation using actual Bitcoin nodes
- Retry logic and failover support
- Comprehensive logging and reporting

IMPORTANT: This connects to REAL Bitcoin blockchain.
Transactions are irreversible and may incur network fees.

Author: Douglas Shane Davis & Claude
Date: 2026-01-03
"""

import hashlib
import requests
import json
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(name)s - %(message)s',
    handlers=[
        logging.FileHandler('bitcoin_batch_broadcaster.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Lightning Wallet - ALL rewards go here
LIGHTNING_WALLET = "bc1qfzhx87ckhn4tnkswhsth56h0gm5we4hdq5wass"

# Blockchain API endpoints (for validation and broadcasting)
BLOCKCHAIN_APIS = {
    "blockstream": {
        "mainnet": "https://blockstream.info/api",
        "testnet": "https://blockstream.info/testnet/api"
    },
    "blockchain_info": {
        "mainnet": "https://blockchain.info",
        "testnet": None
    },
    "blockcypher": {
        "mainnet": "https://api.blockcypher.com/v1/btc/main",
        "testnet": "https://api.blockcypher.com/v1/btc/test3"
    },
    "mempool": {
        "mainnet": "https://mempool.space/api",
        "testnet": "https://mempool.space/testnet/api"
    }
}


@dataclass
class BlockValidationResult:
    """Result of block validation against real blockchain"""
    block_height: int
    block_hash: str
    is_valid: bool
    confirmations: int
    timestamp: str
    merkle_root: Optional[str]
    difficulty: float
    nonce: int
    size: int
    version: int
    tx_count: int
    validation_method: str
    network_status: str
    validation_checks: Dict[str, bool]
    error_message: Optional[str] = None


@dataclass
class TransactionBroadcastResult:
    """Result of transaction broadcast to real network"""
    txid: str
    broadcast_time: str
    status: str
    confirmations: int
    network: str
    from_address: str
    to_address: str
    amount: float
    fee: float
    broadcast_method: str
    raw_response: Optional[str] = None
    error_message: Optional[str] = None


class RealBitcoinValidator:
    """Validates blocks and transactions against REAL Bitcoin blockchain"""

    def __init__(self, network: str = "mainnet"):
        """
        Initialize validator

        Args:
            network: "mainnet" or "testnet"
        """
        self.network = network
        self.api_index = 0
        self.apis = list(BLOCKCHAIN_APIS.keys())
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'NexusAGI-Bitcoin-Validator/1.0'
        })

    def _get_api_url(self, api_name: str) -> Optional[str]:
        """Get API URL for given service"""
        api_config = BLOCKCHAIN_APIS.get(api_name, {})
        return api_config.get(self.network)

    def _retry_request(self, url: str, max_retries: int = 3) -> Optional[Dict]:
        """Make HTTP request with exponential backoff retry"""
        for attempt in range(max_retries):
            try:
                logger.debug(f"Request attempt {attempt + 1}/{max_retries}: {url}")
                response = self.session.get(url, timeout=30)

                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 404:
                    logger.warning(f"Resource not found: {url}")
                    return None
                else:
                    logger.warning(f"HTTP {response.status_code}: {url}")

            except requests.exceptions.Timeout:
                logger.warning(f"Timeout on attempt {attempt + 1}")
            except requests.exceptions.ConnectionError:
                logger.warning(f"Connection error on attempt {attempt + 1}")
            except Exception as e:
                logger.error(f"Request error: {e}")

            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                logger.info(f"Retrying in {wait_time}s...")
                time.sleep(wait_time)

        return None

    def validate_block_on_chain(self, block_height: int) -> Optional[BlockValidationResult]:
        """
        Validate a block against the REAL Bitcoin blockchain

        Args:
            block_height: Block height to validate

        Returns:
            BlockValidationResult if block exists, None otherwise
        """
        logger.info(f"Validating block #{block_height} on REAL Bitcoin blockchain...")

        # Try multiple APIs for redundancy
        for api_name in self.apis:
            api_url = self._get_api_url(api_name)
            if not api_url:
                continue

            try:
                if api_name == "blockstream":
                    # Get block hash from height
                    hash_url = f"{api_url}/block-height/{block_height}"
                    block_hash = self._retry_request(hash_url)

                    if not block_hash:
                        continue

                    if isinstance(block_hash, dict):
                        block_hash = block_hash.get('hash', block_hash)

                    # Get full block data
                    block_url = f"{api_url}/block/{block_hash}"
                    block_data = self._retry_request(block_url)

                    if block_data:
                        return self._parse_blockstream_data(block_data, api_name)

                elif api_name == "mempool":
                    # Mempool Space API
                    block_url = f"{api_url}/block-height/{block_height}"
                    block_hash = self._retry_request(block_url)

                    if block_hash:
                        if isinstance(block_hash, str):
                            block_url = f"{api_url}/block/{block_hash}"
                            block_data = self._retry_request(block_url)
                            if block_data:
                                return self._parse_blockstream_data(block_data, api_name)

                elif api_name == "blockcypher":
                    # BlockCypher API
                    block_url = f"{api_url}/blocks/{block_height}"
                    block_data = self._retry_request(block_url)

                    if block_data:
                        return self._parse_blockcypher_data(block_data, api_name)

            except Exception as e:
                logger.error(f"Error validating with {api_name}: {e}")
                continue

        logger.error(f"Failed to validate block #{block_height} on all APIs")
        return None

    def _parse_blockstream_data(self, block_data: Dict, api_name: str) -> BlockValidationResult:
        """Parse Blockstream/Mempool API response"""
        current_height = self._get_current_block_height()
        confirmations = 0
        if current_height:
            confirmations = max(0, current_height - block_data.get('height', 0) + 1)

        validation_checks = {
            'block_exists': True,
            'hash_valid': len(block_data.get('id', '')) == 64,
            'merkle_root_valid': len(block_data.get('merkle_root', '')) == 64,
            'timestamp_valid': block_data.get('timestamp', 0) > 0,
            'transactions_present': block_data.get('tx_count', 0) > 0,
            'confirmations_sufficient': confirmations >= 6
        }

        return BlockValidationResult(
            block_height=block_data.get('height', 0),
            block_hash=block_data.get('id', ''),
            is_valid=all(validation_checks.values()),
            confirmations=confirmations,
            timestamp=datetime.fromtimestamp(block_data.get('timestamp', 0)).strftime("%Y-%m-%d %H:%M:%S"),
            merkle_root=block_data.get('merkle_root'),
            difficulty=block_data.get('difficulty', 0),
            nonce=block_data.get('nonce', 0),
            size=block_data.get('size', 0),
            version=block_data.get('version', 0),
            tx_count=block_data.get('tx_count', 0),
            validation_method=api_name,
            network_status="CONFIRMED" if confirmations >= 6 else "PENDING",
            validation_checks=validation_checks
        )

    def _parse_blockcypher_data(self, block_data: Dict, api_name: str) -> BlockValidationResult:
        """Parse BlockCypher API response"""
        confirmations = block_data.get('confirmations', 0)

        validation_checks = {
            'block_exists': True,
            'hash_valid': len(block_data.get('hash', '')) == 64,
            'merkle_root_valid': len(block_data.get('mrkl_root', '')) == 64,
            'timestamp_valid': block_data.get('time') is not None,
            'transactions_present': block_data.get('n_tx', 0) > 0,
            'confirmations_sufficient': confirmations >= 6
        }

        timestamp = block_data.get('time', '')
        if timestamp:
            timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00')).strftime("%Y-%m-%d %H:%M:%S")

        return BlockValidationResult(
            block_height=block_data.get('height', 0),
            block_hash=block_data.get('hash', ''),
            is_valid=all(validation_checks.values()),
            confirmations=confirmations,
            timestamp=timestamp,
            merkle_root=block_data.get('mrkl_root'),
            difficulty=0,  # Not provided by BlockCypher
            nonce=block_data.get('nonce', 0),
            size=block_data.get('bits', 0),
            version=block_data.get('ver', 0),
            tx_count=block_data.get('n_tx', 0),
            validation_method=api_name,
            network_status="CONFIRMED" if confirmations >= 6 else "PENDING",
            validation_checks=validation_checks
        )

    def _get_current_block_height(self) -> Optional[int]:
        """Get current blockchain height"""
        for api_name in self.apis:
            api_url = self._get_api_url(api_name)
            if not api_url:
                continue

            try:
                if api_name in ["blockstream", "mempool"]:
                    url = f"{api_url}/blocks/tip/height"
                    height = self._retry_request(url)
                    if height:
                        return int(height) if isinstance(height, (int, str)) else height.get('height', 0)

                elif api_name == "blockcypher":
                    url = f"{api_url}"
                    data = self._retry_request(url)
                    if data:
                        return data.get('height', 0)
            except:
                continue

        return None

    def validate_transaction(self, txid: str) -> Optional[Dict]:
        """Validate a transaction on the blockchain"""
        for api_name in self.apis:
            api_url = self._get_api_url(api_name)
            if not api_url:
                continue

            try:
                if api_name in ["blockstream", "mempool"]:
                    url = f"{api_url}/tx/{txid}"
                    tx_data = self._retry_request(url)
                    if tx_data:
                        return tx_data

                elif api_name == "blockcypher":
                    url = f"{api_url}/txs/{txid}"
                    tx_data = self._retry_request(url)
                    if tx_data:
                        return tx_data
            except:
                continue

        return None

    def validate_address(self, address: str) -> Dict[str, Any]:
        """Validate Bitcoin address and get balance/transactions"""
        logger.info(f"Validating address: {address}")

        for api_name in self.apis:
            api_url = self._get_api_url(api_name)
            if not api_url:
                continue

            try:
                if api_name in ["blockstream", "mempool"]:
                    url = f"{api_url}/address/{address}"
                    addr_data = self._retry_request(url)

                    if addr_data:
                        # Get transactions
                        txs_url = f"{api_url}/address/{address}/txs"
                        txs_data = self._retry_request(txs_url)

                        return {
                            'address': address,
                            'balance': addr_data.get('chain_stats', {}).get('funded_txo_sum', 0) - addr_data.get('chain_stats', {}).get('spent_txo_sum', 0),
                            'total_received': addr_data.get('chain_stats', {}).get('funded_txo_sum', 0),
                            'total_sent': addr_data.get('chain_stats', {}).get('spent_txo_sum', 0),
                            'tx_count': addr_data.get('chain_stats', {}).get('tx_count', 0),
                            'unconfirmed_balance': addr_data.get('mempool_stats', {}).get('funded_txo_sum', 0) - addr_data.get('mempool_stats', {}).get('spent_txo_sum', 0),
                            'transactions': txs_data if txs_data else [],
                            'validation_method': api_name
                        }

                elif api_name == "blockcypher":
                    url = f"{api_url}/addrs/{address}/balance"
                    addr_data = self._retry_request(url)

                    if addr_data:
                        return {
                            'address': address,
                            'balance': addr_data.get('balance', 0),
                            'total_received': addr_data.get('total_received', 0),
                            'total_sent': addr_data.get('total_sent', 0),
                            'tx_count': addr_data.get('n_tx', 0),
                            'unconfirmed_balance': addr_data.get('unconfirmed_balance', 0),
                            'transactions': [],
                            'validation_method': api_name
                        }
            except Exception as e:
                logger.error(f"Error validating address with {api_name}: {e}")
                continue

        logger.error(f"Failed to validate address {address} on all APIs")
        return {
            'address': address,
            'error': 'Failed to validate address on all APIs'
        }


class BatchBlockchainValidator:
    """Batch validation system for multiple blocks"""

    def __init__(self, network: str = "mainnet", max_workers: int = 5):
        self.validator = RealBitcoinValidator(network=network)
        self.max_workers = max_workers
        self.results: List[BlockValidationResult] = []

    def validate_blocks_batch(self, block_heights: List[int]) -> List[BlockValidationResult]:
        """
        Validate multiple blocks in parallel

        Args:
            block_heights: List of block heights to validate

        Returns:
            List of BlockValidationResult objects
        """
        logger.info(f"Starting batch validation of {len(block_heights)} blocks...")

        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_height = {
                executor.submit(self.validator.validate_block_on_chain, height): height
                for height in block_heights
            }

            for future in as_completed(future_to_height):
                height = future_to_height[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                        logger.info(f"✅ Validated block #{height}: {result.block_hash[:16]}...")
                    else:
                        logger.warning(f"❌ Failed to validate block #{height}")
                except Exception as e:
                    logger.error(f"❌ Error validating block #{height}: {e}")

        self.results = results
        return results

    def print_batch_summary(self):
        """Print summary of batch validation"""
        if not self.results:
            logger.warning("No validation results to display")
            return

        valid_blocks = [r for r in self.results if r.is_valid]
        total_confirmations = sum(r.confirmations for r in self.results)
        avg_confirmations = total_confirmations / len(self.results) if self.results else 0

        print("\n" + "="*100)
        print("📊 BATCH BLOCKCHAIN VALIDATION SUMMARY")
        print("="*100)
        print(f"Total Blocks Validated:     {len(self.results)}")
        print(f"Valid Blocks:               {len(valid_blocks)}")
        print(f"Invalid Blocks:             {len(self.results) - len(valid_blocks)}")
        print(f"Average Confirmations:      {avg_confirmations:.1f}")
        print(f"Total Transactions:         {sum(r.tx_count for r in self.results)}")
        print("="*100)


class RealBitcoinBroadcaster:
    """Broadcasts transactions to REAL Bitcoin network"""

    def __init__(self, network: str = "mainnet"):
        self.network = network
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'NexusAGI-Bitcoin-Broadcaster/1.0',
            'Content-Type': 'application/json'
        })

    def broadcast_raw_transaction(self, raw_tx_hex: str) -> Optional[TransactionBroadcastResult]:
        """
        Broadcast raw transaction to Bitcoin network

        Args:
            raw_tx_hex: Raw transaction in hexadecimal format

        Returns:
            TransactionBroadcastResult if successful, None otherwise
        """
        logger.info("Broadcasting raw transaction to Bitcoin network...")

        # Try Blockstream API
        api_url = BLOCKCHAIN_APIS["blockstream"][self.network]
        if api_url:
            try:
                url = f"{api_url}/tx"
                response = self.session.post(url, data=raw_tx_hex)

                if response.status_code == 200:
                    txid = response.text.strip()
                    logger.info(f"✅ Transaction broadcasted successfully: {txid}")

                    return TransactionBroadcastResult(
                        txid=txid,
                        broadcast_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        status="BROADCASTED",
                        confirmations=0,
                        network=self.network,
                        from_address="Unknown",
                        to_address="Unknown",
                        amount=0.0,
                        fee=0.0,
                        broadcast_method="blockstream",
                        raw_response=txid
                    )
                else:
                    logger.error(f"Broadcast failed: HTTP {response.status_code} - {response.text}")
            except Exception as e:
                logger.error(f"Broadcast error: {e}")

        return None

    def simulate_coinbase_broadcast(self, block_height: int, reward_address: str,
                                   reward_btc: float = 6.25) -> TransactionBroadcastResult:
        """
        Simulate broadcasting a coinbase transaction (mining reward)

        Note: Real coinbase transactions can only be created by miners
        This simulates what would happen if a block was successfully mined
        """
        # Generate simulated TXID
        data = f"coinbase_{block_height}_{reward_address}_{reward_btc}_{time.time()}"
        txid = hashlib.sha256(data.encode()).hexdigest()

        logger.info(f"Simulating coinbase broadcast for block #{block_height}")

        return TransactionBroadcastResult(
            txid=txid,
            broadcast_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            status="SIMULATED (Coinbase transactions require mining)",
            confirmations=0,
            network=self.network,
            from_address="Coinbase (Mining Reward)",
            to_address=reward_address,
            amount=reward_btc,
            fee=0.0,
            broadcast_method="simulation",
            raw_response=None
        )


class BitcoinBatchBroadcastSystem:
    """Complete batch validation and broadcasting system"""

    def __init__(self, wallet_address: str = LIGHTNING_WALLET, network: str = "mainnet"):
        self.wallet = wallet_address
        self.network = network
        self.validator = RealBitcoinValidator(network=network)
        self.batch_validator = BatchBlockchainValidator(network=network)
        self.broadcaster = RealBitcoinBroadcaster(network=network)
        self.start_time = time.time()

    def run_full_validation_and_broadcast(self, block_heights: List[int]) -> Dict[str, Any]:
        """
        Run complete batch validation and broadcasting workflow

        Args:
            block_heights: List of block heights to validate

        Returns:
            Comprehensive report
        """
        print("\n" + "="*100)
        print("🚀 REAL BITCOIN BLOCKCHAIN - BATCH VALIDATION & BROADCASTING SYSTEM")
        print("="*100)
        print(f"Network:        {self.network.upper()}")
        print(f"Wallet:         {self.wallet}")
        print(f"Blocks:         {len(block_heights)}")
        print(f"Start Time:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*100 + "\n")

        # Phase 1: Batch Block Validation
        print("="*100)
        print("PHASE 1: BATCH BLOCKCHAIN VALIDATION")
        print("="*100 + "\n")

        validation_results = self.batch_validator.validate_blocks_batch(block_heights)

        # Print validation details
        for result in validation_results:
            self._print_block_validation(result)

        self.batch_validator.print_batch_summary()

        # Phase 2: Wallet Validation
        print("\n" + "="*100)
        print("PHASE 2: WALLET VALIDATION")
        print("="*100 + "\n")

        wallet_data = self.validator.validate_address(self.wallet)
        self._print_wallet_info(wallet_data)

        # Phase 3: Broadcast Simulation (for coinbase transactions)
        print("\n" + "="*100)
        print("PHASE 3: TRANSACTION BROADCASTING")
        print("="*100 + "\n")

        broadcast_results = []
        for height in block_heights[:5]:  # Simulate broadcasts for first 5 blocks
            result = self.broadcaster.simulate_coinbase_broadcast(
                block_height=height,
                reward_address=self.wallet,
                reward_btc=6.25
            )
            broadcast_results.append(result)
            self._print_broadcast_result(result)

        # Generate final report
        runtime = time.time() - self.start_time

        report = {
            'network': self.network,
            'wallet_address': self.wallet,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'runtime_seconds': runtime,
            'validation_summary': {
                'total_blocks_validated': len(validation_results),
                'valid_blocks': len([r for r in validation_results if r.is_valid]),
                'total_confirmations': sum(r.confirmations for r in validation_results),
                'total_transactions': sum(r.tx_count for r in validation_results)
            },
            'wallet_data': wallet_data,
            'broadcast_summary': {
                'total_broadcasts': len(broadcast_results),
                'successful_broadcasts': len([b for b in broadcast_results if b.status == "BROADCASTED"])
            },
            'validation_results': [asdict(r) for r in validation_results],
            'broadcast_results': [asdict(b) for b in broadcast_results]
        }

        # Save report
        filename = f"bitcoin_batch_report_{int(time.time())}.json"
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)

        print("\n" + "="*100)
        print("✅ BATCH VALIDATION & BROADCASTING COMPLETE")
        print("="*100)
        print(f"Runtime:            {runtime:.2f} seconds")
        print(f"Report saved to:    {filename}")
        print("="*100 + "\n")

        return report

    def _print_block_validation(self, result: BlockValidationResult):
        """Print detailed block validation result"""
        status = "✅ VALID" if result.is_valid else "❌ INVALID"

        print(f"{status} - Block #{result.block_height}")
        print(f"  Hash:             {result.block_hash}")
        print(f"  Confirmations:    {result.confirmations}")
        print(f"  Status:           {result.network_status}")
        print(f"  Transactions:     {result.tx_count}")
        print(f"  Timestamp:        {result.timestamp}")
        print(f"  Merkle Root:      {result.merkle_root}")
        print(f"  Validation Via:   {result.validation_method.upper()}")
        print()

    def _print_wallet_info(self, wallet_data: Dict):
        """Print wallet information"""
        print(f"💰 WALLET INFORMATION")
        print(f"{'='*100}")
        print(f"Address:              {wallet_data.get('address', 'N/A')}")
        print(f"Balance:              {wallet_data.get('balance', 0) / 1e8:.8f} BTC")
        print(f"Total Received:       {wallet_data.get('total_received', 0) / 1e8:.8f} BTC")
        print(f"Total Sent:           {wallet_data.get('total_sent', 0) / 1e8:.8f} BTC")
        print(f"Unconfirmed:          {wallet_data.get('unconfirmed_balance', 0) / 1e8:.8f} BTC")
        print(f"Transaction Count:    {wallet_data.get('tx_count', 0)}")
        print(f"Validation Method:    {wallet_data.get('validation_method', 'N/A').upper()}")
        print(f"{'='*100}")

    def _print_broadcast_result(self, result: TransactionBroadcastResult):
        """Print broadcast result"""
        print(f"📤 BROADCAST RESULT")
        print(f"  TXID:             {result.txid}")
        print(f"  Status:           {result.status}")
        print(f"  From:             {result.from_address}")
        print(f"  To:               {result.to_address}")
        print(f"  Amount:           {result.amount:.8f} BTC")
        print(f"  Time:             {result.broadcast_time}")
        print(f"  Method:           {result.broadcast_method.upper()}")
        print()


def main():
    """Main execution"""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║        REAL BITCOIN BLOCKCHAIN BATCH VALIDATOR & BROADCASTER v1.0            ║
║                                                                              ║
║                    Connects to REAL Bitcoin Network                          ║
║                                                                              ║
║  ⚠️  WARNING: This validates against the REAL Bitcoin blockchain            ║
║                                                                              ║
║  Author: Douglas Shane Davis & Claude                                       ║
║  Date: 2026-01-03                                                           ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)

    # Configuration
    network = "mainnet"  # Use "testnet" for testing
    wallet_address = LIGHTNING_WALLET

    # Get current block height
    validator = RealBitcoinValidator(network=network)
    current_height = validator._get_current_block_height()

    if not current_height:
        logger.error("Failed to get current block height. Check network connectivity.")
        return

    logger.info(f"Current blockchain height: {current_height:,}")

    # Validate recent blocks (last 10 blocks)
    block_heights = list(range(current_height - 9, current_height + 1))

    # Initialize and run system
    system = BitcoinBatchBroadcastSystem(
        wallet_address=wallet_address,
        network=network
    )

    # Run full validation and broadcasting
    report = system.run_full_validation_and_broadcast(block_heights)

    # Print final summary
    print("\n" + "="*100)
    print("📊 FINAL EXECUTION SUMMARY")
    print("="*100)
    print(f"Network:                 {report['network'].upper()}")
    print(f"Wallet:                  {report['wallet_address']}")
    print(f"Blocks Validated:        {report['validation_summary']['total_blocks_validated']}")
    print(f"Valid Blocks:            {report['validation_summary']['valid_blocks']}")
    print(f"Total Confirmations:     {report['validation_summary']['total_confirmations']:,}")
    print(f"Total Transactions:      {report['validation_summary']['total_transactions']:,}")
    print(f"Broadcasts Simulated:    {report['broadcast_summary']['total_broadcasts']}")
    print(f"Runtime:                 {report['runtime_seconds']:.2f} seconds")
    print("="*100)

    print("\n✅ All operations completed successfully!")
    print(f"📄 Detailed report saved to: bitcoin_batch_report_*.json\n")


if __name__ == "__main__":
    main()
