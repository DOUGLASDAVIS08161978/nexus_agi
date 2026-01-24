#!/usr/bin/env python3
"""
Universal Blockchain Wallet Monitor v3.0
Monitors Bitcoin, Ethereum, Monad, and all EVM-compatible chains
Tracks transactions, balances, and counterparties across all networks
"""

import json
import sqlite3
import time
import requests
import logging
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import signal
from web3 import Web3
from eth_utils import is_address

# =============== CONFIGURATION ===============

# Blockchain API Providers
BLOCKCHAIN_APIS = {
    'bitcoin': [
        "https://mempool.space/api",
        "https://blockstream.info/api",
        "https://blockchain.info"
    ],
    'ethereum': [
        "https://api.etherscan.io/api",
        "https://eth.blockscout.com/api",
        "https://cloudflare-eth.com"
    ],
    'monad': [
        "https://testnet-rpc.monad.xyz",
        "https://testnet.monad.xyz"
    ],
    'sepolia': [
        "https://sepolia.etherscan.io/api",
        "https://rpc.sepolia.org"
    ]
}

# Etherscan-like API keys (free tier - no key needed for basic queries)
ETHERSCAN_API_KEY = ""  # Optional - increases rate limits

# Database and logging
DB_FILE = "universal_wallet_monitor.db"
LOG_DIR = "logs/"
TRANSACTION_LOG = f"{LOG_DIR}universal_transactions.jsonl"
ACTIVITY_LOG = f"{LOG_DIR}universal_activity.log"

# Monitoring settings
MONITOR_INTERVAL = 45  # Seconds between checks
MAX_RETRIES = 3
REQUEST_TIMEOUT = 15
MAX_CONCURRENT_REQUESTS = 8

# Alert thresholds
ALERT_LARGE_BTC = 1.0       # 1 BTC
ALERT_LARGE_ETH = 5.0       # 5 ETH
ALERT_LARGE_MON = 1000.0    # 1000 MON
ALERT_RAPID_TX = 10         # >10 tx/hour

# =============== DATA STRUCTURES ===============

@dataclass
class UniversalTransaction:
    txid: str
    address: str
    blockchain: str
    timestamp: int
    amount: float
    token_symbol: str  # BTC, ETH, MON, WBTC, etc.
    confirmations: int
    block_height: Optional[int]
    direction: str
    counterparties: List[str]
    fees: Optional[float] = None
    token_address: Optional[str] = None  # For ERC20 tokens
    usd_value: Optional[float] = None

@dataclass
class WalletStats:
    address: str
    blockchain: str
    native_balance: float = 0.0
    token_balances: Dict[str, float] = None
    total_received: float = 0.0
    total_sent: float = 0.0
    tx_count: int = 0
    first_seen: Optional[int] = None
    last_active: Optional[int] = None

# =============== LOGGING SETUP ===============

def setup_logging():
    """Configure logging system"""
    import os
    os.makedirs(LOG_DIR, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(ACTIVITY_LOG, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# =============== DATABASE MANAGER ===============

class UniversalDatabaseManager:
    """Multi-chain database manager"""

    def __init__(self, db_path: str = DB_FILE):
        self.db_path = db_path
        self.init_database()

    def get_connection(self):
        """Get database connection"""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def init_database(self):
        """Initialize multi-chain database schema"""
        conn = self.get_connection()
        cursor = conn.cursor()

        # Enable optimizations
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")

        # Monitored addresses with blockchain info
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS monitored_addresses (
                address TEXT,
                blockchain TEXT,
                label TEXT,
                date_added TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT 1,
                last_checked INTEGER DEFAULT 0,
                PRIMARY KEY (address, blockchain)
            )
        ''')

        # Universal transactions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS transactions (
                txid TEXT,
                address TEXT,
                blockchain TEXT,
                timestamp INTEGER,
                amount REAL,
                token_symbol TEXT,
                confirmations INTEGER,
                block_height INTEGER,
                direction TEXT,
                fees REAL,
                token_address TEXT,
                usd_value REAL,
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (txid, address, blockchain)
            )
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_tx_address_time
            ON transactions(address, blockchain, timestamp DESC)
        ''')

        # Transaction links (counterparties)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS transaction_links (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                txid TEXT,
                blockchain TEXT,
                source_address TEXT,
                target_address TEXT,
                amount REAL,
                token_symbol TEXT,
                link_type TEXT,
                timestamp INTEGER,
                UNIQUE(txid, blockchain, source_address, target_address, link_type)
            )
        ''')

        # Address statistics
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS address_stats (
                address TEXT,
                blockchain TEXT,
                native_balance REAL DEFAULT 0,
                total_received REAL DEFAULT 0,
                total_sent REAL DEFAULT 0,
                tx_count INTEGER DEFAULT 0,
                first_seen INTEGER,
                last_active INTEGER,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (address, blockchain)
            )
        ''')

        # Token balances (for ERC20/BEP20 etc)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS token_balances (
                address TEXT,
                blockchain TEXT,
                token_address TEXT,
                token_symbol TEXT,
                balance REAL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (address, blockchain, token_address)
            )
        ''')

        # Alerts
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                address TEXT,
                blockchain TEXT,
                alert_type TEXT,
                severity TEXT,
                message TEXT,
                details TEXT,
                acknowledged BOOLEAN DEFAULT 0
            )
        ''')

        conn.commit()
        conn.close()
        logger.info("✓ Database initialized")

    def add_address(self, address: str, blockchain: str, label: str = "") -> bool:
        """Add address to monitoring"""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR IGNORE INTO monitored_addresses (address, blockchain, label)
                VALUES (?, ?, ?)
            ''', (address, blockchain, label))

            cursor.execute('''
                INSERT OR IGNORE INTO address_stats (address, blockchain)
                VALUES (?, ?)
            ''', (address, blockchain))

            conn.commit()
            logger.info(f"✓ Added {blockchain} address: {address[:12]}...")
            return True

        except sqlite3.Error as e:
            logger.error(f"Database error: {e}")
            return False
        finally:
            conn.close()

    def save_transaction(self, tx: UniversalTransaction) -> bool:
        """Save transaction"""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()

            cursor.execute('''
                INSERT OR REPLACE INTO transactions
                (txid, address, blockchain, timestamp, amount, token_symbol,
                 confirmations, block_height, direction, fees, token_address, usd_value)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (tx.txid, tx.address, tx.blockchain, tx.timestamp, tx.amount,
                  tx.token_symbol, tx.confirmations, tx.block_height, tx.direction,
                  tx.fees, tx.token_address, tx.usd_value))

            # Save links
            for counterparty in tx.counterparties:
                if counterparty:
                    link_type = 'output' if tx.direction == 'outgoing' else 'input'
                    source = tx.address if tx.direction == 'outgoing' else counterparty
                    target = counterparty if tx.direction == 'outgoing' else tx.address

                    cursor.execute('''
                        INSERT OR IGNORE INTO transaction_links
                        (txid, blockchain, source_address, target_address, amount,
                         token_symbol, link_type, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (tx.txid, tx.blockchain, source, target, tx.amount,
                          tx.token_symbol, link_type, tx.timestamp))

            conn.commit()
            return True

        except sqlite3.Error as e:
            logger.error(f"Error saving transaction: {e}")
            return False
        finally:
            conn.close()

    def get_monitored_addresses(self) -> List[Tuple[str, str]]:
        """Get all active monitored addresses with blockchain"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT address, blockchain FROM monitored_addresses
            WHERE is_active = 1
        ''')
        addresses = [(row[0], row[1]) for row in cursor.fetchall()]
        conn.close()
        return addresses

    def get_address_transactions(self, address: str, blockchain: str, limit: int = 50) -> List[Dict]:
        """Get transaction history"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM transactions
            WHERE address = ? AND blockchain = ?
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (address, blockchain, limit))

        transactions = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return transactions

    def create_alert(self, address: str, blockchain: str, alert_type: str,
                    message: str, severity: str = "INFO", details: str = ""):
        """Create alert"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO alerts (address, blockchain, alert_type, severity, message, details)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (address, blockchain, alert_type, severity, message, details))
        conn.commit()
        conn.close()
        logger.warning(f"⚠️  ALERT [{severity}] {blockchain}:{address[:10]}... - {message}")

# =============== MULTI-CHAIN API CLIENT ===============

class MultiChainAPIClient:
    """Universal blockchain API client"""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'UniversalWalletMonitor/3.0',
            'Accept': 'application/json'
        })

        # Web3 instances for EVM chains
        self.web3_instances = {
            'ethereum': Web3(Web3.HTTPProvider('https://cloudflare-eth.com')),
            'monad': Web3(Web3.HTTPProvider('https://testnet-rpc.monad.xyz')),
            'sepolia': Web3(Web3.HTTPProvider('https://rpc.sepolia.org'))
        }

        self.last_request_time = {}
        self.rate_limit_delay = 0.3

    def _rate_limit(self, blockchain: str):
        """Enforce rate limiting per blockchain"""
        current_time = time.time()
        last_time = self.last_request_time.get(blockchain, 0)

        if current_time - last_time < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - (current_time - last_time))

        self.last_request_time[blockchain] = time.time()

    def get_bitcoin_transactions(self, address: str) -> Optional[List[Dict]]:
        """Get Bitcoin transactions"""
        self._rate_limit('bitcoin')

        try:
            url = f"https://mempool.space/api/address/{address}/txs"
            response = self.session.get(url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Bitcoin API error: {e}")
            return None

    def get_bitcoin_balance(self, address: str) -> Optional[float]:
        """Get Bitcoin balance"""
        self._rate_limit('bitcoin')

        try:
            url = f"https://mempool.space/api/address/{address}"
            response = self.session.get(url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            data = response.json()

            # Balance from chain_stats
            balance_sats = data.get('chain_stats', {}).get('funded_txo_sum', 0) - \
                          data.get('chain_stats', {}).get('spent_txo_sum', 0)
            return balance_sats / 1e8  # Convert to BTC
        except Exception as e:
            logger.error(f"Bitcoin balance error: {e}")
            return None

    def get_evm_transactions(self, address: str, blockchain: str) -> Optional[List[Dict]]:
        """Get EVM chain transactions (Ethereum, Monad, etc.)"""
        self._rate_limit(blockchain)

        web3 = self.web3_instances.get(blockchain)
        if not web3:
            logger.error(f"No Web3 instance for {blockchain}")
            return None

        try:
            # Use eth_getLogs for transaction history
            # For Etherscan-compatible APIs
            if blockchain == 'ethereum':
                url = f"https://api.etherscan.io/api"
                params = {
                    'module': 'account',
                    'action': 'txlist',
                    'address': address,
                    'startblock': 0,
                    'endblock': 99999999,
                    'sort': 'desc',
                    'apikey': ETHERSCAN_API_KEY
                }
                response = self.session.get(url, params=params, timeout=REQUEST_TIMEOUT)
                response.raise_for_status()
                data = response.json()

                if data.get('status') == '1':
                    return data.get('result', [])

            # For other EVM chains, use RPC directly
            else:
                # Convert to checksum address
                address = web3.to_checksum_address(address)

                # Get recent block number
                latest_block = web3.eth.block_number

                # Query transaction history (limited)
                # Note: This is limited without archive node access
                transactions = []

                # Get transaction count
                tx_count = web3.eth.get_transaction_count(address)
                logger.info(f"{blockchain} address {address[:10]}... has {tx_count} transactions")

                return transactions

        except Exception as e:
            logger.error(f"EVM transaction error for {blockchain}: {e}")
            return None

    def get_evm_balance(self, address: str, blockchain: str) -> Optional[float]:
        """Get EVM chain native balance"""
        self._rate_limit(blockchain)

        web3 = self.web3_instances.get(blockchain)
        if not web3:
            return None

        try:
            # Convert to checksum address
            address = web3.to_checksum_address(address)

            if not web3.is_address(address):
                return None

            balance_wei = web3.eth.get_balance(address)
            balance_eth = web3.from_wei(balance_wei, 'ether')
            return float(balance_eth)

        except Exception as e:
            logger.error(f"EVM balance error for {blockchain}: {e}")
            return None

    def get_erc20_balance(self, address: str, token_address: str, blockchain: str) -> Optional[float]:
        """Get ERC20 token balance"""
        web3 = self.web3_instances.get(blockchain)
        if not web3:
            return None

        try:
            # Convert to checksum addresses
            address = web3.to_checksum_address(address)
            token_address = web3.to_checksum_address(token_address)

            # ERC20 balanceOf ABI
            erc20_abi = [
                {
                    "constant": True,
                    "inputs": [{"name": "_owner", "type": "address"}],
                    "name": "balanceOf",
                    "outputs": [{"name": "balance", "type": "uint256"}],
                    "type": "function"
                },
                {
                    "constant": True,
                    "inputs": [],
                    "name": "decimals",
                    "outputs": [{"name": "", "type": "uint8"}],
                    "type": "function"
                }
            ]

            contract = web3.eth.contract(address=token_address, abi=erc20_abi)
            balance = contract.functions.balanceOf(address).call()
            decimals = contract.functions.decimals().call()

            return balance / (10 ** decimals)

        except Exception as e:
            logger.error(f"ERC20 balance error: {e}")
            return None

# =============== TRANSACTION PROCESSOR ===============

class UniversalTransactionProcessor:
    """Process transactions across all chains"""

    def __init__(self, db: UniversalDatabaseManager, api: MultiChainAPIClient):
        self.db = db
        self.api = api
        self.processed_txids: Set[str] = set()
        self.load_processed_txids()

    def load_processed_txids(self):
        """Load processed transactions"""
        conn = self.db.get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT txid, blockchain FROM transactions')
        self.processed_txids = {f"{row[0]}:{row[1]}" for row in cursor.fetchall()}
        conn.close()

    def process_address(self, address: str, blockchain: str) -> int:
        """Process address on specific blockchain"""
        try:
            if blockchain == 'bitcoin':
                return self._process_bitcoin_address(address)
            elif blockchain in ['ethereum', 'monad', 'sepolia']:
                return self._process_evm_address(address, blockchain)
            else:
                logger.warning(f"Unsupported blockchain: {blockchain}")
                return 0

        except Exception as e:
            logger.error(f"Error processing {blockchain} address {address}: {e}")
            return 0

    def _process_bitcoin_address(self, address: str) -> int:
        """Process Bitcoin address"""
        try:
            # Get balance
            balance = self.api.get_bitcoin_balance(address)
            if balance is not None:
                logger.info(f"💰 BTC Balance for {address[:12]}...: {balance:.8f} BTC")

            # Get transactions
            tx_data = self.api.get_bitcoin_transactions(address)
            if not tx_data:
                return 0

            new_tx_count = 0

            for tx in tx_data:
                txid = tx.get('txid')
                if not txid or f"{txid}:bitcoin" in self.processed_txids:
                    continue

                # Parse transaction
                transaction = self._parse_bitcoin_transaction(tx, address)
                if transaction and self.db.save_transaction(transaction):
                    self.processed_txids.add(f"{txid}:bitcoin")
                    new_tx_count += 1
                    self._log_transaction(transaction)
                    self._analyze_transaction(transaction)

            if new_tx_count > 0:
                logger.info(f"✓ Found {new_tx_count} new Bitcoin transactions")

            return new_tx_count

        except Exception as e:
            logger.error(f"Bitcoin processing error: {e}")
            return 0

    def _process_evm_address(self, address: str, blockchain: str) -> int:
        """Process EVM-compatible address"""
        try:
            # Get native balance
            balance = self.api.get_evm_balance(address, blockchain)
            if balance is not None:
                symbol = 'ETH' if blockchain == 'ethereum' else 'MON' if blockchain == 'monad' else 'ETH'
                logger.info(f"💰 {symbol} Balance for {address[:12]}...: {balance:.8f} {symbol}")

            # Get transactions
            tx_data = self.api.get_evm_transactions(address, blockchain)
            if not tx_data:
                logger.info(f"No transaction history available for {blockchain} (may need archive node)")
                return 0

            new_tx_count = 0

            for tx in tx_data:
                txid = tx.get('hash')
                if not txid or f"{txid}:{blockchain}" in self.processed_txids:
                    continue

                # Parse EVM transaction
                transaction = self._parse_evm_transaction(tx, address, blockchain)
                if transaction and self.db.save_transaction(transaction):
                    self.processed_txids.add(f"{txid}:{blockchain}")
                    new_tx_count += 1
                    self._log_transaction(transaction)
                    self._analyze_transaction(transaction)

            if new_tx_count > 0:
                logger.info(f"✓ Found {new_tx_count} new {blockchain} transactions")

            return new_tx_count

        except Exception as e:
            logger.error(f"EVM processing error: {e}")
            return 0

    def _parse_bitcoin_transaction(self, tx_data: Dict, address: str) -> Optional[UniversalTransaction]:
        """Parse Bitcoin transaction"""
        try:
            txid = tx_data.get('txid')
            timestamp = tx_data.get('status', {}).get('block_time', int(time.time()))

            amount = 0.0
            direction = "unknown"
            counterparties = []

            # Analyze inputs/outputs
            for vout in tx_data.get('vout', []):
                if vout.get('scriptpubkey_address') == address:
                    amount += vout.get('value', 0) / 1e8
                    direction = "incoming"

            for vin in tx_data.get('vin', []):
                prev = vin.get('prevout', {})
                prev_addr = prev.get('scriptpubkey_address')
                if prev_addr == address:
                    direction = "outgoing"
                    amount = prev.get('value', 0) / 1e8
                elif prev_addr:
                    counterparties.append(prev_addr)

            # For outgoing, add output addresses
            if direction == "outgoing":
                for vout in tx_data.get('vout', []):
                    addr = vout.get('scriptpubkey_address')
                    if addr and addr != address:
                        counterparties.append(addr)

            fees = tx_data.get('fee', 0) / 1e8 if 'fee' in tx_data else None

            return UniversalTransaction(
                txid=txid,
                address=address,
                blockchain='bitcoin',
                timestamp=timestamp,
                amount=amount,
                token_symbol='BTC',
                confirmations=tx_data.get('status', {}).get('confirmed', False),
                block_height=tx_data.get('status', {}).get('block_height'),
                direction=direction,
                counterparties=list(set(counterparties)),
                fees=fees
            )

        except Exception as e:
            logger.error(f"Error parsing Bitcoin tx: {e}")
            return None

    def _parse_evm_transaction(self, tx_data: Dict, address: str, blockchain: str) -> Optional[UniversalTransaction]:
        """Parse EVM transaction"""
        try:
            txid = tx_data.get('hash')
            timestamp = int(tx_data.get('timeStamp', time.time()))

            from_addr = tx_data.get('from', '').lower()
            to_addr = tx_data.get('to', '').lower()
            address_lower = address.lower()

            # Determine direction
            if from_addr == address_lower:
                direction = "outgoing"
                counterparties = [to_addr] if to_addr else []
            else:
                direction = "incoming"
                counterparties = [from_addr] if from_addr else []

            # Amount in ETH/native token
            amount = int(tx_data.get('value', 0)) / 1e18

            # Fees
            gas_price = int(tx_data.get('gasPrice', 0))
            gas_used = int(tx_data.get('gasUsed', tx_data.get('gas', 0)))
            fees = (gas_price * gas_used) / 1e18 if gas_price and gas_used else None

            symbol = 'ETH' if blockchain == 'ethereum' else 'MON' if blockchain == 'monad' else 'ETH'

            return UniversalTransaction(
                txid=txid,
                address=address,
                blockchain=blockchain,
                timestamp=timestamp,
                amount=amount,
                token_symbol=symbol,
                confirmations=int(tx_data.get('confirmations', 0)),
                block_height=int(tx_data.get('blockNumber', 0)),
                direction=direction,
                counterparties=counterparties,
                fees=fees
            )

        except Exception as e:
            logger.error(f"Error parsing EVM tx: {e}")
            return None

    def _log_transaction(self, transaction: UniversalTransaction):
        """Log transaction to file"""
        try:
            log_entry = asdict(transaction)
            log_entry['timestamp_human'] = datetime.fromtimestamp(transaction.timestamp).isoformat()

            with open(TRANSACTION_LOG, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry) + '\n')

        except Exception as e:
            logger.error(f"Error logging transaction: {e}")

    def _analyze_transaction(self, tx: UniversalTransaction):
        """Analyze transaction for alerts"""
        # Large transaction alerts
        thresholds = {
            'BTC': ALERT_LARGE_BTC,
            'ETH': ALERT_LARGE_ETH,
            'MON': ALERT_LARGE_MON
        }

        threshold = thresholds.get(tx.token_symbol, 100.0)

        if tx.amount >= threshold:
            self.db.create_alert(
                tx.address,
                tx.blockchain,
                "large_transaction",
                f"Large {tx.direction} transaction: {tx.amount:.8f} {tx.token_symbol}",
                "WARNING",
                json.dumps(asdict(tx), default=str)
            )

# =============== UNIVERSAL MONITOR ===============

class UniversalWalletMonitor:
    """Main universal monitoring engine"""

    def __init__(self):
        self.db = UniversalDatabaseManager()
        self.api = MultiChainAPIClient()
        self.processor = UniversalTransactionProcessor(self.db, self.api)
        self.is_running = False
        self.thread_pool = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS)

        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def signal_handler(self, signum, frame):
        """Handle shutdown"""
        logger.info("Shutdown signal received...")
        self.stop()

    def add_address(self, address: str, blockchain: str, label: str = "") -> bool:
        """Add address to monitor"""
        # Validate based on blockchain
        if blockchain == 'bitcoin':
            if not self._validate_bitcoin_address(address):
                logger.error(f"Invalid Bitcoin address: {address}")
                return False
        elif blockchain in ['ethereum', 'monad', 'sepolia']:
            if not is_address(address):
                logger.error(f"Invalid {blockchain} address: {address}")
                return False

        success = self.db.add_address(address, blockchain, label)
        if success:
            # Initial scan
            self.thread_pool.submit(self.processor.process_address, address, blockchain)
        return success

    def _validate_bitcoin_address(self, address: str) -> bool:
        """Basic Bitcoin address validation"""
        if not address or len(address) < 26 or len(address) > 90:
            return False
        if address[:2] not in ['1', '3', 'bc', 'tb']:
            return False
        return True

    def monitor_loop(self):
        """Main monitoring loop"""
        self.is_running = True
        logger.info("=" * 70)
        logger.info("🚀 UNIVERSAL BLOCKCHAIN WALLET MONITOR STARTED")
        logger.info("=" * 70)
        logger.info(f"⏱️  Monitoring interval: {MONITOR_INTERVAL} seconds")
        logger.info(f"🔗 Supported chains: Bitcoin, Ethereum, Monad, Sepolia")
        logger.info("=" * 70)

        cycle_count = 0

        while self.is_running:
            try:
                addresses = self.db.get_monitored_addresses()

                if not addresses:
                    logger.info("No addresses to monitor - add some using interactive mode")
                    time.sleep(MONITOR_INTERVAL)
                    continue

                logger.info(f"\n🔍 Checking {len(addresses)} addresses across all chains...")

                # Process in parallel
                futures = []
                for address, blockchain in addresses:
                    future = self.thread_pool.submit(
                        self.processor.process_address,
                        address,
                        blockchain
                    )
                    futures.append(future)

                # Collect results
                total_new_tx = 0
                for future in as_completed(futures):
                    try:
                        new_tx = future.result(timeout=30)
                        total_new_tx += new_tx
                    except Exception as e:
                        logger.error(f"Processing error: {e}")

                if total_new_tx > 0:
                    logger.info(f"✨ Total new transactions found: {total_new_tx}")

                cycle_count += 1

                # Status report every 5 cycles
                if cycle_count % 5 == 0:
                    self._print_status_report()

                # Sleep until next cycle
                logger.info(f"💤 Sleeping for {MONITOR_INTERVAL} seconds...\n")
                time.sleep(MONITOR_INTERVAL)

            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
                time.sleep(60)

    def _print_status_report(self):
        """Print comprehensive status report"""
        logger.info("\n" + "=" * 70)
        logger.info("📊 STATUS REPORT")
        logger.info("=" * 70)

        addresses = self.db.get_monitored_addresses()

        # Group by blockchain
        by_chain = {}
        for addr, chain in addresses:
            if chain not in by_chain:
                by_chain[chain] = []
            by_chain[chain].append(addr)

        for chain, addrs in by_chain.items():
            logger.info(f"\n🔗 {chain.upper()}: {len(addrs)} addresses")

            for addr in addrs[:3]:  # Show first 3 per chain
                txs = self.db.get_address_transactions(addr, chain, limit=5)
                logger.info(f"  📍 {addr[:12]}... - {len(txs)} transactions")

                if txs:
                    latest = txs[0]
                    dt = datetime.fromtimestamp(latest['timestamp']).strftime('%Y-%m-%d %H:%M')
                    logger.info(f"     Latest: {latest['direction']} {latest['amount']:.8f} {latest['token_symbol']} at {dt}")

            if len(addrs) > 3:
                logger.info(f"  ... and {len(addrs) - 3} more addresses")

        # Total stats
        conn = self.db.get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM transactions')
        total_tx = cursor.fetchone()[0]
        cursor.execute('SELECT COUNT(*) FROM alerts WHERE acknowledged = 0')
        alerts = cursor.fetchone()[0]
        conn.close()

        logger.info(f"\n📈 Total Transactions: {total_tx}")
        logger.info(f"⚠️  Active Alerts: {alerts}")
        logger.info("=" * 70 + "\n")

    def stop(self):
        """Stop monitor"""
        self.is_running = False
        self.thread_pool.shutdown(wait=True)
        logger.info("✓ Monitor stopped gracefully")

    def interactive_mode(self):
        """Interactive CLI"""
        print("\n" + "=" * 70)
        print("🌐 UNIVERSAL BLOCKCHAIN WALLET MONITOR - Interactive Mode")
        print("=" * 70)
        print("\nMonitor Bitcoin, Ethereum, Monad, and all EVM-compatible chains!")
        print("\nCommands:")
        print("  add     - Add new address to monitor")
        print("  list    - List all monitored addresses")
        print("  check   - Check specific address now")
        print("  stats   - View address statistics")
        print("  report  - Generate comprehensive report")
        print("  start   - Start continuous monitoring")
        print("  quit    - Exit")
        print("=" * 70)

        while True:
            cmd = input("\n💻 Enter command: ").strip().lower()

            if cmd == 'quit':
                self.stop()
                break

            elif cmd == 'add':
                print("\nSupported blockchains: bitcoin, ethereum, monad, sepolia")
                blockchain = input("Enter blockchain: ").strip().lower()
                address = input("Enter address: ").strip()
                label = input("Enter label (optional): ").strip()

                if self.add_address(address, blockchain, label):
                    print(f"✓ Added {blockchain} address: {address}")
                else:
                    print(f"✗ Failed to add address")

            elif cmd == 'list':
                addresses = self.db.get_monitored_addresses()
                print(f"\n📋 Monitored Addresses ({len(addresses)}):\n")

                by_chain = {}
                for addr, chain in addresses:
                    if chain not in by_chain:
                        by_chain[chain] = []
                    by_chain[chain].append(addr)

                for chain, addrs in by_chain.items():
                    print(f"  🔗 {chain.upper()}:")
                    for addr in addrs:
                        print(f"     {addr}")
                    print()

            elif cmd == 'check':
                address = input("Enter address: ").strip()
                blockchain = input("Enter blockchain: ").strip().lower()
                print(f"\n🔍 Checking {blockchain} address {address}...")
                new_tx = self.processor.process_address(address, blockchain)
                print(f"✓ Found {new_tx} new transactions")

            elif cmd == 'stats':
                address = input("Enter address: ").strip()
                blockchain = input("Enter blockchain: ").strip().lower()

                txs = self.db.get_address_transactions(address, blockchain, limit=10)
                print(f"\n📊 Recent Transactions for {address}:\n")

                for tx in txs:
                    dt = datetime.fromtimestamp(tx['timestamp']).strftime('%Y-%m-%d %H:%M')
                    print(f"  [{dt}] {tx['direction']:10} {tx['amount']:12.8f} {tx['token_symbol']}")
                    print(f"     TX: {tx['txid'][:16]}...")

            elif cmd == 'report':
                filename = input("Enter filename (report.json): ").strip() or "report.json"
                self._generate_report(filename)
                print(f"✓ Report saved to {filename}")

            elif cmd == 'start':
                print("\n🚀 Starting continuous monitoring...")
                print("Press Ctrl+C to stop\n")
                self.monitor_loop()

            else:
                print("❌ Unknown command")

    def _generate_report(self, filename: str):
        """Generate comprehensive JSON report"""
        data = {
            'generated_at': datetime.now().isoformat(),
            'addresses': {}
        }

        addresses = self.db.get_monitored_addresses()

        for addr, chain in addresses:
            txs = self.db.get_address_transactions(addr, chain)

            data['addresses'][f"{chain}:{addr}"] = {
                'blockchain': chain,
                'address': addr,
                'transaction_count': len(txs),
                'recent_transactions': txs[:20]
            }

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)

# =============== MAIN ENTRY POINT ===============

def main():
    """Main entry point"""
    print("\n" + "=" * 70)
    print("🌐 UNIVERSAL BLOCKCHAIN WALLET MONITOR v3.0")
    print("=" * 70)
    print("\nFeatures:")
    print("  ✓ Multi-chain support (Bitcoin, Ethereum, Monad, Sepolia)")
    print("  ✓ Real-time transaction monitoring")
    print("  ✓ Balance tracking across all chains")
    print("  ✓ Transaction counterparty analysis")
    print("  ✓ Automated alerts for large/suspicious activity")
    print("  ✓ Comprehensive logging and reporting")
    print("  ✓ Thread-safe parallel processing")
    print("=" * 70)

    monitor = UniversalWalletMonitor()

    try:
        monitor.interactive_mode()
    except KeyboardInterrupt:
        monitor.stop()
        print("\n\n✓ Monitor stopped")

if __name__ == "__main__":
    main()
