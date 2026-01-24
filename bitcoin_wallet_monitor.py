#!/usr/bin/env python3
"""
Bitcoin Wallet Activity Monitor v2.0
Monitors BTC addresses and logs transactions with counterparties
Optimized for reliability and performance
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
from queue import Queue
import threading
import signal

# =============== CONFIGURATION ===============
# Public blockchain APIs (no authentication required)
API_PROVIDERS = [
    "https://mempool.space/api",      # Primary - reliable, fast
    "https://blockchain.info",        # Secondary
    "https://blockstream.info/api"    # Backup
]

# Database and file paths
DB_FILE = "wallet_monitor.db"
LOG_DIR = "logs/"
TRANSACTION_LOG = f"{LOG_DIR}transactions.jsonl"
ACTIVITY_LOG = f"{LOG_DIR}activity.log"

# Monitoring settings
MONITOR_INTERVAL = 30        # Seconds between checks
MAX_RETRIES = 3              # API retry attempts
REQUEST_TIMEOUT = 10         # Seconds before timeout
MAX_CONCURRENT_REQUESTS = 5  # Simultaneous API calls

# Alert thresholds (BTC)
ALERT_LARGE_TX = 10.0        # Alert for transactions > 10 BTC
ALERT_RAPID_TX = 5           # Alert for >5 tx/hour

# =============== DATA STRUCTURES ===============
@dataclass
class Transaction:
    txid: str
    address: str
    timestamp: int
    amount: float
    confirmations: int
    block_height: Optional[int]
    direction: str
    counterparties: List[str]
    fees: Optional[float] = None

@dataclass
class AddressStats:
    address: str
    balance: float = 0.0
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
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(ACTIVITY_LOG, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# =============== DATABASE MANAGER ===============
class DatabaseManager:
    """Thread-safe database operations"""

    def __init__(self, db_path: str = DB_FILE):
        self.db_path = db_path
        self._lock = threading.Lock()
        self.init_database()

    def get_connection(self):
        """Get thread-safe database connection"""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def init_database(self):
        """Initialize database schema with performance optimizations"""
        with self._lock:
            conn = self.get_connection()
            cursor = conn.cursor()

            # Enable WAL mode for better concurrency
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute("PRAGMA cache_size=-2000")  # 2MB cache

            # Create tables with indexes
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS monitored_addresses (
                    address TEXT PRIMARY KEY,
                    label TEXT,
                    date_added TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1,
                    last_checked INTEGER DEFAULT 0
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS transactions (
                    txid TEXT PRIMARY KEY,
                    address TEXT,
                    timestamp INTEGER,
                    amount REAL,
                    confirmations INTEGER,
                    block_height INTEGER,
                    direction TEXT,
                    fees REAL,
                    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_address_time (address, timestamp DESC),
                    INDEX idx_timestamp (timestamp),
                    FOREIGN KEY (address) REFERENCES monitored_addresses(address) ON DELETE CASCADE
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS transaction_links (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    txid TEXT,
                    source_address TEXT,
                    target_address TEXT,
                    amount REAL,
                    link_type TEXT,  -- 'input' or 'output'
                    timestamp INTEGER,
                    INDEX idx_source (source_address),
                    INDEX idx_target (target_address),
                    INDEX idx_txid (txid),
                    UNIQUE(txid, source_address, target_address, link_type)
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS address_stats (
                    address TEXT PRIMARY KEY,
                    balance REAL DEFAULT 0,
                    total_received REAL DEFAULT 0,
                    total_sent REAL DEFAULT 0,
                    tx_count INTEGER DEFAULT 0,
                    first_seen INTEGER,
                    last_active INTEGER,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    address TEXT,
                    alert_type TEXT,
                    severity TEXT,
                    message TEXT,
                    details TEXT,
                    acknowledged BOOLEAN DEFAULT 0
                )
            ''')

            conn.commit()
            conn.close()

    def add_address(self, address: str, label: str = "") -> bool:
        """Add address to monitoring list"""
        with self._lock:
            conn = self.get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR IGNORE INTO monitored_addresses (address, label)
                    VALUES (?, ?)
                ''', (address, label))

                # Initialize stats
                cursor.execute('''
                    INSERT OR IGNORE INTO address_stats (address)
                    VALUES (?)
                ''', (address,))

                conn.commit()
                logger.info(f"Added address to monitor: {address}")
                return True

            except sqlite3.Error as e:
                logger.error(f"Database error adding address: {e}")
                return False
            finally:
                conn.close()

    def save_transaction(self, tx: Transaction) -> bool:
        """Save transaction and its relationships"""
        with self._lock:
            conn = self.get_connection()
            try:
                cursor = conn.cursor()

                # Save main transaction
                cursor.execute('''
                    INSERT OR REPLACE INTO transactions
                    (txid, address, timestamp, amount, confirmations, block_height, direction, fees)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (tx.txid, tx.address, tx.timestamp, tx.amount,
                      tx.confirmations, tx.block_height, tx.direction, tx.fees))

                # Save transaction links (counterparties)
                for counterparty in tx.counterparties:
                    if counterparty:  # Skip empty addresses
                        link_type = 'output' if tx.direction == 'outgoing' else 'input'
                        source = tx.address if tx.direction == 'outgoing' else counterparty
                        target = counterparty if tx.direction == 'outgoing' else tx.address

                        cursor.execute('''
                            INSERT OR IGNORE INTO transaction_links
                            (txid, source_address, target_address, amount, link_type, timestamp)
                            VALUES (?, ?, ?, ?, ?, ?)
                        ''', (tx.txid, source, target, tx.amount, link_type, tx.timestamp))

                conn.commit()
                return True

            except sqlite3.Error as e:
                logger.error(f"Database error saving transaction: {e}")
                return False
            finally:
                conn.close()

    def get_monitored_addresses(self) -> List[str]:
        """Get all active monitored addresses"""
        with self._lock:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute('''
                SELECT address FROM monitored_addresses
                WHERE is_active = 1
                ORDER BY date_added
            ''')
            addresses = [row[0] for row in cursor.fetchall()]
            conn.close()
            return addresses

    def get_address_transactions(self, address: str, limit: int = 100) -> List[Dict]:
        """Get transaction history for address"""
        with self._lock:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM transactions
                WHERE address = ?
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (address, limit))

            transactions = [dict(row) for row in cursor.fetchall()]
            conn.close()
            return transactions

    def get_connected_addresses(self, address: str, depth: int = 1) -> List[Dict]:
        """Find addresses connected through transactions"""
        with self._lock:
            conn = self.get_connection()
            cursor = conn.cursor()

            # Direct connections
            cursor.execute('''
                SELECT DISTINCT target_address as connected_address,
                       COUNT(*) as tx_count,
                       SUM(amount) as total_volume
                FROM transaction_links
                WHERE source_address = ?
                GROUP BY target_address
                ORDER BY tx_count DESC
                LIMIT 50
            ''', (address,))

            direct_connections = [dict(row) for row in cursor.fetchall()]
            conn.close()
            return direct_connections

    def create_alert(self, address: str, alert_type: str, message: str,
                    severity: str = "INFO", details: str = ""):
        """Create an alert in database"""
        with self._lock:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO alerts (address, alert_type, severity, message, details)
                VALUES (?, ?, ?, ?, ?)
            ''', (address, alert_type, severity, message, details))
            conn.commit()
            conn.close()
            logger.warning(f"ALERT [{severity}] {address}: {message}")

# =============== BLOCKCHAIN API CLIENT ===============
class BlockchainAPIClient:
    """Smart API client with failover and rate limiting"""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'WalletMonitor/2.0',
            'Accept': 'application/json'
        })
        self.api_index = 0  # Current API provider index
        self.request_count = 0
        self.last_request_time = 0
        self.rate_limit_delay = 0.2  # 200ms between requests

    def _get_api_url(self) -> str:
        """Get current API provider URL with round-robin"""
        return API_PROVIDERS[self.api_index % len(API_PROVIDERS)]

    def _rotate_api(self):
        """Switch to next API provider"""
        self.api_index = (self.api_index + 1) % len(API_PROVIDERS)
        logger.info(f"Rotated to API provider: {API_PROVIDERS[self.api_index]}")

    def _rate_limit(self):
        """Enforce rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last)

        self.last_request_time = time.time()

    def make_request(self, endpoint: str, params: Dict = None, retry: int = 0) -> Optional[Dict]:
        """Make API request with retry logic"""
        self._rate_limit()

        url = f"{self._get_api_url()}/{endpoint.lstrip('/')}"

        try:
            response = self.session.get(url, params=params, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            self.request_count += 1

            # Reset API index on success
            self.api_index = 0
            return response.json()

        except (requests.RequestException, json.JSONDecodeError) as e:
            logger.warning(f"API request failed ({retry+1}/{MAX_RETRIES}): {e}")

            if retry < MAX_RETRIES - 1:
                self._rotate_api()
                time.sleep(1 * (retry + 1))  # Exponential backoff
                return self.make_request(endpoint, params, retry + 1)
            else:
                logger.error(f"All API attempts failed for: {endpoint}")
                return None

    def get_address_transactions(self, address: str) -> Optional[List[Dict]]:
        """Get transactions for address using mempool.space format"""
        data = self.make_request(f"address/{address}/txs")

        if data and isinstance(data, list):
            return data
        return None

    def get_address_info(self, address: str) -> Optional[Dict]:
        """Get address balance and stats"""
        # Try multiple endpoints
        endpoints = [
            f"address/{address}",
            f"q/addressbalance/{address}",
            f"rawaddr/{address}?limit=0"
        ]

        for endpoint in endpoints:
            data = self.make_request(endpoint)
            if data:
                return data
        return None

    def get_transaction_details(self, txid: str) -> Optional[Dict]:
        """Get detailed transaction info"""
        data = self.make_request(f"tx/{txid}")
        return data

# =============== TRANSACTION PROCESSOR ===============
class TransactionProcessor:
    """Process and analyze blockchain transactions"""

    def __init__(self, db: DatabaseManager, api: BlockchainAPIClient):
        self.db = db
        self.api = api
        self.processed_txids: Set[str] = set()
        self.load_processed_txids()

    def load_processed_txids(self):
        """Load already processed transactions"""
        conn = self.db.get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT txid FROM transactions')
        self.processed_txids = {row[0] for row in cursor.fetchall()}
        conn.close()

    def process_address(self, address: str) -> int:
        """Process all transactions for an address, returns new transaction count"""
        try:
            tx_data = self.api.get_address_transactions(address)
            if not tx_data:
                return 0

            new_tx_count = 0

            for tx in tx_data:
                if self._should_process_transaction(tx, address):
                    transaction = self._parse_transaction(tx, address)
                    if transaction:
                        if self.db.save_transaction(transaction):
                            self.processed_txids.add(transaction.txid)
                            new_tx_count += 1

                            # Log and analyze
                            self._log_transaction(transaction)
                            self._analyze_transaction(transaction)

                            # Update address stats
                            self._update_address_stats(address, transaction)

            # Update last checked timestamp
            self._update_last_checked(address)

            if new_tx_count > 0:
                logger.info(f"Found {new_tx_count} new transactions for {address}")

            return new_tx_count

        except Exception as e:
            logger.error(f"Error processing address {address}: {e}")
            return 0

    def _should_process_transaction(self, tx_data: Dict, address: str) -> bool:
        """Check if transaction should be processed"""
        txid = tx_data.get('txid') or tx_data.get('hash')
        if not txid or txid in self.processed_txids:
            return False
        return True

    def _parse_transaction(self, tx_data: Dict, address: str) -> Optional[Transaction]:
        """Parse raw transaction data"""
        try:
            txid = tx_data.get('txid') or tx_data.get('hash')
            timestamp = tx_data.get('status', {}).get('block_time') or tx_data.get('time') or int(time.time())

            # Calculate amount and direction
            amount, direction, counterparties = self._analyze_inputs_outputs(tx_data, address)

            # Get fees if available
            fees = None
            if 'fee' in tx_data:
                fees = tx_data['fee'] / 1e8  # Convert satoshis to BTC

            return Transaction(
                txid=txid,
                address=address,
                timestamp=timestamp,
                amount=amount,
                confirmations=tx_data.get('status', {}).get('confirmed', False) or tx_data.get('confirmations', 0),
                block_height=tx_data.get('status', {}).get('block_height'),
                direction=direction,
                counterparties=counterparties,
                fees=fees
            )

        except Exception as e:
            logger.error(f"Error parsing transaction: {e}")
            return None

    def _analyze_inputs_outputs(self, tx_data: Dict, address: str) -> Tuple[float, str, List[str]]:
        """Analyze transaction to determine amount, direction, and counterparties"""
        amount = 0.0
        direction = "unknown"
        counterparties = []

        # Handle mempool.space API format
        if 'vin' in tx_data and 'vout' in tx_data:
            # Check outputs (receiving)
            for vout in tx_data['vout']:
                if 'scriptpubkey_address' in vout and vout['scriptpubkey_address'] == address:
                    amount += vout['value'] / 1e8  # Convert satoshis to BTC
                    direction = "incoming"

            # Check inputs (sending) and collect counterparties
            for vin in tx_data['vin']:
                if 'prevout' in vin and 'scriptpubkey_address' in vin['prevout']:
                    prev_addr = vin['prevout']['scriptpubkey_address']
                    if prev_addr == address:
                        direction = "outgoing"
                    elif prev_addr:  # Don't add empty/null addresses
                        counterparties.append(prev_addr)

            # For outgoing, also collect output addresses
            if direction == "outgoing":
                for vout in tx_data['vout']:
                    if 'scriptpubkey_address' in vout and vout['scriptpubkey_address'] != address:
                        counterparties.append(vout['scriptpubkey_address'])

        # Handle blockchain.info format
        elif 'inputs' in tx_data and 'out' in tx_data:
            # Similar logic for different API format
            pass

        return amount, direction, list(set(counterparties))  # Remove duplicates

    def _log_transaction(self, transaction: Transaction):
        """Log transaction to file"""
        try:
            log_entry = {
                'timestamp': datetime.fromtimestamp(transaction.timestamp).isoformat(),
                'address': transaction.address,
                'txid': transaction.txid,
                'amount': transaction.amount,
                'direction': transaction.direction,
                'counterparties': transaction.counterparties,
                'confirmations': transaction.confirmations,
                'fees': transaction.fees
            }

            with open(TRANSACTION_LOG, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry) + '\n')

        except Exception as e:
            logger.error(f"Error logging transaction: {e}")

    def _analyze_transaction(self, transaction: Transaction):
        """Analyze transaction for alerts"""
        # Large transaction alert
        if transaction.amount >= ALERT_LARGE_TX:
            self.db.create_alert(
                transaction.address,
                "large_transaction",
                f"Large transaction detected: {transaction.amount:.8f} BTC",
                "WARNING",
                json.dumps(asdict(transaction))
            )

        # Check for rapid transactions
        recent_txs = self.db.get_address_transactions(transaction.address, limit=10)
        if len(recent_txs) >= ALERT_RAPID_TX:
            last_hour = time.time() - 3600
            hourly_txs = [tx for tx in recent_txs if tx['timestamp'] > last_hour]

            if len(hourly_txs) >= ALERT_RAPID_TX:
                self.db.create_alert(
                    transaction.address,
                    "rapid_activity",
                    f"Rapid transaction activity: {len(hourly_txs)} tx/hour",
                    "INFO"
                )

    def _update_address_stats(self, address: str, transaction: Transaction):
        """Update address statistics"""
        # In production, this would update cumulative stats
        pass

    def _update_last_checked(self, address: str):
        """Update last checked timestamp"""
        with self.db._lock:
            conn = self.db.get_connection()
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE monitored_addresses
                SET last_checked = ?
                WHERE address = ?
            ''', (int(time.time()), address))
            conn.commit()
            conn.close()

# =============== MONITORING ENGINE ===============
class WalletMonitor:
    """Main monitoring engine"""

    def __init__(self):
        self.db = DatabaseManager()
        self.api = BlockchainAPIClient()
        self.processor = TransactionProcessor(self.db, self.api)
        self.is_running = False
        self.thread_pool = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS)

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info("Shutdown signal received, stopping monitor...")
        self.stop()

    def add_address(self, address: str, label: str = "") -> bool:
        """Add new address to monitor"""
        # Basic validation
        if not self._validate_address(address):
            logger.error(f"Invalid Bitcoin address: {address}")
            return False

        success = self.db.add_address(address, label)
        if success:
            # Initial scan
            self.thread_pool.submit(self.processor.process_address, address)
        return success

    def _validate_address(self, address: str) -> bool:
        """Basic Bitcoin address validation"""
        if not address or len(address) < 26 or len(address) > 35:
            return False

        # Check for common Bitcoin address formats
        if address[0] not in '13bc1':
            return False

        # In production, use proper validation library
        return True

    def monitor_loop(self):
        """Main monitoring loop"""
        self.is_running = True
        logger.info("Starting Bitcoin Wallet Monitor...")
        logger.info(f"Monitoring interval: {MONITOR_INTERVAL} seconds")

        cycle_count = 0

        while self.is_running:
            try:
                addresses = self.db.get_monitored_addresses()

                if not addresses:
                    logger.info("No addresses to monitor")
                    time.sleep(MONITOR_INTERVAL)
                    continue

                logger.info(f"Monitoring {len(addresses)} addresses")

                # Process addresses in parallel
                futures = []
                for address in addresses:
                    future = self.thread_pool.submit(self.processor.process_address, address)
                    futures.append(future)

                # Wait for completion and log results
                for future in as_completed(futures):
                    try:
                        new_tx = future.result(timeout=30)
                        if new_tx > 0:
                            logger.debug(f"Found {new_tx} new transactions")
                    except Exception as e:
                        logger.error(f"Error processing address: {e}")

                cycle_count += 1

                # Periodic report
                if cycle_count % 10 == 0:  # Every 10 cycles
                    self._print_status_report(addresses)

                # Sleep until next cycle
                time.sleep(MONITOR_INTERVAL)

            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
                time.sleep(60)  # Wait before retry

    def _print_status_report(self, addresses: List[str]):
        """Print periodic status report"""
        logger.info("=" * 60)
        logger.info("STATUS REPORT")
        logger.info(f"Monitored addresses: {len(addresses)}")

        total_tx = 0
        conn = self.db.get_connection()
        cursor = conn.cursor()

        for address in addresses[:5]:  # Show first 5 addresses
            cursor.execute('SELECT COUNT(*) FROM transactions WHERE address = ?', (address,))
            tx_count = cursor.fetchone()[0]
            total_tx += tx_count

            cursor.execute('SELECT MAX(timestamp) FROM transactions WHERE address = ?', (address,))
            last_tx = cursor.fetchone()[0]

            last_seen = datetime.fromtimestamp(last_tx).strftime('%Y-%m-%d %H:%M') if last_tx else "Never"
            logger.info(f"  {address[:10]}...: {tx_count} tx, last: {last_seen}")

        if len(addresses) > 5:
            logger.info(f"  ... and {len(addresses) - 5} more addresses")

        cursor.execute('SELECT COUNT(*) FROM alerts WHERE acknowledged = 0')
        active_alerts = cursor.fetchone()[0]

        logger.info(f"Total transactions: {total_tx}")
        logger.info(f"Active alerts: {active_alerts}")
        logger.info("=" * 60)
        conn.close()

    def stop(self):
        """Stop the monitor gracefully"""
        self.is_running = False
        self.thread_pool.shutdown(wait=True)
        logger.info("Monitor stopped")

    def interactive_mode(self):
        """Interactive command line interface"""
        print("\n" + "=" * 60)
        print("BITCOIN WALLET MONITOR - INTERACTIVE MODE")
        print("=" * 60)

        while True:
            print("\nCommands: add, list, stats, check, export, quit")
            cmd = input("\nEnter command: ").strip().lower()

            if cmd == 'quit':
                self.stop()
                break

            elif cmd == 'add':
                address = input("Enter Bitcoin address: ").strip()
                label = input("Enter label (optional): ").strip()
                if self.add_address(address, label):
                    print(f"✓ Added {address}")
                else:
                    print(f"✗ Failed to add address")

            elif cmd == 'list':
                addresses = self.db.get_monitored_addresses()
                print(f"\nMonitored addresses ({len(addresses)}):")
                for addr in addresses[:20]:  # Show first 20
                    print(f"  {addr}")
                if len(addresses) > 20:
                    print(f"  ... and {len(addresses) - 20} more")

            elif cmd == 'stats':
                address = input("Enter address to check: ").strip()
                txs = self.db.get_address_transactions(address, limit=5)
                print(f"\nRecent transactions for {address}:")
                for tx in txs:
                    date = datetime.fromtimestamp(tx['timestamp']).strftime('%Y-%m-%d %H:%M')
                    print(f"  [{date}] {tx['amount']:.8f} BTC ({tx['direction']})")

            elif cmd == 'check':
                address = input("Enter address to check now: ").strip()
                print(f"Checking {address}...")
                new_tx = self.processor.process_address(address)
                print(f"Found {new_tx} new transactions")

            elif cmd == 'export':
                filename = input("Enter filename for export: ").strip() or "wallet_report.json"
                self.export_data(filename)
                print(f"Data exported to {filename}")

            else:
                print("Unknown command")

    def export_data(self, filename: str):
        """Export monitoring data to JSON file"""
        try:
            data = {
                'export_date': datetime.now().isoformat(),
                'addresses': [],
                'statistics': {}
            }

            addresses = self.db.get_monitored_addresses()

            for address in addresses:
                txs = self.db.get_address_transactions(address)
                connections = self.db.get_connected_addresses(address)

                addr_data = {
                    'address': address,
                    'transaction_count': len(txs),
                    'recent_transactions': txs[:10],
                    'connected_addresses': connections[:20]
                }
                data['addresses'].append(addr_data)

            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)

            logger.info(f"Data exported to {filename}")

        except Exception as e:
            logger.error(f"Export failed: {e}")

# =============== MAIN ENTRY POINT ===============
def main():
    """Main entry point"""
    print("\n" + "=" * 60)
    print("BITCOIN WALLET MONITOR v2.0")
    print("=" * 60)
    print("Features:")
    print("  • Monitor multiple Bitcoin addresses")
    print("  • Track all transactions and counterparties")
    print("  • Automatic logging to database and files")
    print("  • Alerts for large/rapid transactions")
    print("  • Network analysis (connected addresses)")
    print("  • Multi-API failover support")
    print("=" * 60)

    monitor = WalletMonitor()

    # Add some example addresses if none exist
    addresses = monitor.db.get_monitored_addresses()
    if not addresses:
        print("\nNo addresses in database. Add some to monitor.")
        print("Example Bitcoin addresses (testnet for testing):")
        print("  tb1qexample... (Testnet)")
        print("  bc1qexample... (Mainnet)")
        print("\nYou can add addresses in interactive mode.")

    # Start in interactive mode
    try:
        monitor.interactive_mode()
    except KeyboardInterrupt:
        monitor.stop()
        print("\n\nMonitor stopped by user")

if __name__ == "__main__":
    main()
