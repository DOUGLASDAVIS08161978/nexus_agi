import socket
import json
import hashlib
import threading
import time
import os
import sys
import logging
import queue
import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Tuple
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("LuminaMiner")

@dataclass
class MiningStatus:
    is_running: bool = False
    connected: bool = False
    authorized: bool = False
    current_job_id: Optional[str] = None
    pool_difficulty: float = 1.0
    network_difficulty: float = 0.0
    hash_rate: float = 0.0
    accepted_shares: int = 0
    rejected_shares: int = 0
    total_hashes: int = 0
    uptime_seconds: float = 0.0
    active_workers: int = 0
    target_workers: int = 4
    system_load: float = 0.0
    last_block_find: Optional[datetime] = None

@dataclass
class Earnings:
    btc_earned: float = 0.0
    shares_contributed: int = 0
    estimated_btc_per_share: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)

class LuminaBitcoinMiner:
    def __init__(
        self,
        pool_host: str = "public-pool.io",
        pool_port: int = 3333,
        wallet_address: str = "",
        worker_name: str = "lumina-arm-01",
        password: str = "x",
        max_workers: int = 8,
        throttle_load_high: float = 0.8,
        throttle_load_low: float = 0.4,
        throttle_interval: float = 5.0
    ):
        self.pool_host = pool_host
        self.pool_port = pool_port
        self.wallet_address = wallet_address
        self.worker_name = worker_name
        self.password = password
        self.max_workers = max_workers
        self.throttle_load_high = throttle_load_high
        self.throttle_load_low = throttle_load_low
        self.throttle_interval = throttle_interval

        self._sock: Optional[socket.socket] = None
        self._running = False
        self._status = MiningStatus()
        self._earnings = Earnings()
        self._job_queue: queue.Queue = queue.Queue()
        self._workers: List[threading.Thread] = []
        self._lock = threading.Lock()
        self._start_time: float = 0.0
        self._hash_counter = 0
        self._hash_rate_window = 10.0
        self._hash_rate_samples: List[Tuple[float, int]] = []
        self._extranonce1: str = ""
        self._extranonce2_size: int = 8
        self._current_job: Dict[str, Any] = {}
        self._callbacks: Dict[str, List[Callable]] = {
            "share_accepted": [],
            "share_rejected": [],
            "block_found": [],
            "status_update": [],
            "error": []
        }
        self._compute_allocation_ratio = 1.0

    def register_callback(self, event: str, callback: Callable):
        if event in self._callbacks:
            self._callbacks[event].append(callback)

    def _emit(self, event: str, data: Any):
        for cb in self._callbacks.get