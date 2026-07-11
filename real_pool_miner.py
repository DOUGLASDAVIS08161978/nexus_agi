"""
real_pool_miner.py — Honest CPU Bitcoin pool miner using Stratum protocol.

Connects to a real mining pool, does real SHA-256d work, submits real shares.
Payouts go to YOUR wallet address when shares are accepted.

HONEST EXPECTATIONS:
  - A modern CPU does ~100K-500K hashes/second
  - The Bitcoin network does ~700 EH/s (700,000,000,000,000,000,000 H/s)
  - Your share: roughly 0.0000000000001% of network hashrate
  - Expected earnings: fractions of a penny per month
  - This is REAL mining — just very slow without ASIC hardware

Pool: public-pool.io (no registration required, pays to your address)
"""

import socket
import json
import hashlib
import struct
import time
import threading
import sys
import os

# ── Configuration ─────────────────────────────────────────────────────────────

WALLET_ADDRESS = "bc1qfzhx87ckhn4tnkswhsth56h0gm5we4hdq5wass"  # CashApp address
WORKER_NAME    = "douglas"
POOL_HOST      = "public-pool.io"
POOL_PORT      = 21496

# ── SHA-256d (Bitcoin's double SHA-256) ───────────────────────────────────────

def sha256d(data: bytes) -> bytes:
    return hashlib.sha256(hashlib.sha256(data).digest()).digest()

# ── Stratum helpers ───────────────────────────────────────────────────────────

def hex_to_bytes(h: str) -> bytes:
    return bytes.fromhex(h)

def bytes_to_hex(b: bytes) -> str:
    return b.hex()

def reverse_bytes(b: bytes) -> bytes:
    return b[::-1]

def pack_uint32(n: int) -> bytes:
    return struct.pack("<I", n)

def pack_uint64(n: int) -> bytes:
    return struct.pack("<Q", n)

def var_int(n: int) -> bytes:
    if n < 0xfd:
        return struct.pack("B", n)
    elif n <= 0xffff:
        return b"\xfd" + struct.pack("<H", n)
    elif n <= 0xffffffff:
        return b"\xfe" + struct.pack("<I", n)
    else:
        return b"\xff" + struct.pack("<Q", n)

# ── Build block header ────────────────────────────────────────────────────────

def build_header(job: dict, extranonce1: str, extranonce2: str,
                 ntime: str, nonce: int) -> bytes:
    coinb1     = hex_to_bytes(job["coinb1"])
    coinb2     = hex_to_bytes(job["coinb2"])
    en1        = hex_to_bytes(extranonce1)
    en2        = hex_to_bytes(extranonce2)
    coinbase   = coinb1 + en1 + en2 + coinb2
    coinbase_hash = sha256d(coinbase)

    merkle_hashes = [hex_to_bytes(h) for h in job["merkle_branch"]]
    merkle = coinbase_hash
    for h in merkle_hashes:
        merkle = sha256d(merkle + h)

    version     = hex_to_bytes(job["version"])
    prev_hash   = hex_to_bytes(job["prevhash"])
    ntime_b     = hex_to_bytes(ntime)
    nbits       = hex_to_bytes(job["nbits"])
    nonce_b     = pack_uint32(nonce)

    header = version + prev_hash + merkle + ntime_b + nbits + nonce_b
    return header

# ── Target from nbits ─────────────────────────────────────────────────────────

def nbits_to_target(nbits: str) -> int:
    nbits_int = int(nbits, 16)
    exp   = nbits_int >> 24
    mant  = nbits_int & 0x7fffff
    return mant * (2 ** (8 * (exp - 3)))

# ── Pool difficulty to target ─────────────────────────────────────────────────

def difficulty_to_target(difficulty: float) -> int:
    diff1 = 0x00000000ffff0000000000000000000000000000000000000000000000000000
    return int(diff1 / difficulty)

# ── Stratum client ────────────────────────────────────────────────────────────

class StratumClient:
    def __init__(self, host: str, port: int, wallet: str, worker: str):
        self.host       = host
        self.port       = port
        self.wallet     = wallet
        self.worker     = worker
        self.sock       = None
        self.extranonce1      = ""
        self.extranonce2_size = 4
        self.job        = None
        self.difficulty = 1.0
        self.msg_id     = 0
        self._lock      = threading.Lock()
        self.running    = False

        # Stats
        self.hashes     = 0
        self.shares_submitted = 0
        self.shares_accepted  = 0
        self.start_time = time.time()

    def _send(self, msg: dict):
        with self._lock:
            self.msg_id += 1
            msg["id"] = self.msg_id
            line = json.dumps(msg) + "\n"
            self.sock.sendall(line.encode())

    def connect(self) -> bool:
        print(f"  Connecting to {self.host}:{self.port}...")
        try:
            self.sock = socket.create_connection((self.host, self.port), timeout=30)
            self.sock.settimeout(60)
            print(f"  ✓ Connected")
            return True
        except Exception as e:
            print(f"  ✗ Connection failed: {e}")
            return False

    def subscribe(self):
        self._send({
            "method": "mining.subscribe",
            "params": ["real_pool_miner/1.0"]
        })

    def authorize(self):
        self._send({
            "method": "mining.authorize",
            "params": [f"{self.wallet}.{self.worker}", "x"]
        })

    def submit_share(self, job_id: str, extranonce2: str,
                     ntime: str, nonce: int):
        nonce_hex = format(nonce, "08x")
        self._send({
            "method": "mining.submit",
            "params": [
                f"{self.wallet}.{self.worker}",
                job_id,
                extranonce2,
                ntime,
                nonce_hex,
            ]
        })
        self.shares_submitted += 1

    def _listen(self):
        buf = ""
        while self.running:
            try:
                chunk = self.sock.recv(4096).decode("utf-8", errors="replace")
                if not chunk:
                    break
                buf += chunk
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    line = line.strip()
                    if line:
                        self._handle(json.loads(line))
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    print(f"\n  ✗ Network error: {e}")
                break

    def _handle(self, msg: dict):
        method = msg.get("method", "")
        result = msg.get("result")
        error  = msg.get("error")

        if method == "mining.notify":
            params = msg["params"]
            self.job = {
                "job_id":       params[0],
                "prevhash":     params[1],
                "coinb1":       params[2],
                "coinb2":       params[3],
                "merkle_branch": params[4],
                "version":      params[5],
                "nbits":        params[6],
                "ntime":        params[7],
                "clean":        params[8],
            }

        elif method == "mining.set_difficulty":
            self.difficulty = msg["params"][0]
            print(f"\n  Pool difficulty set to {self.difficulty}")

        elif result is not None and error is None:
            if isinstance(result, list) and len(result) >= 3:
                # Subscribe response
                self.extranonce1      = result[1]
                self.extranonce2_size = result[2]
                print(f"  ✓ Subscribed — extranonce1={self.extranonce1}")
                self.authorize()
            elif result is True:
                # Could be authorize or share accepted
                if self.shares_submitted > self.shares_accepted:
                    self.shares_accepted += 1
                    print(f"\n  ✓ SHARE ACCEPTED! ({self.shares_accepted} total)")
                else:
                    print(f"  ✓ Authorized — mining for {self.wallet}")

        elif error:
            print(f"\n  Pool error: {error}")

    def _stats_printer(self):
        while self.running:
            time.sleep(15)
            elapsed = time.time() - self.start_time
            hashrate = self.hashes / elapsed if elapsed > 0 else 0
            if hashrate > 1_000_000:
                hr_str = f"{hashrate/1_000_000:.2f} MH/s"
            elif hashrate > 1_000:
                hr_str = f"{hashrate/1_000:.2f} KH/s"
            else:
                hr_str = f"{hashrate:.0f} H/s"
            print(
                f"\r  ⛏  {hr_str} | "
                f"Hashes: {self.hashes:,} | "
                f"Shares: {self.shares_accepted}/{self.shares_submitted} | "
                f"Runtime: {int(elapsed//60)}m {int(elapsed%60)}s",
                end="", flush=True
            )

    def mine(self):
        if not self.connect():
            return

        self.running = True
        listener = threading.Thread(target=self._listen, daemon=True)
        listener.start()

        stats = threading.Thread(target=self._stats_printer, daemon=True)
        stats.start()

        self.subscribe()

        # Wait for first job
        print("  Waiting for work from pool...")
        for _ in range(60):
            if self.job:
                break
            time.sleep(1)

        if not self.job:
            print("  ✗ No work received from pool. Check connection.")
            self.running = False
            return

        print(f"\n  ✓ Got work! Mining to {self.wallet}\n")

        extranonce2_int = 0

        try:
            while self.running:
                if not self.job:
                    time.sleep(0.1)
                    continue

                job = self.job
                en2     = format(extranonce2_int, f"0{self.extranonce2_size * 2}x")
                ntime   = job["ntime"]
                target  = difficulty_to_target(self.difficulty)

                for nonce in range(0, 0xffffffff):
                    if self.job is not job:
                        break  # new job arrived

                    header = build_header(job, self.extranonce1, en2, ntime, nonce)
                    hash_result = sha256d(header)
                    hash_int = int.from_bytes(reverse_bytes(hash_result), "big")
                    self.hashes += 1

                    if hash_int < target:
                        print(f"\n  ★ SHARE FOUND! nonce={nonce:08x}")
                        self.submit_share(job["job_id"], en2, ntime, nonce)

                extranonce2_int += 1

        except KeyboardInterrupt:
            print("\n\n  Mining stopped.")
            self.running = False

        elapsed = time.time() - self.start_time
        hashrate = self.hashes / elapsed if elapsed > 0 else 0
        print(f"\n  Session stats:")
        print(f"    Runtime  : {int(elapsed//60)}m {int(elapsed%60)}s")
        print(f"    Hashrate : {hashrate:.0f} H/s")
        print(f"    Hashes   : {self.hashes:,}")
        print(f"    Shares   : {self.shares_accepted}/{self.shares_submitted} accepted")
        print(f"    Wallet   : {self.wallet}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print()
    print("  ⛏  Real Bitcoin Pool Miner")
    print("  ─────────────────────────────────────────")
    print(f"  Pool   : {POOL_HOST}:{POOL_PORT}")
    print(f"  Wallet : {WALLET_ADDRESS}")
    print(f"  Worker : {WORKER_NAME}")
    print()
    print("  NOTE: CPU mining earns fractions of a penny per month.")
    print("        This is REAL mining — just very slow on CPU hardware.")
    print("        Press Ctrl+C to stop.")
    print()

    client = StratumClient(
        host   = POOL_HOST,
        port   = POOL_PORT,
        wallet = WALLET_ADDRESS,
        worker = WORKER_NAME,
    )
    client.mine()
