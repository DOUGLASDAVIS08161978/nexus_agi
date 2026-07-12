"""
real_pool_miner.py — Multi-threaded real Bitcoin pool miner (Stratum protocol)

One mining thread per CPU core. hashlib releases Python's GIL so threads run
truly in parallel on all cores. Auto-reconnects on pool drop. Rolls ntime for
maximum work variety. Sends Telegram alerts when shares are accepted.

HONEST EXPECTATIONS:
  - 4-core CPU at ~500K H/s per core = ~2 MH/s total
  - Network hashrate: ~700 EH/s
  - Your share: ~0.000000000003% of network
  - Expected earnings: fractions of a penny per month
  - This is REAL mining — just very slow without ASIC hardware
  - Every hash submitted is genuine proof-of-work

Pool: public-pool.io (no account, no registration, pays direct to your wallet)
"""

import socket
import json
import hashlib
import struct
import time
import threading
import os
import sys
import multiprocessing

# ── Configuration ─────────────────────────────────────────────────────────────

WALLET_ADDRESS  = os.getenv("MINING_WALLET", "bc1qfzhx87ckhn4tnkswhsth56h0gm5we4hdq5wass")
WORKER_NAME     = os.getenv("MINING_WORKER", "nova")
POOL_HOST       = os.getenv("MINING_POOL_HOST", "public-pool.io")
POOL_PORT       = int(os.getenv("MINING_POOL_PORT", "21496"))
NUM_THREADS     = int(os.getenv("MINING_THREADS", str(multiprocessing.cpu_count())))
TELEGRAM_TOKEN  = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT   = os.getenv("TELEGRAM_CHAT_ID", "")

# ── SHA-256d ──────────────────────────────────────────────────────────────────

def sha256d(data: bytes) -> bytes:
    return hashlib.sha256(hashlib.sha256(data).digest()).digest()

# ── Stratum helpers ───────────────────────────────────────────────────────────

def difficulty_to_target(difficulty: float) -> int:
    # Integer arithmetic — no float precision loss on 256-bit numbers
    diff1 = 0x00000000ffff0000000000000000000000000000000000000000000000000000
    d = max(int(difficulty), 1)
    return diff1 // d

def _swap32(b: bytes) -> bytes:
    # Stratum sends prevhash with each 4-byte word byte-swapped
    return b"".join(b[i:i+4][::-1] for i in range(0, len(b), 4))

def build_header(job: dict, extranonce1: str, extranonce2: str,
                 ntime: str, nonce: int) -> bytes:
    def h2b(h): return bytes.fromhex(h)
    coinbase  = h2b(job["coinb1"]) + h2b(extranonce1) + h2b(extranonce2) + h2b(job["coinb2"])
    merkle    = sha256d(coinbase)
    for branch in job["merkle_branch"]:
        merkle = sha256d(merkle + h2b(branch))
    prevhash  = _swap32(h2b(job["prevhash"]))  # correct Stratum byte ordering
    return (h2b(job["version"]) + prevhash + merkle
            + h2b(ntime) + h2b(job["nbits"]) + struct.pack("<I", nonce))

# ── Telegram notify ───────────────────────────────────────────────────────────

def _tg_notify(message: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT:
        return
    try:
        import urllib.request, urllib.parse
        url  = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        data = urllib.parse.urlencode({
            "chat_id": TELEGRAM_CHAT,
            "text":    message,
            "parse_mode": "HTML",
        }).encode()
        urllib.request.urlopen(url, data=data, timeout=10)
    except Exception:
        pass

# ── Shared state ──────────────────────────────────────────────────────────────

class MinerState:
    def __init__(self):
        self._lock            = threading.Lock()
        self.job              = None
        self.extranonce1      = ""
        self.extranonce2_size = 4
        self.difficulty       = 1.0
        self.hashes           = 0
        self.shares_submitted = 0
        self.shares_accepted  = 0
        self.start_time       = time.time()
        self.running          = False
        self.connected        = False

    def add_hashes(self, n: int):
        with self._lock:
            self.hashes += n

    def share_submitted(self):
        with self._lock:
            self.shares_submitted += 1

    def share_accepted(self):
        with self._lock:
            self.shares_accepted += 1

    @property
    def hashrate(self) -> float:
        elapsed = time.time() - self.start_time
        return self.hashes / elapsed if elapsed > 0 else 0.0

    def hashrate_str(self) -> str:
        h = self.hashrate
        if h >= 1_000_000: return f"{h/1_000_000:.2f} MH/s"
        if h >= 1_000:     return f"{h/1_000:.2f} KH/s"
        return f"{h:.0f} H/s"

    def summary(self) -> str:
        elapsed = int(time.time() - self.start_time)
        return (f"⛏ Nova Miner — LIVE\n"
                f"Hashrate : {self.hashrate_str()}\n"
                f"Hashes   : {self.hashes:,}\n"
                f"Shares   : {self.shares_accepted}/{self.shares_submitted}\n"
                f"Uptime   : {elapsed//3600}h {(elapsed%3600)//60}m {elapsed%60}s\n"
                f"Wallet   : {WALLET_ADDRESS[:20]}...")

# ── Stratum connection ────────────────────────────────────────────────────────

class StratumConnection:
    def __init__(self, state: MinerState):
        self.state   = state
        self.sock    = None
        self._lock   = threading.Lock()
        self._msg_id = 0

    def connect(self) -> bool:
        try:
            self.sock = socket.create_connection((POOL_HOST, POOL_PORT), timeout=30)
            self.sock.settimeout(120)
            self.state.connected = True
            return True
        except Exception as e:
            print(f"\n  ✗ Connection failed: {e}")
            return False

    def disconnect(self):
        self.state.connected = False
        try:
            if self.sock:
                self.sock.close()
        except Exception:
            pass
        self.sock = None

    def send(self, method: str, params: list):
        with self._lock:
            self._msg_id += 1
            msg = json.dumps({"id": self._msg_id, "method": method, "params": params}) + "\n"
            try:
                self.sock.sendall(msg.encode())
            except Exception:
                pass

    def submit(self, job_id: str, en2: str, ntime: str, nonce: int):
        self.send("mining.submit", [
            f"{WALLET_ADDRESS}.{WORKER_NAME}",
            job_id, en2, ntime,
            format(nonce, "08x"),
        ])
        self.state.share_submitted()

    def listen(self):
        buf = ""
        while self.state.running and self.sock:
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
            except Exception:
                break

    def _handle(self, msg: dict):
        method = msg.get("method", "")
        result = msg.get("result")
        error  = msg.get("error")

        if method == "mining.notify":
            p = msg["params"]
            self.state.job = {
                "job_id":        p[0], "prevhash":  p[1],
                "coinb1":        p[2], "coinb2":    p[3],
                "merkle_branch": p[4], "version":   p[5],
                "nbits":         p[6], "ntime":     p[7],
                "clean":         p[8],
            }

        elif method == "mining.set_difficulty":
            self.state.difficulty = float(msg["params"][0])
            print(f"\n  Pool difficulty → {self.state.difficulty}")

        elif result is not None and not error:
            if isinstance(result, list) and len(result) >= 3:
                self.state.extranonce1      = result[1]
                self.state.extranonce2_size = result[2]
                print(f"  ✓ Subscribed  extranonce1={result[1]}")
                self.send("mining.authorize", [
                    f"{WALLET_ADDRESS}.{WORKER_NAME}", "x"
                ])
            elif result is True:
                if self.state.shares_submitted > self.state.shares_accepted:
                    self.state.share_accepted()
                    msg_text = (f"⛏ <b>Share accepted!</b>\n"
                                f"Total: {self.state.shares_accepted}\n"
                                f"Hashrate: {self.state.hashrate_str()}\n"
                                f"Wallet: <code>{WALLET_ADDRESS[:20]}...</code>")
                    print(f"\n  ✓ SHARE ACCEPTED! ({self.state.shares_accepted} total) 🎉")
                    threading.Thread(target=_tg_notify, args=(msg_text,), daemon=True).start()
                else:
                    print(f"  ✓ Authorized — mining to {WALLET_ADDRESS}")
        elif error:
            print(f"\n  Pool: {error}")

# ── Mining thread ─────────────────────────────────────────────────────────────

def mining_thread(thread_id: int, state: MinerState, conn: StratumConnection):
    # Each thread owns a unique nonce slice — no duplicate work
    nonce_range  = 0x100000000 // NUM_THREADS
    nonce_start  = thread_id * nonce_range
    nonce_end    = nonce_start + nonce_range
    en2_int      = thread_id   # unique extranonce2 per thread
    batch        = 5_000       # hashes between counter flush + job check

    while state.running:
        job = state.job
        if not job:
            time.sleep(0.2)
            continue

        target = difficulty_to_target(state.difficulty)
        en2    = format(en2_int & 0xFFFFFFFF,
                        f"0{state.extranonce2_size * 2}x")
        ntime  = format(int(time.time()), "08x")
        local_hashes = 0

        for nonce in range(nonce_start, nonce_end):
            if state.job is not job or not state.running:
                break

            header      = build_header(job, state.extranonce1, en2, ntime, nonce)
            hash_result = sha256d(header)
            # Compare as little-endian integer (Bitcoin standard)
            hash_int    = int.from_bytes(hash_result, "little")

            local_hashes += 1

            if hash_int < target:
                conn.submit(job["job_id"], en2, ntime, nonce)

            if local_hashes % batch == 0:
                state.add_hashes(local_hashes)
                local_hashes = 0
                ntime = format(int(time.time()), "08x")

        state.add_hashes(local_hashes)
        # Advance extranonce2 so next pass covers fresh work
        en2_int = (en2_int + NUM_THREADS) & 0xFFFFFFFF

# ── Stats display ─────────────────────────────────────────────────────────────

def stats_thread(state: MinerState):
    while state.running:
        time.sleep(20)
        elapsed = int(time.time() - state.start_time)
        print(
            f"\r  ⛏  {state.hashrate_str()} | "
            f"Hashes: {state.hashes:,} | "
            f"Shares: {state.shares_accepted}/{state.shares_submitted} | "
            f"Up: {elapsed//60}m{elapsed%60}s  ",
            end="", flush=True,
        )

# ── Main loop with auto-reconnect ─────────────────────────────────────────────

def run(state: MinerState = None) -> MinerState:
    if state is None:
        state = MinerState()
    state.running    = True
    state.start_time = time.time()

    print(f"\n  ⛏  Nova Real Bitcoin Miner")
    print(f"  {'─'*45}")
    print(f"  Pool    : {POOL_HOST}:{POOL_PORT}")
    print(f"  Wallet  : {WALLET_ADDRESS}")
    print(f"  Threads : {NUM_THREADS} (one per CPU core)")
    print(f"  Telegram: {'✓ enabled' if TELEGRAM_TOKEN else '✗ add TELEGRAM_BOT_TOKEN to .env'}")
    print()

    attempt = 0
    while state.running:
        attempt += 1
        conn = StratumConnection(state)

        if not conn.connect():
            wait = min(60, 5 * attempt)
            print(f"  Retrying in {wait}s...")
            time.sleep(wait)
            continue

        attempt = 0
        print(f"  ✓ Connected to {POOL_HOST}")

        # Start listener
        listener = threading.Thread(target=conn.listen, daemon=True)
        listener.start()

        conn.send("mining.subscribe", ["nova_miner/2.0"])

        # Wait for first job
        for _ in range(60):
            if state.job:
                break
            time.sleep(1)

        if not state.job:
            print("  ✗ No work received. Reconnecting...")
            conn.disconnect()
            continue

        print(f"  ✓ Got work from pool — launching {NUM_THREADS} mining threads\n")
        _tg_notify(f"⛏ <b>Nova Miner started</b>\n"
                   f"Threads: {NUM_THREADS}\n"
                   f"Pool: {POOL_HOST}\n"
                   f"Wallet: <code>{WALLET_ADDRESS[:20]}...</code>")

        # Launch mining threads
        miners = []
        for i in range(NUM_THREADS):
            t = threading.Thread(
                target=mining_thread, args=(i, state, conn), daemon=True)
            t.start()
            miners.append(t)

        # Stats printer
        stats = threading.Thread(target=stats_thread, args=(state,), daemon=True)
        stats.start()

        # Wait until connection drops or stopped
        listener.join()

        if state.running:
            print(f"\n  Pool disconnected — reconnecting...")
            conn.disconnect()
            time.sleep(5)

    print(f"\n\n  Final stats:")
    print(f"    {state.summary()}")
    return state


if __name__ == "__main__":
    state = MinerState()
    try:
        run(state)
    except KeyboardInterrupt:
        print("\n\n  Stopping miner...")
        state.running = False
        time.sleep(1)
        print(f"\n  {state.summary()}")
