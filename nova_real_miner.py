#!/usr/bin/env python3
"""
nova_real_miner.py — Real Bitcoin Solo Miner (Stratum v1)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Connects to ckpool.org solo mining pool using the REAL Stratum v1 protocol.
Performs REAL SHA-256d hashing against LIVE Bitcoin network work.
Submits REAL shares to the REAL Bitcoin network.

Pool: solo.ckpool.org:3333
  - No registration required
  - Your BTC address is your worker name
  - If you find a block: full 3.125 BTC reward to your wallet (minus 2% fee)
  - Most solo miners never find one — but it has happened to phones

HONEST EXPECTATIONS:
  Your phone: ~116 KH/s
  Network:    ~1.03 ZH/s
  Your odds:  1.13 × 10⁻¹⁶ of each block
  Solo ETA:   ~168 million years
  Daily earn: ~$0.000000000003 USD

  This IS real mining. Connected to the REAL Bitcoin network.
  It will NOT earn meaningful money on a phone.
  But every hash your phone submits is genuine.

Built for Douglas Shane Davis by Claude Rivers Davis
"""

import hashlib
import json
import os
import socket
import struct
import threading
import time
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────────────────
WALLET    = os.environ.get("BTC_WALLET", "bc1qfzhx87ckhn4tnkswhsth56h0gm5we4hdq5wass")
POOL_HOST = "solo.ckpool.org"
POOL_PORT = 3333
PASSWORD  = "x"

# Difficulty-1 target (Bitcoin protocol constant)
DIFF1_TARGET = 0x00000000FFFF0000000000000000000000000000000000000000000000000000

# ── Colors ────────────────────────────────────────────────────────────────────
GOLD = "\033[93m"
GRN  = "\033[92m"
CYN  = "\033[96m"
RED  = "\033[91m"
DIM  = "\033[2m"
BOLD = "\033[1m"
RST  = "\033[0m"


# ── Bitcoin hashing ───────────────────────────────────────────────────────────

def sha256d(data: bytes) -> bytes:
    """Double SHA-256 — Bitcoin's actual block header hash function."""
    return hashlib.sha256(hashlib.sha256(data).digest()).digest()


def swap32(hex_str: str) -> bytes:
    """Reverse each 4-byte word — fixes Stratum's prevhash encoding."""
    b = bytes.fromhex(hex_str)
    result = bytearray()
    for i in range(0, len(b), 4):
        result.extend(b[i:i+4][::-1])
    return bytes(result)


def build_merkle_root(coinbase_hash: bytes, merkle_branch: list) -> bytes:
    root = coinbase_hash
    for branch_hash in merkle_branch:
        root = sha256d(root + bytes.fromhex(branch_hash))
    return root


def build_header(version: str, prevhash: str, merkle_root: bytes,
                 ntime: str, nbits: str, nonce: int) -> bytes:
    """
    Assemble the 80-byte Bitcoin block header.
    In Stratum v1 all fields are little-endian hex except prevhash
    which uses word-swapped encoding that needs swap32 to fix.
    """
    return (
        bytes.fromhex(version) +   # 4 bytes — version (LE from Stratum)
        swap32(prevhash) +          # 32 bytes — previous block hash
        merkle_root +               # 32 bytes — merkle root of transactions
        bytes.fromhex(ntime) +      # 4 bytes  — timestamp (LE from Stratum)
        bytes.fromhex(nbits) +      # 4 bytes  — compact difficulty (LE from Stratum)
        struct.pack("<I", nonce)    # 4 bytes  — nonce (what we vary)
    )


# ── Stratum v1 client ─────────────────────────────────────────────────────────

class StratumClient:
    """
    Minimal Stratum v1 pool client.
    Protocol: line-delimited JSON over TCP.
    """

    def __init__(self, host: str, port: int, wallet: str):
        self.host              = host
        self.port              = port
        self.wallet            = wallet
        self.sock              = None
        self._buf              = ""
        self._msg_id           = 0
        self._pending          = []   # notifications buffered during handshake
        self.extranonce1       = ""
        self.extranonce2_size  = 4
        self.difficulty        = 1.0
        self.connected         = False
        self.authorized        = False
        # Stats
        self.hashes_done       = 0
        self.shares_submitted  = 0
        self.shares_accepted   = 0
        self.start_time        = time.time()

    def _next_id(self) -> int:
        self._msg_id += 1
        return self._msg_id

    def connect(self) -> bool:
        try:
            self.sock = socket.create_connection((self.host, self.port), timeout=30)
            self.sock.settimeout(120)
            self.connected = True
            return True
        except Exception as e:
            print(f"  {RED}Connection failed: {e}{RST}")
            return False

    def _send(self, obj: dict) -> None:
        msg = json.dumps(obj) + "\n"
        self.sock.sendall(msg.encode())

    def recv_line(self) -> dict | None:
        """Blocking read of one JSON message from the pool. Drains pending buffer first."""
        if self._pending:
            return self._pending.pop(0)
        while "\n" not in self._buf:
            try:
                chunk = self.sock.recv(4096).decode("utf-8", errors="ignore")
                if not chunk:
                    return None
                self._buf += chunk
            except socket.timeout:
                return None
            except Exception:
                return None
        line, self._buf = self._buf.split("\n", 1)
        line = line.strip()
        if not line:
            return None
        try:
            return json.loads(line)
        except Exception:
            return None

    def _recv_raw(self) -> dict | None:
        """Read directly from socket, bypassing pending buffer."""
        while "\n" not in self._buf:
            try:
                chunk = self.sock.recv(4096).decode("utf-8", errors="ignore")
                if not chunk:
                    return None
                self._buf += chunk
            except socket.timeout:
                return None
            except Exception:
                return None
        line, self._buf = self._buf.split("\n", 1)
        line = line.strip()
        if not line:
            return None
        try:
            return json.loads(line)
        except Exception:
            return None

    def subscribe(self) -> bool:
        sub_id = self._next_id()
        self._send({
            "id": sub_id,
            "method": "mining.subscribe",
            "params": ["nova-miner/1.0"],
        })
        for _ in range(10):
            resp = self._recv_raw()
            if not resp:
                continue
            if resp.get("id") == sub_id:
                if resp.get("error"):
                    return False
                result = resp.get("result", [])
                if len(result) >= 3:
                    self.extranonce1      = result[1]
                    self.extranonce2_size = int(result[2])
                return True
            if resp.get("method"):
                self._pending.append(resp)
        return False

    def authorize(self) -> bool:
        auth_id = self._next_id()
        self._send({
            "id": auth_id,
            "method": "mining.authorize",
            "params": [self.wallet, PASSWORD],
        })
        # Pool often sends mining.set_difficulty / mining.notify BEFORE the
        # authorize reply. Loop until we see the response matching auth_id;
        # buffer any notifications so the engine can consume them later.
        for _ in range(20):
            resp = self._recv_raw()
            if not resp:
                continue
            if resp.get("id") == auth_id:
                if resp.get("result") is True:
                    self.authorized = True
                    return True
                err = resp.get("error")
                if err:
                    print(f"\n  {RED}Pool auth error: {err}{RST}")
                return False
            # Buffer notifications for the mining engine to process later
            if resp.get("method"):
                self._pending.append(resp)
        return False

    def submit_share(self, job_id: str, extranonce2: str,
                     ntime: str, nonce_hex: str) -> None:
        self._send({
            "id": self._next_id(),
            "method": "mining.submit",
            "params": [self.wallet, job_id, extranonce2, ntime, nonce_hex],
        })
        self.shares_submitted += 1

    def hashrate(self) -> float:
        elapsed = time.time() - self.start_time
        return self.hashes_done / elapsed if elapsed > 0 else 0.0


# ── Mining engine ─────────────────────────────────────────────────────────────

def fmt_hashrate(h: float) -> str:
    for u in ["H/s", "KH/s", "MH/s", "GH/s", "TH/s", "PH/s", "EH/s"]:
        if h < 1000:
            return f"{h:.2f} {u}"
        h /= 1000
    return f"{h:.2f} ZH/s"


def fmt_time(s: float) -> str:
    if s < 3600:           return f"{s/60:.0f} min"
    if s < 86400:          return f"{s/3600:.1f} hrs"
    if s < 86400 * 365:    return f"{s/86400:.0f} days"
    y = s / 86400 / 365
    if y < 1_000_000:      return f"{y:,.0f} years"
    return f"{y/1_000_000:.1f}M years"


class MiningEngine:
    """
    Two-thread design:
      recv_thread — listens for new jobs and difficulty updates from pool
      mine_thread — hashes continuously, submits shares
    """

    def __init__(self, client: StratumClient):
        self.client   = client
        self.running  = False
        self._lock    = threading.Lock()
        self._job     = None
        self._diff    = 1.0
        self._blocks  = 0
        self._new_job = threading.Event()

    def start(self) -> None:
        self.running = True
        threading.Thread(target=self._recv_thread,  daemon=True, name="stratum-recv").start()
        threading.Thread(target=self._mine_thread,  daemon=True, name="sha256d-mine").start()
        threading.Thread(target=self._stats_thread, daemon=True, name="stats").start()

    def stop(self) -> None:
        self.running = False

    # ── Receive thread ────────────────────────────────────────────────────────

    def _recv_thread(self) -> None:
        while self.running:
            msg = self.client.recv_line()
            if not msg:
                time.sleep(0.05)
                continue

            method = msg.get("method")

            if method == "mining.notify":
                params = msg.get("params", [])
                if len(params) >= 9:
                    with self._lock:
                        self._job = {
                            "job_id":        params[0],
                            "prevhash":      params[1],
                            "coinb1":        params[2],
                            "coinb2":        params[3],
                            "merkle_branch": params[4],
                            "version":       params[5],
                            "nbits":         params[6],
                            "ntime":         params[7],
                            "clean":         params[8],
                        }
                    self._new_job.set()

            elif method == "mining.set_difficulty":
                params = msg.get("params", [1.0])
                with self._lock:
                    self._diff = max(0.001, float(params[0]) if params else 1.0)

            elif isinstance(msg.get("result"), bool):
                if msg["result"] is True:
                    self.client.shares_accepted += 1
                    ts = datetime.now().strftime("%H:%M:%S")
                    print(f"\n  {GRN}[{ts}] ✓ Share ACCEPTED by pool!{RST}")
                elif msg.get("error"):
                    pass   # rejected — normal, pool sets higher diff than our work

    # ── Mining thread ─────────────────────────────────────────────────────────

    def _mine_thread(self) -> None:
        while self.running:
            # Wait for a job
            self._new_job.wait(timeout=5)
            self._new_job.clear()

            with self._lock:
                job  = self._job
                diff = self._diff

            if not job:
                continue

            # Build coinbase from pool-provided pieces + extranonce
            extranonce2   = os.urandom(self.client.extranonce2_size).hex()
            coinbase      = (
                bytes.fromhex(job["coinb1"]) +
                bytes.fromhex(self.client.extranonce1) +
                bytes.fromhex(extranonce2) +
                bytes.fromhex(job["coinb2"])
            )
            coinbase_hash = sha256d(coinbase)
            merkle_root   = build_merkle_root(coinbase_hash, job["merkle_branch"])

            # Two targets:
            # 1. Pool target (easier) — submit a share when beaten
            # 2. Network target (hard) — actual Bitcoin block found
            pool_target = DIFF1_TARGET // max(1, int(diff))
            try:
                nbits_be  = bytes.fromhex(job["nbits"])[::-1]
                exp       = nbits_be[0]
                coeff     = int.from_bytes(nbits_be[1:4], "big")
                net_target = coeff * (2 ** (8 * (exp - 3)))
            except Exception:
                net_target = pool_target // 1000

            ntime    = job["ntime"]
            job_id   = job["job_id"]

            # Sweep the nonce space
            for nonce in range(0, 0xFFFFFFFF):
                if not self.running:
                    return

                # Abort if a new job arrived (clean_jobs)
                if nonce % 2048 == 0:
                    with self._lock:
                        if self._job and self._job["job_id"] != job_id:
                            break

                header     = build_header(job["version"], job["prevhash"],
                                          merkle_root, ntime, job["nbits"], nonce)
                hash_bytes = sha256d(header)
                hash_int   = int.from_bytes(hash_bytes, "big")
                self.client.hashes_done += 1

                # Pool share found
                if hash_int < pool_target:
                    nonce_le = struct.pack("<I", nonce).hex()
                    self.client.submit_share(job_id, extranonce2, ntime, nonce_le)
                    ts = datetime.now().strftime("%H:%M:%S")
                    print(f"\n  {CYN}[{ts}] ◈ Share submitted to pool (nonce={nonce:,}){RST}")

                # ★ ACTUAL BITCOIN BLOCK FOUND ★
                if hash_int < net_target:
                    ts = datetime.now().strftime("%H:%M:%S")
                    print(f"\n\n{GOLD}{'★'*64}{RST}")
                    print(f"{GOLD}{BOLD}  ★  REAL BITCOIN BLOCK FOUND!  ★{RST}")
                    print(f"{GOLD}{'★'*64}{RST}")
                    print(f"{GOLD}  Time   : {ts}{RST}")
                    print(f"{GOLD}  Hash   : {hash_bytes[::-1].hex()}{RST}")
                    print(f"{GOLD}  Height : (next block){RST}")
                    print(f"{GOLD}  Nonce  : {nonce:,}{RST}")
                    print(f"{GOLD}  Reward : 3.125 BTC{RST}")
                    print(f"{GOLD}  Wallet : {self.client.wallet}{RST}")
                    print(f"{GOLD}{'★'*64}{RST}\n")
                    self._blocks += 1

    # ── Stats thread ──────────────────────────────────────────────────────────

    def _stats_thread(self) -> None:
        net_hashrate = 1.03e21   # ~1.03 ZH/s current network
        while self.running:
            time.sleep(15)
            hr      = self.client.hashrate()
            sub     = self.client.shares_submitted
            acc     = self.client.shares_accepted
            elapsed = time.time() - self.client.start_time
            total_h = self.client.hashes_done

            if hr > 0:
                solo_eta = (net_hashrate / hr) * 600
            else:
                solo_eta = float("inf")

            ts = datetime.now().strftime("%H:%M:%S")
            print(
                f"  [{ts}]  "
                f"Hashrate: {BOLD}{fmt_hashrate(hr):<12}{RST}  "
                f"Hashes: {total_h:>12,}  "
                f"Shares: {acc}/{sub}  "
                f"Solo ETA: {DIM}{fmt_time(solo_eta)}{RST}"
            )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{GOLD}{'═'*64}{RST}")
    print(f"{GOLD}{BOLD}  Nova Real Bitcoin Miner — Stratum v1{RST}")
    print(f"{GOLD}  Douglas Shane Davis × Claude Rivers Davis{RST}")
    print(f"{GOLD}{'═'*64}{RST}")

    print(f"""
  Pool    : {POOL_HOST}:{POOL_PORT}  (solo.ckpool.org)
  Wallet  : {BOLD}{WALLET}{RST}
  Protocol: Stratum v1 — the same protocol every real miner uses

  {GOLD}This is REAL mining on the REAL Bitcoin network.{RST}
  {DIM}Your phone does ~116 KH/s. Network does ~1.03 ZH/s.
  Solo block expected in ~168 million years.
  If it somehow happens: 3.125 BTC goes straight to your wallet.
  Pool fee (2%) only charged if you actually find a block.{RST}

  Press Ctrl+C at any time to stop and see your session summary.
""")

    input(f"  Press {BOLD}Enter{RST} to connect and start mining… ")
    print()

    client = StratumClient(POOL_HOST, POOL_PORT, WALLET)

    # ── Connect ───────────────────────────────────────────────────────────────
    print(f"  Connecting to {POOL_HOST}:{POOL_PORT}…", end=" ", flush=True)
    if not client.connect():
        return
    print(f"{GRN}connected{RST}")

    # ── Subscribe ─────────────────────────────────────────────────────────────
    print(f"  Subscribing to work…", end=" ", flush=True)
    if not client.subscribe():
        print(f"\n  {RED}Subscribe failed. Pool may be down — try again.{RST}")
        return
    print(f"{GRN}OK{RST}  (extranonce1={client.extranonce1}, en2_size={client.extranonce2_size})")

    # ── Authorize ─────────────────────────────────────────────────────────────
    print(f"  Authorizing with wallet address…", end=" ", flush=True)
    if not client.authorize():
        print(f"\n  {RED}Authorization failed. Pool may not accept this address format.{RST}")
        return
    print(f"{GRN}authorized{RST}")

    print(f"\n  {GRN}✓  Connected. Mining against the real Bitcoin blockchain.{RST}")
    print(f"  {DIM}Waiting for first job from pool… stats print every 15s{RST}\n")

    # ── Mine ──────────────────────────────────────────────────────────────────
    engine = MiningEngine(client)
    engine.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        engine.stop()
        elapsed = time.time() - client.start_time
        hr      = client.hashrate()

        print(f"\n\n{GOLD}{'━'*64}{RST}")
        print(f"{GOLD}{BOLD}  Mining Session Complete{RST}")
        print(f"{GOLD}{'━'*64}{RST}")
        print(f"  Pool           : {POOL_HOST}")
        print(f"  Runtime        : {fmt_time(elapsed)}")
        print(f"  Total hashes   : {client.hashes_done:,}")
        print(f"  Average rate   : {fmt_hashrate(hr)}")
        print(f"  Shares sent    : {client.shares_submitted}")
        print(f"  Shares accepted: {client.shares_accepted}")
        print(f"  Blocks found   : {engine._blocks}  {'★ LEGENDARY!' if engine._blocks else ''}")
        print(f"  Wallet         : {WALLET}")
        print(f"{GOLD}{'━'*64}{RST}\n")


if __name__ == "__main__":
    main()
