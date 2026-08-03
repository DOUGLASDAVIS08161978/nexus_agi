"""
real_pool_miner.py — Multi-threaded real Bitcoin pool miner (Stratum protocol)

One mining thread per CPU core. Uses SHA-256 midstate to skip recomputing
the first 64-byte header block on every nonce — the largest single speedup
available in pure Python. All hashes are genuine proof-of-work.

Bitcoin 80-byte header layout (all fields little-endian):
  [0-3]   version   (4 bytes)
  [4-35]  prevhash  (32 bytes)
  [36-67] merkle    (32 bytes)
  [68-71] ntime     (4 bytes)
  [72-75] nbits     (4 bytes)
  [76-79] nonce     (4 bytes)

SHA-256 processes 64-byte blocks, so:
  Block 1 = version + prevhash + merkle[:28]   ← never changes per nonce
  Block 2 = merkle[28:] + ntime + nbits + nonce ← nonce changes every iteration

Midstate: precompute SHA-256 state after block 1, then per nonce only process
block 2 (16 bytes). Combined with eliminating per-nonce hex decoding this
gives roughly 5-8× the naive throughput.

Pool: public-pool.io (no account, no registration, pays direct to your wallet)
"""

import socket
import json
import hashlib
import struct
import time
import threading
import os
import multiprocessing

try:
    import miner_core as _miner_core
    _USE_C_EXT = True
    _EXT_PATH  = getattr(_miner_core, "path", "C-ext")
except ImportError:
    _miner_core = None
    _USE_C_EXT  = False
    _EXT_PATH   = "Python-midstate"

_C_BATCH = 8_000_000  # nonces per C call; larger = fewer GIL re-checks, better throughput
                      # 8M @ ~6 MH/s per thread ≈ 1.3s between Python overhead pauses

# ── Configuration ─────────────────────────────────────────────────────────────

WALLET_ADDRESS  = os.getenv("MINING_WALLET", "bc1qfzhx87ckhn4tnkswhsth56h0gm5we4hdq5wass")
WORKER_NAME     = os.getenv("MINING_WORKER", "nova")
# MINING_SOLO=1  → solo.ckpool.org (full block reward if found; no partial payouts)
# MINING_SOLO=0  → public-pool.io PPLNS (micro-payouts proportional to hashrate)
_SOLO_MODE      = os.getenv("MINING_SOLO", "0").strip() in ("1", "true", "yes")
if _SOLO_MODE:
    _DEFAULT_HOST = "solo.ckpool.org"
    _DEFAULT_PORT = "3333"
else:
    _DEFAULT_HOST = "public-pool.io"
    _DEFAULT_PORT = "21496"
POOL_HOST       = os.getenv("MINING_POOL_HOST", _DEFAULT_HOST)
POOL_PORT       = int(os.getenv("MINING_POOL_PORT", _DEFAULT_PORT))
NUM_THREADS     = int(os.getenv("MINING_THREADS", str(multiprocessing.cpu_count())))
TELEGRAM_TOKEN  = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT   = os.getenv("TELEGRAM_CHAT_ID", "")
# Suggest difficulty that matches public-pool.io's actual vardiff assignment (~1.0).
# Too-low suggestions cause mid-session difficulty bumps → "Difficulty too low" rejects.
SUGGEST_DIFF    = float(os.getenv("MINING_SUGGEST_DIFF", "1.0"))
# Local soft-share target — checked every nonce, never submitted to pool.
# At SOFT_DIFF=0.00001 and ~40 KH/s you see roughly 1 soft share per second.
# Proves the miner is grinding even when real pool shares take hours.
SOFT_DIFF       = float(os.getenv("MINING_SOFT_DIFF", "0.00001"))

# ── SHA-256d ──────────────────────────────────────────────────────────────────

def sha256d(data: bytes) -> bytes:
    return hashlib.sha256(hashlib.sha256(data).digest()).digest()

# ── Stratum helpers ───────────────────────────────────────────────────────────

def difficulty_to_target(difficulty: float) -> int:
    diff1 = 0x00000000ffff0000000000000000000000000000000000000000000000000000
    if difficulty >= 1.0:
        return diff1 // max(int(difficulty), 1)
    # Sub-1 difficulty: target = diff1 * (1/difficulty) — use integer multiply
    # to avoid converting the 256-bit diff1 to a lossy float.
    multiplier = max(1, round(1.0 / max(difficulty, 1e-15)))
    return min(diff1 * multiplier, (1 << 256) - 1)

def _swap32(b: bytes) -> bytes:
    # Stratum sends prevhash with each 4-byte word byte-swapped
    return b"".join(b[i:i+4][::-1] for i in range(0, len(b), 4))

def compute_merkle(job: dict, en1_b: bytes, en2_b: bytes) -> bytes:
    """Build merkle root from job fields + extranonces. Returns 32 bytes."""
    coinbase = bytes.fromhex(job["coinb1"]) + en1_b + en2_b + bytes.fromhex(job["coinb2"])
    merkle   = sha256d(coinbase)
    for branch in job["merkle_branch"]:
        merkle = sha256d(merkle + bytes.fromhex(branch))
    return merkle

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
        self.soft_shares      = 0
        self.shares_submitted = 0
        self.shares_accepted  = 0
        self.start_time       = time.time()
        self.running          = False
        self.connected        = False

    def add_hashes(self, n: int):
        self.hashes += n  # display-only counter; no lock needed

    def add_soft(self, n: int):
        self.soft_shares += n  # same

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
                # Ask pool for the lowest difficulty it will accept
                self.send("mining.suggest_difficulty", [SUGGEST_DIFF])
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

# ── Mining thread (midstate-optimized) ───────────────────────────────────────

def mining_thread(thread_id: int, state: MinerState, conn: StratumConnection):
    """Each thread covers a unique nonce slice.

    Key optimization: SHA-256 midstate.
    The 80-byte Bitcoin header is hashed in two 64-byte SHA-256 blocks.
    Block 1 (bytes 0-63) = version + prevhash + merkle[:28] — constant per job.
    Block 2 (bytes 64-79) = merkle[28:] + ntime + nbits + nonce — nonce changes per iter.
    We precompute SHA-256 state after block 1 and copy it for each nonce,
    so each nonce only requires one 16-byte SHA-256 update + one 32-byte final SHA-256.
    """
    nonce_range = 0x100000000 // NUM_THREADS
    nonce_start = thread_id * nonce_range
    nonce_end   = nonce_start + nonce_range
    en2_int     = thread_id
    batch       = 50_000    # hashes between stat flush + ntime roll

    while state.running:
        job = state.job
        if not job:
            time.sleep(0.2)
            continue

        cur_diff    = state.difficulty
        target      = difficulty_to_target(cur_diff)
        soft_target = difficulty_to_target(SOFT_DIFF)
        en2    = format(en2_int & 0xFFFFFFFF, f"0{state.extranonce2_size * 2}x")
        ntime  = format(int(time.time()), "08x")

        # Decode job fields once per extranonce2 sweep — not per nonce
        # Pool sends version/nbits/ntime as big-endian hex (.toString(16)) but
        # builds the block header with writeUInt32LE → we must pack little-endian.
        # prevhash is sent word-swapped (swapEndianWords) → _swap32 un-swaps it.
        en1_b      = bytes.fromhex(state.extranonce1)
        en2_b      = bytes.fromhex(en2)
        version_b  = struct.pack("<I", int(job["version"], 16))
        prevhash_b = _swap32(bytes.fromhex(job["prevhash"]))
        nbits_b    = struct.pack("<I", int(job["nbits"], 16))
        ntime_b    = struct.pack("<I", int(ntime, 16))

        # Merkle root: only recomputed when job or extranonce2 changes
        merkle = compute_merkle(job, en1_b, en2_b)

        # first_block = version(4) + prevhash(32) + merkle[:28] — constant per job
        first_block = bytes(version_b + prevhash_b + merkle[:28])

        if _USE_C_EXT:
            # ── C extension inner loop ──────────────────────────────────────
            # mine_range() runs the full SHA-256d loop in C — no Python
            # interpreter overhead per nonce. Called in _C_BATCH-nonce slices
            # so ntime rolls and job-change checks happen between batches.
            # Pool uses le256todouble() — treats hash as LE 256-bit number.
            # Pass target as LE bytes so hash_lt (byte 31 first) compares correctly.
            target_le      = target.to_bytes(32, "little")
            soft_target_le = soft_target.to_bytes(32, "little")

            nonce = nonce_start
            while state.running and nonce < nonce_end:
                cur_ntime     = format(int(time.time()), "08x")
                second_prefix = bytes(merkle[28:32] + struct.pack("<I", int(cur_ntime, 16)) + nbits_b)
                batch_end     = min(nonce + _C_BATCH, nonce_end)

                winner, hashes_done, soft_found = _miner_core.mine_range(
                    first_block, second_prefix, target_le, soft_target_le,
                    nonce, batch_end)

                state.add_hashes(hashes_done)
                state.add_soft(soft_found)

                if winner is not None:
                    # Independently verify with Python hashlib before submitting.
                    # Reconstructs the full 80-byte header and SHA-256d hashes it.
                    # If "FALSE-POSITIVE", the C extension has a bug; if "VALID"
                    # but pool rejects, the mismatch is in pool reconstruction.
                    _hdr = first_block + second_prefix + struct.pack("<I", winner)
                    _h1  = hashlib.sha256(_hdr).digest()
                    _h2  = hashlib.sha256(_h1).digest()
                    # LE comparison matches pool's le256todouble() check
                    _h_int = int.from_bytes(_h2, "little")
                    _valid = _h_int < target
                    print(f"\n  ★ nonce={winner:08x}  hash={_h2.hex()}  "
                          f"{'VALID — submitting' if _valid else 'FALSE-POSITIVE — skipping'}")
                    if _valid:
                        conn.submit(job["job_id"], en2, cur_ntime, winner)
                    else:
                        print(f"     target={target.to_bytes(32,'big').hex()}")

                nonce = batch_end
                if state.job is not job or state.difficulty != cur_diff:
                    break
        else:
            # ── Python midstate inner loop ──────────────────────────────────
            # Precompute SHA-256 midstate over the first 64-byte header block.
            midstate = hashlib.sha256()
            midstate.update(first_block)

            # Mutable 16-byte buffer for the second block.
            # Layout: merkle[28:32](4) | ntime(4) | nbits(4) | nonce(4)
            second_block = bytearray(merkle[28:32] + ntime_b + nbits_b + b'\x00\x00\x00\x00')

            _copy      = midstate.copy
            _sha256    = hashlib.sha256
            _pack_into = struct.pack_into
            _frombytes = int.from_bytes

            local_hashes = 0
            local_soft   = 0
            countdown    = batch

            for nonce in range(nonce_start, nonce_end):
                _pack_into("<I", second_block, 12, nonce)

                h           = _copy()
                h.update(second_block)
                inner       = h.digest()
                hash_result = _sha256(inner).digest()
                hash_int    = _frombytes(hash_result, "little")  # LE = pool's le256todouble

                if hash_int < target:
                    conn.submit(job["job_id"], en2, ntime, nonce)
                elif hash_int < soft_target:
                    local_soft += 1

                local_hashes += 1
                countdown    -= 1
                if not countdown:
                    countdown = batch
                    state.add_hashes(local_hashes)
                    state.add_soft(local_soft)
                    local_hashes = 0
                    local_soft   = 0
                    if state.job is not job or state.difficulty != cur_diff or not state.running:
                        break
                    new_ntime = format(int(time.time()), "08x")
                    if new_ntime != ntime:
                        ntime   = new_ntime
                        ntime_b = struct.pack("<I", int(ntime, 16))
                        second_block[4:8] = ntime_b

            state.add_hashes(local_hashes)
            state.add_soft(local_soft)

        # Advance extranonce2 so next pass covers fresh work
        en2_int = (en2_int + NUM_THREADS) & 0xFFFFFFFF

# ── Stats display ─────────────────────────────────────────────────────────────

_NETWORK_HASHRATE_EH = 700.0   # approximate network hashrate in EH/s (update periodically)
_BLOCK_REWARD_BTC    = 3.125   # current subsidy post-4th-halving

def _eta_str(hashrate: float, difficulty: float) -> str:
    """Human-readable expected time to next pool share at current hashrate + difficulty."""
    if hashrate <= 0:
        return "∞"
    # At difficulty D, expected hashes per share ≈ D * 2^32
    expected = max(difficulty, 1e-12) * 4_294_967_296
    secs = expected / hashrate
    if secs < 90:
        return f"{secs:.0f}s"
    if secs < 5400:
        return f"{secs/60:.1f}m"
    return f"{secs/3600:.1f}h"

def _solo_block_eta_str(hashrate: float) -> str:
    """Expected time to find a solo Bitcoin block at current hashrate."""
    if hashrate <= 0:
        return "∞"
    # Bitcoin difficulty ~= (network_hashrate * 600) / 2^32
    # P(block per hash) = 1 / (network_hashrate * 600)
    network_hps = _NETWORK_HASHRATE_EH * 1e18
    secs = network_hps / max(hashrate, 1.0)
    if secs < 86400:
        return f"{secs/3600:.0f}h"
    if secs < 365.25 * 86400:
        return f"{secs/86400:.0f}d"
    years = secs / (365.25 * 86400)
    if years < 1_000_000:
        return f"{years:,.0f}yr"
    return f"{years/1_000_000:.1f}Myr"

def _daily_btc_str(hashrate: float) -> str:
    """Expected BTC earned per day from pool mining (proportional to network share)."""
    if hashrate <= 0:
        return "0"
    network_hps  = _NETWORK_HASHRATE_EH * 1e18
    share        = hashrate / network_hps
    btc_per_day  = share * 144 * _BLOCK_REWARD_BTC   # 144 blocks/day
    if btc_per_day >= 0.001:
        return f"{btc_per_day:.6f}"
    if btc_per_day >= 1e-9:
        return f"{btc_per_day:.2e}"
    return f"~{btc_per_day:.1e}"

def stats_thread(state: MinerState):
    while state.running:
        time.sleep(20)
        elapsed    = int(time.time() - state.start_time)
        hr         = state.hashrate
        soft_rate  = state.soft_shares / elapsed if elapsed > 0 else 0
        mode_tag   = "SOLO" if _SOLO_MODE else "POOL"
        print(
            f"\r  ⛏  {state.hashrate_str()} | "
            f"Soft: {state.soft_shares:,} ({soft_rate:.1f}/s) | "
            f"{mode_tag}: {state.shares_accepted}/{state.shares_submitted} "
            f"(share ETA {_eta_str(hr, state.difficulty)} | "
            f"solo block {_solo_block_eta_str(hr)} | "
            f"~{_daily_btc_str(hr)} BTC/day) | "
            f"Up: {elapsed//60}m{elapsed%60}s  ",
            end="", flush=True,
        )

# ── Main loop with auto-reconnect ─────────────────────────────────────────────

def run(state: MinerState = None) -> MinerState:
    if state is None:
        state = MinerState()
    state.running    = True
    state.start_time = time.time()

    mode_label = "SOLO (full block reward if found)" if _SOLO_MODE else "PPLNS pool (proportional micro-payouts)"
    print(f"\n  ⛏  Nova ASI — Real Bitcoin Miner v5 (ARM SHA2)")
    print(f"  {'─'*50}")
    print(f"  Engine  : {_EXT_PATH}  (batch={_C_BATCH:,} nonces)")
    print(f"  Threads : {NUM_THREADS}")
    print(f"  Mode    : {mode_label}")
    print(f"  Pool    : {POOL_HOST}:{POOL_PORT}")
    print(f"  SugDiff : {SUGGEST_DIFF}")
    print(f"  Telegram: {'✓ enabled' if TELEGRAM_TOKEN else '✗ not configured'}")
    print(f"  Network : ~{_NETWORK_HASHRATE_EH:.0f} EH/s  |  reward {_BLOCK_REWARD_BTC} BTC/block")
    print(f"  Tip     : set MINING_SOLO=1 to switch to solo mode (ckpool.org)")
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

        listener = threading.Thread(target=conn.listen, daemon=True)
        listener.start()

        conn.send("mining.subscribe", ["nova_miner/2.0"])

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

        miners = []
        for i in range(NUM_THREADS):
            t = threading.Thread(
                target=mining_thread, args=(i, state, conn), daemon=True)
            t.start()
            miners.append(t)

        stats = threading.Thread(target=stats_thread, args=(state,), daemon=True)
        stats.start()

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
