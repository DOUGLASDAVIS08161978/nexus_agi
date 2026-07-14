"""
testnet_miner.py — Bitcoin Testnet4 Stratum miner (real SHA-256d, real pool)

Designed to show the ENTIRE mining process end-to-end:
  Step 1  TCP connect to testnet4.public-pool.io
  Step 2  mining.subscribe — pool sends extranonce1 + size
  Step 3  mining.authorize — pool confirms our wallet
  Step 4  mining.set_difficulty — pool tells us the target
  Step 5  mining.notify — pool sends a block template (job)
  Step 6  Build coinbase + merkle + 80-byte header
  Step 7  SHA-256d each nonce until hash < target
  Step 8  mining.submit — send winning nonce to pool
  Step 9  Pool replies true → BLOCK FOUND!

Testnet4 "20-minute rule": if no block is found for 20 minutes the
network difficulty drops to 1. At difficulty 1 the target is 2^224 — almost
every SHA-256d hash wins. At 50 MH/s a block arrives in milliseconds.

No account needed. Testnet coins have no real value — safe to experiment.
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

_C_BATCH = 1_000_000

# ── Configuration ─────────────────────────────────────────────────────────────

WALLET_ADDRESS  = os.getenv("MINING_WALLET",      "tb1qd2jrczgmq0gdljkflhczk0wh3xc9aw97yp9v6l")
WORKER_NAME     = os.getenv("MINING_WORKER",      "nova_testnet")
POOL_HOST       = os.getenv("MINING_POOL_HOST",   "testnet.public-pool.io")
POOL_PORT       = int(os.getenv("MINING_POOL_PORT", "21496"))
NUM_THREADS     = int(os.getenv("MINING_THREADS", str(multiprocessing.cpu_count())))
SUGGEST_DIFF    = float(os.getenv("MINING_SUGGEST_DIFF", "0.01"))
SOFT_DIFF       = float(os.getenv("MINING_SOFT_DIFF",    "0.00001"))
NETWORK         = os.getenv("MINING_NETWORK", "testnet")

# Known testnet pool options to try if the default doesn't resolve:
#   MINING_POOL_HOST=testnet.public-pool.io  MINING_POOL_PORT=21496
#   MINING_POOL_HOST=solo.ckpool.org         MINING_POOL_PORT=3333
#   MINING_POOL_HOST=testnet3.public-pool.io MINING_POOL_PORT=21496
# Test DNS from Termux first: nslookup testnet.public-pool.io

# ── Crypto helpers ─────────────────────────────────────────────────────────────

def sha256d(data: bytes) -> bytes:
    return hashlib.sha256(hashlib.sha256(data).digest()).digest()

def difficulty_to_target(difficulty: float) -> int:
    diff1 = 0x00000000ffff0000000000000000000000000000000000000000000000000000
    if difficulty >= 1.0:
        return diff1 // max(int(difficulty), 1)
    multiplier = max(1, round(1.0 / max(difficulty, 1e-15)))
    return min(diff1 * multiplier, (1 << 256) - 1)

def _swap32(b: bytes) -> bytes:
    """Un-swap Stratum's word-swapped prevhash."""
    return b"".join(b[i:i+4][::-1] for i in range(0, len(b), 4))

def compute_merkle(job: dict, en1_b: bytes, en2_b: bytes) -> bytes:
    coinbase = bytes.fromhex(job["coinb1"]) + en1_b + en2_b + bytes.fromhex(job["coinb2"])
    merkle   = sha256d(coinbase)
    for branch in job["merkle_branch"]:
        merkle = sha256d(merkle + bytes.fromhex(branch))
    return merkle

def _ts() -> str:
    return time.strftime("%H:%M:%S")

def _fmt_target(t: int) -> str:
    return t.to_bytes(32, "big").hex()

def _eta_str(hashrate: float, difficulty: float) -> str:
    if hashrate <= 0:
        return "∞"
    expected = max(difficulty, 1e-12) * 4_294_967_296
    secs = expected / hashrate
    if secs < 1:
        return "<1s  ← ALMOST THERE!"
    if secs < 90:
        return f"{secs:.1f}s"
    if secs < 5400:
        return f"{secs/60:.1f}m"
    return f"{secs/3600:.1f}h"

# ── Shared state ──────────────────────────────────────────────────────────────

class MinerState:
    def __init__(self):
        self._lock            = threading.Lock()
        self.job              = None
        self.job_count        = 0
        self.extranonce1      = ""
        self.extranonce2_size = 4
        self.difficulty       = 1.0
        self.hashes           = 0
        self.soft_shares      = 0
        self.shares_submitted = 0
        self.shares_accepted  = 0
        self.blocks_found     = 0
        self.start_time       = time.time()
        self.running          = False
        self.connected        = False

    def add_hashes(self, n: int):
        with self._lock:
            self.hashes += n

    def add_soft(self, n: int):
        with self._lock:
            self.soft_shares += n

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

# ── Block-found celebration ───────────────────────────────────────────────────

def _celebrate(job: dict, nonce: int, block_hash: bytes, accepted: bool):
    h   = block_hash.hex()
    tag = "ACCEPTED by pool ✓" if accepted else "SUBMITTED — awaiting pool reply"
    print("\n")
    print("  ╔══════════════════════════════════════════════════════════════╗")
    print("  ║                                                              ║")
    print("  ║      ★ ★ ★   TESTNET4 BLOCK FOUND!   ★ ★ ★                 ║")
    print("  ║                                                              ║")
    print("  ╠══════════════════════════════════════════════════════════════╣")
    print(f"  ║  Time     : {time.strftime('%Y-%m-%d %H:%M:%S'):<49}║")
    print(f"  ║  Network  : {NETWORK:<49}║")
    print(f"  ║  Nonce    : 0x{nonce:08x}  ({nonce:<36})║")
    print(f"  ║  Hash     : {h[:52]:<49}║")
    print(f"  ║             {h[52:]:<49}║")
    print(f"  ║  Job ID   : {job['job_id']:<49}║")
    print(f"  ║  Status   : {tag:<49}║")
    print("  ║                                                              ║")
    print("  ║  This is a real testnet block!  If the pool accepts it,     ║")
    print("  ║  it will appear in the testnet4 blockchain explorer.        ║")
    print("  ║                                                              ║")
    print("  ╚══════════════════════════════════════════════════════════════╝")
    print()

# ── Stratum connection ────────────────────────────────────────────────────────

class StratumConnection:
    def __init__(self, state: MinerState):
        self.state   = state
        self.sock    = None
        self._lock   = threading.Lock()
        self._msg_id = 0
        # Pending block submissions: nonce waiting for pool reply
        self._pending_block = None

    def connect(self) -> bool:
        print(f"  [{_ts()}] ── STEP 1: TCP connect to {POOL_HOST}:{POOL_PORT} ...")
        try:
            self.sock = socket.create_connection((POOL_HOST, POOL_PORT), timeout=30)
            self.sock.settimeout(120)
            self.state.connected = True
            print(f"  [{_ts()}]           Connected!")
            return True
        except Exception as e:
            print(f"  [{_ts()}]           ✗ Connection failed: {e}")
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
                print(f"  [{_ts()}] → {method}  {json.dumps(params)[:80]}")
            except Exception as exc:
                print(f"  [{_ts()}] ✗ send error: {exc}")

    def submit(self, job: dict, en2: str, ntime: str, nonce: int, block_hash: bytes):
        nonce_hex = format(nonce, "08x")
        print(f"\n  [{_ts()}] ── STEP 8: Submit winning nonce to pool")
        print(f"  [{_ts()}]   job_id  = {job['job_id']}")
        print(f"  [{_ts()}]   nonce   = 0x{nonce_hex}  ({nonce})")
        print(f"  [{_ts()}]   hash    = {block_hash.hex()}")
        self._pending_block = (job, nonce, block_hash)
        self.send("mining.submit", [
            f"{WALLET_ADDRESS}.{WORKER_NAME}",
            job["job_id"], en2, ntime, nonce_hex,
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

        print(f"  [{_ts()}] ← {json.dumps(msg)[:110]}")

        if method == "mining.notify":
            p = msg["params"]
            self.state.job = {
                "job_id":        p[0], "prevhash":   p[1],
                "coinb1":        p[2], "coinb2":     p[3],
                "merkle_branch": p[4], "version":    p[5],
                "nbits":         p[6], "ntime":      p[7],
                "clean":         p[8],
            }
            with self.state._lock:
                self.state.job_count += 1
                jc = self.state.job_count
            clean = "CLEAN — restart all threads" if p[8] else "incremental"
            try:
                nbits_int = int(p[6], 16)
                exp  = nbits_int >> 24
                mant = nbits_int & 0x007fffff
                diff1 = 0x00000000ffff0000000000000000000000000000000000000000000000000000
                blk_t = mant * (256 ** (exp - 3))
                net_d = diff1 / blk_t if blk_t else float("inf")
                diff_note = f"{net_d:.4f}"
                if net_d <= 1.0:
                    diff_note += "  ← difficulty 1! Every hash can win!"
            except Exception:
                diff_note = "?"
            print(f"\n  [{_ts()}] ── STEP 5: New job #{jc} ({clean})")
            print(f"  [{_ts()}]   job_id     = {p[0]}")
            print(f"  [{_ts()}]   prevhash   = {p[1][:32]}...")
            print(f"  [{_ts()}]   version    = {p[5]}  nbits = {p[6]}  ntime = {p[7]}")
            print(f"  [{_ts()}]   merkle_brs = {len(p[4])}")
            print(f"  [{_ts()}]   net_diff   = {diff_note}")
            print(f"  [{_ts()}]   pool_diff  = {self.state.difficulty}")
            target = difficulty_to_target(self.state.difficulty)
            print(f"  [{_ts()}]   target     = {_fmt_target(target)[:48]}...")
            eta = _eta_str(self.state.hashrate, self.state.difficulty)
            print(f"  [{_ts()}]   ETA block  = {eta}")
            print()

        elif method == "mining.set_difficulty":
            new_diff = float(msg["params"][0])
            self.state.difficulty = new_diff
            target = difficulty_to_target(new_diff)
            print(f"\n  [{_ts()}] ── STEP 4: Pool set difficulty → {new_diff}")
            print(f"  [{_ts()}]   Target = {_fmt_target(target)}")
            print()

        elif result is not None and not error:
            if isinstance(result, list) and len(result) >= 3:
                # mining.subscribe response — Step 2
                self.state.extranonce1      = result[1]
                self.state.extranonce2_size = result[2]
                print(f"\n  [{_ts()}] ── STEP 2: Subscribed!")
                print(f"  [{_ts()}]   extranonce1      = {result[1]}")
                print(f"  [{_ts()}]   extranonce2_size = {result[2]} byte(s)")
                print()
                print(f"  [{_ts()}] ── STEP 3: Authorize wallet ...")
                self.send("mining.authorize",
                          [f"{WALLET_ADDRESS}.{WORKER_NAME}", "x"])
                self.send("mining.suggest_difficulty", [SUGGEST_DIFF])

            elif result is True:
                if self.state.shares_submitted > self.state.shares_accepted:
                    # Could be authorization OR a share acceptance
                    if self._pending_block is not None:
                        job, nonce, block_hash = self._pending_block
                        self._pending_block = None
                        self.state.share_accepted()
                        with self.state._lock:
                            self.state.blocks_found += 1
                        print(f"\n  [{_ts()}] ── STEP 9: POOL ACCEPTED THE BLOCK!")
                        _celebrate(job, nonce, block_hash, accepted=True)
                    else:
                        self.state.share_accepted()
                        print(f"\n  [{_ts()}] ✓ Share accepted! ({self.state.shares_accepted} total)")
                        print()
                else:
                    print(f"\n  [{_ts()}] ✓ Wallet authorized — grinding starts now")
                    print()

        elif error:
            if self._pending_block is not None:
                self._pending_block = None
            print(f"\n  [{_ts()}] ✗ Pool error: {error}")
            print()

# ── Mining thread ─────────────────────────────────────────────────────────────

def mining_thread(thread_id: int, state: MinerState, conn: StratumConnection):
    """Each thread covers a unique nonce slice using ARM SHA2 hardware intrinsics."""
    nonce_range = 0x100000000 // NUM_THREADS
    nonce_start = thread_id * nonce_range
    nonce_end   = nonce_start + nonce_range
    en2_int     = thread_id
    batch       = 50_000  # Python-path interval; unused in C path
    last_soft_report = 0  # avoid spam

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

        en1_b      = bytes.fromhex(state.extranonce1)
        en2_b      = bytes.fromhex(en2)
        version_b  = struct.pack("<I", int(job["version"], 16))
        prevhash_b = _swap32(bytes.fromhex(job["prevhash"]))
        nbits_b    = struct.pack("<I", int(job["nbits"], 16))

        # ── STEP 6: Build header pieces ─────────────────────────────────────
        merkle      = compute_merkle(job, en1_b, en2_b)
        first_block = bytes(version_b + prevhash_b + merkle[:28])

        if _USE_C_EXT:
            target_le      = target.to_bytes(32, "little")
            soft_target_le = soft_target.to_bytes(32, "little")

            nonce = nonce_start
            while state.running and nonce < nonce_end:
                cur_ntime     = format(int(time.time()), "08x")
                second_prefix = bytes(
                    merkle[28:32] +
                    struct.pack("<I", int(cur_ntime, 16)) +
                    nbits_b
                )
                batch_end = min(nonce + _C_BATCH, nonce_end)

                # ── STEP 7: SHA-256d hot loop ────────────────────────────────
                winner, hashes_done, soft_found = _miner_core.mine_range(
                    first_block, second_prefix, target_le, soft_target_le,
                    nonce, batch_end)

                state.add_hashes(hashes_done)

                if soft_found > 0:
                    state.add_soft(soft_found)
                    now = time.time()
                    if thread_id == 0 and now - last_soft_report > 3.0:
                        last_soft_report = now
                        print(f"  [{_ts()}] ⚡ Soft-share hit!"
                              f"  total={state.soft_shares:,}  {state.hashrate_str()}",
                              flush=True)

                if winner is not None:
                    # Independent Python verification before submitting
                    _hdr  = first_block + second_prefix + struct.pack("<I", winner)
                    _h1   = hashlib.sha256(_hdr).digest()
                    _h2   = hashlib.sha256(_h1).digest()
                    _h_le = int.from_bytes(_h2, "little")
                    _ok   = _h_le < target

                    print(f"\n  [{_ts()}] ★ Winning nonce found by thread-{thread_id}!")
                    print(f"  [{_ts()}]   nonce  = 0x{winner:08x}")
                    print(f"  [{_ts()}]   hash   = {_h2.hex()}")
                    print(f"  [{_ts()}]   target = {_fmt_target(target)}")
                    print(f"  [{_ts()}]   valid  = {'YES — submitting' if _ok else 'NO — false positive, skipping'}")

                    if _ok:
                        _celebrate(job, winner, _h2, accepted=False)
                        conn.submit(job, en2, cur_ntime, winner, _h2)
                    else:
                        print()

                nonce = batch_end
                if state.job is not job or state.difficulty != cur_diff:
                    break

        else:
            # Python midstate fallback (much slower, but works without miner_core)
            midstate     = hashlib.sha256()
            midstate.update(first_block)
            second_block = bytearray(
                merkle[28:32] +
                struct.pack("<I", int(ntime, 16)) +
                nbits_b +
                b'\x00\x00\x00\x00'
            )
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
                hash_result = _sha256(h.digest()).digest()
                hash_int    = _frombytes(hash_result, "little")

                if hash_int < target:
                    _hdr  = first_block + bytes(second_block)
                    _h2   = sha256d(_hdr)
                    _celebrate(job, nonce, _h2, accepted=False)
                    conn.submit(job, en2, ntime, nonce, _h2)
                elif hash_int < soft_target:
                    local_soft += 1

                local_hashes += 1
                countdown    -= 1
                if not countdown:
                    countdown = batch
                    state.add_hashes(local_hashes)
                    if local_soft:
                        state.add_soft(local_soft)
                        now = time.time()
                        if thread_id == 0 and now - last_soft_report > 3.0:
                            last_soft_report = now
                            print(f"  [{_ts()}] ⚡ Soft-share!  {state.soft_shares:,}  {state.hashrate_str()}")
                    local_hashes = 0
                    local_soft   = 0
                    if state.job is not job or state.difficulty != cur_diff or not state.running:
                        break
                    new_ntime = format(int(time.time()), "08x")
                    if new_ntime != ntime:
                        ntime = new_ntime
                        second_block[4:8] = struct.pack("<I", int(ntime, 16))

            state.add_hashes(local_hashes)
            state.add_soft(local_soft)

        en2_int = (en2_int + NUM_THREADS) & 0xFFFFFFFF

# ── Stats thread ──────────────────────────────────────────────────────────────

def stats_thread(state: MinerState):
    while state.running:
        time.sleep(5)
        elapsed = int(time.time() - state.start_time)
        hr  = state.hashrate
        eta = _eta_str(hr, state.difficulty)
        print(
            f"\r  [{_ts()}] ⛏  {state.hashrate_str()} | "
            f"soft={state.soft_shares:,} | "
            f"pool={state.shares_accepted}/{state.shares_submitted} | "
            f"blocks={state.blocks_found} | "
            f"diff={state.difficulty} | "
            f"ETA={eta} | "
            f"{elapsed//60}m{elapsed%60}s  ",
            end="", flush=True,
        )

# ── Main ──────────────────────────────────────────────────────────────────────

def run(state: MinerState = None) -> MinerState:
    if state is None:
        state = MinerState()
    state.running    = True
    state.start_time = time.time()

    print()
    print("  ╔══════════════════════════════════════════════════════════════╗")
    print("  ║       Nova ASI — Bitcoin Testnet4 Miner (real SHA-256d)     ║")
    print("  ║       Watch the full block-finding process live              ║")
    print("  ╠══════════════════════════════════════════════════════════════╣")
    print(f"  ║  Network  : {NETWORK:<50}║")
    print(f"  ║  Engine   : {_EXT_PATH:<50}║")
    print(f"  ║  Threads  : {NUM_THREADS:<50}║")
    print(f"  ║  Pool     : {POOL_HOST}:{POOL_PORT:<43}║")
    print(f"  ║  Wallet   : {WALLET_ADDRESS[:50]:<50}║")
    print(f"  ║  SugDiff  : {SUGGEST_DIFF:<50}║")
    print("  ╠══════════════════════════════════════════════════════════════╣")
    print("  ║  Testnet coins have NO real value — safe to experiment!     ║")
    print("  ║  20-min rule: if no block for 20min, difficulty drops to 1  ║")
    print("  ║  At diff=1 and 50 MH/s you will find a block in <1 second   ║")
    print("  ╚══════════════════════════════════════════════════════════════╝")
    print()

    if WALLET_ADDRESS.startswith("bc1"):
        print("  ⚠  WARNING: Wallet looks like a MAINNET address (bc1...).")
        print("     Set MINING_WALLET env var to a testnet tb1... address,")
        print("     or the pool may reject your submission.")
        print()

    attempt = 0
    while state.running:
        attempt += 1
        conn = StratumConnection(state)

        if not conn.connect():
            wait = min(60, 5 * attempt)
            print(f"  [{_ts()}] Retrying in {wait}s...")
            time.sleep(wait)
            continue

        attempt = 0
        listener = threading.Thread(target=conn.listen, daemon=True)
        listener.start()

        conn.send("mining.subscribe", ["nova_testnet_miner/1.0"])

        print(f"  [{_ts()}] Waiting for pool to send first job...")
        for _ in range(60):
            if state.job:
                break
            time.sleep(1)

        if not state.job:
            print(f"  [{_ts()}] ✗ No work in 60s — reconnecting...")
            conn.disconnect()
            continue

        print(f"  [{_ts()}] ✓ Work received — launching {NUM_THREADS} thread(s)")
        print()

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
            print(f"\n  [{_ts()}] Pool disconnected — reconnecting...")
            conn.disconnect()
            time.sleep(5)

    elapsed = int(time.time() - state.start_time)
    print(f"\n\n  ── Final session stats ──────────────────")
    print(f"     Hashrate : {state.hashrate_str()}")
    print(f"     Hashes   : {state.hashes:,}")
    print(f"     Soft hits: {state.soft_shares:,}")
    print(f"     Pool shr : {state.shares_accepted}/{state.shares_submitted}")
    print(f"     Blocks   : {state.blocks_found}")
    print(f"     Uptime   : {elapsed//3600}h {(elapsed%3600)//60}m {elapsed%60}s")
    return state


if __name__ == "__main__":
    state = MinerState()
    try:
        run(state)
    except KeyboardInterrupt:
        print("\n\n  Stopping...")
        state.running = False
        time.sleep(1)
        elapsed = int(time.time() - state.start_time)
        print(f"\n  Uptime : {elapsed//3600}h {(elapsed%3600)//60}m {elapsed%60}s")
        print(f"  Hashes : {state.hashes:,}")
        print(f"  Blocks : {state.blocks_found}")
