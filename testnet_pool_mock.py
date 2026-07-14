"""
testnet_pool_mock.py — Local mock Stratum pool for Bitcoin mining demo

Runs a complete Stratum v1 server on localhost. Every protocol message
is printed from both the pool's and miner's perspective, so you can watch
the full handshake + block-finding process without needing any internet.

Usage:
  Terminal 1:  python3 testnet_pool_mock.py
  Terminal 2:  MINING_POOL_HOST=localhost MINING_POOL_PORT=3333 python3 testnet_miner.py

At MOCK_DIFFICULTY=1.0 and 50 MH/s × 8 threads you'll find a "block" roughly
every 10 seconds. Crank MOCK_DIFFICULTY down to 0.01 for every-second blocks.
"""

import socket
import json
import threading
import time
import os

MOCK_HOST       = "0.0.0.0"
MOCK_PORT       = int(os.getenv("MOCK_PORT", "3333"))
MOCK_DIFFICULTY = float(os.getenv("MOCK_DIFFICULTY", "1.0"))

EXTRANONCE1      = "aabbccdd"
EXTRANONCE2_SIZE = 4

# Minimal but structurally valid coinbase pieces (hex).
# miner will sha256d(coinb1 + en1 + en2 + coinb2) to get the merkle root.
COINB1 = (
    "01000000"          # tx version
    "01"                # 1 input
    "0000000000000000000000000000000000000000000000000000000000000000"  # prev txid (coinbase)
    "ffffffff"          # prev vout
    "08"                # script length
    "03dead00"          # height push (BIP34: height=0xdead)
    "deadbeef"          # extra nonce space marker
)
COINB2 = (
    "ffffffff"          # sequence
    "01"                # 1 output
    "00f2052a01000000"  # value: 50 BTC in satoshis (little-endian)
    "43"                # script length
    "4104678afdb0fe5548271967f1a67130b7105cd6a828e03909a67962e0ea1f6"
    "1deb649f6bc3f4cef38c4f35504e51ec112de5c384df7ba0b8d578a4c702b6b"
    "f11d5f"
    "ac"                # OP_CHECKSIG
    "00000000"          # locktime
)

# Testnet4-ish block header fields
JOB_VERSION = "20000000"
JOB_NBITS   = "1d00ffff"   # matches difficulty 1 (diff1 target)
PREVHASH    = "0000000000000000000000000000000000000000000000000000000000000000"

_job_counter = 0
_total_blocks = 0
_session_start = time.time()
_lock = threading.Lock()


def _ts():
    return time.strftime("%H:%M:%S")


def _new_job_id():
    global _job_counter
    _job_counter += 1
    return f"job{_job_counter:04d}"


def _make_notify(clean: bool) -> dict:
    return {
        "id": None,
        "method": "mining.notify",
        "params": [
            _new_job_id(),
            PREVHASH,
            COINB1,
            COINB2,
            [],                               # merkle branches (empty = coinbase IS root)
            JOB_VERSION,
            JOB_NBITS,
            format(int(time.time()), "08x"),  # ntime
            clean,
        ],
    }


def _celebrate_pool(nonce_hex: str, job_id: str, blocks: int):
    elapsed = int(time.time() - _session_start)
    print()
    print("  ╔══════════════════════════════════════════════════════════════╗")
    print("  ║       [MOCK POOL]  BLOCK SUBMITTED AND ACCEPTED             ║")
    print(f"  ║       Block #{blocks:<55}║")
    print(f"  ║       Nonce  : 0x{nonce_hex:<52}║")
    print(f"  ║       Job    : {job_id:<52}║")
    print(f"  ║       Time   : {time.strftime('%H:%M:%S'):<52}║")
    print(f"  ║       Up     : {elapsed//60}m{elapsed%60}s{'':48}║")
    print("  ║                                                              ║")
    print("  ║  This is what winning looks like. In real mining the pool   ║")
    print("  ║  verifies the hash, broadcasts the block to the network,    ║")
    print("  ║  and the reward lands in your wallet.                       ║")
    print("  ╚══════════════════════════════════════════════════════════════╝")
    print()


def handle_miner(conn, addr):
    global _total_blocks
    peer = f"{addr[0]}:{addr[1]}"
    print(f"\n  [{_ts()}] [MOCK POOL] ✓ Miner connected from {peer}")

    buf     = ""
    shares  = 0
    blocks  = 0
    authed  = False

    def send(obj):
        raw = json.dumps(obj) + "\n"
        try:
            conn.sendall(raw.encode())
            print(f"  [{_ts()}] [MOCK POOL] → {json.dumps(obj)[:110]}")
        except Exception:
            pass

    try:
        conn.settimeout(300)
        while True:
            chunk = conn.recv(4096).decode("utf-8", errors="replace")
            if not chunk:
                break
            buf += chunk
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                line = line.strip()
                if not line:
                    continue

                msg    = json.loads(line)
                method = msg.get("method", "")
                mid    = msg.get("id", 0)
                print(f"  [{_ts()}] [MOCK POOL] ← {json.dumps(msg)[:110]}")

                if method == "mining.subscribe":
                    # Step 2: send subscription details
                    send({
                        "id": mid,
                        "result": [
                            [["mining.notify", "n1"], ["mining.set_difficulty", "d1"]],
                            EXTRANONCE1,
                            EXTRANONCE2_SIZE,
                        ],
                        "error": None,
                    })
                    # Step 4: set difficulty
                    send({
                        "id": None,
                        "method": "mining.set_difficulty",
                        "params": [MOCK_DIFFICULTY],
                    })
                    # Step 5: send first job
                    send(_make_notify(clean=True))

                elif method == "mining.authorize":
                    wallet = msg.get("params", ["?"])[0]
                    print(f"  [{_ts()}] [MOCK POOL] Authorizing: {wallet}")
                    send({"id": mid, "result": True, "error": None})
                    authed = True

                elif method == "mining.suggest_difficulty":
                    # Log it but keep our configured difficulty
                    suggested = msg.get("params", [MOCK_DIFFICULTY])[0]
                    print(f"  [{_ts()}] [MOCK POOL]   Miner suggested diff={suggested},"
                          f" pool keeping {MOCK_DIFFICULTY}")

                elif method == "mining.submit":
                    params    = msg.get("params", [])
                    job_id    = params[1] if len(params) > 1 else "?"
                    nonce_hex = params[4] if len(params) > 4 else "?"
                    nonce     = int(nonce_hex, 16) if nonce_hex != "?" else 0

                    shares += 1
                    blocks += 1
                    with _lock:
                        _total_blocks += 1
                        total = _total_blocks

                    print(f"\n  [{_ts()}] [MOCK POOL] ★ SHARE RECEIVED #{shares}")
                    print(f"  [{_ts()}]   nonce = 0x{nonce:08x}  ({nonce})")
                    _celebrate_pool(nonce_hex, job_id, total)

                    # Accept and send fresh work
                    send({"id": mid, "result": True, "error": None})
                    send(_make_notify(clean=False))

    except Exception as e:
        print(f"  [{_ts()}] [MOCK POOL] {peer} gone: {e}")
    finally:
        conn.close()
        print(f"  [{_ts()}] [MOCK POOL] Session ended — shares: {shares}  blocks: {blocks}")


def run():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((MOCK_HOST, MOCK_PORT))
    srv.listen(10)

    print()
    print("  ╔══════════════════════════════════════════════════════════════╗")
    print("  ║         Nova ASI — Local Mock Stratum Pool                  ║")
    print("  ║         Complete Stratum v1 protocol — no internet needed   ║")
    print("  ╠══════════════════════════════════════════════════════════════╣")
    print(f"  ║  Listening  : localhost:{MOCK_PORT:<39}║")
    print(f"  ║  Difficulty : {MOCK_DIFFICULTY:<50}║")
    eta_secs = MOCK_DIFFICULTY * 4_294_967_296 / (50_000_000 * 8)
    if eta_secs < 60:
        eta_str = f"~{eta_secs:.0f}s between blocks"
    else:
        eta_str = f"~{eta_secs/60:.1f}m between blocks"
    print(f"  ║  Block rate : {eta_str:<50}║")
    print("  ╠══════════════════════════════════════════════════════════════╣")
    print("  ║  Open a SECOND Termux session and run:                      ║")
    print("  ║                                                              ║")
    print(f"  ║  MINING_POOL_HOST=localhost \\                               ║")
    print(f"  ║  MINING_POOL_PORT={MOCK_PORT} \\                                 ║")
    print(f"  ║  python3 testnet_miner.py                                   ║")
    print("  ║                                                              ║")
    print("  ║  Watch both windows — you'll see every message exchanged    ║")
    print("  ║  between the miner and this pool in real time.              ║")
    print("  ╚══════════════════════════════════════════════════════════════╝")
    print()
    print(f"  [{_ts()}] Waiting for miner to connect...")

    while True:
        try:
            conn, addr = srv.accept()
            t = threading.Thread(target=handle_miner, args=(conn, addr), daemon=True)
            t.start()
        except KeyboardInterrupt:
            print(f"\n\n  [{_ts()}] Mock pool stopped.")
            print(f"  Total blocks found this session: {_total_blocks}")
            break
        except Exception as e:
            print(f"  [{_ts()}] Accept error: {e}")


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        print("\n  Stopped.")
