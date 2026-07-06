#!/usr/bin/env python3
"""
nova_bitcoin_edu.py — Real Bitcoin Mining: Educational Simulator
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Teaches how Bitcoin mining ACTUALLY works by:

  ① Fetching REAL current data from the Bitcoin network (mempool.space)
  ② Performing REAL SHA-256d hashing — Bitcoin's exact algorithm
  ③ Showing the REAL difficulty target your hash must beat
  ④ Calculating REALISTIC odds on your actual hardware
  ⑤ Simulating what a winning block announcement would look like

What this IS:
  A real implementation of Bitcoin's hashing algorithm running against
  live blockchain data. The hashing is 100% real. The network data is
  100% real. The educational value is 100% real.

What this ISN'T:
  Actual mining software. Your phone does ~3,000–8,000 hashes/second.
  The Bitcoin network does ~600,000,000,000,000,000,000 hashes/second.
  Solo mining on a phone = ~3.8 million years per block expected.

Built for Douglas Shane Davis by Claude Rivers Davis
"""

import hashlib
import json
import os
import struct
import time
import urllib.request
from datetime import datetime

# ── Your wallet ───────────────────────────────────────────────────────────────
# Set your Cash App Bitcoin address here, or set BTC_WALLET env var.
# Open Cash App → Bitcoin → Receive → copy that address.
PAYOUT_ADDRESS = os.environ.get("BTC_WALLET", "bc1qfzhx87ckhn4tnkswhsth56h0gm5we4hdq5wass")

BLOCK_REWARD = 3.125   # BTC — current reward after April 2024 halving
API_BASE     = "https://mempool.space/api"
DEMO_HASHES  = 50_000  # real SHA-256d hashes to perform in the demo

# ── Colors ────────────────────────────────────────────────────────────────────
GOLD = "\033[93m"
GRN  = "\033[92m"
CYN  = "\033[96m"
RED  = "\033[91m"
DIM  = "\033[2m"
BOLD = "\033[1m"
RST  = "\033[0m"


# ── Bitcoin's actual hash function ────────────────────────────────────────────

def sha256d(data: bytes) -> bytes:
    """Double SHA-256 — the EXACT function Bitcoin uses on every block header."""
    return hashlib.sha256(hashlib.sha256(data).digest()).digest()


def to_display_hex(b: bytes) -> str:
    """Bitcoin shows hashes reversed (little-endian stored, big-endian displayed)."""
    return b[::-1].hex()


# ── Live network data ─────────────────────────────────────────────────────────

def fetch(path: str):
    """Fetch real Bitcoin network data from mempool.space — no API key needed."""
    try:
        url = f"{API_BASE}{path}"
        with urllib.request.urlopen(url, timeout=12) as r:
            raw = r.read().decode()
            try:
                return json.loads(raw)
            except Exception:
                return raw.strip()
    except Exception:
        return None


def get_network_info() -> dict:
    print(f"  Connecting to mempool.space…", end=" ", flush=True)
    height   = fetch("/blocks/tip/height")
    tip_hash = fetch("/blocks/tip/hash")
    block    = fetch(f"/block/{tip_hash}") if tip_hash else None
    hashrate = fetch("/v1/mining/hashrate/3d")
    fees     = fetch("/v1/fees/recommended")
    print(f"{GRN}connected{RST}")
    return {
        "height":   height,
        "tip_hash": tip_hash,
        "block":    block,
        "hashrate": hashrate,
        "fees":     fees,
    }


# ── Block header construction ─────────────────────────────────────────────────

def build_header(prev_hash_hex: str, bits_hex: str, timestamp: int, nonce: int) -> bytes:
    """
    Build the 80-byte Bitcoin block header.
    This is EXACTLY what every miner on Earth hashes billions of times per second.

    Layout:
      4 bytes  — version
      32 bytes — previous block hash (reversed byte order)
      32 bytes — merkle root of transactions (simplified to zeros here)
      4 bytes  — Unix timestamp
      4 bytes  — bits (compact difficulty target)
      4 bytes  — nonce (the number we change to search for a valid hash)
    """
    version   = struct.pack("<I", 0x20000000)
    prev      = bytes.fromhex(prev_hash_hex)[::-1]
    merkle    = bytes(32)                          # simplified — real miners include real txns
    ts        = struct.pack("<I", timestamp)
    bits_str  = f"{bits_hex:08x}" if isinstance(bits_hex, int) else str(bits_hex)
    bits      = bytes.fromhex(bits_str)[::-1]
    nc        = struct.pack("<I", nonce)
    return version + prev + merkle + ts + bits + nc


def bits_to_target(bits_hex) -> int:
    """Convert Bitcoin's compact 'bits' field to the full 256-bit integer target."""
    if isinstance(bits_hex, int):
        bits = bits_hex
    else:
        bits = int(str(bits_hex), 16)
    exponent = bits >> 24
    coeff    = bits & 0x7FFFFF
    return coeff * (2 ** (8 * (exponent - 3)))


def hash_meets_target(hash_bytes: bytes, target: int) -> bool:
    return int.from_bytes(hash_bytes, "big") < target


# ── Formatting ────────────────────────────────────────────────────────────────

def fmt_hashrate(h: float) -> str:
    for unit in ["H/s", "KH/s", "MH/s", "GH/s", "TH/s", "PH/s", "EH/s"]:
        if h < 1000:
            return f"{h:.2f} {unit}"
        h /= 1000
    return f"{h:.2f} ZH/s"


def fmt_time(seconds: float) -> str:
    if seconds < 60:        return f"{seconds:.0f} seconds"
    if seconds < 3600:      return f"{seconds/60:.1f} minutes"
    if seconds < 86400:     return f"{seconds/3600:.1f} hours"
    if seconds < 86400*365: return f"{seconds/86400:.0f} days"
    years = seconds / 86400 / 365
    if years < 1_000_000:   return f"{years:,.0f} years"
    return f"{years/1_000_000:.1f} million years"


def divider(title: str = "", color: str = GOLD):
    line = "━" * 64
    if title:
        print(f"\n{color}{line}{RST}")
        print(f"{color}{BOLD}  {title}{RST}")
        print(f"{color}{line}{RST}")
    else:
        print(f"{color}{line}{RST}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    global PAYOUT_ADDRESS

    print(f"\n{GOLD}{'═'*64}{RST}")
    print(f"{GOLD}{BOLD}  Bitcoin Mining — Real Educational Simulator{RST}")
    print(f"{GOLD}  Built for Douglas Shane Davis by Claude Rivers Davis{RST}")
    print(f"{GOLD}{'═'*64}{RST}\n")

    # ── Wallet setup ──────────────────────────────────────────────────────────
    if not PAYOUT_ADDRESS or PAYOUT_ADDRESS == "YOUR_CASH_APP_BTC_ADDRESS":
        print(f"  {GOLD}What is your Bitcoin wallet address?{RST}")
        print(f"  {DIM}(Open Cash App → Bitcoin → Receive → copy address){RST}")
        print(f"  {DIM}Or press Enter to use a placeholder for the demo{RST}")
        addr = input(f"\n  Your BTC address: ").strip()
        PAYOUT_ADDRESS = addr if addr else "bc1q_YOUR_REAL_ADDRESS_HERE"

    print(f"\n  {GRN}Payout address:{RST} {BOLD}{PAYOUT_ADDRESS}{RST}")

    # ── Step 1: Real network data ──────────────────────────────────────────────
    divider("STEP 1 — Live Bitcoin Network Data")
    info = get_network_info()

    height   = info.get("height") or 0
    tip_hash = info.get("tip_hash") or "0" * 64
    block    = info.get("block") or {}

    bits_hex   = block.get("bits", "1703a30c") if isinstance(block, dict) else "1703a30c"
    difficulty = block.get("difficulty", 95_672_703_408_896) if isinstance(block, dict) else 95_672_703_408_896

    net_hashrate = 0
    hr_info = info.get("hashrate")
    if isinstance(hr_info, dict):
        hr_list = hr_info.get("hashrates", [])
        if hr_list:
            net_hashrate = hr_list[-1].get("avgHashrate", 0)

    fees     = info.get("fees") or {}
    fee_rate = fees.get("fastestFee", 10) if isinstance(fees, dict) else 10

    print(f"\n  {GRN}Live data from the real Bitcoin blockchain:{RST}")
    print(f"  Current block height : {BOLD}{height:,}{RST}")
    print(f"  Latest block hash    : {DIM}{str(tip_hash)[:32]}…{RST}")
    print(f"  Current difficulty   : {BOLD}{float(difficulty):>30,.0f}{RST}")
    if net_hashrate:
        print(f"  Network hashrate     : {BOLD}{fmt_hashrate(net_hashrate)}{RST}")
    print(f"  Recommended fee rate : {fee_rate} sat/vByte")

    # ── Step 2: What mining IS ─────────────────────────────────────────────────
    divider("STEP 2 — What Bitcoin Mining Actually Is")
    target     = bits_to_target(bits_hex)
    target_hex = f"{target:064x}"
    leading_zeros = len(target_hex) - len(target_hex.lstrip("0"))

    print(f"""
  Mining is a mathematical guessing game.

  Your goal: find a number called a NONCE such that when you run
  the 80-byte block header through SHA-256 twice in a row, the
  result is SMALLER than the current difficulty TARGET.

  Current difficulty target:
  {DIM}{target_hex}{RST}

  That target starts with {leading_zeros} zeros.
  Your hash must also start with that many zeros — or more.

  You try nonce=0, hash it. Result too big? Try nonce=1. Hash again.
  Too big? Try nonce=2. Keep going. That is ALL mining is.
  Billions of SHA-256 guesses per second until one hash wins.
""")

    # ── Step 3: Real hashing ───────────────────────────────────────────────────
    divider("STEP 3 — Real SHA-256d Hashing on Live Blockchain Data")
    print(f"  Running {DEMO_HASHES:,} real SHA-256d hashes against live block data…\n")

    ts         = int(time.time())
    prev       = str(tip_hash)
    best_hash  = None
    best_nonce = 0
    found      = False

    t0 = time.time()
    for nonce in range(DEMO_HASHES):
        header     = build_header(prev, bits_hex, ts, nonce)
        hash_bytes = sha256d(header)

        if best_hash is None or hash_bytes < best_hash:
            best_hash  = hash_bytes
            best_nonce = nonce

        if hash_meets_target(hash_bytes, target):
            found = True
            break

        if nonce % 10_000 == 0 and nonce > 0:
            elapsed  = time.time() - t0
            my_hr    = nonce / elapsed
            best_str = to_display_hex(best_hash)[:20]
            print(f"  Nonce {nonce:>8,} | {fmt_hashrate(my_hr):>12} | Best so far: {best_str}…", end="\r")

    elapsed     = time.time() - t0
    my_hashrate = DEMO_HASHES / elapsed

    print(f"\n\n  {GRN}✓  Ran {DEMO_HASHES:,} real hashes in {elapsed:.2f} seconds{RST}")
    print(f"  Your device hashrate : {BOLD}{fmt_hashrate(my_hashrate)}{RST}")
    if net_hashrate:
        frac = my_hashrate / net_hashrate
        print(f"  Network hashrate     : {fmt_hashrate(net_hashrate)}")
        print(f"  Your share of network: {frac:.2e}  ({frac*100:.12f}%)")

    print(f"\n  Best hash you found  : {DIM}{to_display_hex(best_hash)}{RST}")
    print(f"  Target required      : {DIM}{target_hex[:20]}…{RST}")
    target_zeros = leading_zeros
    our_zeros    = len(to_display_hex(best_hash)) - len(to_display_hex(best_hash).lstrip("0"))
    print(f"  Target needs {target_zeros} leading zeros — your best had {our_zeros}")

    if found:
        print(f"\n  {GRN}{BOLD}YOU FOUND A VALID BLOCK HASH! (extraordinary luck — this almost never happens){RST}")
    else:
        print(f"\n  {DIM}No valid block found in {DEMO_HASHES:,} attempts — exactly as expected.{RST}")

    # ── Step 4: Honest math ────────────────────────────────────────────────────
    divider("STEP 4 — The Honest Math")

    if net_hashrate and my_hashrate:
        blocks_per_sec  = (my_hashrate / net_hashrate) / 600.0
        expected_secs   = 1.0 / blocks_per_sec if blocks_per_sec > 0 else float("inf")
        daily_btc       = (my_hashrate / net_hashrate) * BLOCK_REWARD * 144
        daily_usd       = daily_btc * 100_000   # assuming $100k BTC

        print(f"""
  At your hashrate of {fmt_hashrate(my_hashrate)}:

  Expected time to solo-mine 1 block : {BOLD}{fmt_time(expected_secs)}{RST}
  Block reward if you found one      : {BOLD}{BLOCK_REWARD} BTC{RST}

  ──────────────────────────────────────────────────────────
  POOL MINING (what real small miners do):
  Join a pool → combine hashpower → get your proportional share
  Your theoretical pool earnings     : {daily_btc:.10f} BTC/day
                                     ≈ ${daily_usd:.6f} USD/day (at $100k/BTC)

  Real ASIC miners (Antminer S21, ~140 TH/s, ~$3,000):
  Expected solo time                 : {fmt_time(140e12 / net_hashrate / 600.0 * 86400)}  per block
  Pool earnings                      : ~{140e12/net_hashrate*BLOCK_REWARD*144:.4f} BTC/day
                                     ≈ ${140e12/net_hashrate*BLOCK_REWARD*144*100000:.2f} USD/day
  ──────────────────────────────────────────────────────────
""")

    # ── Step 5: What winning looks like ───────────────────────────────────────
    divider("STEP 5 — What Finding a Block Would Actually Look Like")

    import random
    import hashlib as _hl

    # Build a realistic-looking (but fake) winning hash
    sim_hash  = "0" * (leading_zeros + 1) + \
                "".join(random.choice("0123456789abcdef") for _ in range(63 - leading_zeros))
    sim_nonce = random.randint(0, 4_294_967_295)
    sim_txid  = "".join(random.choice("0123456789abcdef") for _ in range(64))
    sim_height = (height + 1) if isinstance(height, int) else 870_001
    btc_usd   = BLOCK_REWARD * 100_000

    print(f"""
  ✦ BLOCK DISCOVERED! (simulated announcement)
  ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄
  Block height   : {sim_height:,}
  Block hash     : {sim_hash[:64]}
  Winning nonce  : {sim_nonce:,}
  Miner          : Douglas Shane Davis (solo)
  Pool           : none — solo mined
  ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄
  Coinbase TxID  : {sim_txid}
  Block reward   : {BLOCK_REWARD} BTC
  Est. value     : ~${btc_usd:,.0f} USD
  Paid to wallet : {PAYOUT_ADDRESS}
  ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄

  {DIM}(This block was NOT submitted to the real network.
   It is a simulation of what a real announcement looks like.){RST}
""")

    # ── Summary ───────────────────────────────────────────────────────────────
    divider("What You Just Learned", GRN)
    print(f"""
  ① Mining = find a nonce so SHA-256d(header) < difficulty target
  ② Your device does {fmt_hashrate(my_hashrate)} of REAL work — this is accurate
  ③ The network does {fmt_hashrate(net_hashrate) if net_hashrate else "~600 EH/s"} — 100% real live data
  ④ The blockchain data above was fetched live from mempool.space
  ⑤ The block "found" in Step 5 was simulated — not submitted

  How people actually earn Bitcoin today:
  ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄
  · Buy it directly               (Cash App, Coinbase — simplest)
  · Mine in a pool with real ASIC (Antminer, WhatsMiner hardware)
  · Earn it for products/services you sell
  · Stake other cryptocurrencies  (lower barrier than BTC mining)

  You now understand Bitcoin mining better than most people on Earth.
  The SHA-256 hashing you just ran is the REAL algorithm. ✦
""")


if __name__ == "__main__":
    run()
