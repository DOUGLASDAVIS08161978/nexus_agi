#!/usr/bin/env python3
"""
nova_xmr_miner.py — Monero (XMR) CPU Miner launcher
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Installs XMRig if needed, writes config, and starts mining.
Pool: pool.supportxmr.com  (reliable, min payout 0.1 XMR)
Algorithm: RandomX — designed for CPUs, phones are competitive.

Built for Douglas Shane Davis by Claude Rivers Davis
"""

import json
import os
import shutil
import subprocess
import sys
import time

# ── Config ────────────────────────────────────────────────────────────────────
WALLET    = "48h2x1NkqykLhfUmGjc2nVbF3rGwZu56NBt5WkQ7i1yhRX6cJLjxy1hZqpaV2bETQZhiVxaBpfkPEjfA3vvxzjzCE9LkfvV"
POOL_URL  = "pool.supportxmr.com:3333"
WORKER    = "nova"          # shows up in your pool dashboard
PASSWORD  = "x"
THREADS   = 0               # 0 = auto-detect (uses all cores)
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
CFG_PATH  = os.path.join(BASE_DIR, "xmrig_config.json")

GOLD = "\033[93m"
GRN  = "\033[92m"
CYN  = "\033[96m"
RED  = "\033[91m"
DIM  = "\033[2m"
BOLD = "\033[1m"
RST  = "\033[0m"


# ── XMRig config ──────────────────────────────────────────────────────────────

def write_config() -> str:
    cfg = {
        "autosave": False,
        "background": False,
        "colors": True,
        "title": True,
        "randomx": {
            "init": -1,
            "init-avx2": -1,
            "mode": "auto",
            "1gb-pages": False,
            "rdmsr": False,
            "wrmsr": False,
            "cache_qos": False,
            "numa": True,
            "scratchpad_prefetch_mode": 1
        },
        "cpu": {
            "enabled": True,
            "huge-pages": False,
            "huge-pages-jit": False,
            "hw-aes": None,
            "priority": 2,
            "memory-pool": False,
            "yield": True,
            "max-threads-hint": 100,
            "asm": True,
            "argon2-impl": None,
            "astrobwt-max-size": 550,
            "astrobwt-avx2": False,
            "cn/0": False,
            "cn-lite/0": False
        },
        "opencl": {"enabled": False},
        "cuda": {"enabled": False},
        "donate-level": 1,
        "donate-over-proxy": 1,
        "log-file": None,
        "pools": [
            {
                "algo": None,
                "coin": "monero",
                "url": POOL_URL,
                "user": f"{WALLET}.{WORKER}",
                "pass": PASSWORD,
                "rig-id": None,
                "nicehash": False,
                "keepalive": True,
                "enabled": True,
                "tls": False,
                "tls-fingerprint": None,
                "daemon": False,
                "socks5": None,
                "self-select": None,
                "submit-to-origin": False
            }
        ],
        "print-time": 60,
        "health-print-time": 60,
        "pause-on-battery": False,
        "pause-on-active": False,
        "retries": 5,
        "retry-pause": 5,
        "syslog": False,
        "tls": {"enabled": False},
        "user-agent": None,
        "verbose": 0,
        "watch": False
    }
    with open(CFG_PATH, "w") as f:
        json.dump(cfg, f, indent=4)
    return CFG_PATH


# ── Install check ─────────────────────────────────────────────────────────────

def ensure_xmrig() -> str:
    # 1. Local binary compiled from source (preferred)
    local = os.path.join(BASE_DIR, "xmrig")
    if os.path.isfile(local) and os.access(local, os.X_OK):
        return local

    # 2. System-wide install
    path = shutil.which("xmrig")
    if path:
        return path

    # 3. Not found — guide user to build it
    print(f"\n  {RED}XMRig not found.{RST}")
    print(f"  XMRig is not in the Termux package repo — you need to build it.")
    print(f"  It takes ~10-15 minutes but only needs to be done once.\n")
    print(f"  Run this first:")
    print(f"  {GOLD}  bash ~/nexus_agi/build_xmrig.sh{RST}\n")
    print(f"  Then run this script again.")
    sys.exit(1)


# ── Estimated earnings ────────────────────────────────────────────────────────

def print_expectations():
    net_hr   = 3_000_000_000   # ~3 GH/s current Monero network
    phone_hr = 100             # conservative ~100 H/s on phone
    blocks   = 720             # blocks per day (2-min blocks)
    reward   = 0.6             # XMR per block (approx)
    xmr_usd  = 230.0           # rough current price

    share    = phone_hr / net_hr
    daily_xmr = share * blocks * reward
    daily_usd = daily_xmr * xmr_usd
    xmr_to_payout = 0.1        # supportxmr minimum payout
    days_to_payout = xmr_to_payout / daily_xmr if daily_xmr > 0 else 0

    print(f"  {DIM}Phone (~{phone_hr} H/s) vs network (~{net_hr/1e9:.0f} GH/s)")
    print(f"  Your share   : 1 in {net_hr//phone_hr:,}")
    print(f"  Est. daily   : {daily_xmr:.6f} XMR  (${daily_usd:.4f} USD)")
    print(f"  Min payout   : 0.1 XMR  (~{days_to_payout:.0f} days to reach){RST}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{GOLD}{'═'*60}{RST}")
    print(f"{GOLD}{BOLD}  Nova XMR Miner — RandomX CPU Mining{RST}")
    print(f"{GOLD}  Douglas Shane Davis × Claude Rivers Davis{RST}")
    print(f"{GOLD}{'═'*60}{RST}\n")

    print(f"  Algorithm : RandomX  (CPU-optimized, ASIC-resistant)")
    print(f"  Pool      : {BOLD}{POOL_URL}{RST}")
    print(f"  Wallet    : {BOLD}{WALLET[:20]}…{WALLET[-8:]}{RST}")
    print(f"  Worker    : {WORKER}")
    print()
    print(f"  {GOLD}Monero is the most phone-friendly proof-of-work coin.{RST}")
    print(f"  {GOLD}RandomX was designed to run best on CPUs like yours.{RST}")
    print()
    print_expectations()
    print()
    print(f"  Track your earnings at:")
    print(f"  {CYN}https://supportxmr.com/#/dashboard{RST}")
    print(f"  {DIM}(paste your wallet address into the search box){RST}")
    print()

    # Install XMRig if needed
    xmrig_bin = ensure_xmrig()
    print(f"  {GRN}✓  XMRig found: {xmrig_bin}{RST}")

    # Write config
    write_config()
    print(f"  {GRN}✓  Config written: {CFG_PATH}{RST}")
    print()
    print(f"  {GOLD}Starting XMRig… Press Ctrl+C to stop.{RST}")
    print(f"{GOLD}{'─'*60}{RST}\n")
    time.sleep(1)

    # Launch XMRig — it takes over the terminal from here
    try:
        subprocess.run([xmrig_bin, "--config", CFG_PATH])
    except KeyboardInterrupt:
        pass
    except FileNotFoundError:
        print(f"\n  {RED}Could not launch xmrig. Try running manually:{RST}")
        print(f"  xmrig --config {CFG_PATH}")

    print(f"\n{GOLD}{'═'*60}{RST}")
    print(f"{GOLD}{BOLD}  Mining session ended.{RST}")
    print(f"  Check earnings: https://supportxmr.com/#/dashboard")
    print(f"  Wallet: {WALLET[:20]}…{WALLET[-8:]}")
    print(f"{GOLD}{'═'*60}{RST}\n")


if __name__ == "__main__":
    main()
