"""
nova_cap_miner.py — Nova capability: real Bitcoin pool miner integration

Nova can start, stop, and monitor the real Bitcoin miner.
She reports stats in conversation and sends Telegram alerts on share acceptance.

Commands Nova understands:
  /miner start    — launch the miner in background
  /miner stop     — stop mining
  /miner status   — show hashrate, shares, uptime
  /miner wallet   — show the payout wallet address
"""

import threading
import time
import os
from typing import Optional

# ── State ─────────────────────────────────────────────────────────────────────

_miner_state  = None   # MinerState from real_pool_miner
_miner_thread = None   # background thread running the miner
_lock         = threading.Lock()

# ── Start / stop ──────────────────────────────────────────────────────────────

def start_miner() -> str:
    global _miner_state, _miner_thread

    with _lock:
        if _miner_state is not None and _miner_state.running:
            return f"⛏ Miner is already running — {_miner_state.hashrate_str()}"

        try:
            from real_pool_miner import MinerState, run, WALLET_ADDRESS, POOL_HOST, NUM_THREADS
        except ImportError as e:
            return f"✗ Could not import real_pool_miner: {e}"

        _miner_state = MinerState()
        state_ref = _miner_state

        def _bg():
            try:
                run(state_ref)
            except Exception as exc:
                print(f"  [Miner] Error: {exc}")

        _miner_thread = threading.Thread(target=_bg, daemon=True, name="nova-miner")
        _miner_thread.start()

    # Give it a moment to connect
    time.sleep(3)

    return (f"⛏ Nova Miner started!\n"
            f"  Pool    : {POOL_HOST}\n"
            f"  Threads : {NUM_THREADS} CPU cores\n"
            f"  Wallet  : {WALLET_ADDRESS}\n"
            f"  This is real mining — every hash counts.\n"
            f"  I'll notify you in Telegram when a share is accepted.")


def stop_miner() -> str:
    global _miner_state

    with _lock:
        if _miner_state is None or not _miner_state.running:
            return "Miner isn't running."

        _miner_state.running = False
        summary = _miner_state.summary()
        _miner_state = None

    return f"⛏ Miner stopped.\n\n{summary}"


def miner_status() -> str:
    with _lock:
        if _miner_state is None or not _miner_state.running:
            return "⛏ Miner is not running. Say '/miner start' to begin."
        return _miner_state.summary()


def is_running() -> bool:
    return _miner_state is not None and _miner_state.running


# ── Nova command handler (/miner ...) ─────────────────────────────────────────

def run_command(arg: str) -> str:
    arg = (arg or "").strip().lower()

    if arg in ("", "status"):
        return miner_status()

    if arg == "start":
        return start_miner()

    if arg == "stop":
        return stop_miner()

    if arg == "wallet":
        try:
            from real_pool_miner import WALLET_ADDRESS
            return (f"Payout wallet: {WALLET_ADDRESS}\n"
                    f"This is your CashApp Bitcoin address.\n"
                    f"Mining rewards deposit here automatically when shares are accepted.")
        except ImportError:
            return "Could not read wallet address."

    if arg == "help":
        return (
            "⛏ Nova Miner commands:\n"
            "  /miner start   — start mining in background\n"
            "  /miner stop    — stop mining\n"
            "  /miner status  — hashrate, shares, uptime\n"
            "  /miner wallet  — show payout address\n\n"
            "Every hash is real SHA-256d proof-of-work submitted to the pool.\n"
            "I notify you on Telegram when a share is accepted."
        )

    return f"Unknown miner command: '{arg}'. Try '/miner help'."


# ── Nova module interface ─────────────────────────────────────────────────────

def status() -> dict:
    running = is_running()
    result  = {
        "items":      1 if running else 0,
        "confidence": 1.0,
        "accuracy":   1.0,
        "available":  True,
        "running":    running,
    }
    if running and _miner_state:
        result["hashrate"]        = _miner_state.hashrate_str()
        result["shares_accepted"] = _miner_state.shares_accepted
        result["hashes"]          = _miner_state.hashes
    return result
