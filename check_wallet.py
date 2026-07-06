#!/usr/bin/env python3
"""
check_wallet.py — Bitcoin wallet balance checker
Uses mempool.space public API — no API key needed.
"""

import json
import urllib.request
import urllib.error

WALLET = "bc1qfzhx87ckhn4tnkswhsth56h0gm5we4hdq5wass"
API    = "https://mempool.space/api"

GOLD = "\033[93m"
GRN  = "\033[92m"
CYN  = "\033[96m"
DIM  = "\033[2m"
BOLD = "\033[1m"
RST  = "\033[0m"


def fetch(path: str) -> dict:
    url = f"{API}{path}"
    req = urllib.request.Request(url, headers={"User-Agent": "nova-wallet-checker/1.0"})
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read().decode())


def satoshi_to_btc(sats: int) -> float:
    return sats / 1_000_000_00


def main():
    print(f"\n{GOLD}{'═'*56}{RST}")
    print(f"{GOLD}{BOLD}  Nova Wallet Balance Checker{RST}")
    print(f"{GOLD}{'═'*56}{RST}")
    print(f"\n  Wallet : {BOLD}{WALLET}{RST}")
    print(f"  Source : mempool.space (live blockchain data)\n")

    # ── Fetch wallet data ─────────────────────────────────────────────────────
    try:
        data = fetch(f"/address/{WALLET}")
    except urllib.error.URLError as e:
        print(f"  Network error: {e}")
        return

    # ── Parse stats ───────────────────────────────────────────────────────────
    funded   = data.get("chain_stats", {}).get("funded_txo_sum", 0)
    spent    = data.get("chain_stats", {}).get("spent_txo_sum", 0)
    balance  = funded - spent
    tx_count = data.get("chain_stats", {}).get("tx_count", 0)

    mem_in   = data.get("mempool_stats", {}).get("funded_txo_sum", 0)
    mem_out  = data.get("mempool_stats", {}).get("spent_txo_sum", 0)
    pending  = mem_in - mem_out

    btc_bal     = satoshi_to_btc(balance)
    btc_pending = satoshi_to_btc(pending)
    btc_total   = btc_bal + btc_pending

    # ── Fetch live BTC price ──────────────────────────────────────────────────
    btc_usd = 62_000.0
    try:
        prices  = fetch("/v1/prices")
        btc_usd = float(prices.get("USD", 62_000.0))
    except Exception:
        pass

    usd_bal     = btc_bal     * btc_usd
    usd_pending = btc_pending * btc_usd
    usd_total   = btc_total   * btc_usd

    # ── Display ───────────────────────────────────────────────────────────────
    print(f"  {BOLD}Confirmed Balance{RST}")
    print(f"    {GRN}{btc_bal:.8f} BTC{RST}   (${usd_bal:,.2f} USD)")

    if pending != 0:
        print(f"\n  {BOLD}Unconfirmed (mempool){RST}")
        print(f"    {CYN}{btc_pending:.8f} BTC{RST}   (${usd_pending:,.2f} USD)")
        print(f"\n  {BOLD}Total (confirmed + pending){RST}")
        print(f"    {GOLD}{btc_total:.8f} BTC{RST}   (${usd_total:,.2f} USD)")

    print(f"\n  {DIM}Transactions : {tx_count}{RST}")
    print(f"  {DIM}BTC Price    : ${btc_usd:,.2f} USD{RST}")

    if tx_count == 0:
        print(f"\n  {DIM}No transactions found — wallet has never received Bitcoin.{RST}")

    print(f"\n{GOLD}{'═'*56}{RST}\n")


if __name__ == "__main__":
    main()
