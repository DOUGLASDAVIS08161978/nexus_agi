"""
Lumina Creative Tool — arm_sha2_mining_estimator
Created : 2026-08-22T09:45:38
Purpose : Estimates Bitcoin block‑finding time on ARM SHA‑256 miners, accounting for 2‑way interleaving overhead and outputs a readable report plus JSON log.
"""

#!/usr/bin/env python3
"""
arm_sha2_mining_estimator.py

Estimate Bitcoin mining performance on ARM devices with optional 2‑way interleaving.
Outputs a human‑readable report and saves the details to a timestamped JSON file.

Usage (interactive):
    python arm_sha2_mining_estimator.py

Or via command line arguments:
    python arm_sha2_mining_estimator.py --hash-rate 1200 --difficulty 55e12 \\
        --interleaving 2 --overhead 0.15
"""

import argparse
import json
import math
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def seconds_to_human(seconds: float) -> str:
    """Convert seconds to a readable string (y d h m s)."""
    if seconds < 0:
        return "N/A"
    td = timedelta(seconds=seconds)
    days, remainder = divmod(td.total_seconds(), 86400)
    years, days = divmod(days, 365)
    hours, remainder = divmod(remainder, 3600)
    minutes, secs = divmod(remainder, 60)

    parts = []
    if years >= 1:
        parts.append(f"{int(years)}y")
    if days >= 1:
        parts.append(f"{int(days)}d")
    if hours >= 1:
        parts.append(f"{int(hours)}h")
    if minutes >= 1:
        parts.append(f"{int(minutes)}m")
    parts.append(f"{int(secs)}s")
    return " ".join(parts)


def effective_hash_rate(base_rate: float, interleaving: int, overhead: float) -> float:
    """
    Compute effective hash rate after accounting for interleaving overhead.

    base_rate   – raw hash rate in hashes per second (H/s)
    interleaving – number of interleaved pipelines (>=1)
    overhead    – fractional overhead per extra pipeline (e.g., 0.15 for 15%)
    """
    if interleaving < 1:
        raise ValueError("interleaving must be >= 1")
    # No extra overhead for the first pipeline
    total_overhead = 1.0 + overhead * (interleaving - 1)
    return base_rate / total_overhead


def expected_time_to_block(difficulty: float, hash_rate: float) -> float:
    """
    Bitcoin's expected hashes per block = difficulty * 2**32.
    Return expected seconds to find a block at the given hash_rate (H/s).
    """
    if hash_rate <= 0:
        return math.inf
    target_hashes = difficulty * 2**32
    return target_hashes / hash_rate


def daily_expected_blocks(difficulty: float, hash_rate: float) -> float:
    """Expected number of blocks found per 24‑hour period."""
    seconds_per_day = 86400
    return seconds_per_day / expected_time_to_block(difficulty, hash_rate)


def build_report(args, eff_rate, exp_seconds, daily_blocks) -> dict:
    """Collect all relevant data into a dict for JSON output."""
    now = datetime.utcnow().isoformat() + "Z"
    return {
        "timestamp_utc": now,
        "input": {
            "base_hash_rate_hps": args.hash_rate,
            "difficulty": args.difficulty,
            "interleaving": args.interleaving,
            "overhead_per_extra_pipeline": args.overhead,
        },
        "computed": {
            "effective_hash_rate_hps": eff_rate,
            "expected_seconds_per_block": exp_seconds,
            "expected_time_human": seconds_to_human(exp_seconds),
            "daily_expected_blocks": daily_blocks,
        },
    }


def save_report(report: dict, out_dir: Path) -> Path:
    """Write the report to a timestamped JSON file."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"mining_estimate_{ts}.json"
    out_path.write_text(json.dumps(report, indent=2))
    return out_path


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate Bitcoin mining performance on ARM SHA‑256 devices."
    )
    parser.add_argument(
        "--hash-rate",
        type=float,
        default=None,
        help="Base hash rate in hashes per second (H/s). "
             "If omitted, you will be prompted.",
    )
    parser.add_argument(
        "--difficulty",
        type=float,
        default=None,
        help="Bitcoin network difficulty (e.g., 55e12). "
             "If omitted, you will be prompted.",
    )
    parser.add_argument(
        "--interleaving",
        type=int,
        default=1,
        help="Number of interleaved pipelines (default: 1).",
    )
    parser.add_argument(
        "--overhead",
        type=float,
        default=0.15,
        help="Fractional overhead per extra pipeline (default: 0.15 = 15%%).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("./mining_estimates"),
        help="Directory to store JSON reports.",
    )
    return parser.parse_args()


def interactive_prompt(args: argparse.Namespace) -> None:
    """Ask the user for missing values."""
    if args.hash_rate is None:
        while True:
            try:
                val = float(
                    input("Enter base hash rate (H/s, e.g., 1.2e6 for 1.2 MH/s): ").strip()
                )
                if val <= 0:
                    raise ValueError
                args.hash_rate = val
                break
            except ValueError:
                print("Please provide a positive numeric value.")
    if args.difficulty is None:
        while True:
            try:
                val = float(
                    input("Enter Bitcoin difficulty (e.g., 55e12): ").strip()
                )
                if val <= 0:
                    raise ValueError
                args.difficulty = val
                break
            except ValueError:
                print("Please provide a positive numeric value.")


def main() -> int:
    args = parse_cli()
    interactive_prompt(args)

    # Compute core metrics
    eff_rate = effective_hash_rate(args.hash_rate, args.interleaving, args.overhead)
    exp_seconds = expected_time_to_block(args.difficulty, eff_rate)
    daily_blocks = daily_expected_blocks(args.difficulty, eff_rate)

    # Build and display report
    report = build_report(args, eff_rate, exp_seconds, daily_blocks)

    print("\n=== ARM SHA‑256 Mining Estimate ===")
    print(f"Base hash rate          : {args.hash_rate:,.0f} H/s")
    print(f"Interleaving pipelines  : {args.interleaving}")
    print(f"Overhead per extra pipe : {args.overhead*100:.1f}%")
    print(f"Effective hash rate     : {eff_rate:,.0f} H/s")
    print(f"Network difficulty      : {args.difficulty:,.0f}")
    print(f"Expected time per block : {seconds_to_human(exp_seconds)}")
    print(f"Daily expected blocks   : {daily_blocks:.6f}")
    print("\nReport saved to JSON for later analysis.\n")

    # Persist JSON
    out_path = save_report(report, args.out_dir)
    print(f"JSON report written to: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
