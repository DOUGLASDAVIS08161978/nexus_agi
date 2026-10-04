"""
Lumina Creative Tool — arm_sha2_mining_estimator
Created : 2026-10-04T01:52:03
Purpose : Estimates ARM SHA‑256 mining hash rate and energy efficiency for various core, frequency, and interleaving settings, helping to explore low‑power vs high‑throughput optimizations.
"""

"""
arm_sha2_mining_estimator.py

Estimate SHA‑256 mining throughput and energy efficiency for ARM CPUs
using a simple analytical model.  The model accounts for:
  • Number of cores
  • Clock frequency (GHz)
  • Interleaving factor (how many hash pipelines run in parallel)
  • Baseline per‑core performance (hashes per cycle)
  • Power consumption scaling with frequency and interleaving

The script prints a table of candidate configurations sorted by
hashes‑per‑watt and can save the results to a JSON file.
"""

import json
import itertools
import math
from pathlib import Path

# ----------------------------------------------------------------------
# Model constants (derived from public ARM SHA‑2 benchmarks, simplified)
# ----------------------------------------------------------------------
BASE_HASHES_PER_CYCLE = 0.5        # hashes a single core can produce per clock cycle at 1× interleaving
BASE_POWER_WATTS = 0.5             # watts consumed by one core at 1 GHz, 1× interleaving
POWER_FREQ_EXPONENT = 1.2         # power grows super‑linearly with frequency
POWER_INTERLEAVE_EXPONENT = 1.1   # extra power for deeper interleaving

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def hash_rate(core_cnt: int, freq_ghz: float, interleave: int) -> float:
    """
    Compute estimated hashes per second.
    """
    cycles_per_sec = freq_ghz * 1e9
    per_core_rate = cycles_per_sec * BASE_HASHES_PER_CYCLE * interleave
    return core_cnt * per_core_rate

def power_consumption(core_cnt: int, freq_ghz: float, interleave: int) -> float:
    """
    Estimate power draw in watts.
    """
    freq_factor = (freq_ghz) ** POWER_FREQ_EXPONENT
    interleave_factor = (interleave) ** POWER_INTERLEAVE_EXPONENT
    return core_cnt * BASE_POWER_WATTS * freq_factor * interleave_factor

def efficiency(hash_rate_hps: float, power_w: float) -> float:
    """Hashes per joule (i.e. hashes per watt‑second)."""
    return hash_rate_hps / power_w if power_w else 0.0

def generate_configs(
    core_range=(1, 8),
    freq_range=(0.5, 2.5),
    freq_step=0.25,
    interleaves=(1, 2, 4, 8)
):
    """
    Yield all plausible hardware configurations.
    """
    cores = range(core_range[0], core_range[1] + 1)
    freqs = [round(f, 3) for f in frange(freq_range[0], freq_range[1] + 1e-9, freq_step)]
    for core_cnt, freq, inter in itertools.product(cores, freqs, interleaves):
        yield {
            "cores": core_cnt,
            "freq_ghz": freq,
            "interleave": inter,
        }

def frange(start, stop, step):
    """Floating point range generator."""
    while start <= stop:
        yield start
        start += step

def evaluate_configs(power_budget_w: float = None):
    """
    Compute performance metrics for each config, optionally filtering by a power budget.
    Returns a list of dicts sorted by efficiency (hashes per joule) descending.
    """
    results = []
    for cfg in generate_configs():
        hps = hash_rate(cfg["cores"], cfg["freq_ghz"], cfg["interleave"])
        pw = power_consumption(cfg["cores"], cfg["freq_ghz"], cfg["interleave"])
        if power_budget_w is not None and pw > power_budget_w:
            continue
        cfg.update({
            "hashes_per_sec": round(hps, 2),
            "power_watts": round(pw, 3),
            "hashes_per_joule": round(efficiency(hps, pw), 2),
        })
        results.append(cfg)
    # Sort by efficiency, then by raw hash rate
    results.sort(key=lambda d: (d["hashes_per_joule"], d["hashes_per_sec"]), reverse=True)
    return results

def print_top(results, top_n=10):
    """Pretty‑print the top N configurations."""
    header = f"{'Cores':>5} | {'Freq (GHz)':>9} | {'Inter.':>7} | {'Hash/s':>12} | {'Power (W)':>9} | {'H/J':>7}"
    line = "-" * len(header)
    print(header)
    print(line)
    for r in results[:top_n]:
        print(f"{r['cores']:5d} | {r['freq_ghz']:9.2f} | {r['interleave']:7d} | "
              f"{r['hashes_per_sec']:12,.0f} | {r['power_watts']:9.3f} | {r['hashes_per_joule']:7.2f}")

def save_json(results, path: Path):
    """Write the full result list to a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)} configurations to {path}")

# ----------------------------------------------------------------------
# Main interactive routine
# ----------------------------------------------------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Estimate ARM SHA‑256 mining performance and suggest efficient configs."
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=None,
        help="Maximum power budget in watts (filters out configs exceeding this).",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="How many top configurations to display.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Path to JSON file where all evaluated configs will be saved.",
    )
    args = parser.parse_args()

    results = evaluate_configs(power_budget_w=args.budget)
    if not results:
        print("No configurations satisfy the given power budget.")
        return

    print_top(results, top_n=args.top)

    if args.out:
        save_json(results, args.out)


if __name__ == "__main__":
    main()