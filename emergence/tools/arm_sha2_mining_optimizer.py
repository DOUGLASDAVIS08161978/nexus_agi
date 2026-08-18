"""
Lumina Creative Tool — arm_sha2_mining_optimizer
Created : 2026-08-18T13:58:16
Purpose : Suggests feasible ARM‑SHA‑2 mining configurations for given core count, clock speed, and power budget, estimating hash‑rate performance.
"""

#!/usr/bin/env python3
"""
arm_sha2_mining_optimizer

Generate feasible ARM SHA‑2 mining configurations based on
hardware constraints and estimate their hash‑rate performance.

Outputs:
  - An ASCII table printed to stdout.
  - A JSON file "sha2_mining_options.json" with the same data.
"""

import json
import itertools
import math
from pathlib import Path
from collections import namedtuple

# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------
Config = namedtuple(
    "Config",
    "cores freq_mhz interleaving unroll power_w est_hash_rate_mhs"
)

# ----------------------------------------------------------------------
# Simple performance model
# ----------------------------------------------------------------------
def estimate_hash_rate(cores: int, freq_mhz: float,
                       interleaving: int, unroll: int,
                       power_w: float) -> float:
    """
    Estimate hash rate in Mega‑Hashes per second (MH/s).

    Model assumptions (purely illustrative):
      * Base cycles per hash = 200 (typical for SHA‑256 on ARM).
      * Each core can issue one instruction per cycle.
      * Interleaving reduces effective cycles per hash linearly.
      * Loop unroll gives a modest 5 % speedup per factor of 2.
      * Power limits effective frequency: if power budget is low,
        we scale frequency down proportionally.
    """
    base_cycles = 200.0
    # Adjust frequency for power budget (very rough)
    max_power_per_core = 0.5  # watts per core at full freq (example)
    power_limit_factor = min(1.0, power_w / (cores * max_power_per_core))
    effective_freq = freq_mhz * power_limit_factor

    # Interleaving factor: higher interleaving reduces parallelism
    interleaving_penalty = interleaving

    # Unroll factor: each doubling gives ~5% boost
    unroll_bonus = 1.0 + 0.05 * math.log2(unroll) if unroll > 1 else 1.0

    # Hashes per second per core
    hashes_per_sec = (
        effective_freq * 1e6 / (base_cycles * interleaving_penalty)
    ) * unroll_bonus

    total_hashes = hashes_per_sec * cores
    return total_hashes / 1e6  # convert to MH/s


# ----------------------------------------------------------------------
# Configuration generator
# ----------------------------------------------------------------------
def generate_configs(cores: int, freq_mhz: float,
                     power_w: float,
                     interleaving_options=(1, 2, 4),
                     unroll_options=(1, 2, 4, 8)):
    """Yield Config objects that respect the power budget."""
    for inter, unroll in itertools.product(interleaving_options,
                                            unroll_options):
        est = estimate_hash_rate(cores, freq_mhz,
                                 inter, unroll, power_w)
        # Discard absurdly low rates (<0.1 MH/s) as impractical
        if est < 0.1:
            continue
        yield Config(
            cores=cores,
            freq_mhz=freq_mhz,
            interleaving=inter,
            unroll=unroll,
            power_w=power_w,
            est_hash_rate_mhs=round(est, 3)
        )


# ----------------------------------------------------------------------
# Presentation helpers
# ----------------------------------------------------------------------
def ascii_table(configs):
    """Return a formatted ASCII table string."""
    header = (
        f"{'Cores':>5} | {'Freq(MHz)':>9} | {'Inter.':>7} | "
        f"{'Unroll':>6} | {'Power(W)':>8} | {'Est MH/s':>9}"
    )
    line = "-" * len(header)
    rows = [header, line]
    for cfg in configs:
        rows.append(
            f"{cfg.cores:5d} | {cfg.freq_mhz:9.1f} | {cfg.interleaving:7d} | "
            f"{cfg.unroll:6d} | {cfg.power_w:8.2f} | {cfg.est_hash_rate_mhs:9.3f}"
        )
    return "\n".join(rows)


def configs_to_json(configs):
    """Serialize Config objects to a JSON‑serializable list."""
    return [
        {
            "cores": cfg.cores,
            "freq_mhz": cfg.freq_mhz,
            "interleaving": cfg.interleaving,
            "unroll": cfg.unroll,
            "power_w": cfg.power_w,
            "est_hash_rate_mhs": cfg.est_hash_rate_mhs,
        }
        for cfg in configs
    ]


# ----------------------------------------------------------------------
# Main execution
# ----------------------------------------------------------------------
def main():
    # Example scenario – can be edited or driven by CLI later
    example_devices = [
        # (cores, freq_mhz, power_w, description)
        (4, 1200.0, 2.0, "Low‑power phone SOC"),
        (8, 1800.0, 5.0, "Mid‑range tablet SOC"),
        (16, 2500.0, 15.0, "High‑end server‑grade ARM"),
    ]

    all_results = []

    for cores, freq, power, desc in example_devices:
        print(f"\nDevice: {desc}")
        configs = list(generate_configs(cores, freq, power))
        if not configs:
            print("  No viable configurations under given power budget.")
            continue
        print(ascii_table(configs))
        all_results.extend(configs)

    # Save JSON file
    out_path = Path("sha2_mining_options.json")
    out_path.write_text(json.dumps(configs_to_json(all_results), indent=2))
    print(f"\nSaved {len(all_results)} configurations to {out_path}")

if __name__ == "__main__":
    main()