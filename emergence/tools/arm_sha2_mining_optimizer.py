"""
Lumina Creative Tool — arm_sha2_mining_optimizer
Created : 2026-10-03T22:51:07
Purpose : Estimates daily profit for ARM‑based SHA‑256 miners and outputs the most profitable device configuration as JSON and an ASCII table.
"""

"""
arm_sha2_mining_optimizer.py

Estimate daily profit for a set of ARM‑based SHA‑256 mining devices and
suggest the most profitable configuration.

The model uses the standard Bitcoin mining formula:
    hashes_per_day = hash_rate * 86400
    probability_of_block = hashes_per_day / (difficulty * 2**32)
    expected_BTC = probability_of_block * block_reward
    revenue_usd = expected_BTC * btc_price
    cost_usd = power_watts * 24 / 1000 * electricity_usd_per_kwh
    profit_usd = revenue_usd - cost_usd
"""

import json
import math
import sys
from dataclasses import dataclass, asdict
from typing import List

# ----------------------------------------------------------------------
# Data definitions
# ----------------------------------------------------------------------
@dataclass
class Device:
    name: str
    hash_rate_mh: float   # Mega‑hashes per second
    power_w: float        # Watts

    @property
    def hash_rate_h(self) -> float:
        """Convert MH/s to hashes per second."""
        return self.hash_rate_mh * 1_000_000


# ----------------------------------------------------------------------
# Sample device catalog (feel free to extend)
# ----------------------------------------------------------------------
DEVICES: List[Device] = [
    Device("Raspberry Pi 4 (overclocked)", 0.12, 5.0),
    Device("Odroid N2+", 0.25, 7.0),
    Device("Rockchip RK3399 (optimized)", 0.45, 10.0),
    Device("Custom ARM64 board", 1.0, 15.0),
    Device("High‑end ARM server", 3.5, 45.0),
]

# ----------------------------------------------------------------------
# Core profitability calculation
# ----------------------------------------------------------------------
def daily_profit(device: Device,
                 difficulty: float,
                 block_reward: float,
                 btc_price: float,
                 electricity_usd_per_kwh: float) -> float:
    """Return expected daily profit in USD for a single device."""
    # Expected number of blocks found per day
    hashes_per_day = device.hash_rate_h * 86400
    prob_block = hashes_per_day / (difficulty * 2**32)
    expected_btc = prob_block * block_reward

    revenue = expected_btc * btc_price
    cost = device.power_w * 24 / 1000 * electricity_usd_per_kwh
    return revenue - cost


# ----------------------------------------------------------------------
# Reporting utilities
# ----------------------------------------------------------------------
def format_currency(v: float) -> str:
    return f"${v:,.2f}" if v >= 0 else f"-${abs(v):,.2f}"


def ascii_bar(value: float, max_value: float, width: int = 30) -> str:
    """Simple horizontal bar proportional to value / max_value."""
    if max_value <= 0:
        return ""
    filled = int(round(width * value / max_value))
    return "█" * filled + " " * (width - filled)


def generate_report(devices: List[Device],
                    difficulty: float,
                    block_reward: float,
                    btc_price: float,
                    electricity_usd_per_kwh: float) -> List[dict]:
    """Compute profit for each device and return a sorted list of dicts."""
    results = []
    for dev in devices:
        profit = daily_profit(dev, difficulty, block_reward,
                              btc_price, electricity_usd_per_kwh)
        results.append({
            "name": dev.name,
            "hash_rate_MH/s": dev.hash_rate_mh,
            "power_W": dev.power_w,
            "daily_profit_USD": profit,
        })
    results.sort(key=lambda x: x["daily_profit_USD"], reverse=True)
    return results


def print_ascii_table(results: List[dict]) -> None:
    """Print a human‑readable table with an ASCII profit bar."""
    if not results:
        print("No results to display.")
        return

    max_profit = max(r["daily_profit_USD"] for r in results)
    header = f"{'Device':30} {'Hash (MH/s)':>12} {'Power (W)':>10} {'Profit/Day':>12}  Bar"
    print(header)
    print("-" * len(header))
    for r in results:
        bar = ascii_bar(r["daily_profit_USD"], max_profit)
        print(f"{r['name'][:30]:30} {r['hash_rate_MH/s']:12.2f} {r['power_W']:10.1f} "
              f"{format_currency(r['daily_profit_USD']):12}  {bar}")


def save_best_config(results: List[dict], filename: str = "optimal_mining_config.json") -> None:
    """Save the top‑profit device configuration to a JSON file."""
    if not results:
        return
    best = results[0]
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)
    print(f"\nBest configuration saved to {filename}")


# ----------------------------------------------------------------------
# Command‑line interface
# ----------------------------------------------------------------------
def parse_float(arg: str, name: str) -> float:
    try:
        return float(arg)
    except ValueError:
        sys.exit(f"Invalid {name}: {arg}")


def main(argv: List[str]) -> None:
    if len(argv) != 5:
        print("Usage: python arm_sha2_mining_optimizer.py <difficulty> <block_reward> "
              "<btc_price_usd> <electricity_usd_per_kwh>")
        sys.exit(1)

    difficulty = parse_float(argv[0], "difficulty")
    block_reward = parse_float(argv[1], "block_reward")
    btc_price = parse_float(argv[2], "btc_price_usd")
    electricity = parse_float(argv[3], "electricity_usd_per_kwh")

    results = generate_report(DEVICES, difficulty, block_reward,
                              btc_price, electricity)

    print_ascii_table(results)
    save_best_config(results)


if __name__ == "__main__":
    # Example defaults (can be overridden via CLI)
    # Difficulty ~ 50 T, block reward 6.25 BTC, price $27 000, electricity $0.12/kWh
    example_args = ["50000000000000", "6.25", "27000", "0.12"]
    if len(sys.argv) == 1:
        # No arguments supplied – run with example values
        main(example_args)
    else:
        main(sys.argv[1:])
