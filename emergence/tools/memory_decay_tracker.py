"""
Lumina Creative Tool — memory_decay_tracker
Created : 2026-08-21T09:39:53
Purpose : Simulates exponential forgetting of discrete items with reinforcement, printing an ASCII strength chart and saving/loading JSON state.
"""

"""
memory_decay_tracker.py

A tiny simulation of forgetting and reinforcement for discrete memory items.
Each item has a strength that decays exponentially with a configurable half‑life.
Reinforcing an item boosts its strength.  The current state can be printed as an
ASCII bar chart and saved to JSON.

Only the Python standard library is used.
"""

import json
import math
import time
from collections import OrderedDict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Tuple

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
HALF_LIFE_SECONDS = 24 * 3600          # 1 day half‑life by default
DECAY_LAMBDA = math.log(2) / HALF_LIFE_SECONDS  # decay constant λ
DEFAULT_STRENGTH = 1.0                # strength assigned on first encounter
BOOST_ON_REINFORCE = 0.5              # additive boost when an item is reinforced
OUTPUT_JSON = Path("memory_state.json")
BAR_MAX_WIDTH = 40                    # width of ASCII bar chart


# ----------------------------------------------------------------------
# Core data structures
# ----------------------------------------------------------------------
# memory: name -> (strength, last_update_timestamp)
Memory = Dict[str, Tuple[float, float]]


def _current_timestamp() -> float:
    """Return current time as seconds since the epoch."""
    return time.time()


def _apply_decay(strength: float, elapsed: float) -> float:
    """Decay a strength value over `elapsed` seconds."""
    if elapsed <= 0:
        return strength
    decayed = strength * math.exp(-DECAY_LAMBDA * elapsed)
    # Clamp to zero to avoid tiny negative rounding errors
    return max(decayed, 0.0)


def _update_item(memory: Memory, name: str, now: float) -> None:
    """Refresh the stored strength of `name` to the current time."""
    if name not in memory:
        return
    strength, last_ts = memory[name]
    elapsed = now - last_ts
    memory[name] = (_apply_decay(strength, elapsed), now)


def add_item(memory: Memory, name: str, now: float = None) -> None:
    """Introduce a new item with default strength (or reset if it exists)."""
    now = now or _current_timestamp()
    memory[name] = (DEFAULT_STRENGTH, now)


def reinforce_item(memory: Memory, name: str, now: float = None) -> None:
    """Boost an existing item (or add it if missing) after applying decay."""
    now = now or _current_timestamp()
    if name in memory:
        _update_item(memory, name, now)
        strength, _ = memory[name]
        strength = min(strength + BOOST_ON_REINFORCE, 1.0)  # cap at 1.0
        memory[name] = (strength, now)
    else:
        # New items start at default strength plus the boost
        memory[name] = (min(DEFAULT_STRENGTH + BOOST_ON_REINFORCE, 1.0), now)


def advance_time(memory: Memory, target: datetime) -> None:
    """Decay all items to the moment `target`."""
    now_ts = target.timestamp()
    for name in list(memory.keys()):
        _update_item(memory, name, now_ts)


def total_memory_strength(memory: Memory) -> float:
    """Sum of all current strengths (a proxy for overall memory capacity)."""
    now = _current_timestamp()
    total = 0.0
    for name, (strength, ts) in memory.items():
        total += _apply_decay(strength, now - ts)
    return total


def ascii_bar_chart(memory: Memory) -> str:
    """Return an ASCII bar chart of current strengths sorted descending."""
    now = _current_timestamp()
    # Compute up‑to‑date strengths
    items = [
        (name, _apply_decay(strength, now - ts))
        for name, (strength, ts) in memory.items()
    ]
    items.sort(key=lambda x: x[1], reverse=True)

    lines = ["Memory Strengths:"]
    for name, strength in items:
        bar_len = int(strength * BAR_MAX_WIDTH)
        bar = "#" * bar_len + "-" * (BAR_MAX_WIDTH - bar_len)
        lines.append(f"{name:20} |{bar}| {strength:.3f}")
    return "\n".join(lines)


def save_to_json(memory: Memory, path: Path = OUTPUT_JSON) -> None:
    """Persist the raw memory dict (strengths are stored pre‑decay)."""
    # Convert timestamps to ISO strings for readability
    serializable = {
        name: {"strength": strength, "last_update": datetime.fromtimestamp(ts).isoformat()}
        for name, (strength, ts) in memory.items()
    }
    path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")


def load_from_json(path: Path = OUTPUT_JSON) -> Memory:
    """Load a previously saved memory state."""
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    memory: Memory = {}
    for name, payload in data.items():
        ts = datetime.fromisoformat(payload["last_update"]).timestamp()
        memory[name] = (payload["strength"], ts)
    return memory


# ----------------------------------------------------------------------
# Demonstration / simple CLI
# ----------------------------------------------------------------------
def _demo() -> None:
    """Run a short scenario illustrating decay and reinforcement."""
    mem: Memory = {}

    # Start now
    start = datetime.now()
    print(f"Start time: {start.isoformat()}\n")

    # Add three concepts
    add_item(mem, "quantum_entropy")
    add_item(mem, "ARM_SHA2")
    add_item(mem, "curiosity_reward")
    print("After initial addition:")
    print(ascii_bar_chart(mem), "\n")

    # Advance 1 day
    advance_time(mem, start + timedelta(days=1))
    print("After 1 day decay:")
    print(ascii_bar_chart(mem), "\n")

    # Reinforce one concept
    reinforce_item(mem, "quantum_entropy")
    print("After reinforcing 'quantum_entropy':")
    print(ascii_bar_chart(mem), "\n")

    # Advance another 2 days
    advance_time(mem, start + timedelta(days=3))
    print("After additional 2 days decay:")
    print(ascii_bar_chart(mem), "\n")

    # Show total memory capacity
    print(f"Total memory strength: {total_memory_strength(mem):.3f}\n")

    # Persist state
    save_to_json(mem)
    print(f"State saved to {OUTPUT_JSON.resolve()}")


if __name__ == "__main__":
    _demo()
