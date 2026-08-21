"""
Lumina Creative Tool — memory_consolidation_simulator
Created : 2026-08-21T16:36:28
Purpose : Simulates short‑term memory decay and consolidates high‑information entries into a persistent JSON long‑term store.
"""

"""
memory_consolidation_simulator.py

Simulates short‑term memory (STM) that decays each step and consolidates
entries into long‑term memory (LTM) when their cumulative information gain
exceeds a threshold.

Outputs:
  - console summary of the simulation
  - "long_term_memory.json" containing consolidated entries
"""

import json
import math
import random
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple

# ----------------------------------------------------------------------
# Configuration parameters (tweakable)
# ----------------------------------------------------------------------
DECAY_RATE = 0.85          # each step multiplies info_gain by this factor
CONSOLIDATION_THRESHOLD = 12.0  # minimum cumulative info_gain to move to LTM
MAX_STEPS = 100            # maximum number of processing steps
SEED = 42                  # reproducibility

random.seed(SEED)

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def tokenize(text: str) -> List[str]:
    """Very simple whitespace + punctuation tokenizer."""
    return [word.strip(".,!?;:()[]\"'").lower() for word in text.split() if word]

def compute_global_frequencies(entries: List[str]) -> Counter:
    """Count word occurrences across all entries."""
    freq = Counter()
    for e in entries:
        freq.update(tokenize(e))
    return freq

def entry_info_gain(entry: str, global_freq: Counter, total_words: int) -> float:
    """
    Approximate information gain of an entry as sum(-log(p(word))) where
    p(word) = freq/total_words. Rare words contribute more.
    """
    gain = 0.0
    for w in tokenize(entry):
        freq = global_freq.get(w, 1)  # avoid zero
        p = freq / total_words
        gain += -math.log(p + 1e-12)  # small epsilon for safety
    return gain

# ----------------------------------------------------------------------
# Memory data structures
# ----------------------------------------------------------------------
class STMEntry:
    """Short‑term memory entry."""
    def __init__(self, text: str, info_gain: float, timestamp: float):
        self.text = text
        self.cumulative_gain = info_gain
        self.timestamp = timestamp  # seconds since start

    def decay(self):
        self.cumulative_gain *= DECAY_RATE

    def is_ready(self) -> bool:
        return self.cumulative_gain >= CONSOLIDATION_THRESHOLD

class MemorySimulator:
    def __init__(self, entries: List[str]):
        self.entries = entries
        self.global_freq = compute_global_frequencies(entries)
        self.total_words = sum(self.global_freq.values())
        self.stm: List[STMEntry] = []
        self.ltm: List[Dict] = []  # list of dicts for JSON export
        self.step = 0
        self.start_time = time.time()

    def process_next(self):
        """Process a single entry (or decay if none left)."""
        if self.step < len(self.entries):
            raw = self.entries[self.step]
            gain = entry_info_gain(raw, self.global_freq, self.total_words)
            entry = STMEntry(
                text=raw,
                info_gain=gain,
                timestamp=time.time() - self.start_time,
            )
            self.stm.append(entry)
        # decay all STM entries
        for e in self.stm:
            e.decay()
        # consolidate ready entries
        self._consolidate()
        self.step += 1

    def _consolidate(self):
        ready = [e for e in self.stm if e.is_ready()]
        for e in ready:
            self.ltm.append({
                "text": e.text,
                "consolidated_at": datetime.utcnow().isoformat() + "Z",
                "cumulative_gain": round(e.cumulative_gain, 3),
            })
        # keep only non‑ready entries in STM
        self.stm = [e for e in self.stm if not e.is_ready()]

    def run(self):
        max_steps = max(MAX_STEPS, len(self.entries) * 2)
        while self.step < max_steps:
            self.process_next()
            # early stop if nothing left in STM and all entries processed
            if self.step >= len(self.entries) and not self.stm:
                break

    def summary(self):
        print("\n=== Memory Consolidation Simulation Summary ===")
        print(f"Total entries processed      : {len(self.entries)}")
        print(f"STM entries remaining        : {len(self.stm)}")
        print(f"LTM entries consolidated    : {len(self.ltm)}")
        if self.ltm:
            print("\nSample consolidated entries (up to 3):")
            for item in self.ltm[:3]:
                print(f"- [{item['consolidated_at']}] {item['text'][:60]}...")
        else:
            print("\nNo entries reached the consolidation threshold.")

    def export_ltm(self, path: Path):
        path.write_text(json.dumps(self.ltm, indent=2, ensure_ascii=False))
        print(f"\nLong‑term memory saved to: {path}")

# ----------------------------------------------------------------------
# Example usage (can be replaced with reading a file)
# ----------------------------------------------------------------------
def load_example_entries() -> List[str]:
    """Generate or load a modest list of journal‑like sentences."""
    sample = [
        "The relationship between entropy and perplexity can guide model scaling.",
        "ARM SHA2 mining on low‑power devices faces thermal constraints.",
        "Dynamic category memory updates mimic Bayesian belief revision.",
        "Curiosity drives reward processing in the ventral striatum and prefrontal cortex.",
        "Information gain from rare tokens is higher than from common ones.",
        "Consolidation may require a threshold of surprise to store long‑term.",
        "Simulating decay helps model forgetting in short‑term buffers.",
        "Perplexity reduction often correlates with entropy minimization.",
        "Hardware acceleration can offset the energy cost of SHA‑256 hashing.",
        "Long‑term memory should be sparse yet high‑information.",
    ]
    # Optionally shuffle to emulate streaming
    random.shuffle(sample)
    return sample

def main():
    entries = load_example_entries()
    sim = MemorySimulator(entries)
    sim.run()
    sim.summary()
    sim.export_ltm(Path("long_term_memory.json"))

if __name__ == "__main__":
    main()
