"""
Lumina Creative Tool — decaying_category_manager
Created : 2026-08-24T19:38:47
Purpose : Incrementally learns token frequencies for named categories while applying exponential time decay, enabling a simple simulation of dynamic long‑term memory.
"""

import json
import sys
import time
import math
import string
from collections import defaultdict, Counter
from datetime import datetime, timedelta

# ---------- Helper functions ----------
def tokenize(text: str) -> list[str]:
    """Very simple tokenizer: lower‑case, strip punctuation, split on whitespace."""
    translator = str.maketrans("", "", string.punctuation)
    return text.lower().translate(translator).split()

def now_ts() -> float:
    """Current time as Unix timestamp (seconds)."""
    return time.time()

# ---------- Core class ----------
class DecayingCategoryManager:
    """
    Maintains token counts per category with exponential time decay.
    New documents increase counts; older counts fade automatically.
    """
    def __init__(self, half_life_seconds: float = 86400.0):
        """
        half_life_seconds: time for a token count to halve if not reinforced.
        """
        self.half_life = half_life_seconds
        self.decay_rate = 0.5 ** (1.0 / self.half_life)  # per‑second multiplier
        self.categories: dict[str, dict] = {}  # name -> {"counts": Counter, "last": ts}

    def _ensure_category(self, name: str):
        if name not in self.categories:
            self.categories[name] = {"counts": Counter(), "last": now_ts()}

    def _apply_decay(self, name: str):
        """Decay all token counts for a category based on elapsed time."""
        cat = self.categories[name]
        elapsed = now_ts() - cat["last"]
        if elapsed <= 0:
            return
        factor = self.decay_rate ** elapsed
        # Apply decay to each token count
        for token in list(cat["counts"]):
            decayed = cat["counts"][token] * factor
            if decayed < 0.01:  # prune near‑zero entries
                del cat["counts"][token]
            else:
                cat["counts"][token] = decayed
        cat["last"] = now_ts()

    def add_document(self, text: str, category: str = "default"):
        """Tokenize `text` and update the specified category."""
        self._ensure_category(category)
        self._apply_decay(category)
        tokens = tokenize(text)
        self.categories[category]["counts"].update(tokens)
        # Record fresh timestamp after update
        self.categories[category]["last"] = now_ts()

    def top_tokens(self, category: str, n: int = 10) -> list[tuple[str, float]]:
        """Return the `n` most frequent tokens (after decay) for a category."""
        if category not in self.categories:
            return []
        self._apply_decay(category)
        return self.categories[category]["counts"].most_common(n)

    def snapshot(self) -> dict:
        """Export the full state (rounded counts) as a JSON‑serialisable dict."""
        out = {}
        for name, data in self.categories.items():
            self._apply_decay(name)
            out[name] = {tok: round(cnt, 3) for tok, cnt in data["counts"].items()}
        return out

    def save(self, path: str = "categories.json"):
        """Write the current snapshot to a JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.snapshot(), f, indent=2, ensure_ascii=False)

# ---------- Simple CLI ----------
def parse_input_line(line: str) -> tuple[str, str]:
    """
    Expected format:
        CategoryName: actual text
    If no colon is present, the line belongs to the default category.
    """
    if ":" in line:
        cat, txt = line.split(":", 1)
        return cat.strip(), txt.strip()
    return "default", line.strip()

def main():
    manager = DecayingCategoryManager(half_life_seconds=48 * 3600)  # 2‑day half‑life
    print("Enter lines (empty line to finish). Prefix with 'Category:' to assign.")
    for raw in sys.stdin:
        line = raw.rstrip("\n")
        if not line:
            break
        cat, txt = parse_input_line(line)
        manager.add_document(txt, cat)
        top = manager.top_tokens(cat, n=5)
        print(f"[{cat}] Top tokens: {', '.join(f'{t}:{c:.2f}' for t,c in top)}")

    # Final report
    print("\n=== Category snapshots ===")
    snapshot = manager.snapshot()
    for cat, tokens in snapshot.items():
        top = sorted(tokens.items(), key=lambda kv: kv[1], reverse=True)[:5]
        print(f"{cat}: " + ", ".join(f"{t}:{c:.2f}" for t,c in top))

    manager.save()
    print("\nState saved to 'categories.json'.")

if __name__ == "__main__":
    main()
