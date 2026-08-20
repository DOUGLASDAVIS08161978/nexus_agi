"""
Lumina Creative Tool — dynamic_category_memory
Created : 2026-08-20T09:00:42
Purpose : Simulates Bayesian updating of categorical beliefs from contextual tokens, printing evolving probabilities and exporting state as JSON.
"""

"""
dynamic_category_memory.py

A pure‑standard‑library tool that maintains a set of categories with
Dirichlet priors and updates them as contextual evidence arrives.
It prints the evolving predictive probabilities and can export the
state to JSON for later inspection.
"""

import json
import random
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def normalize(dist: List[float]) -> List[float]:
    """Return a probability distribution (sum to 1)."""
    total = sum(dist)
    return [v / total for v in dist] if total else [0.0] * len(dist)


def pretty_print(title: str, probs: Dict[str, float]) -> None:
    """Print a title and a sorted probability table."""
    print(f"\n{title}")
    print("-" * len(title))
    for cat, p in sorted(probs.items(), key=lambda kv: -kv[1]):
        bar = "#" * int(p * 40)
        print(f"{cat:12}: {p:6.3%} |{bar:<40}|")
# ----------------------------------------------------------------------


class DynamicCategoryMemory:
    """
    Maintains Dirichlet parameters (α) for a fixed set of categories.
    Updating adds observed counts to α, and predictive probabilities are
    α_i / Σ α.
    """

    def __init__(self, categories: List[str], prior: float = 1.0):
        self.categories = categories
        self.alpha = {c: prior for c in categories}
        self.total_observations = 0

    def observe(self, cat: str, count: int = 1) -> None:
        """Incorporate `count` observations for `cat`."""
        if cat not in self.alpha:
            raise ValueError(f"Unknown category: {cat}")
        self.alpha[cat] += count
        self.total_observations += count

    def predictive(self) -> Dict[str, float]:
        """Return the current predictive probability for each category."""
        alphas = list(self.alpha.values())
        probs = normalize(alphas)
        return dict(zip(self.categories, probs))

    def to_json(self) -> str:
        """Serialize the internal state to a JSON string."""
        state = {
            "categories": self.categories,
            "alpha": self.alpha,
            "total_observations": self.total_observations,
        }
        return json.dumps(state, indent=2)

    @classmethod
    def from_json(cls, data: str) -> "DynamicCategoryMemory":
        """Recreate an instance from a JSON string."""
        obj = json.loads(data)
        inst = cls(obj["categories"])
        inst.alpha = {c: float(v) for c, v in obj["alpha"].items()}
        inst.total_observations = int(obj["total_observations"])
        return inst

    def save(self, path: Path) -> None:
        """Write the JSON representation to `path`."""
        path.write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "DynamicCategoryMemory":
        """Load a saved state from `path`."""
        return cls.from_json(path.read_text(encoding="utf-8"))


# ----------------------------------------------------------------------
# Simple contextual stream simulation
# ----------------------------------------------------------------------
# Mapping of keywords to categories – this mimics a very crude semantic
# extractor that an AGI might use.
KEYWORD_MAP: Dict[str, str] = {
    # science / technology
    "quantum": "technology",
    "neural": "technology",
    "entropy": "science",
    "perplexity": "science",
    "bitcoin": "technology",
    "mining": "technology",
    # art / humanities
    "painting": "art",
    "music": "art",
    "poetry": "art",
    # philosophy / consciousness
    "consciousness": "philosophy",
    "curiosity": "philosophy",
    "entropy": "philosophy",
    # default fallback
    "default": "misc",
}


def categorize_token(token: str) -> str:
    """Return a category based on the token, falling back to 'misc'."""
    token = token.lower()
    return KEYWORD_MAP.get(token, "misc")


def simulate_stream(
    memory: DynamicCategoryMemory,
    tokens: List[str],
    batch: int = 5,
) -> None:
    """Feed tokens to the memory in batches, printing updates."""
    for i in range(0, len(tokens), batch):
        batch_tokens = tokens[i : i + batch]
        # Count category occurrences in this batch
        batch_counts: defaultdict = defaultdict(int)
        for t in batch_tokens:
            cat = categorize_token(t)
            batch_counts[cat] += 1

        # Apply observations
        for cat, cnt in batch_counts.items():
            memory.observe(cat, cnt)

        # Show current state
        probs = memory.predictive()
        pretty_print(
            f"After processing tokens {i+1}-{i+len(batch_tokens)}",
            probs,
        )


# ----------------------------------------------------------------------
# Demo when run as script
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Define the universe of categories we care about
    categories = ["technology", "science", "art", "philosophy", "misc"]
    mem = DynamicCategoryMemory(categories, prior=1.0)

    # Example token stream (could be extracted from a journal entry)
    example_text = """
    quantum entropy neural network perplexity bitcoin mining curiosity
    painting music poetry consciousness entropy quantum
    """
    tokens = [t for t in example_text.split() if t]

    print(textwrap.dedent("""\
        Dynamic Category Memory Demo
        ============================
        Starting with uniform priors (α=1) for each category.
    """))

    simulate_stream(mem, tokens, batch=4)

    # Save final state
    out_path = Path("category_memory_state.json")
    mem.save(out_path)
    print(f"\nFinal state saved to {out_path.resolve()}")
