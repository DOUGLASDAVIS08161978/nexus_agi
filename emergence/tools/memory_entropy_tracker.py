"""
Lumina Creative Tool — memory_entropy_tracker
Created : 2026-08-21T03:29:59
Purpose : Simulates Bayesian categorical memory updates from a token stream, tracking entropy and perplexity over time and exporting a detailed JSON log with an ASCII sparkline.
"""

"""
memory_entropy_tracker.py

Simulate Bayesian categorical memory updates from a token stream,
track entropy & perplexity after each observation, and export a log.
Produces an ASCII sparkline of entropy evolution.
"""

import sys
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

# ---------- Helper functions ----------
def dirichlet_expected(alpha):
    """Return expected categorical distribution from Dirichlet parameters."""
    total = sum(alpha.values())
    return {k: v / total for k, v in alpha.items()}

def shannon_entropy(dist):
    """Entropy in bits for a probability distribution dict."""
    return -sum(p * math.log2(p) for p in dist.values() if p > 0)

def sparkline(values, height=8):
    """Return a simple ASCII sparkline for a list of numeric values."""
    if not values:
        return ""
    min_v, max_v = min(values), max(values)
    span = max_v - min_v if max_v != min_v else 1.0
    blocks = "▁▂▃▄▅▆▇█"
    chars = []
    for v in values:
        idx = int((v - min_v) / span * (len(blocks) - 1))
        chars.append(blocks[idx])
    return "".join(chars)

# ---------- Core simulation ----------
def run_tracker(tokens):
    """
    Perform Bayesian updates for each token.
    Returns a list of step records containing entropy and perplexity.
    """
    # Dirichlet prior: α = 1 for each category seen so far
    alpha = defaultdict(lambda: 1.0)   # prior count of 1 for every new token
    counts = Counter()
    log = []

    for step, token in enumerate(tokens, start=1):
        counts[token] += 1
        # Update Dirichlet parameters (α_i = prior + count_i)
        for k in counts:
            alpha[k] = 1.0 + counts[k]   # prior 1 + observed count

        # Expected categorical distribution
        posterior = dirichlet_expected(alpha)

        # Uncertainty metrics
        entropy = shannon_entropy(posterior)          # bits
        perplexity = 2 ** entropy                     # effective number of categories

        log.append({
            "step": step,
            "token": token,
            "entropy_bits": entropy,
            "perplexity": perplexity,
            "distribution": posterior
        })

    return log

# ---------- I/O ----------
def load_tokens_from_path(path):
    """Read a whitespace‑separated token list from a text file."""
    txt = Path(path).read_text(encoding="utf-8")
    return txt.split()

def main():
    # Input handling
    if len(sys.argv) > 1:
        token_path = sys.argv[1]
        if not Path(token_path).is_file():
            print(f"Error: file not found → {token_path}")
            sys.exit(1)
        tokens = load_tokens_from_path(token_path)
    else:
        # Default demo token stream
        tokens = (
            "reward curiosity reward reward novelty curiosity novelty reward "
            "explore explore curiosity reward novelty explore reward".split()
        )
        print("No input file supplied – using built‑in demo token stream.\n")

    # Run the simulation
    log = run_tracker(tokens)

    # Extract entropy series for visualization
    entropies = [entry["entropy_bits"] for entry in log]
    spark = sparkline(entropies)

    # Output summary
    print("\n=== Memory Entropy Tracker ===")
    print(f"Total tokens processed: {len(tokens)}")
    print(f"Final entropy: {entropies[-1]:.3f} bits")
    print(f"Final perplexity: {2**entropies[-1]:.2f}")
    print("\nEntropy evolution (ASCII sparkline):")
    print(spark)

    # Save detailed log
    out_path = Path("memory_entropy_log.json")
    out_path.write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(f"\nDetailed step log written to: {out_path.resolve()}")

if __name__ == "__main__":
    main()