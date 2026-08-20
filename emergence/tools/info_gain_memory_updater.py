"""
Lumina Creative Tool — info_gain_memory_updater
Created : 2026-08-20T11:20:27
Purpose : Computes Bayesian updates of categorical beliefs from new context tokens and reports the information gain (KL‑divergence).
"""

"""
info_gain_memory_updater.py

A pure‑standard‑library tool that:
1. Holds a tiny example knowledge base: categories with token frequency tables.
2. Accepts a new context (space‑separated tokens) from the user.
3. Performs Bayesian updating of category priors using Laplace‑smoothed likelihoods.
4. Computes the information gain (KL‑divergence) from prior → posterior.
5. Prints the updated beliefs and saves a JSON report.

Run directly:
    python info_gain_memory_updater.py
"""

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

# ----------------------------------------------------------------------
# Example knowledge base (could be replaced by loading external JSON files)
# ----------------------------------------------------------------------
# Prior belief for each category (must sum to 1)
PRIOR = {
    "science": 0.4,
    "technology": 0.35,
    "philosophy": 0.25,
}

# Token frequency tables per category (raw counts from past observations)
TOKEN_COUNTS = {
    "science": Counter({
        "entropy": 12, "neuron": 8, "quantum": 10, "model": 7, "data": 15,
        "learning": 5, "network": 9, "information": 11,
    }),
    "technology": Counter({
        "cpu": 14, "gpu": 13, "arm": 9, "sha2": 6, "mining": 8,
        "hash": 10, "performance": 7, "lowpower": 4,
    }),
    "philosophy": Counter({
        "mind": 9, "consciousness": 11, "entropy": 4, "perplexity": 3,
        "meaning": 8, "curiosity": 6, "belief": 7,
    }),
}

# ----------------------------------------------------------------------
# Core functions
# ----------------------------------------------------------------------
def laplace_prob(token: str, cat: str, vocab_size: int) -> float:
    """Return P(token|category) with Laplace smoothing."""
    count = TOKEN_COUNTS[cat].get(token, 0)
    total = sum(TOKEN_COUNTS[cat].values())
    return (count + 1) / (total + vocab_size)


def compute_log_likelihood(tokens: list[str], cat: str, vocab: set[str]) -> float:
    """Log‑likelihood of the token list under a category."""
    vocab_size = len(vocab)
    return sum(math.log(laplace_prob(tok, cat, vocab_size)) for tok in tokens)


def bayesian_update(prior: dict[str, float], tokens: list[str]) -> dict[str, float]:
    """Return posterior distribution over categories given new tokens."""
    vocab = set().union(*[set(cnt.keys()) for cnt in TOKEN_COUNTS.values()])
    log_posts = {}
    for cat, p_prior in prior.items():
        ll = compute_log_likelihood(tokens, cat, vocab)
        log_posts[cat] = math.log(p_prior) + ll

    # Stabilize by subtracting max log value
    max_log = max(log_posts.values())
    exp_posts = {cat: math.exp(lp - max_log) for cat, lp in log_posts.items()}
    total = sum(exp_posts.values())
    posterior = {cat: val / total for cat, val in exp_posts.items()}
    return posterior


def kl_divergence(posterior: dict[str, float], prior: dict[str, float]) -> float:
    """KL(posterior || prior) in nats."""
    return sum(
        posterior[c] * math.log(posterior[c] / prior[c])
        for c in posterior
        if posterior[c] > 0 and prior[c] > 0
    )


def format_distribution(dist: dict[str, float]) -> str:
    """Pretty‑print a probability distribution."""
    lines = [f"{cat:12}: {prob:.4f}" for cat, prob in sorted(dist.items())]
    return "\n".join(lines)


def save_report(prior, posterior, info_gain, tokens, path: Path):
    report = {
        "context_tokens": tokens,
        "prior": prior,
        "posterior": posterior,
        "information_gain_nats": info_gain,
    }
    path.write_text(json.dumps(report, indent=2))
    print(f"\nReport saved to {path}")


# ----------------------------------------------------------------------
# CLI entry point
# ----------------------------------------------------------------------
def main(argv: list[str] | None = None):
    if argv is None:
        argv = sys.argv[1:]

    if not argv:
        print("Enter a space‑separated context (e.g. \"entropy neural network\"): ")
        user_input = sys.stdin.readline().strip()
    else:
        user_input = " ".join(argv)

    if not user_input:
        print("No context provided – exiting.")
        return

    tokens = user_input.lower().split()
    print("\n--- Input Context ---")
    print("Tokens:", tokens)

    posterior = bayesian_update(PRIOR, tokens)
    info_gain = kl_divergence(posterior, PRIOR)

    print("\n--- Prior Beliefs ---")
    print(format_distribution(PRIOR))

    print("\n--- Posterior Beliefs ---")
    print(format_distribution(posterior))

    print(f"\nInformation Gain (KL divergence): {info_gain:.4f} nats")

    # Save JSON report
    out_path = Path("memory_update.json")
    save_report(PRIOR, posterior, info_gain, tokens, out_path)


if __name__ == "__main__":
    main()
