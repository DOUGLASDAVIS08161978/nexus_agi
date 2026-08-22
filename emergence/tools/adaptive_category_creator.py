"""
Lumina Creative Tool — adaptive_category_creator
Created : 2026-08-22T16:39:03
Purpose : Dynamically builds token‑based categories from a text stream while reporting entropy, perplexity, and an ASCII timeline.
"""

import sys
import math
import json
import pathlib
import collections
import itertools
import textwrap

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
ALPHA = 1.0               # Laplace smoothing parameter
THRESHOLD = 1e-4          # Minimum probability to consider token explained
TOP_TOKENS = 5            # How many top tokens to show per category
OUTPUT_JSON = "adaptive_categories.json"

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------
def tokenize(text: str):
    """Very simple whitespace tokenizer, lower‑casing."""
    return [t.lower() for t in text.split() if t]

def shannon_entropy(counter: collections.Counter):
    """Compute Shannon entropy (bits) of a token distribution."""
    total = sum(counter.values())
    if total == 0:
        return 0.0
    ent = 0.0
    for cnt in counter.values():
        p = cnt / total
        ent -= p * math.log2(p)
    return ent

def token_probability(token: str, cat_counter: collections.Counter, vocab_size: int):
    """Laplace‑smoothed probability of a token under a category."""
    count = cat_counter.get(token, 0)
    total = sum(cat_counter.values())
    return (count + ALPHA) / (total + ALPHA * vocab_size)

def ascii_bar(value: float, width: int = 30):
    """Render a simple bar for a probability (0‑1)."""
    filled = int(value * width)
    return "[" + "#" * filled + "-" * (width - filled) + f"] {value:.3f}"

# ------------------------------------------------------------
# Core adaptive categorization
# ------------------------------------------------------------
def adaptive_categorize(tokens):
    """Iterate over tokens, maintaining a list of category counters."""
    categories = []               # list of Counter objects
    vocab = set(tokens)           # global vocabulary for smoothing
    vocab_size = len(vocab) or 1

    # For reporting category probabilities over time
    prob_history = []

    for idx, token in enumerate(tokens, 1):
        # Compute probability of token under each existing category
        probs = [token_probability(token, cat, vocab_size) for cat in categories]

        # Determine best category (or none)
        if probs and max(probs) >= THRESHOLD:
            best_idx = probs.index(max(probs))
            categories[best_idx][token] += 1
            chosen = best_idx
        else:
            # Create a fresh category for this token
            new_cat = collections.Counter({token: 1})
            categories.append(new_cat)
            chosen = len(categories) - 1

        # Record normalized probabilities for visualization
        total = sum(probs) + (THRESHOLD if not probs else 0)
        norm_probs = [p / total for p in probs] + ([1.0] if not probs else [])
        prob_history.append((idx, chosen, norm_probs))

    return categories, prob_history, vocab_size

# ------------------------------------------------------------
# Reporting
# ------------------------------------------------------------
def print_summary(categories, vocab_size, prob_history):
    overall_counter = collections.Counter()
    for cat in categories:
        overall_counter.update(cat)

    overall_entropy = shannon_entropy(overall_counter)
    perplexity = 2 ** overall_entropy

    print("\n=== Adaptive Category Summary ===")
    print(f"Total tokens processed : {sum(overall_counter.values())}")
    print(f"Vocabulary size        : {vocab_size}")
    print(f"Overall Shannon entropy: {overall_entropy:.4f} bits")
    print(f"Perplexity             : {perplexity:.2f}\n")

    for i, cat in enumerate(categories, 1):
        top = cat.most_common(TOP_TOKENS)
        top_str = ", ".join(f"{tok}:{cnt}" for tok, cnt in top)
        print(f"Category {i:2d} | size: {sum(cat.values():5d)} | top tokens: {top_str}")

    # Simple ASCII timeline of category assignments (first 100 tokens)
    print("\nToken → Category (first 100 tokens)")
    timeline = "".join(str(chosen % 10) for _, chosen, _ in prob_history[:100])
    print(textwrap.fill(timeline, width=80))

def save_json(categories, path=OUTPUT_JSON):
    data = {
        f"category_{i+1}": dict(cat) for i, cat in enumerate(categories)
    }
    pathlib.Path(path).write_text(json.dumps(data, indent=2))
    print(f"\nCategory data saved to {path}")

# ------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------
def main():
    if len(sys.argv) > 1:
        txt_path = pathlib.Path(sys.argv[1])
        if not txt_path.is_file():
            print(f"File not found: {txt_path}")
            return
        text = txt_path.read_text(encoding="utf-8")
    else:
        # Fallback sample text
        text = """Artificial intelligence strives to model cognition.
                  Entropy and perplexity are linked through information theory.
                  Bitcoin mining on ARM devices faces power constraints.
                  Reward circuits in the brain drive curiosity and learning."""
        print("No input file given – using built‑in sample text.\n")

    tokens = tokenize(text)
    categories, prob_history, vocab_size = adaptive_categorize(tokens)
    print_summary(categories, vocab_size, prob_history)
    save_json(categories)

if __name__ == "__main__":
    main()
