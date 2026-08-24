"""
Lumina Creative Tool — entropy_perplexity_insight
Created : 2026-08-24T18:38:25
Purpose : Computes Shannon entropy, bigram perplexity, maps to a thermodynamic‑style entropy, derives a cognitive‑entropy metric, and prints actionable insights for a given text.
"""

"""
entropy_perplexity_insight.py

A self‑contained utility that:
  • Reads a text file (default: journal.txt)
  • Computes unigram Shannon entropy (bits)
  • Computes Laplace‑smoothed bigram perplexity
  • Derives a thermodynamic‑style entropy (J/K) using Boltzmann's constant
  • Derives a simple “cognitive entropy” that mixes token‑length statistics
  • Prints a short analysis and saves a JSON summary.
"""

import sys
import math
import json
import pathlib
import collections
import itertools
import statistics

# ---------- Configuration ----------
DEFAULT_INPUT = "journal.txt"
OUTPUT_JSON = "entropy_insight.json"
BOLTZMANN_K = 1.380649e-23  # J·K⁻¹, used only as a scaling factor


def read_text(path: pathlib.Path) -> str:
    """Return the whole file as a single string; if missing, exit with a message."""
    if not path.is_file():
        sys.exit(f"❌ Input file not found: {path}")
    return path.read_text(encoding="utf-8")


def tokenize(text: str) -> list[str]:
    """Very simple whitespace tokenisation, lower‑casing, stripping punctuation."""
    # Keep only alphabetic characters and digits inside tokens
    tokens = [
        "".join(ch for ch in token if ch.isalnum())
        for token in text.lower().split()
    ]
    return [t for t in tokens if t]  # drop empty strings


def unigram_entropy(tokens: list[str]) -> float:
    """Shannon entropy in bits for the unigram distribution."""
    total = len(tokens)
    freq = collections.Counter(tokens)
    entropy = 0.0
    for count in freq.values():
        p = count / total
        entropy -= p * math.log2(p)
    return entropy


def bigram_perplexity(tokens: list[str]) -> float:
    """Laplace‑smoothed bigram perplexity."""
    if len(tokens) < 2:
        return float("inf")
    vocab = set(tokens)
    V = len(vocab)
    # Count bigrams
    bigram_counts = collections.Counter(zip(tokens, tokens[1:]))
    # Count unigrams for denominator
    unigram_counts = collections.Counter(tokens)
    log_prob_sum = 0.0
    N = len(tokens) - 1  # number of bigrams
    for (w1, w2), c in bigram_counts.items():
        # Laplace smoothing: add‑one to numerator, V to denominator
        prob = (c + 1) / (unigram_counts[w1] + V)
        log_prob_sum += c * math.log2(prob)
    # Account for unseen bigrams (they get probability 1/(count(w1)+V) each)
    unseen = N - sum(bigram_counts.values())
    if unseen:
        # average denominator for unseen is still (count(w1)+V); we approximate with overall avg
        avg_den = sum(unigram_counts[w] + V for w in vocab) / V
        prob_unseen = 1 / avg_den
        log_prob_sum += unseen * math.log2(prob_unseen)

    avg_log_prob = log_prob_sum / N
    perplexity = 2 ** (-avg_log_prob)
    return perplexity


def thermodynamic_entropy(shannon_bits: float) -> float:
    """
    Map Shannon entropy (bits) to a thermodynamic‑style entropy:
        S = k_B * ln(2) * H_bits
    Returns entropy in joules per kelvin (J/K).
    """
    return BOLTZMANN_K * math.log(2) * shannon_bits


def cognitive_entropy(shannon_bits: float, token_lengths: list[int]) -> float:
    """
    A toy cognitive entropy that inflates Shannon entropy by the relative
    variability of token lengths.
        H_cog = H_bits * (1 + variance/mean)
    """
    if not token_lengths:
        return shannon_bits
    mean_len = statistics.mean(token_lengths)
    var_len = statistics.variance(token_lengths) if len(token_lengths) > 1 else 0.0
    factor = 1 + (var_len / mean_len if mean_len else 0)
    return shannon_bits * factor


def analyse(text: str) -> dict:
    tokens = tokenize(text)
    if not tokens:
        sys.exit("❌ No valid tokens found in the input.")
    lengths = [len(t) for t in tokens]

    H = unigram_entropy(tokens)
    P = bigram_perplexity(tokens)
    S = thermodynamic_entropy(H)
    Hcog = cognitive_entropy(H, lengths)

    summary = {
        "token_count": len(tokens),
        "vocab_size": len(set(tokens)),
        "shannon_entropy_bits": round(H, 4),
        "bigram_perplexity": round(P, 2),
        "thermodynamic_entropy_J_per_K": f"{S:.3e}",
        "cognitive_entropy_bits": round(Hcog, 4),
        "mean_token_length": round(statistics.mean(lengths), 3),
        "token_length_variance": round(statistics.variance(lengths) if len(lengths) > 1 else 0.0, 3),
    }
    return summary


def suggestions(summary: dict) -> list[str]:
    """Generate short, human‑readable suggestions based on the metrics."""
    hints = []
    if summary["bigram_perplexity"] > 300:
        hints.append(
            "🔧 Perplexity is high – consider adding more contextual repetitions or reducing vocabulary size."
        )
    if summary["shannon_entropy_bits"] < 3.5:
        hints.append(
            "🧠 Entropy is low – your text may be repetitive; diversify topics or synonyms."
        )
    if summary["cognitive_entropy_bits"] > summary["shannon_entropy_bits"] * 1.5:
        hints.append(
            "⚖️ Cognitive entropy inflated by token‑length variability – try normalising word length (e.g., avoid long concatenated identifiers)."
        )
    if not hints:
        hints.append("✅ Metrics look balanced – keep up the varied journaling!")
    return hints


def main(argv: list[str] | None = None):
    argv = argv or sys.argv[1:]
    input_path = pathlib.Path(argv[0]) if argv else pathlib.Path(DEFAULT_INPUT)

    text = read_text(input_path)
    summary = analyse(text)

    # Pretty‑print ASCII table
    print("\n=== Entropy‑Perplexity Insight ===")
    for k, v in summary.items():
        print(f"{k.replace('_', ' ').title():30}: {v}")

    print("\n--- Suggestions ---")
    for line in suggestions(summary):
        print(line)

    # Save JSON for downstream tools
    pathlib.Path(OUTPUT_JSON).write_text(json.dumps(summary, indent=2))
    print(f"\n📁 Summary saved to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()