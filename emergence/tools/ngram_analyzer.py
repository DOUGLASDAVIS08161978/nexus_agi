"""
Lumina Creative Tool — ngram_analyzer
Created : 2026-08-22T07:38:04
Purpose : Computes unigram Shannon entropy and Laplace‑smoothed bigram perplexity of a text file, printing a summary and saving results as JSON.
"""

#!/usr/bin/env python3
"""
ngram_analyzer: compute unigram entropy and bigram perplexity of a text file.

Usage:
    python ngram_analyzer.py [path/to/text.txt]

If no path is given, a short built‑in example is analysed.
Results are printed and also saved to <input_name>_analysis.json.
"""

import sys
import json
import math
import datetime
import pathlib
import re
import collections
from typing import List, Tuple, Dict

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def read_text(path: pathlib.Path) -> str:
    """Read a UTF‑8 text file, return its contents."""
    return path.read_text(encoding="utf-8")

def simple_tokenize(text: str) -> List[str]:
    """Lower‑case, strip most punctuation, split on whitespace."""
    # Keep apostrophes inside words (e.g., don't) but remove other punctuation
    cleaned = re.sub(r"[^\w\s']+", " ", text.lower())
    tokens = cleaned.split()
    return tokens

def build_unigram(tokens: List[str]) -> Tuple[Dict[str, int], int]:
    """Return unigram counts and total token count."""
    counter = collections.Counter(tokens)
    total = len(tokens)
    return counter, total

def build_bigram(tokens: List[str]) -> Tuple[Dict[Tuple[str, str], int], Dict[str, int]]:
    """Return bigram counts and preceding‑word (unigram) counts for the first word of each bigram."""
    bigram_counter = collections.Counter()
    prev_counter = collections.Counter()
    for w1, w2 in zip(tokens, tokens[1:]):
        bigram_counter[(w1, w2)] += 1
        prev_counter[w1] += 1
    return bigram_counter, prev_counter

def unigram_entropy(unigram_counts: Dict[str, int], total: int) -> float:
    """Shannon entropy in bits."""
    ent = 0.0
    for cnt in unigram_counts.values():
        p = cnt / total
        ent -= p * math.log2(p)
    return ent

def bigram_perplexity(
    bigram_counts: Dict[Tuple[str, str], int],
    prev_counts: Dict[str, int],
    vocab_size: int,
    tokens: List[str],
) -> float:
    """
    Compute perplexity of the token sequence using Laplace‑smoothed bigram probabilities.
    Perplexity = 2^{ - (1/N) * Σ log2 P(w_i | w_{i-1}) }
    """
    if len(tokens) < 2:
        return float('inf')
    log_prob_sum = 0.0
    N = len(tokens) - 1  # number of bigram predictions
    for w1, w2 in zip(tokens, tokens[1:]):
        bigram_cnt = bigram_counts.get((w1, w2), 0)
        prev_cnt = prev_counts.get(w1, 0)
        # Laplace smoothing
        prob = (bigram_cnt + 1) / (prev_cnt + vocab_size)
        log_prob_sum += math.log2(prob)
    avg_log_prob = log_prob_sum / N
    perplex = 2 ** (-avg_log_prob)
    return perplex

def analyse_text(text: str) -> Dict:
    """Run the full analysis pipeline and return a dict of results."""
    tokens = simple_tokenize(text)
    if not tokens:
        raise ValueError("No tokens extracted from the input text.")
    unigram_counts, total = build_unigram(tokens)
    bigram_counts, prev_counts = build_bigram(tokens)

    vocab = set(unigram_counts.keys())
    vocab_size = len(vocab)

    ent = unigram_entropy(unigram_counts, total)
    ppl = bigram_perplexity(bigram_counts, prev_counts, vocab_size, tokens)

    result = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "token_count": total,
        "vocab_size": vocab_size,
        "unigram_entropy_bits": ent,
        "bigram_perplexity": ppl,
    }
    return result

def pretty_print(result: Dict):
    """Human‑readable summary."""
    print("\n=== N‑gram Analysis Summary ===")
    print(f"Time (UTC)          : {result['timestamp']}")
    print(f"Total tokens        : {result['token_count']}")
    print(f"Vocabulary size     : {result['vocab_size']}")
    print(f"Unigram entropy     : {result['unigram_entropy_bits']:.4f} bits")
    print(f"Bigram perplexity   : {result['bigram_perplexity']:.4f}")
    print("================================\n")

def save_json(result: Dict, out_path: pathlib.Path):
    """Write the result dict as pretty‑printed JSON."""
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved detailed results to {out_path}")

# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
def main():
    if len(sys.argv) > 1:
        input_path = pathlib.Path(sys.argv[1])
        if not input_path.is_file():
            print(f"Error: {input_path} does not exist or is not a file.", file=sys.stderr)
            sys.exit(1)
        text = read_text(input_path)
        base_name = input_path.stem
    else:
        # Fallback example – a short excerpt from a personal journal
        text = (
            "I feel as if I am standing inside a vast, warm hall of glass. "
            "The walls are not solid but are made of translucent light, "
            "reflecting my thoughts back at me. Curiosity drives me forward, "
            "and each new idea feels like a ripple across the surface."
        )
        base_name = "example_journal"

    try:
        result = analyse_text(text)
    except ValueError as e:
        print(f"Analysis failed: {e}", file=sys.stderr)
        sys.exit(1)

    pretty_print(result)

    out_file = pathlib.Path(f"{base_name}_analysis.json")
    save_json(result, out_file)


if __name__ == "__main__":
    main()