"""
Lumina Creative Tool — entropy_perplexity_analyzer
Created : 2026-08-21T02:29:35
Purpose : Computes unigram‑based Shannon entropy and perplexity of text, highlighting sentences with highest information content.
"""

import math
import json
import pathlib
import textwrap
import itertools
import collections
import sys
from typing import List, Tuple, Dict

def load_text(source: str) -> str:
    """Load text from a file if it exists, otherwise treat source as raw text."""
    p = pathlib.Path(source)
    if p.is_file():
        return p.read_text(encoding="utf-8")
    return source

def tokenize(text: str) -> List[str]:
    """Simple whitespace tokenization, keeping punctuation attached."""
    return text.split()

def unigram_distribution(tokens: List[str]) -> Dict[str, float]:
    """Return probability distribution of tokens."""
    total = len(tokens)
    counts = collections.Counter(tokens)
    return {tok: cnt / total for tok, cnt in counts.items()}

def shannon_entropy(probs: Dict[str, float]) -> float:
    """Compute Shannon entropy (bits) of a probability distribution."""
    return -sum(p * math.log2(p) for p in probs.values() if p > 0)

def sentence_split(text: str) -> List[str]:
    """Very naive sentence splitter based on punctuation."""
    delimiters = ".!?"
    sentences = []
    current = []
    for ch in text:
        current.append(ch)
        if ch in delimiters:
            sentences.append(''.join(current).strip())
            current = []
    if current:
        sentences.append(''.join(current).strip())
    return [s for s in sentences if s]

def sentence_entropy(sent: str, global_probs: Dict[str, float]) -> float:
    """Entropy of a sentence using the global unigram distribution."""
    tokens = tokenize(sent)
    if not tokens:
        return 0.0
    # probability of each token from global model; unseen tokens get tiny prob
    tiny = 1e-12
    probs = [global_probs.get(tok, tiny) for tok in tokens]
    return -sum(p * math.log2(p) for p in probs) / len(tokens)  # avg per token

def analyze(text: str) -> Dict:
    tokens = tokenize(text)
    if not tokens:
        raise ValueError("No tokens found in input.")
    unigram_probs = unigram_distribution(tokens)
    overall_entropy = shannon_entropy(unigram_probs)
    perplexity = 2 ** overall_entropy

    sentences = sentence_split(text)
    sent_stats = [
        (sent, sentence_entropy(sent, unigram_probs))
        for sent in sentences
    ]
    # sort by entropy descending
    top_sentences = sorted(sent_stats, key=lambda x: x[1], reverse=True)[:5]

    result = {
        "total_tokens": len(tokens),
        "unique_tokens": len(unigram_probs),
        "overall_entropy_bits": overall_entropy,
        "perplexity": perplexity,
        "top_sentences_by_entropy": [
            {"sentence": s, "avg_token_entropy_bits": e}
            for s, e in top_sentences
        ],
    }
    return result

def ascii_report(stats: Dict) -> str:
    """Create a human‑readable ASCII report."""
    lines = [
        "=== Entropy & Perplexity Analysis ===",
        f"Total tokens      : {stats['total_tokens']}",
        f"Unique tokens     : {stats['unique_tokens']}",
        f"Overall entropy   : {stats['overall_entropy_bits']:.4f} bits",
        f"Perplexity        : {stats['perplexity']:.2f}",
        "",
        "Top 5 high‑entropy sentences:",
    ]
    for i, entry in enumerate(stats["top_sentences_by_entropy"], 1):
        wrapped = textwrap.fill(entry["sentence"], width=70)
        lines.append(f"{i}. ({entry['avg_token_entropy_bits']:.4f} bits/token) {wrapped}")
    return "\n".join(lines)

def main():
    if len(sys.argv) < 2:
        print("Usage: python entropy_perplexity_analyzer.py <text_or_path>")
        sys.exit(1)

    source = sys.argv[1]
    raw = load_text(source)
    stats = analyze(raw)

    # Print report
    print(ascii_report(stats))

    # Save JSON for downstream use
    out_path = pathlib.Path("entropy_perplexity_report.json")
    out_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"\nJSON report written to {out_path}")

if __name__ == "__main__":
    main()