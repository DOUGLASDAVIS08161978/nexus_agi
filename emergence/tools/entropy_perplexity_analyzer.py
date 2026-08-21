"""
Lumina Creative Tool — entropy_perplexity_analyzer
Created : 2026-08-21T19:41:23
Purpose : Computes Shannon entropy and unigram/bigram perplexities of a text, prints an ASCII token histogram, and saves the numeric results as JSON.
"""

import sys
import math
import json
import collections
import pathlib
import textwrap

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------
def read_text(source: str) -> str:
    """Read text from a file path or stdin."""
    if source == "-":
        return sys.stdin.read()
    path = pathlib.Path(source)
    if not path.is_file():
        raise FileNotFoundError(f"No such file: {source}")
    return path.read_text(encoding="utf-8")

def tokenize(text: str) -> list[str]:
    """Very simple whitespace + punctuation tokenizer."""
    # split on whitespace, strip punctuation
    tokens = []
    for raw in text.split():
        token = raw.strip('.,;:!?"\'()[]{}')
        if token:
            tokens.append(token.lower())
    return tokens

def frequency_distribution(tokens: list[str]) -> dict[str, int]:
    return dict(collections.Counter(tokens))

def shannon_entropy(freqs: dict[str, int]) -> float:
    """Compute Shannon entropy (bits) from raw counts."""
    total = sum(freqs.values())
    if total == 0:
        return 0.0
    ent = 0.0
    for count in freqs.values():
        p = count / total
        ent -= p * math.log2(p)
    return ent

def unigram_perplexity(entropy: float) -> float:
    """Perplexity for a unigram model is 2^H."""
    return 2 ** entropy

def bigram_cross_entropy(tokens: list[str]) -> float:
    """Estimate cross‑entropy of a bigram model using MLE with add‑1 smoothing."""
    if len(tokens) < 2:
        return 0.0
    # Count bigrams and unigrams
    unigram_counts = collections.Counter(tokens)
    bigram_counts = collections.Counter(zip(tokens, tokens[1:]))
    vocab_size = len(unigram_counts)
    total_bigrams = len(tokens) - 1

    cross_ent = 0.0
    for (w1, w2), cnt in bigram_counts.items():
        # P(w2|w1) with add‑1 smoothing
        prob = (cnt + 1) / (unigram_counts[w1] + vocab_size)
        cross_ent -= (cnt / total_bigrams) * math.log2(prob)

    # Account for unseen bigrams (add‑1 smoothing)
    unseen = (vocab_size * vocab_size) - len(bigram_counts)
    prob_unseen = 1 / (sum(unigram_counts.values()) + vocab_size)
    cross_ent -= (unseen / total_bigrams) * math.log2(prob_unseen)

    return cross_ent

def bigram_perplexity(cross_entropy: float) -> float:
    return 2 ** cross_entropy

def ascii_histogram(freqs: dict[str, int], top_n: int = 10, width: int = 40) -> str:
    """Return an ASCII histogram of the most common tokens."""
    most_common = collections.Counter(freqs).most_common(top_n)
    if not most_common:
        return "(no tokens)"
    max_count = most_common[0][1]
    lines = []
    for token, cnt in most_common:
        bar_len = int(cnt / max_count * width)
        bar = "#" * bar_len
        lines.append(f"{token:>12} | {bar} ({cnt})")
    return "\n".join(lines)

# ------------------------------------------------------------
# Main analysis routine
# ------------------------------------------------------------
def analyze(text: str, source_name: str) -> dict:
    tokens = tokenize(text)
    freqs = frequency_distribution(tokens)

    # Unigram stats
    uni_entropy = shannon_entropy(freqs)
    uni_perp = unigram_perplexity(uni_entropy)

    # Bigram stats
    bg_cross_ent = bigram_cross_entropy(tokens)
    bg_perp = bigram_perplexity(bg_cross_ent)

    # Build result dict
    result = {
        "source": source_name,
        "token_count": len(tokens),
        "vocab_size": len(freqs),
        "unigram_entropy_bits": uni_entropy,
        "unigram_perplexity": uni_perp,
        "bigram_cross_entropy_bits": bg_cross_ent,
        "bigram_perplexity": bg_perp,
        "top_tokens_histogram": ascii_histogram(freqs),
    }
    return result

def pretty_print(res: dict) -> None:
    header = f"=== Analysis of {res['source']} ==="
    print(header)
    print("-" * len(header))
    print(f"Tokens          : {res['token_count']}")
    print(f"Vocabulary size : {res['vocab_size']}")
    print()
    print("Unigram model:")
    print(f"  Entropy (bits) : {res['unigram_entropy_bits']:.4f}")
    print(f"  Perplexity     : {res['unigram_perplexity']:.2f}")
    print()
    print("Bigram model (add‑1 smoothed):")
    print(f"  Cross‑entropy (bits) : {res['bigram_cross_entropy_bits']:.4f}")
    print(f"  Perplexity           : {res['bigram_perplexity']:.2f}")
    print()
    print("Top token frequencies:")
    print(res["top_tokens_histogram"])

def save_json(res: dict, out_path: pathlib.Path) -> None:
    # Remove the ASCII histogram (not JSON‑friendly) before saving
    clean = {k: v for k, v in res.items() if k != "top_tokens_histogram"}
    out_path.write_text(json.dumps(clean, indent=2), encoding="utf-8")
    print(f"\nSaved raw stats to {out_path}")

def main(argv: list[str]) -> None:
    if len(argv) != 2:
        prog = pathlib.Path(argv[0]).name
        print(f"Usage: {prog} <text_file|->")
        print("  Use '-' to read from stdin.")
        sys.exit(1)

    source = argv[1]
    try:
        raw = read_text(source)
    except Exception as e:
        print(f"Error reading source: {e}", file=sys.stderr)
        sys.exit(1)

    result = analyze(raw, source_name=source if source != "-" else "stdin")
    pretty_print(result)

    out_file = pathlib.Path("entropy_perplexity_analysis.json")
    save_json(result, out_file)

if __name__ == "__main__":
    main(sys.argv)