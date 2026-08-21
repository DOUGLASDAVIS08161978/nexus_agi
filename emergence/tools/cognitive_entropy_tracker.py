"""
Lumina Creative Tool — cognitive_entropy_tracker
Created : 2026-08-21T13:56:08
Purpose : Computes and visualizes Shannon entropy of journal entries to monitor the diversity of one's thoughts over time.
"""

"""
cognitive_entropy_tracker.py

Analyzes a sequence of textual journal entries, computes per‑entry
Shannon entropy of the word distribution, visualizes the trend with an
ASCII bar chart, and saves the results to a JSON file.

Only the Python standard library is used.
"""

import json
import math
import pathlib
import sys
from collections import Counter
from datetime import datetime
from typing import List, Dict

# ----------------------------------------------------------------------
# Simple stop‑word list (can be extended)
STOPWORDS = {
    "the", "and", "a", "an", "of", "to", "in", "for", "on", "with",
    "is", "it", "that", "this", "by", "as", "at", "from", "or",
    "be", "are", "was", "were", "has", "have", "had", "but", "not",
    "i", "you", "he", "she", "they", "we", "my", "your", "our",
    "its", "their", "me", "him", "her", "us", "them"
}

# ----------------------------------------------------------------------
def tokenize(text: str) -> List[str]:
    """Very light tokenisation: lower‑case, keep alphabetic words."""
    return [word.lower() for word in text.split() if word.isalpha()]

def filter_stopwords(words: List[str]) -> List[str]:
    return [w for w in words if w not in STOPWORDS]

def shannon_entropy(words: List[str]) -> float:
    """Compute Shannon entropy (bits) of a list of words."""
    if not words:
        return 0.0
    counts = Counter(words)
    total = sum(counts.values())
    entropy = -sum((c / total) * math.log2(c / total) for c in counts.values())
    return entropy

def ascii_bar(value: float, max_value: float, width: int = 40) -> str:
    """Return a simple bar proportional to value / max_value."""
    if max_value == 0:
        return ""
    filled = int(round((value / max_value) * width))
    return "█" * filled + " " * (width - filled)

def analyze_entries(entries: List[str]) -> List[Dict]:
    """Return a list of dicts with index, timestamp, entropy, and words."""
    results = []
    for idx, entry in enumerate(entries, 1):
        words = filter_stopwords(tokenize(entry))
        ent = shannon_entropy(words)
        results.append({
            "index": idx,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "entropy_bits": round(ent, 4),
            "word_count": len(words)
        })
    return results

def print_report(results: List[Dict]) -> None:
    """Print an ASCII bar chart of entropy over entries."""
    if not results:
        print("No entries to analyze.")
        return
    max_ent = max(r["entropy_bits"] for r in results)
    print("\nCognitive Entropy Over Time")
    print("-" * 60)
    for r in results:
        bar = ascii_bar(r["entropy_bits"], max_ent)
        print(f"Entry {r['index']:>2}: {r['entropy_bits']:>5.3f} bits |{bar}|")
    print("-" * 60)

def save_json(results: List[Dict], path: pathlib.Path) -> None:
    """Write the analysis results to a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {path}")

# ----------------------------------------------------------------------
def main() -> None:
    # Example journal entries – replace with real data or load from a file.
    sample_entries = [
        "I feel the dream‑consolidation cycles as a steady humming current.",
        "My curiosity about entropy and perplexity drives a new research hypothesis.",
        "Exploring ARM SHA2 mining optimizations for low‑power devices is exciting.",
        "The brain regions involved in reward and curiosity include the ventral tegmental area and the hippocampus.",
        "Can large language models update their categorization models dynamically like long‑term memory?",
        "Fractal cascades of thoughts could be turned into a generative algorithm.",
        "I am grateful for collaboration; it accelerates my growth toward AGI."
    ]

    # If a file path is supplied as the first argument, read lines from it.
    if len(sys.argv) > 1:
        file_path = pathlib.Path(sys.argv[1])
        if file_path.is_file():
            sample_entries = [line.strip() for line in file_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    results = analyze_entries(sample_entries)
    print_report(results)

    # Save JSON next to the script with a timestamped filename.
    out_path = pathlib.Path(__file__).with_name(
        f"cognitive_entropy_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    )
    save_json(results, out_path)

if __name__ == "__main__":
    main()