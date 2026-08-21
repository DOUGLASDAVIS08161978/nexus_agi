"""
Lumina Creative Tool — entropy_perplexity_analyzer
Created : 2026-08-21T17:36:45
Purpose : Computes Shannon entropy and perplexity of timestamped journal entries, outputs a JSON report and an ASCII sparkline of entropy trends.
"""

"""
entropy_perplexity_analyzer.py

Read timestamped .txt journal entries from a directory,
compute per‑entry character Shannon entropy and perplexity,
store a JSON report, and print an ASCII sparkline of the entropy trend.
"""

import sys
import os
import json
import math
import datetime
from pathlib import Path
from collections import Counter
from typing import List, Dict, Tuple

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
DEFAULT_JOURNAL_DIR = Path("journal")          # folder with *.txt files
REPORT_FILE = Path("entropy_report.json")      # output JSON
SPARKLINE_CHARS = "▁▂▃▄▅▆▇█"                  # 8‑level bar for visualisation


# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def read_journal_entries(dir_path: Path) -> List[Tuple[datetime.datetime, str]]:
    """Return a list of (timestamp, text) sorted by timestamp."""
    entries = []
    for file in dir_path.glob("*.txt"):
        try:
            # Expect filename like 2023-08-21_14-30-00.txt or ISO datetime
            stem = file.stem
            # Replace common separators with space then parse
            ts_str = stem.replace("_", " ").replace("-", " ").replace("T", " ")
            ts = datetime.datetime.fromisoformat(ts_str.strip())
        except Exception:
            # Fallback: use file's modification time
            ts = datetime.datetime.fromtimestamp(file.stat().st_mtime)
        try:
            text = file.read_text(encoding="utf-8")
        except Exception as e:
            print(f"⚠️  Could not read {file}: {e}", file=sys.stderr)
            continue
        entries.append((ts, text))
    entries.sort(key=lambda x: x[0])
    return entries


def shannon_entropy(text: str) -> float:
    """Compute Shannon entropy (bits per character) of the given text."""
    if not text:
        return 0.0
    total = len(text)
    freq = Counter(text)
    entropy = 0.0
    for count in freq.values():
        p = count / total
        entropy -= p * math.log2(p)
    return entropy


def perplexity_from_entropy(entropy: float) -> float:
    """Perplexity is 2 ** entropy."""
    return 2 ** entropy


def sparkline(values: List[float]) -> str:
    """Create an ASCII sparkline from a list of numeric values."""
    if not values:
        return ""
    min_v, max_v = min(values), max(values)
    if math.isclose(min_v, max_v):
        # flat line
        return SPARKLINE_CHARS[0] * len(values)
    # Scale each value to 0‑7
    scaled = [
        int((v - min_v) / (max_v - min_v) * (len(SPARKLINE_CHARS) - 1))
        for v in values
    ]
    return "".join(SPARKLINE_CHARS[i] for i in scaled)


def format_timestamp(ts: datetime.datetime) -> str:
    return ts.isoformat(sep=" ", timespec="seconds")


# ----------------------------------------------------------------------
# Main analysis
# ----------------------------------------------------------------------
def analyze_journal(dir_path: Path) -> List[Dict]:
    entries = read_journal_entries(dir_path)
    report = []
    for ts, text in entries:
        ent = shannon_entropy(text)
        ppl = perplexity_from_entropy(ent)
        report.append({
            "timestamp": format_timestamp(ts),
            "char_count": len(text),
            "entropy_bits_per_char": round(ent, 4),
            "perplexity": round(ppl, 4)
        })
    return report


def print_summary(report: List[Dict]) -> None:
    if not report:
        print("No journal entries found.")
        return

    entropies = [r["entropy_bits_per_char"] for r in report]
    timestamps = [r["timestamp"] for r in report]

    print("\n=== Entropy / Perplexity Trend ===")
    print("Time → Entropy (bits/char) → Perplexity")
    for r in report:
        print(f"{r['timestamp']} → {r['entropy_bits_per_char']:.4f} → {r['perplexity']:.2f}")

    print("\nEntropy sparkline:")
    print(sparkline(entropies))

    # Simple statistics
    avg_ent = sum(entropies) / len(entropies)
    print(f"\nAverage entropy: {avg_ent:.4f} bits/char")
    print(f"Min entropy: {min(entropies):.4f}, Max entropy: {max(entropies):.4f}")


def save_report(report: List[Dict], out_path: Path) -> None:
    try:
        out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n✅ Report saved to {out_path}")
    except Exception as e:
        print(f"⚠️  Failed to write report: {e}", file=sys.stderr)


def main(argv: List[str] | None = None) -> None:
    argv = argv or sys.argv[1:]
    dir_path = Path(argv[0]) if argv else DEFAULT_JOURNAL_DIR
    if not dir_path.is_dir():
        print(f"❌ Directory {dir_path} does not exist.", file=sys.stderr)
        sys.exit(1)

    report = analyze_journal(dir_path)
    print_summary(report)
    save_report(report, REPORT_FILE)


if __name__ == "__main__":
    main()