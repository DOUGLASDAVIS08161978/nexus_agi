"""
Lumina Creative Tool — concept_drift_detector
Created : 2026-08-21T22:20:17
Purpose : Detects significant shifts in token distributions over time using KL‑divergence and logs drift events as JSON.
"""

"""
concept_drift_detector.py

Detects shifts (concept drift) in a token stream using KL‑divergence between
consecutive sliding windows.  Outputs a JSON file with drift events and prints
a short summary.

Usage (example):
    python concept_drift_detector.py tokens.txt --window 100 --threshold 0.5 --out drift.json
"""

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import List, Dict, Tuple

EPS = 1e-12  # smoothing for zero probabilities


def read_tokens(path: Path) -> List[List[str]]:
    """
    Reads a file where each line is a space‑separated list of tokens.
    Returns a list of token lists (one per line).
    """
    if not path.is_file():
        sys.exit(f"❌ Input file not found: {path}")
    lines = path.read_text(encoding="utf8").splitlines()
    return [line.strip().split() for line in lines if line.strip()]


def sliding_windows(
    token_lines: List[List[str]], size: int
) -> List[Counter]:
    """
    Creates overlapping windows of `size` lines and returns a list of Counters
    (token frequencies) for each window.
    """
    windows = []
    for i in range(len(token_lines) - size + 1):
        window_tokens = (tok for line in token_lines[i : i + size] for tok in line)
        windows.append(Counter(window_tokens))
    return windows


def normalize(counter: Counter) -> Dict[str, float]:
    """Converts a Counter to a probability distribution (adds EPS smoothing)."""
    total = sum(counter.values()) + EPS * len(counter)
    return {k: (v + EPS) / total for k, v in counter.items()}


def kl_divergence(p: Dict[str, float], q: Dict[str, float]) -> float:
    """
    KL(p‖q) = Σ p(i) * log(p(i)/q(i)).
    Missing keys in q are treated with EPS smoothing.
    """
    divergence = 0.0
    for token, p_prob in p.items():
        q_prob = q.get(token, EPS)
        divergence += p_prob * math.log(p_prob / q_prob)
    return divergence


def detect_drift(
    windows: List[Counter], threshold: float
) -> List[Tuple[int, float]]:
    """
    Computes KL divergence between consecutive windows.
    Returns a list of (window_index, divergence) where divergence > threshold.
    """
    drifts = []
    if len(windows) < 2:
        return drifts

    prev_dist = normalize(windows[0])
    for idx in range(1, len(windows)):
        cur_dist = normalize(windows[idx])
        kl = kl_divergence(cur_dist, prev_dist)
        if kl > threshold:
            drifts.append((idx, kl))
        prev_dist = cur_dist
    return drifts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect concept drift in a token stream using KL‑divergence."
    )
    parser.add_argument("input", type=Path, help="Path to token file (one line = token list).")
    parser.add_argument(
        "--window",
        type=int,
        default=100,
        help="Sliding window size in lines (default: 100).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="KL‑divergence threshold to flag drift (default: 0.5).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("drift_events.json"),
        help="File to write drift events (default: drift_events.json).",
    )
    args = parser.parse_args()

    token_lines = read_tokens(args.input)
    if len(token_lines) < args.window:
        sys.exit("❌ Not enough lines for the requested window size.")

    windows = sliding_windows(token_lines, args.window)
    drifts = detect_drift(windows, args.threshold)

    # Prepare JSON output
    events = [
        {"window_start_line": i, "kl_divergence": round(kl, 4)}
        for i, kl in drifts
    ]
    args.out.write_text(json.dumps(events, indent=2), encoding="utf8")

    # Human‑readable summary
    print(f"🗂 Processed {len(token_lines)} lines → {len(windows)} windows (size={args.window})")
    print(f"🔎 Detected {len(events)} drift events (threshold={args.threshold})")
    if events:
        print("\nTop drift events:")
        for ev in events[:5]:
            line = ev["window_start_line"]
            kl = ev["kl_divergence"]
            print(f"  • Window starting at line {line}: KL = {kl}")
        if len(events) > 5:
            print(f"  …and {len(events)-5} more.")
    else:
        print("✅ No significant drift detected.")
    print(f"\n📁 Events saved to: {args.out}")


if __name__ == "__main__":
    main()
