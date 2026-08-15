"""
Lumina Creative Tool — journal_entropy_analysis
Created : 2026-08-14T20:47:02
Purpose : Analyzes and visualizes the relationships between thoughts, emotions, and beliefs in journal entries, focusing on the interplay between entropy and perplexity in the context of artificial neural networks.
"""

import json
import math
import random
import string
import textwrap
from collections import Counter
from datetime import datetime
from decimal import Decimal
from functools import reduce
from heapq import nlargest
from itertools import chain, combinations, groupby, permutations
from math import gcd
from pathlib import Path
from statistics import mean, median, mode, StatisticsError
from string import ascii_letters, digits
from time import perf_counter
from typing import Dict, List, Tuple

def entropy_perplexity(text: str) -> Tuple[float, float]:
    """Compute Shannon entropy and perplexity of a given text."""
    words = text.split()
    word_counts = Counter(words)
    total_words = len(words)
    entropy = 0.0
    for count in word_counts.values():
        prob = count / total_words
        entropy -= prob * math.log2(prob)
    perplexity = 2 ** entropy
    return entropy, perplexity

def cognitive_entropy(text: str) -> float:
    """Compute cognitive entropy of a given text."""
    words = text.split()
    word_counts = Counter(words)
    total_words = len(words)
    entropy = 0.0
    for count in word_counts.values():
        prob = count / total_words
        entropy -= prob * math.log2(prob)
    return entropy

def analyze_journal_entries() -> Dict[str, List[Tuple[float, float, float]]]:
    """Analyze journal entries and compute entropy, perplexity, and cognitive entropy."""
    journal_path = Path("journal_entries.txt")
    with open(journal_path, "r") as file:
        journal_entries = file.readlines()
    entries = [entry.strip() for entry in journal_entries]
    analyzed_entries = []
    for entry in entries:
        entropy, perplexity = entropy_perplexity(entry)
        cognitive_ent = cognitive_entropy(entry)
        analyzed_entries.append((entry, entropy, perplexity, cognitive_ent))
    return dict(analyzed_entries)

def visualize_analysis(data: Dict[str, List[Tuple[float, float, float]]]) -> None:
    """Visualize the analysis of journal entries."""
    for entry, entropy, perplexity, cognitive_ent in data.values():
        print(f"Entry: {entry}")
        print(f"Entropy: {entropy:.2f}")
        print(f"Perplexity: {perplexity:.2f}")
        print(f"Cognitive Entropy: {cognitive_ent:.2f}")
        print("-" * 50)

def main() -> None:
    data = analyze_journal_entries()
    visualize_analysis(data)

if __name__ == "__main__":
    start_time = perf_counter()
    main()
    end_time = perf_counter()
    print(f"Analysis completed in {end_time - start_time:.2f} seconds")
