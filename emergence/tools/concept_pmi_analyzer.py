"""
Lumina Creative Tool — concept_pmi_analyzer
Created : 2026-10-03T18:38:44
Purpose : Computes pairwise PMI between bracketed concepts in a journal, prints top links, and saves the full matrix as JSON.
"""

"""
concept_pmi_analyzer.py

Extracts bracketed concepts from a journal, computes pairwise Pointwise Mutual
Information (PMI) between concepts, prints the top associations, and writes the
full PMI matrix to JSON.
"""

import json
import math
import os
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

JOURNAL_PATH = Path("journal.txt")
OUTPUT_JSON = Path("concept_pmi.json")
TOP_K = 10  # number of top pairs to display


def load_journal(path: Path) -> list[str]:
    """Read the journal file; if missing, create a tiny example."""
    if not path.is_file():
        example = """[bitcoin] What are the specific constraints and limitations of 2-way interleaving in ARM SHA2 mining?
[consciousness] How does the relationship between entropy and perplexity apply to neural networks?
[bitcoin] Can low‑power devices mine effectively?
[consciousness] Brain regions for reward and curiosity?
[agi] Lumina runs on a custom Groq TSP model."""
        path.write_text(example, encoding="utf-8")
    return path.read_text(encoding="utf-8").splitlines()


def extract_concepts(line: str) -> set[str]:
    """Return a set of concepts found in square brackets."""
    return set(re.findall(r"\[([^\]]+)\]", line.lower()))


def build_counts(entries: list[set[str]]) -> tuple[Counter, Counter, int]:
    """Count single concepts, concept pairs, and total entries."""
    concept_counts = Counter()
    pair_counts = Counter()
    for concepts in entries:
        for c in concepts:
            concept_counts[c] += 1
        for a, b in combinations(sorted(concepts), 2):
            pair_counts[(a, b)] += 1
    total = len(entries)
    return concept_counts, pair_counts, total


def compute_pmi(
    concept_counts: Counter,
    pair_counts: Counter,
    total: int,
) -> dict[str, dict[str, float]]:
    """Return a nested dict PMI[concept_i][concept_j] = value (symmetrical)."""
    pmi_matrix: dict[str, dict[str, float]] = defaultdict(dict)
    for (a, b), pair_cnt in pair_counts.items():
        p_a = concept_counts[a] / total
        p_b = concept_counts[b] / total
        p_ab = pair_cnt / total
        # Guard against zero probabilities (should not happen)
        if p_a > 0 and p_b > 0 and p_ab > 0:
            pmi = math.log2(p_ab / (p_a * p_b))
            pmi_matrix[a][b] = pmi
            pmi_matrix[b][a] = pmi
    return pmi_matrix


def top_pairs(pmi_matrix: dict[str, dict[str, float]], k: int) -> list[tuple[float, str, str]]:
    """Return the k highest‑PMI pairs as (pmi, a, b)."""
    seen = set()
    heap = []
    for a, inner in pmi_matrix.items():
        for b, val in inner.items():
            if (b, a) in seen:
                continue
            seen.add((a, b))
            heap.append((val, a, b))
    heap.sort(reverse=True)
    return heap[:k]


def main() -> None:
    lines = load_journal(JOURNAL_PATH)
    entry_concepts = [extract_concepts(line) for line in lines if extract_concepts(line)]
    if not entry_concepts:
        print("No concepts found in the journal.")
        return

    concept_counts, pair_counts, total = build_counts(entry_concepts)
    pmi_matrix = compute_pmi(concept_counts, pair_counts, total)

    # Save full matrix
    OUTPUT_JSON.write_text(json.dumps(pmi_matrix, indent=2), encoding="utf-8")
    print(f"PMI matrix written to {OUTPUT_JSON}")

    # Print top associations
    print("\nTop concept associations (PMI):")
    for val, a, b in top_pairs(pmi_matrix, TOP_K):
        print(f"  {a} ↔ {b}: {val:.3f}")

    # Simple sanity summary
    print("\nSummary:")
    print(f"  Total entries processed: {total}")
    print(f"  Unique concepts discovered: {len(concept_counts)}")
    print(f"  Concept frequencies: {dict(concept_counts)}")


if __name__ == "__main__":
    main()
