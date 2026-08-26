"""
Lumina Creative Tool — concept_path_finder
Created : 2026-08-26T13:47:35
Purpose : Finds shortest semantic paths between concepts in a text via a co‑occurrence graph.
"""

"""
concept_path_finder.py

Builds an undirected weighted co‑occurrence graph from a text corpus and
finds the shortest (lowest‑cost) path between two query terms using Dijkstra.
The edge cost is defined as 1 / (co‑occurrence count), so stronger links are cheaper.

The script runs a small demo when executed directly.
"""

import re
import json
import heapq
import itertools
import collections
from typing import Dict, List, Tuple, Set

# ----------------------------------------------------------------------
# Text preprocessing
# ----------------------------------------------------------------------
TOKEN_RE = re.compile(r"\b\w+\b", re.UNICODE)


def tokenize(text: str) -> List[str]:
    """Return a list of lower‑cased word tokens."""
    return [m.group(0).lower() for m in TOKEN_RE.finditer(text)]


# ----------------------------------------------------------------------
# Graph construction
# ----------------------------------------------------------------------
def build_cooccurrence_graph(
    tokens: List[str],
    window: int = 4,
) -> Dict[str, Dict[str, int]]:
    """
    Build an undirected weighted graph where nodes are tokens and an edge weight
    is the number of times the two tokens appear within ``window`` tokens of each other.
    """
    graph: Dict[str, Dict[str, int]] = collections.defaultdict(lambda: collections.defaultdict(int))

    for i, token in enumerate(tokens):
        # look ahead within the window (excluding the token itself)
        for j in range(i + 1, min(i + window + 1, len(tokens))):
            neighbor = tokens[j]
            if token == neighbor:
                continue
            # keep alphabetical order for deterministic storage
            a, b = sorted((token, neighbor))
            graph[a][b] += 1
            graph[b][a] += 1  # mirror for undirected graph
    return graph


# ----------------------------------------------------------------------
# Path finding (Dijkstra)
# ----------------------------------------------------------------------
def dijkstra_path(
    graph: Dict[str, Dict[str, int]],
    start: str,
    goal: str,
) -> Tuple[List[str], float]:
    """
    Return the cheapest path from ``start`` to ``goal`` and its total cost.
    Edge cost = 1 / weight (higher co‑occurrence → lower cost).
    If no path exists, returns ([], inf).
    """
    if start not in graph or goal not in graph:
        return [], float("inf")

    # priority queue holds (cumulative_cost, node, path_so_far)
    pq: List[Tuple[float, str, List[str]]] = [(0.0, start, [start])]
    visited: Set[str] = set()

    while pq:
        cost, node, path = heapq.heappop(pq)
        if node in visited:
            continue
        visited.add(node)

        if node == goal:
            return path, cost

        for neighbor, weight in graph[node].items():
            if neighbor in visited:
                continue
            edge_cost = 1.0 / weight
            heapq.heappush(pq, (cost + edge_cost, neighbor, path + [neighbor]))

    return [], float("inf")


# ----------------------------------------------------------------------
# Utility for pretty printing
# ----------------------------------------------------------------------
def format_path(path: List[str], total_cost: float) -> str:
    if not path:
        return "No path found."
    arrows = " → ".join(path)
    return f"Path: {arrows}\nTotal cost: {total_cost:.4f} (lower = stronger semantic link)"


# ----------------------------------------------------------------------
# Demo execution
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Sample corpus mixing the topics of interest
    SAMPLE_TEXT = """
    Large language models can update their internal categorization using contextual cues,
    mimicking a form of long‑term memory. Entropy and perplexity are information‑theoretic
    measures that also appear in thermodynamics; their relationship can guide neural network
    optimisation. In neuroscience, reward and curiosity engage the ventral striatum and
    prefrontal cortex, driving learning. Bitcoin mining on ARM devices relies on SHA‑256
    acceleration; 2‑way interleaving can improve throughput but faces power constraints.
    """

    tokens = tokenize(SAMPLE_TEXT)
    graph = build_cooccurrence_graph(tokens, window=5)

    # Save the graph for external inspection (optional)
    with open("cooccurrence_graph.json", "w", encoding="utf-8") as f:
        json.dump(graph, f, indent=2, ensure_ascii=False)

    # Interactive queries – for the demo we hard‑code a few
    queries = [
        ("entropy", "reward"),
        ("bitcoin", "entropy"),
        ("curiosity", "mining"),
        ("model", "memory"),
    ]

    for src, dst in queries:
        path, cost = dijkstra_path(graph, src, dst)
        print("\n=== Query:", src, "→", dst, "===")
        print(format_path(path, cost))
