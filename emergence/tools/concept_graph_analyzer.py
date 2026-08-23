"""
Lumina Creative Tool — concept_graph_analyzer
Created : 2026-08-23T11:31:34
Purpose : Builds a weighted co‑occurrence graph from text, computes degree centrality and PageRank for each concept, saves the graph as JSON, and prints the top concepts.
"""

import sys
import json
import re
import math
from collections import defaultdict, Counter

# ------------------------------------------------------------
# Configuration
WINDOW_SIZE = 2          # co‑occurrence window (tokens)
TOP_N = 10               # how many top concepts to display
PAGE_RANK_ITER = 30
DAMPING = 0.85
OUTPUT_FILE = "concept_graph.json"
# ------------------------------------------------------------

def tokenize(text: str):
    """Simple word tokenizer: lower‑case, keep alphanumerics."""
    return re.findall(r"\b\w+\b", text.lower())

def build_cooccurrence(tokens):
    """Build weighted undirected co‑occurrence graph."""
    graph = defaultdict(Counter)   # node -> Counter(neighbor: weight)
    for i, token in enumerate(tokens):
        # look ahead within window
        for j in range(i + 1, min(i + 1 + WINDOW_SIZE, len(tokens))):
            neighbor = tokens[j]
            if token == neighbor:
                continue
            # increment both directions
            graph[token][neighbor] += 1
            graph[neighbor][token] += 1
    return graph

def degree_centrality(graph):
    """Sum of edge weights for each node."""
    return {node: sum(neigh.values()) for node, neigh in graph.items()}

def pagerank(graph):
    """Simple PageRank on weighted undirected graph."""
    nodes = list(graph.keys())
    N = len(nodes)
    if N == 0:
        return {}
    rank = {node: 1.0 / N for node in nodes}
    for _ in range(PAGE_RANK_ITER):
        new_rank = {}
        for node in nodes:
            inbound = 0.0
            for neighbor, weight in graph[node].items():
                total_out = sum(graph[neighbor].values())
                if total_out > 0:
                    inbound += rank[neighbor] * (weight / total_out)
            new_rank[node] = (1 - DAMPING) / N + DAMPING * inbound
        rank = new_rank
    return rank

def top_items(scores, n=TOP_N):
    """Return top‑n items sorted by score descending."""
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:n]

def analyze_text(text: str):
    tokens = tokenize(text)
    if not tokens:
        print("No tokens found.")
        return

    graph = build_cooccurrence(tokens)
    deg = degree_centrality(graph)
    pr = pagerank(graph)

    # Combine scores for output
    output = {
        "nodes": [],
        "edges": []
    }
    for node in graph:
        output["nodes"].append({
            "concept": node,
            "degree_centrality": deg.get(node, 0),
            "pagerank": pr.get(node, 0.0)
        })
        for neighbor, weight in graph[node].items():
            # store each undirected edge once (lexicographic order)
            if node < neighbor:
                output["edges"].append({
                    "source": node,
                    "target": neighbor,
                    "weight": weight
                })

    # Save JSON
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # Print top concepts by each metric
    print("\nTop concepts by degree centrality:")
    for concept, score in top_items(deg):
        print(f"  {concept:20} {score}")

    print("\nTop concepts by PageRank:")
    for concept, score in top_items(pr):
        print(f"  {concept:20} {score:.6f}")

    print(f"\nGraph saved to {OUTPUT_FILE}")

def main():
    if len(sys.argv) > 1:
        path = sys.argv[1]
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception as e:
            print(f"Failed to read {path}: {e}")
            return
    else:
        # Sample placeholder text if no file supplied
        text = """
        curiosity reward learning entropy perplexity neural networks
        entropy information theory cognitive entropy thermodynamic entropy
        bitcoin mining ARM SHA2 optimization low‑power high‑performance
        AGI long‑term memory contextual updating categories dynamic refinement
        """
        print("No input file supplied – using built‑in sample text.\n")

    analyze_text(text)

if __name__ == "__main__":
    main()
