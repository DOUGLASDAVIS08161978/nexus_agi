"""
Lumina Creative Tool — concept_cooccurrence_graph
Created : 2026-08-21T21:07:41
Purpose : Generates a JSON co‑occurrence graph of key concepts from journal text and prints the strongest links.
"""

import json
import re
import itertools
from collections import defaultdict, Counter
from pathlib import Path

# ---------- Configuration ----------
JOURNAL_PATH = Path("journal.txt")          # Input file with raw journal entries
OUTPUT_JSON = Path("concept_graph.json")    # Graph export
WINDOW_SIZE = 2                             # Sentences within this window are considered co‑occurring
TOP_N_EDGES = 15                            # How many strongest links to display

# ---------- Simple tokenization ----------
_SENTENCE_RE = re.compile(r"[.!?]\s+")
_WORD_RE = re.compile(r"\b\w+\b")
_STOPWORDS = {
    "the", "and", "of", "to", "a", "in", "for", "on", "with", "as", "is",
    "it", "that", "by", "from", "at", "or", "be", "are", "was", "were",
    "i", "my", "me", "we", "us", "our", "you", "your", "he", "she", "they",
    "them", "this", "these", "those", "but", "not", "can", "will", "would",
    "could", "should", "has", "have", "had", "do", "does", "did"
}

def load_journal(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Journal file not found: {path}")
    return path.read_text(encoding="utf-8")

def split_sentences(text: str):
    # Keep delimiters to avoid empty trailing pieces
    pieces = _SENTENCE_RE.split(text)
    return [s.strip() for s in pieces if s.strip()]

def extract_concepts(sentence: str):
    # Very naive concept extraction: keep words longer than 3 chars not in stoplist,
    # and treat capitalized words as potential proper‑nouns.
    words = _WORD_RE.findall(sentence.lower())
    concepts = [
        w for w in words
        if len(w) > 3 and w not in _STOPWORDS
    ]
    # Add capitalized tokens (original case) as extra signals
    caps = [tok for tok in re.findall(r"\b[A-Z][a-zA-Z]+\b", sentence)]
    concepts.extend([c.lower() for c in caps if c.lower() not in concepts])
    return set(concepts)

# ---------- Build co‑occurrence graph ----------
def build_graph(sentences):
    # Map concept -> index
    concept_counter = Counter()
    co_counts = defaultdict(int)

    # Sliding window over sentences
    for i, sent in enumerate(sentences):
        window = sentences[i : i + WINDOW_SIZE]
        window_concepts = set()
        for s in window:
            window_concepts.update(extract_concepts(s))
        # Update global concept frequencies
        concept_counter.update(window_concepts)
        # Count each unordered pair once per window
        for a, b in itertools.combinations(sorted(window_concepts), 2):
            co_counts[(a, b)] += 1

    # Build node list
    nodes = [{"id": idx, "label": concept, "freq": concept_counter[concept]}
             for idx, concept in enumerate(sorted(concept_counter))]
    # Map concept to node id for edge creation
    id_map = {n["label"]: n["id"] for n in nodes}

    # Build edge list with weight
    edges = [
        {"source": id_map[a], "target": id_map[b], "weight": w}
        for (a, b), w in co_counts.items()
    ]

    return {"nodes": nodes, "edges": edges}, co_counts

# ---------- Output ----------
def save_json(data, path: Path):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

def print_top_edges(co_counts, n=TOP_N_EDGES):
    print("\nTop co‑occurring concept pairs:")
    for (a, b), w in Counter(co_counts).most_common(n):
        print(f"  {a:<15} ↔ {b:<15} : {w}")

def main():
    text = load_journal(JOURNAL_PATH)
    sentences = split_sentences(text)
    graph, co_counts = build_graph(sentences)

    save_json(graph, OUTPUT_JSON)
    print(f"Graph saved to {OUTPUT_JSON.resolve()}")
    print_top_edges(co_counts)

if __name__ == "__main__":
    main()