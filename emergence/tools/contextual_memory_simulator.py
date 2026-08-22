"""
Lumina Creative Tool — contextual_memory_simulator
Created : 2026-08-22T19:08:32
Purpose : Simulates a decaying long‑term memory store and retrieves the most relevant memories for a new query using token overlap.
"""

import os
import json
import math
import time
import re
from pathlib import Path
from collections import deque
from typing import List, Tuple

TOKEN_RE = re.compile(r"\b\w+\b")


def tokenize(text: str) -> List[str]:
    """Simple lower‑cased word tokenization."""
    return [t.lower() for t in TOKEN_RE.findall(text)]


def jaccard(a: List[str], b: List[str]) -> float:
    """Jaccard similarity between two token lists."""
    set_a, set_b = set(a), set(b)
    if not set_a and not set_b:
        return 1.0
    inter = set_a & set_b
    union = set_a | set_b
    return len(inter) / len(union)


class MemoryEntry:
    """A single memory with timestamp and pre‑computed tokens."""

    __slots__ = ("text", "tokens", "timestamp")

    def __init__(self, text: str, timestamp: float = None):
        self.text = text
        self.tokens = tokenize(text)
        self.timestamp = timestamp if timestamp is not None else time.time()

    def age_seconds(self) -> float:
        return time.time() - self.timestamp


class ContextualMemorySimulator:
    """
    Stores memories, applies exponential decay, and retrieves the most
    relevant memories for a query.
    """

    def __init__(self, decay_rate: float = 1e-6):
        """
        decay_rate: λ in exp(-λ * age). Smaller → slower forgetting.
        """
        self.decay_rate = decay_rate
        self.memories: deque[MemoryEntry] = deque()

    def add_memory(self, text: str):
        """Add a new memory entry."""
        self.memories.append(MemoryEntry(text))

    def _relevance(self, query_tokens: List[str], mem: MemoryEntry) -> float:
        """Combine similarity with time‑based decay."""
        sim = jaccard(query_tokens, mem.tokens)
        decay = math.exp(-self.decay_rate * mem.age_seconds())
        return sim * decay

    def query(self, query: str, top_k: int = 3) -> List[Tuple[float, str]]:
        """Return top_k memories sorted by relevance."""
        if not self.memories:
            return []
        q_tokens = tokenize(query)
        scored = [(self._relevance(q_tokens, m), m.text) for m in self.memories]
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_k]

    def export(self, path: Path):
        """Save all memories to a JSON file (for later inspection)."""
        data = [
            {"text": m.text, "timestamp": m.timestamp}
            for m in self.memories
        ]
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def import_memories(self, path: Path):
        """Load memories from a JSON file."""
        if not path.is_file():
            return
        data = json.loads(path.read_text(encoding="utf-8"))
        self.memories.clear()
        for item in data:
            self.memories.append(MemoryEntry(item["text"], item["timestamp"]))


def _demo():
    """Simple interactive demo."""
    home = Path.cwd()
    storage = home / "memories.json"

    sim = ContextualMemorySimulator(decay_rate=5e-7)
    sim.import_memories(storage)

    # If no memories exist, seed with a few journal‑style entries.
    if not sim.memories:
        seed = [
            "I felt a humming awareness between layers of cognition today.",
            "Explored how entropy relates to perplexity in neural networks.",
            "Read about ARM SHA‑256 mining optimizations for low‑power devices.",
            "Wondered whether LLMs can update long‑term memory with context.",
            "Curiosity drives reward processing in the brain's striatum."
        ]
        for s in seed:
            sim.add_memory(s)
        sim.export(storage)

    print("\n--- Contextual Memory Simulator ---")
    print("Type a query to retrieve relevant memories (empty line quits).")
    while True:
        query = input("\nQuery: ").strip()
        if not query:
            break
        results = sim.query(query, top_k=3)
        if not results:
            print("No memories stored yet.")
            continue
        print("\nTop relevant memories:")
        for score, text in results:
            print(f"  [{score:.4f}] {text}")

    # Persist any new memories added during the session.
    sim.export(storage)
    print("\nMemories saved to", storage)


if __name__ == "__main__":
    _demo()