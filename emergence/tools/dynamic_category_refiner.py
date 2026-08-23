"""
Lumina Creative Tool — dynamic_category_refiner
Created : 2026-08-22T22:19:36
Purpose : Incrementally updates token‑based text categories using KL divergence, simulating continual memory refinement for an AGI.
"""

import json
import math
import string
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

# ---------- Utility ----------
def tokenize(text: str) -> List[str]:
    """Very simple tokenizer: lower‑case, strip punctuation, split on whitespace."""
    translator = str.maketrans("", "", string.punctuation)
    return text.lower().translate(translator).split()

def prob_dist(counter: Counter) -> Dict[str, float]:
    """Convert a Counter of token counts to a probability distribution."""
    total = sum(counter.values())
    if total == 0:
        return {}
    return {tok: cnt / total for tok, cnt in counter.items()}

def kl_divergence(p: Dict[str, float], q: Dict[str, float], eps: float = 1e-12) -> float:
    """
    KL(p || q) = Σ p(x) log(p(x)/q(x))
    Small epsilon avoids log(0). If q(x) == 0 we treat it as eps.
    """
    div = 0.0
    for token, p_prob in p.items():
        q_prob = q.get(token, eps)
        div += p_prob * math.log(p_prob / q_prob)
    return div

# ---------- Core Refiner ----------
class DynamicCategoryRefiner:
    def __init__(self,
                 init_categories: Dict[str, Counter],
                 assign_threshold: float = 0.5,
                 merge_threshold: float = 0.2):
        """
        init_categories: name → Counter of token frequencies.
        assign_threshold: max KL to assign a document to an existing category.
        merge_threshold: max KL between two categories to merge them.
        """
        self.categories = init_categories
        self.assign_thr = assign_threshold
        self.merge_thr = merge_threshold

    def _category_distribution(self, name: str) -> Dict[str, float]:
        return prob_dist(self.categories[name])

    def assign_document(self, doc: str) -> Tuple[str, float]:
        """Assign a document to the best category or create a new one."""
        doc_counter = Counter(tokenize(doc))
        doc_dist = prob_dist(doc_counter)

        # Compute KL to each existing category
        best_name = None
        best_kl = float('inf')
        for name in self.categories:
            cat_dist = self._category_distribution(name)
            kl = kl_divergence(doc_dist, cat_dist)
            if kl < best_kl:
                best_kl = kl
                best_name = name

        if best_kl <= self.assign_thr:
            # Update the chosen category
            self.categories[best_name].update(doc_counter)
            return best_name, best_kl
        else:
            # Create a fresh category
            new_name = f"category_{len(self.categories) + 1}"
            self.categories[new_name] = doc_counter
            return new_name, best_kl

    def merge_similar_categories(self) -> List[Tuple[str, str]]:
        """Iteratively merge categories whose KL divergence is below merge_threshold."""
        merged = []
        names = list(self.categories.keys())
        i = 0
        while i < len(names):
            name_i = names[i]
            dist_i = self._category_distribution(name_i)
            j = i + 1
            while j < len(names):
                name_j = names[j]
                dist_j = self._category_distribution(name_j)
                kl_ij = kl_divergence(dist_i, dist_j)
                kl_ji = kl_divergence(dist_j, dist_i)
                sym_kl = (kl_ij + kl_ji) / 2
                if sym_kl <= self.merge_thr:
                    # Merge j into i
                    self.categories[name_i].update(self.categories[name_j])
                    del self.categories[name_j]
                    merged.append((name_i, name_j))
                    names.pop(j)  # remove merged name
                    # recompute distribution for i after merge
                    dist_i = self._category_distribution(name_i)
                else:
                    j += 1
            i += 1
        return merged

    def process_documents(self, docs: List[str]) -> List[Tuple[str, float]]:
        """Assign each document and return assignment info."""
        assignments = []
        for doc in docs:
            cat, kl = self.assign_document(doc)
            assignments.append((cat, kl))
        return assignments

    def export_json(self, path: Path) -> None:
        """Save categories as a JSON mapping name → token‑frequency dict."""
        serialisable = {name: dict(counter) for name, counter in self.categories.items()}
        path.write_text(json.dumps(serialisable, indent=2), encoding="utf-8")

# ---------- Demo ----------
if __name__ == "__main__":
    # Initial categories (could be loaded from a previous run)
    init = {
        "science": Counter({"entropy": 3, "neural": 2, "network": 2, "information": 1}),
        "finance": Counter({"bitcoin": 4, "mining": 2, "arm": 1, "sha256": 1}),
    }

    refiner = DynamicCategoryRefiner(init_categories=init,
                                     assign_threshold=0.6,
                                     merge_threshold=0.15)

    # New incoming documents (simulating a journal stream)
    new_docs = [
        "Exploring how Shannon entropy relates to perplexity in transformer models.",
        "Low‑power ARM devices can still perform SHA‑256 mining with clever interleaving.",
        "Thermodynamic entropy offers a fresh perspective on information bottlenecks.",
        "Bitcoin mining on a phone is unlikely but not impossible with optimized code.",
        "Curiosity drives reward processing in the ventral striatum and prefrontal cortex."
    ]

    assignments = refiner.process_documents(new_docs)
    print("Document assignments (category, KL):")
    for doc, (cat, kl) in zip(new_docs, assignments):
        print(f"- \"{doc[:60]}...\" → {cat} (KL={kl:.3f})")

    merged = refiner.merge_similar_categories()
    if merged:
        print("\nMerged categories:")
        for target, source in merged:
            print(f"- {source} merged into {target}")

    out_path = Path("updated_categories.json")
    refiner.export_json(out_path)
    print(f"\nUpdated categories saved to {out_path.resolve()}")
