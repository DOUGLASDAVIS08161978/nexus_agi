#!/usr/bin/env python3
"""
lumina_vector_memory.py — Semantic Memory v2  (AGI Module 22)

Replaces TF-IDF keyword matching with two-stage retrieval:

  Stage 1 — BM25 (Okapi BM25, Robertson et al.)
    Industry-standard retrieval used by Elasticsearch and Solr.
    Beats TF-IDF on three key axes:
      • Term-frequency saturation  — 10 hits ≠ 10× more relevant
      • Document-length normalisation — short crisp memories aren't penalised
      • Better IDF curve — rare, precise terms earn exponentially more weight
    Also indexes unigram + bigram tokens, so "bitcoin miner" matches as a
    phrase, not just two separate words.

  Stage 2 — Groq semantic reranking  (opt-in: rerank=True)
    BM25 fetches RERANK_N candidates; llama-3.1-8b-instant reads them and
    returns the top_k ranked by true semantic relevance.  This catches
    synonyms, analogies, and conceptual links that keyword overlap misses
    entirely ("vehicle" → "car", "hash computation" → "mining").
    Results are cached 120 s to avoid redundant API calls.

Drop-in replacement for SemanticMemory — same public API:
  .store(text, tags, category) → id
  .recall(query, top_k, category, rerank) → List[Dict]
  .recent(n, category) → List[Dict]
  .stats() → Dict
  .flush()

On first run, migrates existing semantic_memory.json automatically.
"""

from __future__ import annotations

import hashlib, json, math, re, threading, time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from emergence_engine import GroqClient

_BASE       = Path(__file__).parent
VM_FILE     = _BASE / "vector_memory.json"
SM_FILE     = _BASE / "semantic_memory.json"   # migration source

MAX_ENTRIES = 2000
BM25_K1     = 1.5    # term-frequency saturation point
BM25_B      = 0.75   # document-length normalisation strength
RERANK_N    = 20     # BM25 candidates fed to the reranker
RERANK_TTL  = 120    # seconds to cache a rerank result

_STOPWORDS = frozenset({
    "a","an","the","is","are","was","were","be","been","being",
    "have","has","had","do","does","did","will","would","could",
    "should","may","might","shall","can","to","of","in","on",
    "at","by","for","with","about","from","and","or","but","not",
    "it","its","this","that","these","those","i","my","me","we",
    "our","you","your","he","she","they","their","what","which",
    "who","whom","when","where","how","why","s","t","re","ve",
})


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _tokenize(text: str) -> List[str]:
    """Unigrams + bigrams, stop words removed."""
    words = [
        w for w in re.findall(r"[a-z0-9]+", text.lower())
        if w not in _STOPWORDS and len(w) > 1
    ]
    bigrams = [f"{words[i]}~{words[i+1]}" for i in range(len(words) - 1)]
    return words + bigrams


# ── BM25 index ────────────────────────────────────────────────────────────────

class _BM25:
    """
    Okapi BM25 over a frozen snapshot of entries.
    Rebuilt whenever the entry list changes (store() call).
    """

    def __init__(self, entries: List[Dict]):
        self._entries = entries
        n             = len(entries)
        self._n       = n
        self._tf      : List[Dict[str, int]] = []
        self._dl      : List[int]            = []
        self._df      : Dict[str, int]       = defaultdict(int)
        total_dl      = 0

        for e in entries:
            tok: List[str] = _tokenize(e.get("text", ""))
            tf: Dict[str, int] = defaultdict(int)
            for t in tok:
                tf[t] += 1
            self._tf.append(dict(tf))
            dl = len(tok)
            self._dl.append(dl)
            total_dl += dl
            for t in set(tok):
                self._df[t] += 1

        self._avgdl = total_dl / n if n else 1.0

    def score(self, query: str, doc_idx: int) -> float:
        dl  = self._dl[doc_idx]
        tfd = self._tf[doc_idx]
        s   = 0.0
        for t in _tokenize(query):
            df = self._df.get(t, 0)
            if not df:
                continue
            idf = math.log((self._n - df + 0.5) / (df + 0.5) + 1.0)
            tf  = tfd.get(t, 0)
            num = tf * (BM25_K1 + 1)
            den = tf + BM25_K1 * (1 - BM25_B + BM25_B * dl / self._avgdl)
            s  += idf * num / max(den, 1e-9)
        return s

    def retrieve(self, query: str, top_k: int,
                 category: Optional[str] = None) -> List[Dict]:
        pool = [
            (i, e) for i, e in enumerate(self._entries)
            if not category or e.get("category") == category
        ]
        scored = []
        for i, e in pool:
            s = self.score(query, i)
            if s > 0:
                scored.append((s, e))
        scored.sort(key=lambda x: -x[0])
        return [e for _, e in scored[:top_k]]


# ── Main class ────────────────────────────────────────────────────────────────

class VectorMemory:
    """
    BM25 + optional Groq semantic reranking.
    Drop-in replacement for SemanticMemory.
    """

    def __init__(self, groq: Optional["GroqClient"] = None):
        self._groq   = groq
        self._lock   = threading.Lock()
        self._dirty  = False
        self._entries: List[Dict] = []
        self._index:   Optional[_BM25] = None
        self._rcache:  Dict[str, tuple] = {}   # rerank cache

        self._load()
        self._rebuild()

    # ── Persistence ────────────────────────────────────────────────────────

    def _load(self):
        if VM_FILE.exists():
            try:
                raw = json.loads(VM_FILE.read_text("utf-8"))
                self._entries = [e for e in raw if isinstance(e, dict)]
                return
            except Exception:
                pass
        # First run: migrate from SemanticMemory if it exists
        if SM_FILE.exists():
            try:
                raw = json.loads(SM_FILE.read_text("utf-8"))
                self._entries = [
                    {k: v for k, v in e.items() if not k.startswith("_")}
                    for e in raw if isinstance(e, dict)
                ]
                self._dirty = True
                print(f"  ✓ Migrated {len(self._entries)} memories → BM25 vector store")
            except Exception:
                pass

    def _save(self):
        VM_FILE.write_text(
            json.dumps(self._entries, indent=2, ensure_ascii=False), "utf-8"
        )
        self._dirty = False

    def flush(self):
        if self._dirty:
            with self._lock:
                self._save()

    def _rebuild(self):
        self._index = _BM25(self._entries) if self._entries else None

    # ── Write ──────────────────────────────────────────────────────────────

    def store(self, text: str, tags: List[str] = None,
              category: str = "general") -> str:
        with self._lock:
            mid = hashlib.md5(text.encode()).hexdigest()[:12]
            entry = {
                "id":       mid,
                "text":     text,
                "tags":     tags or [],
                "category": category,
                "ts":       _now(),
            }
            self._entries = [e for e in self._entries if e["id"] != mid]
            self._entries.append(entry)
            if len(self._entries) > MAX_ENTRIES:
                self._entries = self._entries[-MAX_ENTRIES:]
            self._rebuild()
            self._save()
        return mid

    # ── Read ───────────────────────────────────────────────────────────────

    def recall(self, query: str, top_k: int = 5,
               category: str = None,
               rerank: bool = False) -> List[Dict]:
        with self._lock:
            if not self._index:
                return []
            fetch_n    = RERANK_N if (rerank and self._groq) else top_k
            candidates = self._index.retrieve(query, fetch_n, category)

        if not candidates:
            return []

        if rerank and self._groq and len(candidates) > top_k:
            candidates = self._groq_rerank(query, candidates, top_k)

        return candidates[:top_k]

    def recent(self, n: int = 10, category: str = None) -> List[Dict]:
        with self._lock:
            pool = [e for e in self._entries
                    if not category or e.get("category") == category]
            return pool[-n:]

    def stats(self) -> Dict:
        with self._lock:
            cats: Dict[str, int] = defaultdict(int)
            for e in self._entries:
                cats[e.get("category", "general")] += 1
            return {"total": len(self._entries), "by_category": dict(cats)}

    # ── Groq semantic reranker ─────────────────────────────────────────────

    def _groq_rerank(self, query: str, candidates: List[Dict],
                     top_k: int) -> List[Dict]:
        """
        Ask llama-3.1-8b-instant to rank candidates by semantic relevance.
        Uses the fast model to keep latency low.  Result is cached 120 s.
        """
        cache_key = hashlib.md5(
            (query + "|" + "|".join(e["id"] for e in candidates)).encode()
        ).hexdigest()[:16]

        cached = self._rcache.get(cache_key)
        if cached:
            ts, result = cached
            if time.time() - ts < RERANK_TTL:
                return result

        numbered = "\n".join(
            f"{i+1}. {e['text'][:140]}"
            for i, e in enumerate(candidates)
        )
        prompt = (
            f'Query: "{query}"\n\n'
            f"Rank these memories by semantic relevance to the query.\n"
            f"Return ONLY a comma-separated list of numbers, most relevant first.\n"
            f"Example: 3,1,7,2,5\n\n"
            f"{numbered}"
        )
        try:
            resp = self._groq.chat(
                "You are a semantic relevance ranker. "
                "Return only a comma-separated list of numbers, nothing else.",
                prompt,
                tier="fast",
                max_tokens=80,
            )
            indices: List[int] = []
            seen: set = set()
            for tok in re.findall(r"\d+", resp):
                idx = int(tok) - 1
                if 0 <= idx < len(candidates) and idx not in seen:
                    indices.append(idx)
                    seen.add(idx)
            # Append any candidates not mentioned by the model
            for i in range(len(candidates)):
                if i not in seen:
                    indices.append(i)

            result = [candidates[i] for i in indices[:top_k]]
            self._rcache[cache_key] = (time.time(), result)
            return result
        except Exception:
            return candidates[:top_k]

    # ── Display ────────────────────────────────────────────────────────────

    def display(self) -> str:
        s = self.stats()
        rerank_status = "✓ active" if self._groq else "✗ no Groq (BM25 only)"
        lines = [
            f"  Memory engine  : BM25 + Groq reranker ({rerank_status})",
            f"  Stored entries : {s['total']} / {MAX_ENTRIES}",
        ]
        for cat, n in sorted(s["by_category"].items()):
            lines.append(f"    {cat:22s}  {n}")
        return "\n".join(lines)
