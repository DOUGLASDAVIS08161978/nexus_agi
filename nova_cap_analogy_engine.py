"""
Nova ASI — Analogy Engine
Proposed autonomously via /evolve
"""

"""
AnalogyEngine — structural analogy discovery and plain-English explanation.

Finds mappings between concepts using TF-IDF-weighted feature vectors,
Bayesian confidence scoring, and EMA-updated similarity learning.
Each analogy stores (source, target, mapping, confidence, use_count).
Confidence is updated via Bayesian posterior on successful retrievals.
"""

import sqlite3
import json
import math
import time
import re
import hashlib
from collections import defaultdict
from typing import Dict, List, Tuple, Optional


class AnalogyEngine:
    """Structural analogy discovery with Bayesian confidence and EMA learning."""

    # Seed knowledge: concept -> {feature: weight}
    _SEED: Dict[str, Dict[str, float]] = {
        "electricity": {"flow": 1.0, "pressure": 0.9, "resistance": 0.8, "source": 0.7, "network": 0.6},
        "water":       {"flow": 1.0, "pressure": 0.9, "resistance": 0.7, "source": 0.8, "network": 0.5},
        "brain":       {"network": 1.0, "signal": 0.9, "memory": 0.8, "learning": 0.9, "processing": 0.8},
        "computer":    {"network": 0.8, "signal": 0.7, "memory": 1.0, "learning": 0.6, "processing": 1.0},
        "heart":       {"flow": 0.9, "pump": 1.0, "pressure": 0.8, "rhythm": 0.7, "network": 0.5},
        "economy":     {"flow": 0.8, "network": 0.9, "pressure": 0.6, "growth": 0.9, "feedback": 0.8},
        "evolution":   {"learning": 0.9, "feedback": 0.8, "growth": 0.7, "selection": 1.0, "adaptation": 0.9},
        "language":    {"signal": 0.8, "network": 0.7, "pattern": 0.9, "learning": 0.8, "structure": 0.9},
        "gravity":     {"force": 1.0, "attraction": 0.9, "field": 0.8, "network": 0.4, "pressure": 0.5},
        "democracy":   {"network": 0.8, "feedback": 0.9, "selection": 0.8, "pressure": 0.6, "growth": 0.5},
    }

    _TEMPLATES = [
        "{a} is like {b}: both involve {shared}.",
        "Think of {a} as {b} — the key parallel is {shared}.",
        "Just as {b} relies on {shared}, so does {a}.",
        "{a} mirrors {b} through the shared principle of {shared}.",
    ]

    def __init__(self) -> None:
        self._conn = sqlite3.connect(":memory:", check_same_thread=False)
        self._build_schema()
        self._ema_similarity: Dict[str, float] = {}   # key -> EMA score
        self._alpha: float = 0.15                      # EMA learning rate
        self._idf: Dict[str, float] = {}
        self._compute_idf()
        self._seed_db()

    # ------------------------------------------------------------------ #
    def _build_schema(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS analogies (
                id       TEXT PRIMARY KEY,
                concept_a TEXT NOT NULL,
                concept_b TEXT NOT NULL,
                mapping   TEXT NOT NULL,
                confidence REAL DEFAULT 0.5,
                use_count  INTEGER DEFAULT 0,
                created_at REAL
            )""")
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS features (
                concept TEXT NOT NULL,
                feature TEXT NOT NULL,
                weight  REAL NOT NULL,
                PRIMARY KEY (concept, feature)
            )""")
        self._conn.commit()

    def _compute_idf(self) -> None:
        N = len(self._SEED)
        feat_doc_count: Dict[str, int] = defaultdict(int)
        for features in self._SEED.values():
            for f in features:
                feat_doc_count[f] += 1
        self._idf = {f: math.log(N / (1 + cnt)) for f, cnt in feat_doc_count.items()}

    def _seed_db(self) -> None:
        for concept, features in self._SEED.items():
            for feat, w in features.items():
                self._conn.execute(
                    "INSERT OR IGNORE INTO features VALUES (?,?,?)", (concept, feat, w))
        self._conn.commit()

    def _get_features(self, concept: str) -> Dict[str, float]:
        rows = self._conn.execute(
            "SELECT feature, weight FROM features WHERE concept=?", (concept,)).fetchall()
        if rows:
            return {r[0]: r[1] for r in rows}
        # Auto-extract features from concept string via character n-gram hashing
        tokens = re.findall(r"[a-z]+", concept.lower())
        return {t: 1.0 / (i + 1) for i, t in enumerate(tokens[:8])}

    def _tfidf_vector(self, features: Dict[str, float]) -> Dict[str, float]:
        return {f: w * self._idf.get(f, 1.0) for f, w in features.items()}

    def _cosine(self, va: Dict[str, float], vb: Dict[str, float]) -> float:
        shared = set(va) & set(vb)
        if not shared:
            return 0.0
        dot = sum(va[k] * vb[k] for k in shared)
        na = math.sqrt(sum(v ** 2 for v in va.values()))
        nb = math.sqrt(sum(v ** 2 for v in vb.values()))
        return dot / (na * nb + 1e-9)

    def _bayesian_update(self, prior: float, likelihood: float) -> float:
        p = likelihood * prior
        q = (1 - likelihood) * (1 - prior)
        return p / (p + q + 1e-9)

    def _analogy_id(self, a: str, b: str) -> str:
        key = f"{a.lower()}|{b.lower()}"
        return hashlib.md5(key.encode()).hexdigest()[:12]

    # ------------------------------------------------------------------ #
    def find_analogy(self, concept_a: str, concept_b: str) -> Dict:
        """Map structural similarities between two concepts; return scored mapping dict."""
        fa = self._get_features(concept_a.lower())
        fb = self._get_features(concept_b.lower())
        va = self._tfidf_vector(fa)
        vb = self._tfidf_vector(fb)
        shared_keys = sorted(set(va) & set(vb), key=lambda k: va[k] * vb[k], reverse=True)
        raw_sim = self._cosine(va, vb)

        # EMA update for this pair
        ema_key = self._analogy_id(concept_a, concept_b)
        prev = self._ema_similarity.get(ema_key, raw_sim)
        self._ema_similarity[ema_key] = self._alpha * raw_sim + (1 - self._alpha) * prev

        # Bayesian confidence update
        row = self._conn.execute(
            "SELECT confidence, use_count FROM analogies WHERE id=?", (ema_key,)).fetchone()
        if row:
            prior, use_count = row[0], row[1]
            confidence = self._bayesian_update(prior, raw_sim)
            self._conn.execute(
                "UPDATE analogies SET confidence=?, use_count=? WHERE id=?",
                (confidence, use_count + 1, ema_key))
        else:
            confidence = raw_sim
            mapping_json = json.dumps(shared_keys[:6])
            self._conn.execute(
                "INSERT INTO analogies VALUES (?,?,?,?,?,?,?)",
                (ema_key, concept_a, concept_b, mapping_json, confidence, 1, time.time()))
        self._conn.commit()

        return {
            "concept_a": concept_a,
            "concept_b": concept_b,
            "shared_features": shared_keys[:6],
            "similarity": round(raw_sim, 4),
            "ema_similarity": round(self._ema_similarity[ema_key], 4),
            "confidence": round(confidence, 4),
        }

    def explain_via_analogy(self, complex_idea: str, familiar_domain: str) -> str:
        """Return a plain-English explanation of complex_idea using familiar_domain as anchor."""
        mapping = self.find_analogy(complex_idea, familiar_domain)
        shared = mapping["shared_features"]
        if not shared:
            shared_str = "underlying structural patterns"
        elif len(shared) == 1:
            shared_str = shared[0]
        else:
            shared_str = ", ".join(shared[:-1]) + f" and {shared[-1]}"

        import random
        template = random.choice(self._TEMPLATES)
        base = template.format(a=complex_idea, b=familiar_domain, shared=shared_str)
        confidence = mapping["confidence"]
        qualifier = (
            "This is a strong analogy." if confidence > 0.7 else
            "This analogy holds moderately well." if confidence > 0.4 else
            "This is a loose analogy — treat it as a starting point."
        )
        return f"{base} {qualifier} (confidence={confidence:.2f})"

    def add_concept(self, concept: str, features: Dict[str, float]) -> None:
        """Register a new concept with its feature-weight map for future analogy searches."""
        for feat, w in features.items():
            self._conn.execute(
                "INSERT OR REPLACE INTO features VALUES (?,?,?)", (concept.lower(), feat, float(w)))
        self._conn.commit()
        # Recompute IDF incrementally
        self._SEED[concept.lower()] = features
        self._compute_idf()

    def best_analogies(self, concept: str, top_k: int = 3) -> List[Dict]:
        """Return the top-k most structurally similar concepts to the given one."""
        candidates = [c for c in self._SEED if c != concept.lower()]
        scored = [self.find_analogy(concept, c) for c in candidates]
        scored.sort(key=lambda x: x["similarity"], reverse=True)
        return scored[:top_k]

    def confidence_report(self) -> List[Dict]:
        """Return all stored analogies sorted by Bayesian confidence descending."""
        rows = self._conn.execute(
            "SELECT concept_a, concept_b, mapping, confidence, use_count "
            "FROM analogies ORDER BY confidence DESC").fetchall()
        return [
            {"a": r[0], "b": r[1], "shared": json.loads(r[2]),
             "confidence": round(r[3], 4), "uses": r[4]}
            for r in rows
        ]

    def forget_weak(self, threshold: float = 0.2) -> int:
        """Delete analogies whose confidence has fallen below threshold; return count removed."""
        cur = self._conn.execute(
            "DELETE FROM analogies WHERE confidence < ?", (threshold,))
        self._conn.commit()
        return cur.rowcount

# Usage: obj = AnalogyEngine() | result = obj.find_analogy("electricity", "water")