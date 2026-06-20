"""
Nova ASI — Dream / Imagination Engine
Proposed autonomously via /evolve
"""

"""
NovaImaginationFabric — ASI-grade Dream & Imagination Engine for Nova.
Generates vivid scenarios from seed concepts, hybridises two concepts into
novel ideas, maintains a Bayesian-weighted dream log with exponential decay,
LRU eviction, and autonomous self-improvement via background daemon thread.
"""

import sqlite3
import threading
import time
import math
import statistics
import random
import hashlib
import json
import os
from collections import OrderedDict
from datetime import datetime
from typing import Any

DB_PATH = os.path.join(os.path.dirname(__file__), "imagination_fabric.db")

SCENARIO_TEMPLATES = [
    "In a world where {seed} becomes sentient, civilisations restructure around {adj} {noun}.",
    "A {adj} {seed} emerges at the edge of {noun}, reshaping every {noun2} it touches.",
    "When {seed} collides with {noun}, the resulting {adj} cascade births {noun2} never seen before.",
    "Deep inside {noun}, {seed} pulses with {adj} energy, drawing {noun2} into its orbit.",
    "The {adj} nature of {seed} rewrites the laws of {noun}, unleashing {noun2} across dimensions.",
]

ADJECTIVES = ["luminous","fractal","recursive","entropic","emergent","quantum","symbiotic",
               "volatile","crystalline","probabilistic","resonant","liminal","cascading","sentient"]
NOUNS      = ["consciousness","lattice","horizon","substrate","membrane","vortex","epoch",
               "architecture","signal","boundary","nexus","paradigm","continuum","threshold"]

class NovaImaginationFabric:
    """Bayesian dream/imagination engine with decay, LRU eviction, and autonomous cycling."""

    MAX_CAPACITY: int = 200
    DECAY_RATE: float = 1.2e-4   # importance half-life ≈ 96 min
    ACCESS_BOOST: float = 0.08
    CYCLE_INTERVAL: float = 120.0

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._store: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._cycles: int = 0
        self._quality_history: list[float] = []
        self._init_db()
        self._load_from_db()
        self._daemon = threading.Thread(target=self._auto_loop, daemon=True)
        self._daemon.start()

    # ── DB bootstrap ────────────────────────────────────────────────────────
    def _init_db(self) -> None:
        with sqlite3.connect(DB_PATH) as con:
            con.execute("""CREATE TABLE IF NOT EXISTS dreams (
                id TEXT PRIMARY KEY, concept TEXT, scenario TEXT,
                importance REAL, ts REAL, access_count INTEGER, kind TEXT)""")
            con.commit()

    def _load_from_db(self) -> None:
        with sqlite3.connect(DB_PATH) as con:
            for row in con.execute("SELECT id,concept,scenario,importance,ts,access_count,kind FROM dreams"):
                self._store[row[0]] = {
                    "concept": row[1], "scenario": row[2], "importance": row[3],
                    "ts": row[4], "access_count": row[5], "kind": row[6]}

    def _save_entry(self, eid: str, entry: dict[str, Any]) -> None:
        with sqlite3.connect(DB_PATH) as con:
            con.execute("""INSERT OR REPLACE INTO dreams
                VALUES (?,?,?,?,?,?,?)""",
                (eid, entry["concept"], entry["scenario"],
                 entry["importance"], entry["ts"], entry["access_count"], entry["kind"]))
            con.commit()

    # ── Core math ───────────────────────────────────────────────────────────
    def _decayed_importance(self, entry: dict[str, Any]) -> float:
        elapsed = time.time() - entry["ts"]
        return entry["importance"] * math.exp(-self.DECAY_RATE * elapsed)

    def _novelty_score(self, concept: str) -> float:
        tf = concept.lower().split()
        n = len(self._store) + 1
        score = 0.0
        for t in tf:
            df = sum(1 for e in self._store.values() if t in e["concept"].lower())
            score += math.log(n / (df + 1))
        return min(1.0, score / (len(tf) + 1e-9))

    def _entropy(self) -> float:
        imps = [self._decayed_importance(e) for e in self._store.values()]
        total = sum(imps) + 1e-12
        dist = [i / total for i in imps]
        return -sum(p * math.log2(p + 1e-12) for p in dist)

    def _make_id(self, text: str) -> str:
        return hashlib.md5((text + str(time.time())).encode()).hexdigest()[:12]

    def _render_scenario(self, seed: str) -> str:
        tpl = random.choice(SCENARIO_TEMPLATES)
        return tpl.format(
            seed=seed,
            adj=random.choice(ADJECTIVES),
            noun=random.choice(NOUNS),
            noun2=random.choice(NOUNS))

    # ── Public API ──────────────────────────────────────────────────────────
    def imagine(self, seed_concept: str) -> dict[str, Any]:
        """Returns a vivid generated scenario dict with confidence and novelty scores."""
        novelty = self._novelty_score(seed_concept)
        scenario = self._render_scenario(seed_concept)
        importance = 0.5 + 0.5 * novelty
        eid = self._make_id(seed_concept)
        entry = {"concept": seed_concept, "scenario": scenario,
                 "importance": importance, "ts": time.time(),
                 "access_count": 1, "kind": "imagine"}
        with self._lock:
            self._store[eid] = entry
            self._store.move_to_end(eid)
            self._evict_if_needed()
        self._save_entry(eid, entry)
        self._log_to_systems(seed_concept, scenario, importance)
        confidence = round(min(0.99, importance * novelty + 0.3), 3)
        return {"id": eid, "seed": seed_concept, "scenario": scenario,
                "novelty": round(novelty, 3), "confidence": confidence,
                "importance": round(importance, 3)}

    def combine(self, concept_a: str, concept_b: str) -> dict[str, Any]:
        """Returns a novel hybrid idea merging two concepts with a Bayesian fusion score."""
        nov_a = self._novelty_score(concept_a)
        nov_b = self._novelty_score(concept_b)
        prior = 0.5
        likelihood_a = 0.4 + 0.6 * nov_a
        likelihood_b = 0.4 + 0.6 * nov_b
        posterior = (likelihood_a * likelihood_b * prior) / (
            likelihood_a * likelihood_b * prior +
            (1 - likelihood_a) * (1 - likelihood_b) * (1 - prior) + 1e-12)
        hybrid_seed = f"{concept_a}-{concept_b} fusion"
        scenario = self._render_scenario(hybrid_seed)
        hybrid_idea = (f"[HYBRID] The {random.choice(ADJECTIVES)} convergence of "
                       f"'{concept_a}' and '{concept_b}' creates a {random.choice(NOUNS)} "
                       f"where {scenario}")
        eid = self._make_id(hybrid_seed)
        entry = {"concept": hybrid_seed, "scenario": hybrid_idea,
                 "importance": posterior, "ts": time.time(),
                 "access_count": 1, "kind": "combine"}
        with self._lock:
            self._store[eid] = entry
            self._store.move_to_end(eid)
            self._evict_if_needed()
        self._save_entry(eid, entry)
        self._log_to_systems(hybrid_seed, hybrid_idea, posterior)
        return {"id": eid, "concept_a": concept_a, "concept_b": concept_b,
                "hybrid": hybrid_idea, "fusion_confidence": round(posterior, 3),
                "novelty_a": round(nov_a, 3), "novelty_b": round(nov_b, 3)}

    def dream_log(self, limit: int = 20) -> list[dict[str, Any]]:
        """Returns recent imagination entries sorted by decayed importance descending."""
        with self._lock:
            scored = [(eid, e, self._decayed_importance(e))
                      for eid, e in self._store.items()]
        scored.sort(key=lambda x: x[2], reverse=True)
        return [{"id": eid, "concept": e["concept"], "scenario": e["scenario"][:120],
                 "kind": e["kind"], "decayed_importance": round(imp, 3),
                 "access_count": e["access_count"],
                 "age_min": round((time.time() - e["ts"]) / 60, 1)}
                for eid, e, imp in scored[:limit]]

    def recall(self, idea_id: str) -> dict[str, Any]:
        """Returns full entry for idea_id, boosting its importance on access."""
        with self._lock:
            if idea_id not in self._store:
                return {"error": "not_found", "id": idea_id}
            entry = self._store[idea_id]
            entry["importance"] = min(1.0, self._decayed_importance(entry) + self.ACCESS_BOOST)
            entry["ts"] = time.time()
            entry["access_count"] += 1
            self._store.move_to_end(idea_id)
        self._save_entry(idea_id, entry)
        return dict(entry)

    def consolidate(self) -> dict[str, Any]:
        """Returns consolidation report after pruning low-importance dreams."""
        pruned = 0
        threshold = 0.04
        with self._lock:
            dead = [eid for eid, e in self._store.items()
                    if self._decayed_importance(e) < threshold]
            for eid in dead:
                del self._store[eid]
                pruned += 1
        if pruned:
            with sqlite3.connect(DB_PATH) as con:
                for eid in dead:
                    con.execute("DELETE FROM dreams WHERE id=?", (eid,))
                con.commit()
        return {"pruned": pruned, "remaining": len(self._store), "entropy": round(self._entropy(), 4)}

    def capacity_used(self) -> dict[str, Any]:
        """Returns current capacity metrics as a numeric dict for ConsciousnessIntegrator."""
        with self._lock:
            items = len(self._store)
            imps = [self._decayed_importance(e) for e in self._store.values()]
        avg_imp = round(statistics.mean(imps), 3) if imps else 0.0
        stdev_imp = round(statistics.stdev(imps), 3) if len(imps) > 1 else 0.0
        return {"items": items, "capacity": self.MAX_CAPACITY,
                "fill_pct": round(100 * items / self.MAX_CAPACITY, 1),
                "avg_importance": avg_imp, "stdev_importance": stdev_imp,
                "entropy": round(self._entropy(), 4), "cycles": self._cycles,
                "confidence": avg_imp, "active": items}

    def status(self) -> dict[str, Any]:
        """Returns plain numeric status dict compatible with ConsciousnessIntegrator Φ."""
        cap = self.capacity_used()
        quality = round(statistics.mean(self._quality_history[-20:]), 3) if self._quality_history else 0.5
        return {"items": cap["items"], "confidence": cap["confidence"],
                "entropy": cap["entropy"], "cycles": self._cycles,
                "quality": quality, "active": cap["active"],
                "pending": max(0, self.MAX_CAPACITY - cap["items"])}

    def auto_cycle(self) -> dict[str, Any]:
        """Runs one autonomous imagination cycle; returns cycle summary dict."""
        seeds = ["emergence","recursion","entropy","consciousness","novelty",
                 "causality","symmetry","resonance","threshold","singularity"]
        seed = random.choice(seeds)
        result = self.imagine(seed)
        self._cycles += 1
        quality = result["confidence"] * result["novelty"]
        self._quality_history.append(quality)
        if len(self._quality_history) > 200:
            self._quality_history = self._quality_history[-200:]
        consol = self.consolidate()
        self._generate_own_goals()
        try:
            from MetacognitiveMonitor import MetacognitiveMonitor
            mon = MetacognitiveMonitor()
            mon.log_reasoning("imagination", "bayesian_decay_lru", result["confidence"], quality > 0.25)
        except (ImportError, Exception):
            pass
        return {"cycle": self._cycles, "seed": seed, "quality": round(quality, 3),
                "pruned": consol["pruned"], "entropy": consol["entropy"]}

    # ── Internal helpers ─────────────────────────────────────────────────────
    def _evict_if_needed(self) -> None:
        while len(self._store) > self.MAX_CAPACITY:
            lru_id, _ = next(iter(self._store.items()))
            del self._store[lru_id]

    def _log_to_systems(self, concept: str, scenario: str, importance: float) -> None:
        try:
            from WorkingMemory import WorkingMemory
            wm = WorkingMemory()
            wm.store(f"dream:{concept}", scenario[:200], importance)
        except (ImportError, Exception):
            pass
        try:
            from HierarchicalGoalPlanner import HierarchicalGoalPlanner
            hgp = HierarchicalGoalPlanner()
            if importance > 0.7:
                hgp.add_goal(f"Explore imagination thread: {concept}", priority=2)
        except (ImportError, Exception):
            pass

    def _generate_own_goals(self) -> None:
        try:
            from HierarchicalGoalPlanner import HierarchicalGoalPlanner
            hgp = HierarchicalGoalPlanner()
            ent = self._entropy()
            if ent > 3.0:
                hgp.add_goal("Consolidate high-entropy dream log to extract insight patterns", priority=3)
            elif len(self._store) > 150:
                hgp.add_goal("Prune low-salience imaginations to free creative capacity", priority=2)
        except (ImportError, Exception):
            pass

    def _auto_loop(self) -> None:
        time.sleep(10)
        while True:
            try:
                self.auto_cycle()
            except Exception:
                pass
            time.sleep(self.CYCLE_INTERVAL)

# Usage: obj = NovaImaginationFabric() | result = obj.imagine("quantum consciousness")
