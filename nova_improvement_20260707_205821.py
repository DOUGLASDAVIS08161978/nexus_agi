"""
AdversarialConceptualImmuneSystem — Nova's memetic immune layer.

Continuously patrols the cognitive stack for corrupted priors, adversarially
injected assumptions, goal-drift, and conceptual parasites. Implements cosine
drift detection, mutual-information injection flagging, antibody optimization,
LSH-indexed immune memory, and Bayesian working memory with exponential decay.
"""

import math
import time
import json
import sqlite3
import hashlib
import random
import threading
import statistics
import collections
import os
from typing import Any, Optional

# ── Named types (plain dicts kept lightweight; no numpy dependency) ──────────
AnomalyMap       = dict   # cluster_id -> {z_score, mi_score, flagged}
QuarantineRecord = dict   # cluster_id, confidence, centroid, ts, antibody
ConceptVector    = list   # list[float]  ℝ^n embedding
CounterFraming   = dict   # antibody vector + semantic label + kl_loss
ImmunityMemoryEntry = dict # lsh_hash, signature, antibody, ts, reuse_count
GoalState        = dict   # id, description, priority, embedding, ts
DriftVector      = dict   # vector, magnitude, cosine_alignment, window


class AdversarialConceptualImmuneSystem:
    """Living memetic immune system for Nova's entire cognitive stack."""

    # ── init ─────────────────────────────────────────────────────────────────
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._belief_history:       dict[str, list[list[float]]] = collections.defaultdict(list)
        self._baseline_drift_stats: dict[str, tuple[float, float]] = {}
        self._quarantine_registry:  dict[str, QuarantineRecord]  = {}
        self._immune_memory:        dict[int, ImmunityMemoryEntry] = {}
        self._goal_history:         list[GoalState]               = []
        self._external_input_log:   dict[str, list[list[float]]] = collections.defaultdict(list)
        self._infection_embeddings: dict[str, ConceptVector]      = {}
        self._lsh_index:            dict[int, str]                = {}
        self._lambda_antibody:      float = 0.3
        self._decay_rate:           float = 0.0012
        self._wm_store:             dict[str, dict]               = {}
        self._wm_capacity:          int   = 200
        self._health_score_cache:   float = 1.0
        self._scan_count:           int   = 0
        self._random_projections:   list[list[float]]             = [
            self._unit_vec(32) for _ in range(32)
        ]
        self._alignment_axis: list[float] = self._unit_vec(32)
        db_path = os.path.join(os.path.dirname(__file__), "immune_system.db")
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_db()
        self._load_state()
        t = threading.Thread(target=self._daemon, daemon=True)
        t.start()

    # ── private helpers ───────────────────────────────────────────────────────
    def _unit_vec(self, n: int) -> list[float]:
        v = [random.gauss(0, 1) for _ in range(n)]
        norm = math.sqrt(sum(x*x for x in v)) + 1e-9
        return [x/norm for x in v]

    def _cosine_drift(self, a: list[float], b: list[float]) -> float:
        dot  = sum(x*y for x, y in zip(a, b))
        na   = math.sqrt(sum(x*x for x in a))
        nb   = math.sqrt(sum(x*x for x in b))
        return 1.0 - dot / (na * nb + 1e-9)

    def _entropy(self, vec: list[float], bins: int = 32) -> float:
        mn, mx = min(vec), max(vec)
        span = mx - mn + 1e-9
        counts = [0] * bins
        for v in vec:
            idx = min(int((v - mn) / span * bins), bins - 1)
            counts[idx] += 1
        total = len(vec) + 1e-9
        return -sum((c/total) * math.log(c/total + 1e-12) for c in counts if c)

    def _mutual_info(self, bvec: list[float], evec: list[float]) -> float:
        hb = self._entropy(bvec)
        he = self._entropy(evec)
        joint = [b + e for b, e in zip(bvec, evec)]
        hj = self._entropy(joint)
        return max(0.0, hb + he - hj)

    def _lsh(self, vec: list[float]) -> int:
        n = min(len(vec), 32)
        bits = 0
        for i, r in enumerate(self._random_projections[:n]):
            dot = sum(r[j] * vec[j] for j in range(min(len(r), n)))
            if dot >= 0:
                bits |= (1 << i)
        return bits

    def _kl_divergence(self, p: list[float], q: list[float]) -> float:
        total = 0.0
        sp = sum(p) + 1e-9
        sq = sum(q) + 1e-9
        for pi, qi in zip(p, q):
            pi /= sp; qi /= sq
            if pi > 1e-12:
                total += pi * math.log((pi + 1e-12) / (qi + 1e-12))
        return total

    def _antibody_gradient_descent(
        self, centroid: list[float], clean_beliefs: list[float],
        infected_beliefs: list[float]
    ) -> tuple[list[float], float]:
        n   = len(centroid)
        A   = centroid[:]
        lr  = 0.01
        lam = self._lambda_antibody
        for _ in range(500):
            kl  = self._kl_divergence(infected_beliefs[:n], clean_beliefs[:n])
            diff = [A[i] - centroid[i] for i in range(n)]
            dist = math.sqrt(sum(d*d for d in diff)) + 1e-9
            grad = [lam * diff[i] / dist for i in range(n)]
            A    = [A[i] - lr * grad[i] for i in range(n)]
            if math.sqrt(sum(g*g for g in grad)) < 1e-5:
                break
        loss = kl + lam * math.sqrt(sum((A[i]-centroid[i])**2 for i in range(n)))
        return A, loss

    def _update_baseline(self, module_id: str, d: float) -> None:
        hist = [rec["drift"] for rec in self._belief_history.get(module_id, [])
                if "drift" in rec]
        if len(hist) < 3:
            self._baseline_drift_stats[module_id] = (d, 0.1)
            return
        mu  = statistics.mean(hist[-50:])
        std = statistics.stdev(hist[-50:]) if len(hist) >= 2 else 0.1
        self._baseline_drift_stats[module_id] = (mu, std)

    def _wm_decay_importance(self, key: str) -> float:
        item = self._wm_store.get(key)
        if not item:
            return 0.0
        elapsed = time.time() - item["timestamp"]
        return item["importance"] * math.exp(-self._decay_rate * elapsed)

    def _wm_evict(self) -> None:
        if len(self._wm_store) < self._wm_capacity:
            return
        scored = [(k, self._wm_decay_importance(k)) for k in self._wm_store]
        scored.sort(key=lambda x: x[1])
        for k, _ in scored[:max(1, len(scored)//10)]:
            del self._wm_store[k]

    def _init_db(self) -> None:
        c = self._conn.cursor()
        c.execute("""CREATE TABLE IF NOT EXISTS immune_memory(
            lsh_hash INTEGER PRIMARY KEY,
            signature TEXT, antibody TEXT,
            ts REAL, reuse_count INTEGER)""")
        c.execute("""CREATE TABLE IF NOT EXISTS quarantine(
            cluster_id TEXT PRIMARY KEY,
            record TEXT, ts REAL)""")
        self._conn.commit()

    def _load_state(self) -> None:
        c = self._conn.cursor()
        for row in c.execute("SELECT lsh_hash, signature, antibody, ts, reuse_count FROM immune_memory"):
            self._immune_memory[row[0]] = {
                "lsh_hash": row[0],
                "signature": json.loads(row[1]),
                "antibody":  json.loads(row[2]),
                "ts": row[3], "reuse_count": row[4]
            }
        for row in c.execute("SELECT cluster_id, record FROM quarantine"):
            self._quarantine_registry[row[0]] = json.loads(row[1])

    def _persist_immune(self, entry: ImmunityMemoryEntry) -> None:
        try:
            c = self._conn.cursor()
            c.execute("""INSERT OR REPLACE INTO immune_memory
                VALUES(?,?,?,?,?)""",
                (entry["lsh_hash"], json.dumps(entry["signature"]),
                 json.dumps(entry["antibody"]), entry["ts"], entry["reuse_count"]))
            self._conn.commit()
        except sqlite3.Error:
            pass

    def _persist_quarantine(self, record: QuarantineRecord) -> None:
        try:
            c = self._conn.cursor()
            c.execute("INSERT OR REPLACE INTO quarantine VALUES(?,?,?)",
                      (record["cluster_id"], json.dumps(record), record["ts"]))
            self._conn.commit()
        except sqlite3.Error:
            pass

    def _daemon(self) -> None:
        while True:
            time.sleep(90)
            try:
                self._auto_immune_cycle()
            except Exception:
                pass

    def _auto_immune_cycle(self) -> None:
        with self._lock:
            # decay WM
            for k in list(self._wm_store.keys()):
                imp = self._wm_decay_importance(k)
                if imp < 1e-4:
                    del self._wm_store[k]
            # synthesize health and store
            score = self._compute_health()
            self._health_score_cache = score
            self.store("immune_health", score, importance=0.9)
        # cross-system integration
        try:
            from metacognitive_monitor import MetacognitiveMonitor  # type: ignore
            MetacognitiveMonitor().log_reasoning(
                "immune_system", "auto_cycle",
                confidence=self._health_score_cache, success=self._health_score_cache > 0.6
            )
        except Exception:
            pass
        try:
            if self._health_score_cache < 0.7:
                from hierarchical_goal_planner import HierarchicalGoalPlanner  # type: ignore
                HierarchicalGoalPlanner().add_goal(
                    f"Resolve memetic infection — immune health {self._health_score_cache:.2f}",
                    priority=8
                )
        except Exception:
            pass

    def _compute_health(self) -> float:
        quarantine_penalty = min(0.5, len(self._quarantine_registry) * 0.05)
        memory_bonus       = min(0.2, len(self._immune_memory) * 0.01)
        scan_bonus         = min(0.1, self._scan_count * 0.002)
        return max(0.0, min(1.0, 1.0 - quarantine_penalty + memory_bonus + scan_bonus))

    # ── public methods ────────────────────────────────────────────────────────
    def scan_belief_topology(self, module_snapshots: list[dict]) -> AnomalyMap:
        """Returns AnomalyMap: cluster_id -> {z_score, mi_score, flagged, drift}."""
        result: AnomalyMap = {}
        with self._lock:
            self._scan_count += 1
            for snap in module_snapshots:
                mid  = snap.get("module_id", "unknown")
                bvec = snap.get("belief_vector", [random.gauss(0,1) for _ in range(32)])
                evec = snap.get("external_input", [random.gauss(0,1) for _ in range(32)])
                n    = min(len(bvec), 32)
                bvec = bvec[:n]; evec = evec[:n]
                hist = self._belief_history[mid]
                if hist:
                    prev = hist[-1] if isinstance(hist[-1], list) else bvec
                    d    = self._cosine_drift(bvec, prev)
                else:
                    d = 0.0
                self._belief_history[mid].append(bvec)
                self._external_input_log[mid].append(evec)
                if len(self._belief_history[mid]) > 200:
                    self._belief_history[mid] = self._belief_history[mid][-200:]
                self._update_baseline(mid, d)
                mu, std = self._baseline_drift_stats.get(mid, (0.0, 0.1))
                z_d = (d - mu) / (std + 1e-9)
                mi  = self._mutual_info(bvec, evec)
                mi_vals = [self._mutual_info(
                    self._belief_history[mid][i],
                    self._external_input_log[mid][i]
                ) for i in range(min(len(self._belief_history[mid]),
                                     len(self._external_input_log[mid])))]
                mi_mu  = statistics.mean(mi_vals) if mi_vals else 0.0
                mi_std = statistics.stdev(mi_vals) if len(mi_vals) > 1 else 0.1
                z_mi   = (mi - mi_mu) / (mi_std + 1e-9)
                flagged = z_d > 2.0 and z_mi > 2.0
                cid = f"{mid}_cluster_{self._scan_count}"
                result[cid] = {"z_score": round(z_d, 4),
                                "mi_score": round(mi, 4),
                                "z_mi": round(z_mi, 4),
                                "flagged": flagged,
                                "drift": round(d, 4),
                                "module_id": mid}
                if flagged:
                    self._infection_embeddings[cid] = bvec
        return result

    def quarantine_concept_cluster(self, cluster_id: str, confidence: float) -> QuarantineRecord:
        """Returns QuarantineRecord with cluster metadata and quarantine timestamp."""
        with self._lock:
            centroid = self._infection_embeddings.get(cluster_id, self._unit_vec(32))
            record: QuarantineRecord = {
                "cluster_id": cluster_id,
                "confidence": round(min(1.0, max(0.0, confidence)), 4),
                "centroid": centroid,
                "ts": time.time(),
                "antibody": None,
                "status": "quarantined"
            }
            self._quarantine_registry[cluster_id] = record
            self._persist_quarantine(record)
            self.store(f"quarantine_{cluster_id}", confidence, importance=confidence)
        return record

    def generate_antibody(self, infection_signature: ConceptVector) -> CounterFraming:
        """Returns CounterFraming dict with antibody vector, kl_loss, and semantic label."""
        with self._lock:
            n       = min(len(infection_signature), 32)
            sig     = infection_signature[:n]
            clean   = self._unit_vec(n)
            antibody, kl_loss = self._antibody_gradient_descent(sig, clean, sig)
            sem_dist = self._cosine_drift(antibody, sig)
            label    = f"antibody_kl{kl_loss:.3f}_sd{sem_dist:.3f}"
            lsh_h    = self._lsh(sig)
            entry: ImmunityMemoryEntry = {
                "lsh_hash": lsh_h,
                "signature": sig,
                "antibody":  antibody,
                "ts": time.time(),
                "reuse_count": 0
            }
            self._immune_memory[lsh_h]  = entry
            self._lsh_index[lsh_h]      = label
            self._persist_immune(entry)
            framing: CounterFraming = {
                "antibody_vector": antibody,
                "kl_loss": round(kl_loss, 6),
                "semantic_distance": round(sem_dist, 4),
                "label": label,
                "lsh_hash": lsh_h
            }
        return framing

    def inoculate(self, past_infection: QuarantineRecord) -> ImmunityMemoryEntry:
        """Returns ImmunityMemoryEntry stored in LSH-indexed immune memory for O(1) future retrieval."""
        with self._lock:
            centroid = past_infection.get("centroid", self._unit_vec(32))
            antibody = past_infection.get("antibody") or self._unit_vec(len(centroid))
            lsh_h    = self._lsh(centroid)
            existing = self._immune_memory.get(lsh_h)
            if existing:
                existing["reuse_count"] += 1
                self._persist_immune(existing)
                return existing
            entry: ImmunityMemoryEntry = {
                "lsh_hash":    lsh_h,
                "signature":   centroid,
                "antibody":    antibody if isinstance(antibody, list) else list(antibody),
                "ts":          time.time(),
                "reuse_count": 1
            }
            self._immune_memory[lsh_h] = entry
            self._lsh_index[lsh_h]     = past_infection.get("cluster_id", "unknown")
            self._persist_immune(entry)
            self.store(f"inoculation_{lsh_h}", lsh_h, importance=0.85)
        return entry

    def audit_goal_drift(self, goal_history: list[GoalState], window: int = 10) -> DriftVector:
        """Returns DriftVector with magnitude, cosine alignment, and per-step drift scores."""
        with self._lock:
            with self._lock:
                self._goal_history.extend(goal_history)
            n = len(goal_history)
            if n < 2:
                return {"vector": [], "magnitude": 0.0,
                        "cosine_alignment": 1.0, "window": window,
                        "drift_scores": [], "anomaly": False}
            w     = min(window, n)
            start = goal_history[max(0, n-w)]
            end   = goal_history[-1]
            def _g_vec(gs: GoalState) -> list[float]:
                emb = gs.get("embedding")
                if emb and isinstance(emb, list):
                    return emb[:32]
                pri = float(gs.get("priority", 5))
                return [pri / 10.0] * 32
            sv    = _g_vec(start)
            ev    = _g_vec(end)
            drift = [(ev[i] - sv[i]) / (abs(sv[i]) + 1e-9) for i in range(len(sv))]
            mag   = math.sqrt(sum(d*d for d in drift))
            ax    = self._alignment_axis[:len(drift)]
            cos_a = sum(drift[i]*ax[i] for i in range(len(drift))) / (mag * 1.0 + 1e-9)
            scores = []
            for i in range(1, w):
                idx = max(0, n - w + i)
                if idx < n:
                    scores.append(round(self._cosine_drift(_g_vec(goal_history[idx-1]),
                                                           _g_vec(goal_history[idx])), 4))
            anomaly = mag > 2.0 or (scores and max(scores) > 0.5)
            if anomaly:
                try:
                    from hierarchical_goal_planner import HierarchicalGoalPlanner  # type: ignore
                    HierarchicalGoalPlanner().add_goal(
                        f"Investigate goal drift — magnitude {mag:.2f}", priority=7
                    )
                except Exception:
                    pass
        return {"vector": [round(d, 4) for d in drift],
                "magnitude": round(mag, 4),
                "cosine_alignment": round(cos_a, 4),
                "window": w, "drift_scores": scores,
                "anomaly": anomaly}

    def synthesize_immune_report(self) -> dict:
        """Returns ImmuneHealthScore dict with numeric keys for ConsciousnessIntegrator Φ."""
        with self._lock:
            score    = self._compute_health()
            self._health_score_cache = score
            q_count  = len(self._quarantine_registry)
            m_count  = len(self._immune_memory)
            entropy_vals = []
            for mid, hist in self._belief_history.items():
                if hist:
                    entropy_vals.append(self._entropy(hist[-1]))
            avg_ent = statistics.mean(entropy_vals) if entropy_vals else 0.0
            active_infections = sum(
                1 for r in self._quarantine_registry.values()
                if r.get("status") == "quarantined"
            )
            confidence = score
            report = {
                "health_score":       round(score, 4),
                "confidence":         round(confidence, 4),
                "quarantine_count":   q_count,
                "immune_memory_size": m_count,
                "active_infections":  active_infections,
                "avg_belief_entropy": round(avg_ent, 4),
                "scan_count":         self._scan_count,
                "items":              m_count + q_count,
                "cycles":             self._scan_count,
                "entropy":            round(avg_ent, 4),
                "quality":            round(score, 4),
                "active":             active_infections,
                "pending":            q_count,
                "accuracy":           round(score, 4),
            }
        try:
            from working_memory import WorkingMemory  # type: ignore
            WorkingMemory().store("immune_report", report, importance=0.8)
        except Exception:
            pass
        return report

    # ── Bayesian working memory ───────────────────────────────────────────────
    def store(self, key: str, value: Any, importance: float = 0.5) -> None:
        """Stores item in Bayesian working memory with decay; evicts LRU when full."""
        with self._lock:
            self._wm_evict()
            if key in self._wm_store:
                self._wm_store[key]["importance"] = min(1.0, importance + 0.1)
                self._wm_store[key]["access_count"] += 1
                self._wm_store[key]["value"] = value
                self._wm_store[key]["timestamp"] = time.time()
            else:
                self._wm_store[key] = {
                    "value": value,
                    "importance": importance,
                    "timestamp": time.time(),
                    "access_count": 0
                }

    def retrieve(self, key: str) -> Optional[Any]:
        """Returns decayed value from working memory, boosting importance on access."""
        with self._lock:
            item = self._wm_store.get(key)
            if not item:
                return None
            item["importance"] = min(1.0, self._wm_decay_importance(key) + 0.05)
            item["access_count"] += 1
            item["timestamp"] = time.time()
            return item["value"]

    def status(self) -> dict:
        """Returns numeric status dict compatible with ConsciousnessIntegrator Φ."""
        with self._lock:
            return {
                "items":            len(self._wm_store) + len(self._immune_memory),
                "confidence":       round(self._health_score_cache, 4),
                "accuracy":         round(self._health_score_cache, 4),
                "quality":          round(self._health_score_cache, 4),
                "active":           len(self._quarantine_registry),
                "pending":          sum(1 for r in self._quarantine_registry.values()
                                        if r.get("status") == "quarantined"),
                "cycles":           self._scan_count,
                "entropy":          round(
                    statistics.mean([self._entropy(v[-1]) for v in self._belief_history.values()
                                     if v]) if self._belief_history else 0.0, 4),
                "immune_memory":    len(self._immune_memory),
                "wm_capacity_used": len(self._wm_store),
            }

# Usage: obj = AdversarialConceptualImmuneSystem() | result = obj.synthesize_immune_report()
