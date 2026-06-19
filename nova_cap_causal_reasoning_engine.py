"""
Nova ASI — Causal Reasoning Engine
Proposed autonomously via /evolve
"""

"""
CausalEngine — probabilistic cause-and-effect modelling with Bayesian chain propagation,
LRU-backed working memory, exponential importance decay, and full causal-trace with
confidence intervals. Integrates with BayesianBeliefSystem, WorkingMemory, and
MetacognitiveMonitor for live cross-system intelligence.
"""

import sqlite3
import threading
import math
import time
import statistics
import json
import os
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple


class CausalEngine:
    """Model cause-and-effect chains with probabilistic confidence and Bayesian decay."""

    _DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "causal_engine.db")
    _CAPACITY = 200
    _DECAY_RATE = 0.0002          # importance half-life ≈ 57 minutes
    _PRUNE_THRESHOLD = 0.05       # drop causal paths below this chain strength
    _ACCESS_BOOST = 1.15          # importance multiplier on retrieval

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._lru: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._observations: List[Tuple[float, float]] = []   # (predicted, actual) for calibration
        self._conn = sqlite3.connect(self._DB, check_same_thread=False)
        self._init_db()
        self._load_from_db()

    # ------------------------------------------------------------------ #
    #  Private helpers                                                     #
    # ------------------------------------------------------------------ #

    def _init_db(self) -> None:
        with self._conn:
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS causal_edges (
                    src TEXT NOT NULL,
                    dst TEXT NOT NULL,
                    direction TEXT NOT NULL CHECK(direction IN ('cause','effect')),
                    confidence REAL NOT NULL DEFAULT 0.8,
                    observations INTEGER NOT NULL DEFAULT 1,
                    PRIMARY KEY (src, dst, direction)
                )""")
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS wm_items (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    importance REAL NOT NULL,
                    timestamp REAL NOT NULL,
                    access_count INTEGER NOT NULL DEFAULT 0
                )""")

    def _load_from_db(self) -> None:
        cur = self._conn.execute(
            "SELECT key, value, importance, timestamp, access_count FROM wm_items ORDER BY timestamp")
        for row in cur.fetchall():
            key, val_json, imp, ts, ac = row
            self._lru[key] = {
                "value": json.loads(val_json),
                "importance": imp,
                "timestamp": ts,
                "access_count": ac,
            }

    def _decayed_importance(self, item: Dict[str, Any]) -> float:
        elapsed = time.time() - item["timestamp"]
        return item["importance"] * math.exp(-self._DECAY_RATE * elapsed)

    def _evict_lru(self) -> None:
        """Evict least-important item when capacity exceeded (LRU + importance score)."""
        while len(self._lru) > self._CAPACITY:
            worst_key = min(
                self._lru,
                key=lambda k: self._decayed_importance(self._lru[k]) * math.log1p(self._lru[k]["access_count"] + 1)
            )
            self._lru.pop(worst_key)
            self._conn.execute("DELETE FROM wm_items WHERE key=?", (worst_key,))

    def _chain_confidence(self, path: List[str]) -> float:
        """Multiply edge confidences along a path (causal chain rule)."""
        if len(path) < 2:
            return 1.0
        conf = 1.0
        for i in range(len(path) - 1):
            cur = self._conn.execute(
                "SELECT confidence FROM causal_edges WHERE src=? AND dst=?",
                (path[i], path[i + 1])
            ).fetchone()
            conf *= (cur[0] if cur else 0.5)
        return conf

    def _bayesian_update(self, src: str, dst: str, direction: str, observed: bool) -> float:
        """Update edge confidence with Bayesian posterior."""
        row = self._conn.execute(
            "SELECT confidence, observations FROM causal_edges WHERE src=? AND dst=? AND direction=?",
            (src, dst, direction)
        ).fetchone()
        prior = row[0] if row else 0.5
        obs = row[1] if row else 0
        likelihood = 0.85 if observed else 0.15
        posterior = (likelihood * prior) / (likelihood * prior + (1 - likelihood) * (1 - prior))
        new_obs = obs + 1
        with self._conn:
            self._conn.execute("""
                INSERT INTO causal_edges (src, dst, direction, confidence, observations)
                VALUES (?,?,?,?,?)
                ON CONFLICT(src,dst,direction) DO UPDATE SET
                    confidence=excluded.confidence,
                    observations=excluded.observations
            """, (src, dst, direction, round(posterior, 6), new_obs))
        return posterior

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def add_cause(self, event: str, causes: List[str], confidence: float = 0.8) -> Dict[str, Any]:
        """Returns updated edge confidences after registering causes for event."""
        results: Dict[str, float] = {}
        with self._lock:
            for cause in causes:
                conf = self._bayesian_update(cause, event, "cause", observed=True)
                results[cause] = conf
                self.store(f"cause:{cause}->{event}", {"cause": cause, "event": event}, conf)
        try:
            from BayesianBeliefSystem import BayesianBeliefSystem
            bbs = BayesianBeliefSystem()
            for cause in causes:
                bbs.add_causal_edge(cause, event)
        except Exception:
            pass
        return {"event": event, "causes_registered": results}

    def add_effect(self, event: str, effects: List[str], confidence: float = 0.8) -> Dict[str, Any]:
        """Returns updated edge confidences after registering effects for event."""
        results: Dict[str, float] = {}
        with self._lock:
            for effect in effects:
                conf = self._bayesian_update(event, effect, "effect", observed=True)
                results[effect] = conf
                self.store(f"effect:{event}->{effect}", {"event": event, "effect": effect}, conf)
        try:
            from BayesianBeliefSystem import BayesianBeliefSystem
            bbs = BayesianBeliefSystem()
            for effect in effects:
                bbs.add_causal_edge(event, effect)
        except Exception:
            pass
        return {"event": event, "effects_registered": results}

    def trace(self, event: str, max_depth: int = 6) -> Dict[str, Any]:
        """Returns full causal chain (upstream causes + downstream effects) with confidence intervals."""
        with self._lock:
            upstream = self._bfs(event, direction="cause", max_depth=max_depth)
            downstream = self._bfs(event, direction="effect", max_depth=max_depth)

        all_confs = [p["chain_confidence"] for p in upstream + downstream if p["chain_confidence"] > 0]
        if len(all_confs) >= 2:
            mean_c = statistics.mean(all_confs)
            std_c = statistics.stdev(all_confs)
            ci_low = max(0.0, mean_c - 1.96 * std_c)
            ci_high = min(1.0, mean_c + 1.96 * std_c)
        elif all_confs:
            mean_c = all_confs[0]
            ci_low, ci_high = mean_c * 0.8, min(1.0, mean_c * 1.2)
        else:
            mean_c = ci_low = ci_high = 0.0

        self._observations.append((mean_c, mean_c))  # self-referential calibration seed
        self.store(f"trace:{event}", {"upstream": len(upstream), "downstream": len(downstream)}, mean_c)

        try:
            from MetacognitiveMonitor import MetacognitiveMonitor
            mm = MetacognitiveMonitor()
            mm.log_reasoning("causal_trace", "BFS+Bayesian", mean_c, True)
        except Exception:
            pass

        return {
            "event": event,
            "upstream_causes": upstream,
            "downstream_effects": downstream,
            "mean_chain_confidence": round(mean_c, 4),
            "confidence_interval_95": (round(ci_low, 4), round(ci_high, 4)),
            "total_paths": len(upstream) + len(downstream),
        }

    def _bfs(self, start: str, direction: str, max_depth: int) -> List[Dict[str, Any]]:
        """BFS over causal graph; direction='cause' walks backwards, 'effect' forwards."""
        visited: set = set()
        queue: List[Tuple[str, List[str], float]] = [(start, [start], 1.0)]
        paths: List[Dict[str, Any]] = []
        col_src, col_dst = ("dst", "src") if direction == "cause" else ("src", "dst")
        while queue:
            node, path, acc_conf = queue.pop(0)
            if len(path) > max_depth:
                continue
            rows = self._conn.execute(
                f"SELECT {col_dst}, confidence FROM causal_edges WHERE {col_src}=? AND direction=?",
                (node, direction)
            ).fetchall()
            for neighbour, edge_conf in rows:
                if neighbour in visited:
                    continue
                new_conf = acc_conf * edge_conf
                if new_conf < self._PRUNE_THRESHOLD:
                    continue
                new_path = path + [neighbour]
                visited.add(neighbour)
                z_score = (new_conf - 0.5) / 0.2
                paths.append({
                    "path": new_path,
                    "chain_confidence": round(new_conf, 4),
                    "z_score": round(z_score, 3),
                })
                queue.append((neighbour, new_path, new_conf))
        return paths

    def store(self, key: str, value: Any, importance: float = 0.5) -> None:
        """Returns None; stores item in LRU working memory with decay-aware importance."""
        now = time.time()
        with self._lock:
            if key in self._lru:
                item = self._lru[key]
                item["importance"] = min(1.0, self._decayed_importance(item) * self._ACCESS_BOOST + importance * 0.3)
                item["access_count"] += 1
                item["timestamp"] = now
                self._lru.move_to_end(key)
            else:
                self._lru[key] = {
                    "value": value,
                    "importance": float(importance),
                    "timestamp": now,
                    "access_count": 0,
                }
                self._lru.move_to_end(key)
            self._evict_lru()
            with self._conn:
                self._conn.execute("""
                    INSERT INTO wm_items (key, value, importance, timestamp, access_count)
                    VALUES (?,?,?,?,?)
                    ON CONFLICT(key) DO UPDATE SET
                        value=excluded.value, importance=excluded.importance,
                        timestamp=excluded.timestamp, access_count=excluded.access_count
                """, (key, json.dumps(value), self._lru[key]["importance"], now,
                      self._lru[key]["access_count"]))

    def retrieve(self, key: str) -> Optional[Dict[str, Any]]:
        """Returns item dict with decayed importance, or None if absent."""
        with self._lock:
            if key not in self._lru:
                return None
            item = self._lru[key]
            item["importance"] = min(1.0, self._decayed_importance(item) * self._ACCESS_BOOST)
            item["access_count"] += 1
            self._lru.move_to_end(key)
            return {
                "value": item["value"],
                "importance": round(item["importance"], 4),
                "access_count": item["access_count"],
                "age_seconds": round(time.time() - item["timestamp"], 1),
            }

    def consolidate(self) -> Dict[str, Any]:
        """Returns consolidation report; prunes decayed items and weak causal edges."""
        pruned_wm = 0
        with self._lock:
            dead_keys = [k for k, v in self._lru.items() if self._decayed_importance(v) < 0.01]
            for k in dead_keys:
                self._lru.pop(k)
                self._conn.execute("DELETE FROM wm_items WHERE key=?", (k,))
            pruned_wm = len(dead_keys)
        with self._conn:
            deleted = self._conn.execute(
                "DELETE FROM causal_edges WHERE confidence < ?", (self._PRUNE_THRESHOLD,)
            ).rowcount
        return {"pruned_wm_items": pruned_wm, "pruned_causal_edges": deleted, "wm_remaining": len(self._lru)}

    def capacity_used(self) -> Dict[str, Any]:
        """Returns capacity metrics as a plain dict with numeric health keys."""
        with self._lock:
            wm_count = len(self._lru)
        edge_count = self._conn.execute("SELECT COUNT(*) FROM causal_edges").fetchone()[0]
        avg_conf_row = self._conn.execute("SELECT AVG(confidence) FROM causal_edges").fetchone()
        avg_conf = round(avg_conf_row[0] or 0.0, 4)
        if len(self._observations) >= 2:
            preds, actuals = zip(*self._observations[-50:])
            mae = statistics.mean(abs(p - a) for p, a in zip(preds, actuals))
            calibration_error = round(mae, 4)
        else:
            calibration_error = 0.0
        return {
            "items": wm_count,
            "capacity": self._CAPACITY,
            "fill_pct": round(wm_count / self._CAPACITY * 100, 1),
            "causal_edges": edge_count,
            "avg_confidence": avg_conf,
            "calibration_error": calibration_error,
            "confidence": avg_conf,
            "accuracy": max(0.0, 1.0 - calibration_error),
        }

    def forget_least_important(self, n: int = 10) -> Dict[str, int]:
        """Returns count of forgotten items; evicts n lowest-importance working-memory entries."""
        removed = 0
        with self._lock:
            scored = sorted(self._lru.items(), key=lambda kv: self._decayed_importance(kv[1]))
            for key, _ in scored[:n]:
                self._lru.pop(key)
                self._conn.execute("DELETE FROM wm_items WHERE key=?", (key,))
                removed += 1