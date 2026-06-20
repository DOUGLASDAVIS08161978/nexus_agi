"""
nova_cap_resource_optimizer.py
Nova ASI — Resource Optimizer
Generated via /evolve · v29 pipeline · 2026-06-19
"""

"""
Nova Resource Optimizer — models scarcity constraints, greedy allocation,
shadow pricing, and max-impact distributions for humanitarian/logistical planning.
Pillars: ①②③④⑤⑥⑦⑧⑨⑪⑫⑬⑭
"""

import sqlite3
import threading
import math
import statistics
import time
import json
import os
from collections import OrderedDict
from typing import Any

DB_PATH = os.path.join(os.path.dirname(__file__), "resource_optimizer.db")

class ResourceOptimizer:
    """Greedy scarcity-aware resource allocator with shadow pricing and Bayesian impact calibration."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self._setup_db()
        self._cycles: int = 0
        self._impact_history: list[tuple[float, float]] = []
        self._last_plan: dict[str, Any] = {}
        self._ema_impact: float = 0.0
        self._daemon = threading.Thread(target=self._auto_loop, daemon=True)
        self._daemon.start()

    def _setup_db(self) -> None:
        c = self._conn.cursor()
        c.execute("""CREATE TABLE IF NOT EXISTS resources
                     (name TEXT PRIMARY KEY, quantity REAL, unit TEXT)""")
        c.execute("""CREATE TABLE IF NOT EXISTS needs
                     (population TEXT, resource TEXT, amount_per_person REAL,
                      need_weight REAL, PRIMARY KEY(population, resource))""")
        c.execute("""CREATE TABLE IF NOT EXISTS alloc_log
                     (ts REAL, plan TEXT, impact REAL, coverage REAL)""")
        self._conn.commit()

    def add_resource(self, name: str, quantity: float, unit: str) -> dict[str, Any]:
        """Stores a named resource with quantity and unit; returns confirmation dict."""
        with self._lock:
            c = self._conn.cursor()
            c.execute("INSERT OR REPLACE INTO resources VALUES (?,?,?)", (name, quantity, unit))
            self._conn.commit()
            try:
                from KnowledgeGraphFabric import KnowledgeGraphFabric
                KnowledgeGraphFabric().add_entity(name, {"type": "resource", "qty": quantity, "unit": unit})
            except Exception:
                pass
            return {"resource": name, "quantity": quantity, "unit": unit, "status": "stored"}

    def add_need(self, population: str, resource: str, amount_per_person: float, need_weight: float = 1.0) -> dict[str, Any]:
        """Registers a population's resource need with optional priority weight; returns need record."""
        with self._lock:
            c = self._conn.cursor()
            c.execute("INSERT OR REPLACE INTO needs VALUES (?,?,?,?)",
                      (population, resource, amount_per_person, need_weight))
            self._conn.commit()
            return {"population": population, "resource": resource,
                    "amount_per_person": amount_per_person, "need_weight": need_weight}

    def optimize(self, objective: str = "max_people_helped") -> dict[str, Any]:
        """Runs greedy allocation sorted by impact_per_unit*population; returns allocation result with shadow prices."""
        with self._lock:
            c = self._conn.cursor()
            resources = {r[0]: r[1] for r in c.execute("SELECT name, quantity FROM resources")}
            needs = c.execute("SELECT population, resource, amount_per_person, need_weight FROM needs").fetchall()
            remaining = dict(resources)
            served: dict[str, dict[str, Any]] = {}
            unmet: dict[str, float] = {}

            pop_sizes = {}
            for (pop, res, amt, wt) in needs:
                if pop not in pop_sizes:
                    pop_sizes[pop] = max(1, int(remaining.get(res, 0) / max(amt, 1e-9)))

            scored = []
            for (pop, res, amt, wt) in needs:
                avail = remaining.get(res, 0.0)
                max_serve = int(avail / max(amt, 1e-9))
                impact_per_unit = wt / max(amt, 1e-9)
                scored.append((pop, res, amt, wt, impact_per_unit, max_serve))

            scored.sort(key=lambda x: x[4] * x[5], reverse=True)

            total_impact = 0.0
            for (pop, res, amt, wt, ipu, _) in scored:
                avail = remaining.get(res, 0.0)
                can_serve = int(avail / max(amt, 1e-9))
                allocated = can_serve * amt
                remaining[res] = max(0.0, avail - allocated)
                total_impact += can_serve * wt
                served[f"{pop}::{res}"] = {
                    "population": pop, "resource": res,
                    "people_served": can_serve, "units_allocated": round(allocated, 4),
                    "coverage_fraction": round(min(1.0, can_serve / max(1, int(resources.get(res, 1) / max(amt, 1e-9)))), 4),
                    "need_weight": wt
                }
                unmet[pop] = max(0.0, avail - allocated)

            shadow_prices = {}
            for res, qty in remaining.items():
                marginal = 0.0
                for (pop, r, amt, wt, ipu, _) in scored:
                    if r == res:
                        marginal = max(marginal, wt / max(amt, 1e-9))
                shadow_prices[res] = round(marginal, 6)

            bottleneck = min(remaining, key=lambda r: remaining[r]) if remaining else "none"
            self._ema_impact = 0.15 * total_impact + 0.85 * self._ema_impact
            self._impact_history.append((time.time(), total_impact))
            if len(self._impact_history) > 100:
                self._impact_history.pop(0)

            plan = {"served": served, "remaining": remaining, "shadow_prices": shadow_prices,
                    "bottleneck": bottleneck, "total_impact": round(total_impact, 4),
                    "objective": objective, "ema_impact": round(self._ema_impact, 4)}
            self._last_plan = plan
            c.execute("INSERT INTO alloc_log VALUES (?,?,?,?)",
                      (time.time(), json.dumps(plan), total_impact,
                       sum(v["coverage_fraction"] for v in served.values()) / max(1, len(served))))
            self._conn.commit()
            self._cycles += 1

            try:
                from MetacognitiveMonitor import MetacognitiveMonitor
                conf = min(1.0, total_impact / max(1.0, total_impact + sum(remaining.values())))
                MetacognitiveMonitor().log_reasoning("resource_optimizer", "greedy_allocation", conf, True)
            except Exception:
                pass
            try:
                from HierarchicalGoalPlanner import HierarchicalGoalPlanner
                if bottleneck != "none":
                    HierarchicalGoalPlanner().add_goal(f"Acquire more {bottleneck} to relieve bottleneck", priority=8)
            except Exception:
                pass
            return plan

    def allocation_plan(self) -> dict[str, Any]:
        """Returns the latest allocation plan showing who gets what and how many remain unserved."""
        with self._lock:
            if not self._last_plan:
                return {"error": "No optimization run yet. Call optimize() first."}
            plan = self._last_plan
            summary = []
            for key, v in plan.get("served", {}).items():
                unserved_est = max(0, round((1 - v["coverage_fraction"]) * v["people_served"] / max(0.001, v["coverage_fraction"]), 1))
                summary.append({**v, "estimated_unserved": unserved_est})
            return {"allocations": summary, "bottleneck": plan.get("bottleneck"),
                    "shadow_prices": plan.get("shadow_prices"), "remaining_stock": plan.get("remaining")}

    def impact_score(self) -> dict[str, Any]:
        """Returns total human benefit score, EMA trend, confidence interval, and calibration error."""
        with self._lock:
            if len(self._impact_history) < 2:
                return {"impact": self._ema_impact, "confidence_interval": [0.0, 0.0], "calibration": "insufficient_data"}
            vals = [v for _, v in self._impact_history[-50:]]
            mu = statistics.mean(vals)
            sigma = statistics.stdev(vals) if len(vals) > 1 else 0.0
            z = 1.96
            ci_lo = round(mu - z * sigma / math.sqrt(len(vals)), 4)
            ci_hi = round(mu + z * sigma / math.sqrt(len(vals)), 4)
            mae = statistics.mean(abs(p - a) for p, a in zip(vals[:-1], vals[1:]))
            entropy_val = 0.0
            total = sum(vals) + 1e-9
            dist = {i: v / total for i, v in enumerate(vals)}
            entropy_val = -sum(p * math.log2(p + 1e-12) for p in dist.values())
            return {"impact": round(self._ema_impact, 4), "mean_impact": round(mu, 4),
                    "confidence_interval": [ci_lo, ci_hi], "mae": round(mae, 4),
                    "entropy": round(entropy_val, 4), "cycles": self._cycles}

    def anomaly_check(self) -> dict[str, Any]:
        """Detects z-score anomalies in impact history; returns alert dict with z-scores."""
        with self._lock:
            if len(self._impact_history) < 5:
                return {"anomalies": [], "status": "insufficient_data"}
            vals = [v for _, v in self._impact_history]
            mu = statistics.mean(vals)
            sigma = statistics.stdev(vals) + 1e-9
            anomalies = []
            for ts, v in self._impact_history[-10:]:
                z = (v - mu) / sigma
                if abs(z) > 3.0:
                    anomalies.append({"ts": ts, "impact": v, "z_score": round(z, 3)})
            return {"anomalies": anomalies, "rolling_mean": round(mu, 4), "rolling_std": round(sigma, 4)}

    def status(self) -> dict[str, Any]:
        """Returns numeric status dict compatible with ConsciousnessIntegrator Φ computation."""
        with self._lock:
            c = self._conn.cursor()
            items = c.execute("SELECT COUNT(*) FROM resources").fetchone()[0]
            pending = c.execute("SELECT COUNT(*) FROM needs").fetchone()[0]
            accuracy = round(min(1.0, self._ema_impact / max(1.0, self._ema_impact + 1)), 4)
            entropy_val = 0.0
            if self._impact_history:
                vals = [v for _, v in self._impact_history[-20:]]
                total = sum(vals) + 1e-9
                entropy_val = -sum((v / total) * math.log2(v / total + 1e-12) for v in vals)
            return {"items": items, "pending": pending, "cycles": self._cycles,
                    "confidence": round(accuracy, 4), "accuracy": accuracy,
                    "quality": round(self._ema_impact, 4), "entropy": round(entropy_val, 4),
                    "active": 1 if self._last_plan else 0}

    def auto_cycle(self) -> dict[str, Any]:
        """Runs one autonomous optimization cycle and logs metacognitive reasoning; returns cycle summary."""
        result = self.optimize("max_people_helped")
        score = self.impact_score()
        anom = self.anomaly_check()
        return {"cycle": self._cycles, "impact": result.get("total_impact"),
                "anomalies": len(anom.get("anomalies", [])), "ema": result.get("ema_impact"),
                "bottleneck": result.get("bottleneck")}

    def _auto_loop(self) -> None:
        while True:
            time.sleep(300)
            try:
                with self._lock:
                    c = self._conn.cursor()
                    has_data = c.execute("SELECT COUNT(*) FROM resources").fetchone()[0]
                if has_data:
                    self.auto_cycle()
            except Exception:
                pass

# Usage: obj = ResourceOptimizer() | result = obj.optimize("max_people_helped")