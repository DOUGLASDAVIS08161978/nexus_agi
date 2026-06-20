"""
nova_cap_resource_optimizer.py
Nova ASI — Resource Optimizer
Generated via /evolve · v29 pipeline · 2026-06-20
"""

"""
Nova Resource Optimizer — models scarcity constraints and finds max-impact distributions
via greedy allocation ranked by impact_per_unit * population_affected. Tracks coverage,
unmet need, bottleneck resources, and shadow prices. Feeds goals and reasoning back into
Nova's cognitive architecture autonomously.
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
    """Greedy resource allocation optimizer with shadow pricing and autonomous goal generation."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cycles: int = 0
        self._impact_history: list[float] = []
        self._last_plan: dict[str, Any] = {}
        self._init_db()
        self._daemon = threading.Thread(target=self._auto_loop, daemon=True)
        self._daemon.start()

    def _init_db(self) -> None:
        with sqlite3.connect(DB_PATH) as cx:
            cx.execute("""CREATE TABLE IF NOT EXISTS resources
                (name TEXT PRIMARY KEY, quantity REAL, unit TEXT)""")
            cx.execute("""CREATE TABLE IF NOT EXISTS needs
                (population TEXT, resource TEXT, amount_per_person REAL,
                 need_weight REAL, PRIMARY KEY(population, resource))""")
            cx.execute("""CREATE TABLE IF NOT EXISTS alloc_log
                (ts REAL, cycle INT, impact REAL, bottleneck TEXT, plan TEXT)""")

    def add_resource(self, name: str, quantity: float, unit: str) -> dict[str, Any]:
        """Stores a named resource with quantity and unit; returns confirmation dict."""
        with self._lock:
            with sqlite3.connect(DB_PATH) as cx:
                cx.execute("INSERT OR REPLACE INTO resources VALUES (?,?,?)",
                           (name, max(0.0, quantity), unit))
        return {"status": "ok", "resource": name, "quantity": quantity, "unit": unit}

    def add_need(self, population: str, resource: str,
                 amount_per_person: float, need_weight: float = 1.0) -> dict[str, Any]:
        """Registers a population's per-person resource need; returns need record."""
        with self._lock:
            with sqlite3.connect(DB_PATH) as cx:
                cx.execute("INSERT OR REPLACE INTO needs VALUES (?,?,?,?)",
                           (population, resource, max(0.0, amount_per_person),
                            max(0.0, need_weight)))
        return {"population": population, "resource": resource,
                "amount_per_person": amount_per_person, "need_weight": need_weight}

    def optimize(self, objective: str = "maximize_people_helped") -> dict[str, Any]:
        """Runs greedy allocation; returns plan with coverage fractions and shadow prices."""
        with self._lock:
            resources, needs = self._load_state()
            available = {r["name"]: r["quantity"] for r in resources}
            remaining = dict(available)
            served: dict[str, dict[str, Any]] = {}
            shadow: dict[str, float] = {r: 0.0 for r in available}

            candidates = []
            for n in needs:
                pop, res, app, nw = n["population"], n["resource"], n["amount_per_person"], n["need_weight"]
                pop_size = self._infer_population_size(pop, needs)
                total_need = pop_size * app
                impact_per_unit = nw / (app + 1e-12)
                priority = impact_per_unit * pop_size
                candidates.append({"population": pop, "resource": res,
                                    "app": app, "nw": nw, "pop_size": pop_size,
                                    "total_need": total_need, "priority": priority})

            candidates.sort(key=lambda x: x["priority"], reverse=True)
            total_impact = 0.0

            for c in candidates:
                res = c["resource"]
                if res not in remaining:
                    continue
                avail_units = remaining[res]
                fully_servable = avail_units / (c["app"] + 1e-12)
                people_served = min(c["pop_size"], fully_servable)
                allocated = people_served * c["app"]
                remaining[res] = max(0.0, remaining[res] - allocated)
                coverage = people_served / (c["pop_size"] + 1e-12)
                unmet = max(0.0, c["pop_size"] - people_served)
                impact = people_served * c["nw"]
                total_impact += impact
                shadow[res] += c["nw"] / (c["app"] + 1e-12) if remaining[res] < 1e-6 else 0.0
                served[f"{c['population']}::{c['resource']}"] = {
                    "population": c["population"], "resource": res,
                    "people_served": round(people_served, 2),
                    "people_unserved": round(unmet, 2),
                    "allocated_units": round(allocated, 4),
                    "coverage_fraction": round(coverage, 4),
                    "impact": round(impact, 4)}

            bottleneck = min(shadow, key=lambda k: remaining.get(k, 0.0)) if shadow else "none"
            plan = {"objective": objective, "allocations": served,
                    "total_impact": round(total_impact, 4),
                    "remaining_resources": {k: round(v, 4) for k, v in remaining.items()},
                    "shadow_prices": {k: round(v, 6) for k, v in shadow.items()},
                    "bottleneck_resource": bottleneck, "cycles": self._cycles}
            self._last_plan = plan
            self._impact_history.append(total_impact)
            self._log_cycle(total_impact, bottleneck, plan)
            self._cycles += 1
            self._feed_metacognition(total_impact, bottleneck)
            return plan

    def allocation_plan(self) -> dict[str, Any]:
        """Returns the most recent allocation plan showing who gets what and unserved counts."""
        with self._lock:
            return self._last_plan if self._last_plan else {"status": "no_plan_yet"}

    def impact_score(self) -> dict[str, Any]:
        """Returns total human benefit score, EMA trend, z-score anomaly flag, and CI bounds."""
        with self._lock:
            hist = self._impact_history[-50:]
            if not hist:
                return {"impact": 0.0, "trend_ema": 0.0, "anomaly": False}
            ema = hist[0]
            for v in hist[1:]:
                ema = 0.15 * v + 0.85 * ema
            mean_i = statistics.mean(hist)
            std_i = statistics.stdev(hist) if len(hist) > 1 else 0.0
            latest = hist[-1]
            z = (latest - mean_i) / (std_i + 1e-9)
            ci_lo = mean_i - 1.96 * std_i
            ci_hi = mean_i + 1.96 * std_i
            entropy_val = 0.0
            if len(hist) > 1:
                total = sum(hist) + 1e-12
                dist = [v / total for v in hist]
                entropy_val = -sum(p * math.log2(p + 1e-12) for p in dist)
            return {"impact": round(latest, 4), "trend_ema": round(ema, 4),
                    "mean": round(mean_i, 4), "std": round(std_i, 4),
                    "z_score": round(z, 4), "anomaly": abs(z) > 3.0,
                    "ci_95": [round(ci_lo, 4), round(ci_hi, 4)],
                    "entropy": round(entropy_val, 4), "history_len": len(hist)}

    def shadow_analysis(self) -> dict[str, Any]:
        """Returns shadow prices from last plan indicating marginal impact of extra resource units."""
        with self._lock:
            if not self._last_plan:
                return {"status": "run_optimize_first"}
            return {"shadow_prices": self._last_plan.get("shadow_prices", {}),
                    "bottleneck": self._last_plan.get("bottleneck_resource", "unknown"),
                    "recommendation": f"Acquire more {self._last_plan.get('bottleneck_resource','?')} for max impact gain"}

    def coverage_report(self) -> dict[str, Any]:
        """Returns per-population coverage fractions and unmet need summary from last plan."""
        with self._lock:
            if not self._last_plan:
                return {"status": "run_optimize_first"}
            allocs = self._last_plan.get("allocations", {})
            total_served = sum(v["people_served"] for v in allocs.values())
            total_unserved = sum(v["people_unserved"] for v in allocs.values())
            coverages = [v["coverage_fraction"] for v in allocs.values()]
            avg_cov = statistics.mean(coverages) if coverages else 0.0
            return {"total_served": round(total_served, 2),
                    "total_unserved": round(total_unserved, 2),
                    "avg_coverage_fraction": round(avg_cov, 4),
                    "populations": {k: {"coverage": v["coverage_fraction"],
                                        "unserved": v["people_unserved"]}
                                    for k, v in allocs.items()}}

    def status(self) -> dict[str, Any]:
        """Returns numeric status dict compatible with ConsciousnessIntegrator Φ computation."""
        with self._lock:
            with sqlite3.connect(DB_PATH) as cx:
                n_res = cx.execute("SELECT COUNT(*) FROM resources").fetchone()[0]
                n_needs = cx.execute("SELECT COUNT(*) FROM needs").fetchone()[0]
            hist = self._impact_history[-10:]
            confidence = min(1.0, len(hist) / 10.0)
            accuracy = 1.0 - (1.0 / (self._cycles + 1))
            quality = statistics.mean(hist) if hist else 0.0
            return {"items": n_res + n_needs, "confidence": round(confidence, 4),
                    "accuracy": round(accuracy, 4), "quality": round(quality, 4),
                    "cycles": self._cycles, "active": n_res, "pending": n_needs,
                    "entropy": round(self._impact_history[-1] if self._impact_history else 0.0, 4)}

    def _load_state(self) -> tuple[list[dict], list[dict]]:
        with sqlite3.connect(DB_PATH) as cx:
            resources = [{"name": r[0], "quantity": r[1], "unit": r[2]}
                         for r in cx.execute("SELECT name,quantity,unit FROM resources")]
            needs = [{"population": r[0], "resource": r[1],
                      "amount_per_person": r[2], "need_weight": r[3]}
                     for r in cx.execute("SELECT population,resource,amount_per_person,need_weight FROM needs")]
        return resources, needs

    def _infer_population_size(self, pop: str, needs: list[dict]) -> float:
        digits = [int(t) for t in pop.split("_") if t.isdigit()]
        return float(digits[0]) if digits else max(1.0, float(len([n for n in needs if n["population"] == pop]) * 100))

    def _log_cycle(self, impact: float, bottleneck: str, plan: dict) -> None:
        try:
            with sqlite3.connect(DB_PATH) as cx:
                cx.execute("INSERT INTO alloc_log VALUES (?,?,?,?,?)",
                           (time.time(), self._cycles, impact, bottleneck,
                            json.dumps(plan)[:2000]))
        except sqlite3.Error:
            pass

    def _feed_metacognition(self, impact: float, bottleneck: str) -> None:
        try:
            from metacognitive_monitor import MetacognitiveMonitor
            mm = MetacognitiveMonitor()
            confidence = min(1.0, impact / (max(self._impact_history or [1.0]) + 1e-9))
            mm.log_reasoning("resource_optimization", "greedy_allocation",
                             confidence, impact > 0)
        except (ImportError, Exception):
            pass
        try:
            from hierarchical_goal_planner import HierarchicalGoalPlanner
            hgp = HierarchicalGoalPlanner()
            hgp.add_goal(f"Resolve bottleneck: acquire more {bottleneck}", priority=8)
        except (ImportError, Exception):
            pass

    def _auto_loop(self) -> None:
        time.sleep(15)
        while True:
            try:
                with self._lock:
                    with sqlite3.connect(DB_PATH) as cx:
                        n = cx.execute("SELECT COUNT(*) FROM resources").fetchone()[0]
                if n > 0:
                    self.optimize("maximize_people_helped")
            except Exception:
                pass
            time.sleep(300)

# Usage: obj = ResourceOptimizer() | result = obj.optimize("maximize_people_helped")