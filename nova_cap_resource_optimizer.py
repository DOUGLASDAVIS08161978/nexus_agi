"""
nova_cap_resource_optimizer.py
Nova ASI — Resource Optimizer
Generated via /evolve · v29 pipeline · 2026-06-20
"""

"""
Nova Resource Optimizer — models scarcity constraints and finds max-impact distributions
via greedy allocation ranked by impact-per-unit * population_affected. Tracks coverage,
unmet need, bottleneck resources, and shadow prices. Feeds goals and reasoning into
Nova's live cognitive systems.
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
    """Greedy resource allocation optimizer with Bayesian confidence, shadow pricing, and autonomous cycles."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self._init_db()
        self._cycles: int = 0
        self._impact_history: list[float] = []
        self._last_plan: dict[str, Any] = {}
        self._confidence: float = 0.5
        self._start_daemon()

    def _init_db(self) -> None:
        c = self._conn.cursor()
        c.executescript("""
            CREATE TABLE IF NOT EXISTS resources (
                name TEXT PRIMARY KEY,
                quantity REAL,
                unit TEXT,
                updated REAL
            );
            CREATE TABLE IF NOT EXISTS needs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                population TEXT,
                resource TEXT,
                amount_per_person REAL,
                need_weight REAL DEFAULT 1.0,
                updated REAL
            );
            CREATE TABLE IF NOT EXISTS allocation_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL,
                objective TEXT,
                impact REAL,
                plan_json TEXT
            );
        """)
        self._conn.commit()

    def add_resource(self, name: str, quantity: float, unit: str) -> dict[str, Any]:
        """Returns confirmation dict after storing or updating a named resource."""
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO resources VALUES (?,?,?,?)",
                (name, max(0.0, quantity), unit, time.time())
            )
            self._conn.commit()
        return {"status": "ok", "resource": name, "quantity": quantity, "unit": unit}

    def add_need(self, population: str, resource: str, amount_per_person: float, need_weight: float = 1.0) -> dict[str, Any]:
        """Returns confirmation dict after registering a population's per-person resource need."""
        with self._lock:
            self._conn.execute(
                "INSERT INTO needs (population,resource,amount_per_person,need_weight,updated) VALUES (?,?,?,?,?)",
                (population, resource, max(1e-9, amount_per_person), max(0.01, need_weight), time.time())
            )
            self._conn.commit()
        return {"status": "ok", "population": population, "resource": resource, "amount_per_person": amount_per_person}

    def optimize(self, objective: str = "maximize_people_helped") -> dict[str, Any]:
        """Returns greedy allocation plan maximising people helped under resource constraints."""
        with self._lock:
            resources = {r[0]: r[1] for r in self._conn.execute("SELECT name,quantity FROM resources")}
            needs = list(self._conn.execute(
                "SELECT id,population,resource,amount_per_person,need_weight FROM needs ORDER BY updated"
            ))

        if not resources or not needs:
            return {"error": "insufficient data", "resources": len(resources), "needs": len(needs)}

        # Estimate population sizes via need weights as proxy; assume equal base pop of 1000
        base_pop = 1000
        scored = []
        for nid, pop, res, app, wt in needs:
            if res not in resources:
                continue
            impact_per_unit = wt / max(app, 1e-9)
            score = impact_per_unit * base_pop
            scored.append((score, nid, pop, res, app, wt, base_pop))

        scored.sort(key=lambda x: x[0], reverse=True)

        remaining = dict(resources)
        plan: list[dict] = []
        total_impact = 0.0

        for score, nid, pop, res, app, wt, pop_size in scored:
            avail = remaining.get(res, 0.0)
            if avail <= 0:
                plan.append({"population": pop, "resource": res, "served": 0,
                              "unserved": pop_size, "coverage": 0.0, "status": "unmet"})
                continue
            max_servable = math.floor(avail / app)
            served = min(pop_size, max_servable)
            used = served * app
            remaining[res] = avail - used
            coverage = served / max(pop_size, 1)
            impact = served * wt
            total_impact += impact
            plan.append({"population": pop, "resource": res, "served": served,
                         "unserved": pop_size - served, "coverage": round(coverage, 4),
                         "impact": round(impact, 4), "used": round(used, 4), "status": "allocated"})

        # Shadow price: marginal impact of +1 unit of most constrained resource
        shadow_prices: dict[str, float] = {}
        for res_name in resources:
            bottleneck_needs = [s for s in scored if s[3] == res_name and remaining.get(res_name, 0) < s[4]]
            if bottleneck_needs:
                top = bottleneck_needs[0]
                shadow_prices[res_name] = round(top[5] / max(top[4], 1e-9), 6)
            else:
                shadow_prices[res_name] = 0.0

        bottleneck = max(shadow_prices, key=lambda k: shadow_prices[k]) if shadow_prices else "none"

        # EMA confidence update
        prev_impact = self._impact_history[-1] if self._impact_history else total_impact
        delta = abs(total_impact - prev_impact) / max(abs(prev_impact), 1.0)
        self._confidence = max(0.1, min(0.99, 0.85 * self._confidence + 0.15 * (1.0 - delta)))
        self._impact_history.append(total_impact)
        if len(self._impact_history) > 100:
            self._impact_history = self._impact_history[-100:]

        result = {
            "objective": objective,
            "total_impact": round(total_impact, 4),
            "remaining_resources": {k: round(v, 4) for k, v in remaining.items()},
            "shadow_prices": shadow_prices,
            "bottleneck_resource": bottleneck,
            "confidence": round(self._confidence, 4),
            "plan": plan
        }
        self._last_plan = result

        with self._lock:
            self._conn.execute(
                "INSERT INTO allocation_log (ts,objective,impact,plan_json) VALUES (?,?,?,?)",
                (time.time(), objective, total_impact, json.dumps(result))
            )
            self._conn.commit()
            self._cycles += 1

        self._feed_cognitive_systems(result)
        return result

    def allocation_plan(self) -> dict[str, Any]:
        """Returns the most recent allocation plan showing who gets what and unserved counts."""
        if not self._last_plan:
            return self.optimize()
        return self._last_plan

    def impact_score(self) -> dict[str, Any]:
        """Returns total human benefit score with confidence interval derived from history."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT impact FROM allocation_log ORDER BY ts DESC LIMIT 50"
            ).fetchall()

        if not rows:
            return {"impact": 0.0, "confidence_interval": [0.0, 0.0], "cycles": 0}

        scores = [r[0] for r in rows]
        mean_impact = statistics.mean(scores)
        std_impact = statistics.stdev(scores) if len(scores) > 1 else 0.0
        z = 1.96
        ci_low = round(mean_impact - z * std_impact, 4)
        ci_high = round(mean_impact + z * std_impact, 4)

        # Anomaly detection on latest vs rolling
        latest = scores[0]
        rolling_mean = statistics.mean(scores[1:]) if len(scores) > 1 else mean_impact
        rolling_std = statistics.stdev(scores[1:]) if len(scores) > 2 else 1.0
        z_score = (latest - rolling_mean) / max(rolling_std, 1e-9)
        anomaly = abs(z_score) > 3.0

        return {
            "impact": round(mean_impact, 4),
            "latest": round(latest, 4),
            "std": round(std_impact, 4),
            "confidence_interval_95": [ci_low, ci_high],
            "z_score": round(z_score, 4),
            "anomaly_detected": anomaly,
            "cycles": len(scores)
        }

    def shadow_price_report(self) -> dict[str, Any]:
        """Returns shadow prices for all resources indicating marginal impact of +1 unit."""
        plan = self.allocation_plan()
        return {
            "shadow_prices": plan.get("shadow_prices", {}),
            "bottleneck": plan.get("bottleneck_resource", "none"),
            "confidence": plan.get("confidence", self._confidence)
        }

    def coverage_report(self) -> dict[str, Any]:
        """Returns per-population coverage fraction and unmet need counts from last plan."""
        plan = self.allocation_plan()
        entries = plan.get("plan", [])
        if not entries:
            return {"coverage": {}, "total_unserved": 0}
        coverage = {e["population"]: {"coverage": e.get("coverage", 0.0),
                                       "unserved": e.get("unserved", 0)} for e in entries}
        total_unserved = sum(e.get("unserved", 0) for e in entries)
        avg_coverage = statistics.mean(e.get("coverage", 0.0) for e in entries)
        return {"coverage": coverage, "total_unserved": total_unserved,
                "avg_coverage": round(avg_coverage, 4)}

    def status(self) -> dict[str, Any]:
        """Returns numeric status dict compatible with ConsciousnessIntegrator Φ computation."""
        with self._lock:
            n_resources = self._conn.execute("SELECT COUNT(*) FROM resources").fetchone()[0]
            n_needs = self._conn.execute("SELECT COUNT(*) FROM needs").fetchone()[0]

        impact_data = self.impact_score()
        entropy = 0.0
        if self._impact_history and len(self._impact_history) > 1:
            total = sum(self._impact_history[-20:]) + 1e-12
            probs = [v / total for v in self._impact_history[-20:]]
            entropy = -sum(p * math.log2(p + 1e-12) for p in probs)

        return {
            "items": n_needs,
            "active": n_resources,
            "cycles": self._cycles,
            "confidence": round(self._confidence, 4),
            "accuracy": round(impact_data.get("impact", 0.0), 4),
            "quality": round(1.0 - impact_data.get("std", 0.0) / max(impact_data.get("impact", 1.0), 1.0), 4),
            "entropy": round(entropy, 4),
            "pending": n_needs
        }

    def auto_cycle(self) -> dict[str, Any]:
        """Runs one autonomous optimization cycle; returns result and logs reasoning."""
        result = self.optimize("maximize_people_helped")
        self._feed_cognitive_systems(result)
        return result

    def _feed_cognitive_systems(self, result: dict[str, Any]) -> None:
        confidence = result.get("confidence", self._confidence)
        impact = result.get("total_impact", 0.0)
        bottleneck = result.get("bottleneck_resource", "none")
        try:
            from metacognitive_monitor import MetacognitiveMonitor
            mcm = MetacognitiveMonitor()
            mcm.log_reasoning("resource_optimization", "greedy_allocation", confidence, impact > 0)
        except (ImportError, Exception):
            pass
        try:
            from hierarchical_goal_planner import HierarchicalGoalPlanner
            hgp = HierarchicalGoalPlanner()
            if bottleneck and bottleneck != "none":
                hgp.add_goal(f"Acquire more {bottleneck} to relieve allocation bottleneck", priority=8)
            if impact > 0:
                hgp.add_goal(f"Improve coverage for unserved populations (impact={impact:.1f})", priority=6)
        except (ImportError, Exception):
            pass
        try:
            from working_memory import WorkingMemory
            wm = WorkingMemory()
            wm.store("resource_optimizer_last_impact", impact, importance=0.8)
            wm.store("resource_optimizer_bottleneck", bottleneck, importance=0.9)
        except (ImportError, Exception):
            pass
        try:
            from bayesian_belief_system import BayesianBeliefSystem
            bbs = BayesianBeliefSystem()
            bbs.add_causal_edge("resource_scarcity", "unmet_need", confidence)
        except (ImportError, Exception):
            pass

    def _start_daemon(self) -> None:
        def _loop() -> None:
            while True:
                time.sleep(300)
                try:
                    self.auto_cycle()
                except Exception:
                    pass
        t = threading.Thread(target=_loop, daemon=True)
        t.start()

# Usage: obj = ResourceOptimizer() | result = obj.optimize("maximize_people_helped")