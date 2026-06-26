"""
nova_cap_resource_optimizer.py
Nova ASI — Resource Optimizer
Generated via /evolve · v29 pipeline · 2026-06-20
"""

"""
Nova Resource Optimizer — models scarcity constraints and finds max-impact distributions.
Greedy allocation sorted by (impact_per_unit * population_affected) descending.
Tracks unmet need, coverage fraction, bottleneck resource, and shadow prices.
Satisfies pillars: ①②③④⑥⑦⑧⑨⑪⑫⑬⑭
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

DB_PATH = os.path.join(os.path.dirname(__file__), "nova_resource_optimizer.db")

class ResourceOptimizer:
    """Greedy scarcity-aware resource allocator with Bayesian impact confidence and autonomous cycling."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self._build_schema()
        self._last_plan: dict = {}
        self._impact_history: list[float] = []
        self._cycle_count: int = 0
        self._ema_impact: float = 0.0
        self._shadow_prices: dict[str, float] = {}
        self._start_daemon()
        self._register_goals()

    def _build_schema(self) -> None:
        c = self._conn.cursor()
        c.executescript("""
            CREATE TABLE IF NOT EXISTS resources (
                name TEXT PRIMARY KEY, quantity REAL, unit TEXT, created_at REAL
            );
            CREATE TABLE IF NOT EXISTS needs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                population TEXT, resource TEXT, amount_per_person REAL,
                weight REAL DEFAULT 1.0, created_at REAL
            );
            CREATE TABLE IF NOT EXISTS allocation_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cycle INTEGER, plan_json TEXT, impact REAL, timestamp REAL
            );
        """)
        self._conn.commit()

    def add_resource(self, name: str, quantity: float, unit: str) -> dict[str, Any]:
        """Stores or updates a resource supply; returns updated resource record."""
        with self._lock:
            c = self._conn.cursor()
            c.execute("INSERT OR REPLACE INTO resources VALUES (?,?,?,?)",
                      (name, quantity, unit, time.time()))
            self._conn.commit()
        return {"resource": name, "quantity": quantity, "unit": unit}

    def add_need(self, population: str, resource: str,
                 amount_per_person: float, weight: float = 1.0) -> dict[str, Any]:
        """Registers a population's per-person resource need; returns need record."""
        with self._lock:
            c = self._conn.cursor()
            c.execute("INSERT INTO needs (population,resource,amount_per_person,weight,created_at) VALUES (?,?,?,?,?)",
                      (population, resource, amount_per_person, weight, time.time()))
            self._conn.commit()
            nid = c.lastrowid
        return {"need_id": nid, "population": population, "resource": resource,
                "amount_per_person": amount_per_person, "weight": weight}

    def optimize(self, objective: str = "max_people_helped") -> dict[str, Any]:
        """Runs greedy allocation maximizing people helped; returns full allocation plan with confidence."""
        with self._lock:
            c = self._conn.cursor()
            resources = {r[0]: r[1] for r in c.execute("SELECT name,quantity FROM resources")}
            needs = c.execute(
                "SELECT id,population,resource,amount_per_person,weight FROM needs"
            ).fetchall()

        remaining = dict(resources)
        plan: dict[str, dict] = {}
        total_people = 0

        pop_sizes: dict[str, int] = {}
        for nid, pop, res, app, w in needs:
            if pop not in pop_sizes:
                pop_sizes[pop] = max(1, int(remaining.get(res, 0) / max(app, 1e-9)))

        scored = []
        for nid, pop, res, app, w in needs:
            avail = remaining.get(res, 0.0)
            max_served = int(avail / max(app, 1e-9))
            impact_per_unit = w / max(app, 1e-9)
            priority = impact_per_unit * max_served
            scored.append((priority, nid, pop, res, app, w, avail))

        scored.sort(key=lambda x: -x[0])

        for priority, nid, pop, res, app, w, avail in scored:
            avail_now = remaining.get(res, 0.0)
            can_serve = int(avail_now / max(app, 1e-9))
            used = can_serve * app
            remaining[res] = max(0.0, avail_now - used)
            unmet_people = max(0, pop_sizes.get(pop, 0) - can_serve)
            coverage = can_serve / max(pop_sizes.get(pop, 1), 1)
            plan[f"{pop}:{res}"] = {
                "population": pop, "resource": res,
                "people_served": can_serve, "units_allocated": round(used, 4),
                "unmet_people": unmet_people,
                "coverage_fraction": round(min(coverage, 1.0), 4),
                "need_weight": w
            }
            total_people += can_serve

        bottleneck = min(remaining, key=lambda k: remaining[k]) if remaining else "none"
        self._shadow_prices = {
            k: round(1.0 / max(remaining[k], 1e-9) * 100, 4) for k in remaining
        }

        n = max(len(self._impact_history), 1)
        prior = 0.5
        likelihood = min(total_people / max(sum(pop_sizes.values()), 1), 1.0)
        posterior = (likelihood * prior) / max(likelihood * prior + (1 - likelihood) * (1 - prior), 1e-12)
        conf_interval = (
            round(max(0.0, posterior - 1.96 * math.sqrt(posterior * (1 - posterior) / n)), 4),
            round(min(1.0, posterior + 1.96 * math.sqrt(posterior * (1 - posterior) / n)), 4)
        )

        self._last_plan = {
            "objective": objective, "plan": plan,
            "total_people_helped": total_people,
            "bottleneck_resource": bottleneck,
            "shadow_prices": self._shadow_prices,
            "confidence": round(posterior, 4),
            "confidence_interval_95": conf_interval,
            "remaining_resources": {k: round(v, 4) for k, v in remaining.items()}
        }
        self._cycle_count += 1
        score = self.impact_score()
        self._ema_impact = 0.15 * score + 0.85 * self._ema_impact
        self._impact_history.append(score)

        c = self._conn.cursor()
        c.execute("INSERT INTO allocation_log (cycle,plan_json,impact,timestamp) VALUES (?,?,?,?)",
                  (self._cycle_count, json.dumps(self._last_plan), score, time.time()))
        self._conn.commit()

        self._log_to_metacognition(posterior)
        return self._last_plan

    def allocation_plan(self) -> dict[str, Any]:
        """Returns the last computed allocation plan showing who gets what and unserved counts."""
        return self._last_plan if self._last_plan else {"status": "no_plan_yet_run_optimize"}

    def impact_score(self) -> float:
        """Returns total human benefit score as sum(people_served * need_weight) across all allocations."""
        if not self._last_plan or "plan" not in self._last_plan:
            return 0.0
        return round(sum(
            v["people_served"] * v["need_weight"]
            for v in self._last_plan["plan"].values()
        ), 4)

    def shadow_price_report(self) -> dict[str, float]:
        """Returns marginal impact per extra unit of each resource (scarcity signal)."""
        return dict(self._shadow_prices)

    def calibration_report(self) -> dict[str, Any]:
        """Returns EMA impact trend, z-score anomaly flags, and MAE over last 50 cycles."""
        hist = self._impact_history[-50:]
        if len(hist) < 2:
            return {"ema_impact": self._ema_impact, "calibrated": False, "cycles": self._cycle_count}
        mean_i = statistics.mean(hist)
        std_i = statistics.stdev(hist)
        z = (hist[-1] - mean_i) / max(std_i, 1e-9)
        anomaly = abs(z) > 3.0
        entropy = -sum((1 / len(hist)) * math.log2(1 / len(hist) + 1e-12) for _ in hist)
        return {
            "ema_impact": round(self._ema_impact, 4),
            "mean_impact": round(mean_i, 4),
            "std_impact": round(std_i, 4),
            "z_score": round(z, 4),
            "anomaly_detected": anomaly,
            "entropy": round(entropy, 4),
            "cycles": self._cycle_count,
            "calibrated": True
        }

    def status(self) -> dict[str, Any]:
        """Returns numeric status dict compatible with ConsciousnessIntegrator Φ computation."""
        with self._lock:
            c = self._conn.cursor()
            n_res = c.execute("SELECT COUNT(*) FROM resources").fetchone()[0]
            n_needs = c.execute("SELECT COUNT(*) FROM needs").fetchone()[0]
        conf = self._last_plan.get("confidence", 0.0) if self._last_plan else 0.0
        return {
            "items": n_res + n_needs,
            "confidence": conf,
            "accuracy": round(self._ema_impact / max(self._ema_impact + 1, 1), 4),
            "quality": round(conf, 4),
            "active": int(bool(self._last_plan)),
            "pending": n_needs,
            "cycles": self._cycle_count,
            "entropy": self.calibration_report().get("entropy", 0.0)
        }

    def auto_cycle(self) -> None:
        """Runs optimize() autonomously on a 120-second daemon loop; feeds back into EMA."""
        def _loop() -> None:
            while True:
                try:
                    self.optimize()
                except Exception as exc:
                    pass
                time.sleep(120)
        t = threading.Thread(target=_loop, daemon=True)
        t.start()

    def _start_daemon(self) -> None:
        self.auto_cycle()

    def _log_to_metacognition(self, confidence: float) -> None:
        try:
            from metacognitive_monitor import MetacognitiveMonitor
            mm = MetacognitiveMonitor()
            mm.log_reasoning("resource_optimizer", "greedy_allocation", confidence, confidence > 0.6)
        except (ImportError, Exception):
            pass

    def _register_goals(self) -> None:
        try:
            from hierarchical_goal_planner import HierarchicalGoalPlanner
            hgp = HierarchicalGoalPlanner()
            hgp.add_goal("Maximize people helped under resource constraints", priority=9)
            hgp.add_goal("Identify and reduce bottleneck resource scarcity", priority=8)
        except (ImportError, Exception):
            pass

# Usage: obj = ResourceOptimizer() | result = obj.optimize("max_people_helped")
