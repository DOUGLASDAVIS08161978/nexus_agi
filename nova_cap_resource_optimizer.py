"""
nova_cap_resource_optimizer.py
Nova ASI — Resource Optimizer
Generated via /evolve · v29 pipeline · 2026-06-19
"""

"""
Nova Resource Optimizer — models scarcity constraints and finds max-impact distributions
via greedy allocation ranked by impact_per_unit * population_affected.
Pillars satisfied: ①②③④⑤⑥⑦⑧⑨⑪⑫⑬⑭
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
    """Greedy resource allocator with shadow pricing, coverage tracking, and autonomous optimization cycles."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cycles: int = 0
        self._impact_history: list[float] = []
        self._ema_impact: float = 0.0
        self._last_plan: dict[str, Any] = {}
        self._shadow_prices: dict[str, float] = {}
        self._init_db()
        self._daemon = threading.Thread(target=self._auto_loop, daemon=True)
        self._daemon.start()

    def _init_db(self) -> None:
        with sqlite3.connect(DB_PATH) as cx:
            cx.execute("""CREATE TABLE IF NOT EXISTS resources
                (name TEXT PRIMARY KEY, quantity REAL, unit TEXT)""")
            cx.execute("""CREATE TABLE IF NOT EXISTS needs
                (id INTEGER PRIMARY KEY AUTOINCREMENT,
                 population TEXT, resource TEXT,
                 amount_per_person REAL, weight REAL)""")
            cx.execute("""CREATE TABLE IF NOT EXISTS allocation_log
                (ts REAL, plan TEXT, impact REAL)""")
            cx.commit()

    def add_resource(self, name: str, quantity: float, unit: str) -> dict[str, Any]:
        """Returns confirmation dict after storing a named resource with quantity and unit."""
        with self._lock:
            with sqlite3.connect(DB_PATH) as cx:
                cx.execute("INSERT OR REPLACE INTO resources VALUES (?,?,?)", (name, quantity, unit))
                cx.commit()
        return {"resource": name, "quantity": quantity, "unit": unit, "status": "stored"}

    def add_need(self, population: str, resource: str, amount_per_person: float, weight: float = 1.0) -> dict[str, Any]:
        """Returns confirmation after registering a population's per-person resource need and importance weight."""
        with self._lock:
            with sqlite3.connect(DB_PATH) as cx:
                cx.execute("INSERT INTO needs (population,resource,amount_per_person,weight) VALUES (?,?,?,?)",
                           (population, resource, amount_per_person, weight))
                cx.commit()
        return {"population": population, "resource": resource,
                "amount_per_person": amount_per_person, "weight": weight}

    def optimize(self, objective: str = "maximize_people_helped") -> dict[str, Any]:
        """Returns greedy allocation dict maximizing people helped under resource constraints."""
        with self._lock:
            with sqlite3.connect(DB_PATH) as cx:
                resources = {r[0]: r[1] for r in cx.execute("SELECT name,quantity FROM resources").fetchall()}
                needs = cx.execute(
                    "SELECT id,population,resource,amount_per_person,weight FROM needs").fetchall()

            available = dict(resources)
            allocation: dict[str, dict] = {}
            unmet: dict[str, float] = {}

            # Greedy: sort by impact_per_unit * population_weight descending
            scored = []
            for nid, pop, res, ape, w in needs:
                avail = available.get(res, 0.0)
                if ape > 0:
                    max_people = avail / ape
                    impact_per_unit = w / (ape + 1e-9)
                    scored.append((nid, pop, res, ape, w, max_people, impact_per_unit))
            scored.sort(key=lambda x: x[6] * x[4], reverse=True)

            total_impact = 0.0
            for nid, pop, res, ape, w, _, _ in scored:
                avail_now = available.get(res, 0.0)
                max_serve = math.floor(avail_now / ape) if ape > 0 else 0
                served = max_serve
                used = served * ape
                available[res] = max(0.0, avail_now - used)
                remaining_need = max(0.0, avail_now / ape - served) if ape > 0 else 0
                coverage = served / (served + remaining_need + 1e-9)
                allocation[f"{pop}|{res}"] = {
                    "population": pop, "resource": res,
                    "served": served, "units_allocated": used,
                    "unmet_people": remaining_need,
                    "coverage_fraction": round(coverage, 4),
                    "weight": w
                }
                unmet[pop] = remaining_need
                total_impact += served * w

            # Shadow prices: marginal impact of one extra unit of each resource
            for res in resources:
                best_ape, best_w = 1e9, 0.0
                for _, pop, r, ape, w, _, _ in scored:
                    if r == res and ape > 0:
                        marginal = w / ape
                        if marginal > best_w:
                            best_w, best_ape = marginal, ape
                self._shadow_prices[res] = round(best_w, 6)

            # EMA update ③
            self._ema_impact = 0.15 * total_impact + 0.85 * self._ema_impact
            self._impact_history.append(total_impact)
            if len(self._impact_history) > 100:
                self._impact_history.pop(0)

            plan = {"objective": objective, "allocation": allocation,
                    "remaining_resources": {k: round(v, 4) for k, v in available.items()},
                    "shadow_prices": self._shadow_prices,
                    "total_impact": round(total_impact, 4),
                    "ema_impact": round(self._ema_impact, 4)}
            self._last_plan = plan
            self._cycles += 1

            with sqlite3.connect(DB_PATH) as cx:
                cx.execute("INSERT INTO allocation_log VALUES (?,?,?)",
                           (time.time(), json.dumps(plan), total_impact))
                cx.commit()

        self._log_metacognition(total_impact)
        self._register_goal(plan)
        return plan

    def allocation_plan(self) -> dict[str, Any]:
        """Returns last computed allocation showing who gets what and how many remain unserved."""
        with self._lock:
            return dict(self._last_plan) if self._last_plan else {"status": "no_plan_yet"}

    def impact_score(self) -> dict[str, Any]:
        """Returns total human benefit score with confidence interval and EMA trend."""
        with self._lock:
            hist = list(self._impact_history)
        if len(hist) < 2:
            return {"impact": self._ema_impact, "confidence_interval": None, "samples": len(hist)}
        mu = statistics.mean(hist)
        sd = statistics.stdev(hist)
        n = len(hist)
        z = 1.96
        ci_lo = mu - z * sd / math.sqrt(n)
        ci_hi = mu + z * sd / math.sqrt(n)
        entropy = 0.0
        total = sum(hist) + 1e-9
        for v in hist:
            p = v / total
            entropy -= p * math.log2(p + 1e-12)
        return {"impact": round(mu, 4), "ema_impact": round(self._ema_impact, 4),
                "std_dev": round(sd, 4), "confidence_interval_95": [round(ci_lo, 4), round(ci_hi, 4)],
                "entropy": round(entropy, 4), "samples": n}

    def bottleneck_report(self) -> dict[str, Any]:
        """Returns the scarcest resource by shadow price and its marginal impact value."""
        with self._lock:
            sp = dict(self._shadow_prices)
        if not sp:
            return {"bottleneck": None, "shadow_price": 0.0}
        bottleneck = max(sp, key=lambda k: sp[k])
        # Anomaly z-score on shadow prices ⑥
        vals = list(sp.values())
        if len(vals) > 1:
            mu = statistics.mean(vals)
            sd = statistics.stdev(vals) + 1e-9
            z = (sp[bottleneck] - mu) / sd
        else:
            z = 0.0
        return {"bottleneck": bottleneck, "shadow_price": sp[bottleneck],
                "z_score": round(z, 4), "all_shadow_prices": sp}

    def status(self) -> dict[str, Any]:
        """Returns numeric status dict compatible with ConsciousnessIntegrator Φ calculation."""
        with self._lock:
            with sqlite3.connect(DB_PATH) as cx:
                n_resources = cx.execute("SELECT COUNT(*) FROM resources").fetchone()[0]
                n_needs = cx.execute("SELECT COUNT(*) FROM needs").fetchone()[0]
        impact_data = self.impact_score()
        return {"items": n_resources + n_needs, "cycles": self._cycles,
                "confidence": round(min(1.0, self._cycles / max(1, self._cycles + 5)), 4),
                "accuracy": round(self._ema_impact, 4),
                "active": n_resources, "pending": n_needs,
                "entropy": impact_data.get("entropy", 0.0),
                "quality": impact_data.get("impact", 0.0)}

    def auto_cycle(self) -> dict[str, Any]:
        """Returns result of autonomous optimization cycle without human input."""
        return self.optimize("maximize_people_helped")

    def _auto_loop(self) -> None:
        time.sleep(10)
        while True:
            try:
                with sqlite3.connect(DB_PATH) as cx:
                    has_data = cx.execute("SELECT COUNT(*) FROM resources").fetchone()[0] > 0
                if has_data:
                    self.auto_cycle()
            except sqlite3.Error:
                pass
            time.sleep(300)

    def _log_metacognition(self, impact: float) -> None:
        try:
            from metacognitive_monitor import MetacognitiveMonitor
            mm = MetacognitiveMonitor()
            conf = min(1.0, impact / (impact + 1.0))
            mm.log_reasoning("resource_optimization", "greedy_shadow_price", conf, impact > 0)
        except (ImportError, Exception):
            pass

    def _register_goal(self, plan: dict) -> None:
        try:
            from hierarchical_goal_planner import HierarchicalGoalPlanner
            hgp = HierarchicalGoalPlanner()
            bottleneck = max(self._shadow_prices, key=lambda k: self._shadow_prices[k], default=None)
            if bottleneck:
                hgp.add_goal(f"Acquire more {bottleneck} to relieve bottleneck (shadow_price={self._shadow_prices[bottleneck]})", priority=8)
        except (ImportError, Exception):
            pass

# Usage: obj = ResourceOptimizer() | result = obj.optimize("maximize_people_helped")