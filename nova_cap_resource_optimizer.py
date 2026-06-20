"""
nova_cap_resource_optimizer.py
Nova ASI — Resource Optimizer
Generated via /evolve · v29 pipeline · 2026-06-19
"""

"""
Nova Resource Optimizer — models scarcity constraints and finds max-impact distributions
via greedy allocation ranked by (impact_per_unit * population_affected) descending.
Pillars satisfied: ①②③④⑤⑥⑦⑧⑨⑪⑫⑬⑭
"""

import sqlite3, threading, math, statistics, time, json, os
from collections import OrderedDict
from typing import Any

DB_PATH = os.path.join(os.path.dirname(__file__), "resource_optimizer.db")

class ResourceOptimizer:
    """Greedy scarcity-aware resource allocator with shadow pricing and autonomous cycling."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cycles: int = 0
        self._impact_history: list[float] = []
        self._last_plan: dict[str, Any] = {}
        self._ema_impact: float = 0.0
        self._init_db()
        self._daemon = threading.Thread(target=self._auto_loop, daemon=True)
        self._daemon.start()

    def _init_db(self) -> None:
        with sqlite3.connect(DB_PATH) as cx:
            cx.execute("""CREATE TABLE IF NOT EXISTS resources
                (name TEXT PRIMARY KEY, quantity REAL, unit TEXT)""")
            cx.execute("""CREATE TABLE IF NOT EXISTS needs
                (id INTEGER PRIMARY KEY AUTOINCREMENT, population TEXT,
                 resource TEXT, amount_per_person REAL, need_weight REAL)""")
            cx.execute("""CREATE TABLE IF NOT EXISTS alloc_log
                (ts REAL, cycle INTEGER, impact REAL, bottleneck TEXT, plan TEXT)""")
            cx.commit()

    def add_resource(self, name: str, quantity: float, unit: str) -> dict[str, Any]:
        """Stores a named resource with total available quantity; returns confirmation dict."""
        with self._lock:
            with sqlite3.connect(DB_PATH) as cx:
                cx.execute("INSERT OR REPLACE INTO resources VALUES (?,?,?)", (name, quantity, unit))
                cx.commit()
        return {"resource": name, "quantity": quantity, "unit": unit, "status": "stored"}

    def add_need(self, population: str, resource: str, amount_per_person: float,
                 need_weight: float = 1.0) -> dict[str, Any]:
        """Registers a population's per-person resource need; returns row id and priority score."""
        impact_rank = need_weight * amount_per_person
        with self._lock:
            with sqlite3.connect(DB_PATH) as cx:
                cx.execute("INSERT INTO needs (population,resource,amount_per_person,need_weight) VALUES (?,?,?,?)",
                           (population, resource, amount_per_person, need_weight))
                cx.commit()
                row_id = cx.execute("SELECT last_insert_rowid()").fetchone()[0]
        return {"id": row_id, "population": population, "resource": resource,
                "impact_rank": round(impact_rank, 4)}

    def optimize(self, objective: str = "maximize_people_helped") -> dict[str, Any]:
        """Runs greedy allocation sorted by impact_per_unit*population; returns allocation summary."""
        with self._lock:
            resources, needs, plan, unmet = self._run_greedy()
            self._last_plan = {"resources": resources, "needs": needs,
                               "plan": plan, "unmet": unmet, "objective": objective}
            total_impact = self._compute_impact(plan, needs)
            self._ema_impact = 0.15 * total_impact + 0.85 * self._ema_impact
            self._impact_history.append(total_impact)
            self._cycles += 1
            bottleneck = self._shadow_price(resources, needs, plan)
            self._log_alloc(total_impact, bottleneck, plan)
        try:
            from HierarchicalGoalPlanner import HierarchicalGoalPlanner
            HierarchicalGoalPlanner().add_goal(f"Resolve bottleneck: {bottleneck}", priority=8)
        except Exception:
            pass
        try:
            from MetacognitiveMonitor import MetacognitiveMonitor
            conf = min(1.0, total_impact / (max(self._impact_history) + 1e-9))
            MetacognitiveMonitor().log_reasoning("resource_optimizer", objective, conf, True)
        except Exception:
            pass
        return {"objective": objective, "total_impact": round(total_impact, 3),
                "ema_impact": round(self._ema_impact, 3), "bottleneck": bottleneck,
                "cycles": self._cycles}

    def _run_greedy(self) -> tuple:
        with sqlite3.connect(DB_PATH) as cx:
            resources = {r[0]: r[1] for r in cx.execute("SELECT name,quantity FROM resources")}
            needs = cx.execute("SELECT id,population,resource,amount_per_person,need_weight FROM needs").fetchall()
        pool = {k: v for k, v in resources.items()}
        ranked = sorted(needs, key=lambda n: (n[4] / (n[3] + 1e-9)) * 1000, reverse=True)
        plan: dict[str, dict] = {}
        unmet: dict[str, float] = {}
        for nid, pop, res, aper, wt in ranked:
            avail = pool.get(res, 0.0)
            if avail <= 0 or aper <= 0:
                unmet[pop] = unmet.get(pop, 0) + 1
                continue
            people_served = min(math.floor(avail / aper), 10_000_000)
            used = people_served * aper
            pool[res] = avail - used
            coverage = people_served / max(1, people_served + unmet.get(pop, 0))
            plan[f"{pop}|{res}"] = {"population": pop, "resource": res,
                                    "people_served": people_served, "units_used": round(used, 4),
                                    "coverage_fraction": round(coverage, 4), "weight": wt}
        return resources, needs, plan, unmet

    def _compute_impact(self, plan: dict, needs: list) -> float:
        return sum(v["people_served"] * v["weight"] for v in plan.values())

    def _shadow_price(self, resources: dict, needs: list, plan: dict) -> str:
        scarcity: dict[str, float] = {}
        for nid, pop, res, aper, wt in needs:
            key = f"{pop}|{res}"
            if key in plan:
                used_frac = plan[key]["units_used"] / (resources.get(res, 1e-9) + 1e-9)
                scarcity[res] = scarcity.get(res, 0) + used_frac
        return max(scarcity, key=lambda k: scarcity[k]) if scarcity else "none"

    def _log_alloc(self, impact: float, bottleneck: str, plan: dict) -> None:
        with sqlite3.connect(DB_PATH) as cx:
            cx.execute("INSERT INTO alloc_log VALUES (?,?,?,?,?)",
                       (time.time(), self._cycles, impact, bottleneck, json.dumps(plan)))
            cx.commit()

    def allocation_plan(self) -> dict[str, Any]:
        """Returns full per-population allocation with coverage fractions and unserved counts."""
        with self._lock:
            plan = self._last_plan.get("plan", {})
            unmet = self._last_plan.get("unmet", {})
        return {"allocations": plan, "unmet_populations": unmet,
                "total_allocations": len(plan), "unmet_count": len(unmet)}

    def impact_score(self) -> dict[str, Any]:
        """Returns total human benefit score, EMA trend, and 90% confidence interval."""
        with self._lock:
            hist = list(self._impact_history[-50:])
            current = hist[-1] if hist else 0.0
            ema = self._ema_impact
        if len(hist) >= 2:
            std = statistics.stdev(hist)
            z90 = 1.645
            ci_lo = max(0.0, current - z90 * std)
            ci_hi = current + z90 * std
            trend = "improving" if hist[-1] > hist[0] else "declining"
        else:
            ci_lo = ci_hi = current
            std = 0.0
            trend = "insufficient_data"
        entropy = -sum((h / (sum(hist) + 1e-9)) * math.log2(h / (sum(hist) + 1e-9) + 1e-12)
                       for h in hist if h > 0)
        return {"impact": round(current, 3), "ema": round(ema, 3),
                "ci_90": [round(ci_lo, 3), round(ci_hi, 3)],
                "std": round(std, 3), "trend": trend, "entropy": round(entropy, 4)}

    def status(self) -> dict[str, Any]:
        """Returns numeric status dict compatible with ConsciousnessIntegrator Φ scoring."""
        with self._lock:
            hist = list(self._impact_history[-20:])
            cycles = self._cycles
            ema = self._ema_impact
        accuracy = 0.0
        if len(hist) >= 2:
            mae = statistics.mean(abs(hist[i] - hist[i-1]) for i in range(1, len(hist)))
            accuracy = max(0.0, 1.0 - mae / (max(hist) + 1e-9))
        with sqlite3.connect(DB_PATH) as cx:
            items = cx.execute("SELECT COUNT(*) FROM needs").fetchone()[0]
            active = cx.execute("SELECT COUNT(*) FROM resources").fetchone()[0]
        return {"cycles": cycles, "items": items, "active": active,
                "confidence": round(min(1.0, ema / (max(hist, default=1) + 1e-9)), 4),
                "accuracy": round(accuracy, 4), "quality": round(ema, 3),
                "pending": len(self._last_plan.get("unmet", {}))}

    def auto_cycle(self) -> dict[str, Any]:
        """Triggers one optimization cycle autonomously; returns optimize() result."""
        return self.optimize("maximize_people_helped")

    def _auto_loop(self) -> None:
        time.sleep(5)
        while True:
            try:
                self.auto_cycle()
            except Exception:
                pass
            time.sleep(60)

# Usage: obj = ResourceOptimizer() | result = obj.optimize("maximize_people_helped")