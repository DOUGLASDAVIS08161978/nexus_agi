"""
nova_cap_resource_optimizer.py
Nova ASI — Resource Optimizer
Generated via /evolve · v29 pipeline · 2026-06-20
"""

"""
Nova Resource Optimizer — greedy scarcity-constrained allocation engine.
Models resources, needs, and populations; maximizes human impact under hard constraints.
Satisfies pillars: ①②③④⑤⑥⑦⑧⑨⑪⑫⑬⑭
"""

import sqlite3, threading, math, statistics, time, json, os
from collections import OrderedDict
from typing import Any

DB_PATH = os.path.join(os.path.dirname(__file__), "resource_optimizer.db")

class ResourceOptimizer:
    """Greedy impact-maximizing resource allocator with shadow pricing and autonomous re-optimization."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cycles: int = 0
        self._impact_history: list[float] = []
        self._ema_impact: float = 0.0
        self._last_plan: dict[str, Any] = {}
        self._bottleneck: str = ""
        self._init_db()
        self._daemon = threading.Thread(target=self._auto_loop, daemon=True)
        self._daemon.start()

    # ── DB ──────────────────────────────────────────────────────────────────
    def _init_db(self) -> None:
        with sqlite3.connect(DB_PATH) as c:
            c.execute("""CREATE TABLE IF NOT EXISTS resources
                         (name TEXT PRIMARY KEY, quantity REAL, unit TEXT)""")
            c.execute("""CREATE TABLE IF NOT EXISTS needs
                         (id INTEGER PRIMARY KEY AUTOINCREMENT,
                          population TEXT, resource TEXT,
                          amount_per_person REAL, weight REAL)""")
            c.execute("""CREATE TABLE IF NOT EXISTS alloc_log
                         (ts REAL, cycles INTEGER, impact REAL, bottleneck TEXT)""")
            c.commit()

    def _load_resources(self) -> dict[str, dict]:
        with sqlite3.connect(DB_PATH) as c:
            rows = c.execute("SELECT name,quantity,unit FROM resources").fetchall()
        return {r[0]: {"quantity": r[1], "unit": r[2]} for r in rows}

    def _load_needs(self) -> list[dict]:
        with sqlite3.connect(DB_PATH) as c:
            rows = c.execute(
                "SELECT id,population,resource,amount_per_person,weight FROM needs"
            ).fetchall()
        return [{"id": r[0], "population": r[1], "resource": r[2],
                 "amount_per_person": r[3], "weight": r[4]} for r in rows]

    # ── PUBLIC METHODS ───────────────────────────────────────────────────────
    def add_resource(self, name: str, quantity: float, unit: str) -> dict[str, Any]:
        """Returns confirmation dict after persisting a named resource with quantity."""
        with self._lock:
            with sqlite3.connect(DB_PATH) as c:
                c.execute("INSERT OR REPLACE INTO resources VALUES (?,?,?)",
                          (name, max(0.0, quantity), unit))
                c.commit()
        return {"status": "added", "resource": name, "quantity": quantity, "unit": unit}

    def add_need(self, population: str, resource: str,
                 amount_per_person: float, weight: float = 1.0) -> dict[str, Any]:
        """Returns need record after registering a population's per-person resource requirement."""
        if amount_per_person <= 0:
            return {"error": "amount_per_person must be positive"}
        with self._lock:
            with sqlite3.connect(DB_PATH) as c:
                c.execute("INSERT INTO needs (population,resource,amount_per_person,weight) VALUES (?,?,?,?)",
                          (population, resource, amount_per_person, max(0.01, weight)))
                nid = c.lastrowid
                c.commit()
        return {"id": nid, "population": population, "resource": resource,
                "amount_per_person": amount_per_person, "weight": weight}

    def optimize(self, objective: str = "max_people_helped") -> dict[str, Any]:
        """Returns greedy allocation plan maximizing impact under resource constraints."""
        resources = self._load_resources()
        needs = self._load_needs()
        if not resources or not needs:
            return {"error": "insufficient data", "resources": len(resources), "needs": len(needs)}

        available = {k: v["quantity"] for k, v in resources.items()}
        # Greedy sort: impact_per_unit * population proxy (weight * 1/amount_per_person)
        scored = []
        for n in needs:
            if n["resource"] in available and n["amount_per_person"] > 0:
                max_people = math.floor(available[n["resource"]] / n["amount_per_person"])
                impact_per_unit = n["weight"] / n["amount_per_person"]
                scored.append({**n, "max_people": max_people,
                               "impact_per_unit": impact_per_unit})
        scored.sort(key=lambda x: x["impact_per_unit"] * x["max_people"], reverse=True)

        plan, shadow = {}, {}
        for s in scored:
            res = s["resource"]
            avail = available.get(res, 0.0)
            if avail <= 0:
                plan[f"{s['population']}:{res}"] = {"served": 0, "unmet": s["max_people"],
                                                      "coverage": 0.0}
                continue
            can_serve = min(s["max_people"], math.floor(avail / s["amount_per_person"]))
            used = can_serve * s["amount_per_person"]
            available[res] = max(0.0, avail - used)
            coverage = can_serve / max(1, s["max_people"])
            plan[f"{s['population']}:{res}"] = {
                "served": can_serve, "unmet": max(0, s["max_people"] - can_serve),
                "coverage": round(coverage, 4), "units_used": round(used, 4),
                "weight": s["weight"]
            }
            # Shadow price: marginal impact of one more unit
            shadow[res] = shadow.get(res, 0.0) + s["weight"] / s["amount_per_person"]

        bottleneck = min(shadow, key=lambda k: available.get(k, 0.0)) if shadow else ""
        with self._lock:
            self._last_plan = plan
            self._bottleneck = bottleneck
            self._cycles += 1

        try:
            from HierarchicalGoalPlanner import HierarchicalGoalPlanner
            HierarchicalGoalPlanner().add_goal(
                f"Resolve bottleneck resource: {bottleneck}", priority=8)
        except Exception:
            pass
        try:
            from MetacognitiveMonitor import MetacognitiveMonitor
            conf = min(1.0, sum(v["coverage"] for v in plan.values()) / max(1, len(plan)))
            MetacognitiveMonitor().log_reasoning("resource_optimizer", objective, conf, True)
        except Exception:
            pass

        return {"plan": plan, "shadow_prices": shadow, "bottleneck": bottleneck,
                "objective": objective, "cycles": self._cycles}

    def allocation_plan(self) -> dict[str, Any]:
        """Returns last computed allocation showing who gets what and unserved counts."""
        with self._lock:
            plan = dict(self._last_plan)
        total_served = sum(v.get("served", 0) for v in plan.values())
        total_unmet = sum(v.get("unmet", 0) for v in plan.values())
        return {"allocations": plan, "total_served": total_served,
                "total_unmet": total_unmet, "bottleneck": self._bottleneck}

    def impact_score(self) -> dict[str, Any]:
        """Returns total weighted human benefit score with EMA trend and confidence interval."""
        with self._lock:
            plan = dict(self._last_plan)
        if not plan:
            return {"impact": 0.0, "confidence": 0.0, "ci_low": 0.0, "ci_high": 0.0}
        raw = sum(v.get("served", 0) * v.get("weight", 1.0) for v in plan.values())
        self._ema_impact = 0.15 * raw + 0.85 * self._ema_impact
        with self._lock:
            self._impact_history.append(raw)
            hist = list(self._impact_history[-50:])
        std = statistics.stdev(hist) if len(hist) > 1 else 0.0
        z = 1.96
        ci_low, ci_high = raw - z * std, raw + z * std
        coverage_vals = [v.get("coverage", 0.0) for v in plan.values()]
        conf = statistics.mean(coverage_vals) if coverage_vals else 0.0
        # Anomaly z-score
        mean_h = statistics.mean(hist) if hist else raw
        std_h = statistics.stdev(hist) if len(hist) > 1 else 1e-9
        z_score = (raw - mean_h) / (std_h + 1e-9)
        anomaly = abs(z_score) > 3.0
        with sqlite3.connect(DB_PATH) as c:
            c.execute("INSERT INTO alloc_log VALUES (?,?,?,?)",
                      (time.time(), self._cycles, raw, self._bottleneck))
            c.commit()
        return {"impact": round(raw, 4), "ema_impact": round(self._ema_impact, 4),
                "ci_low": round(ci_low, 4), "ci_high": round(ci_high, 4),
                "confidence": round(conf, 4), "z_score": round(z_score, 4),
                "anomaly": anomaly}

    def shadow_prices(self) -> dict[str, float]:
        """Returns marginal impact per additional unit for each resource (scarcity signal)."""
        result = self.optimize()
        return result.get("shadow_prices", {})

    def coverage_report(self) -> dict[str, Any]:
        """Returns per-population coverage fraction and unmet need summary."""
        with self._lock:
            plan = dict(self._last_plan)
        report = {}
        for key, val in plan.items():
            pop, res = key.split(":", 1)
            total = val.get("served", 0) + val.get("unmet", 0)
            report[key] = {
                "population": pop, "resource": res,
                "coverage_fraction": val.get("coverage", 0.0),
                "served": val.get("served", 0),
                "unmet": val.get("unmet", 0),
                "total_need": total
            }
        return report

    def status(self) -> dict[str, Any]:
        """Returns numeric-keyed health dict compatible with ConsciousnessIntegrator Φ."""
        resources = self._load_resources()
        needs = self._load_needs()
        with self._lock:
            plan = dict(self._last_plan)
            cycles = self._cycles
            ema = self._ema_impact
        coverage_vals = [v.get("coverage", 0.0) for v in plan.values()]
        avg_cov = statistics.mean(coverage_vals) if coverage_vals else 0.0
        hist = self._impact_history[-20:]
        entropy_val = 0.0
        if hist:
            total = sum(hist) + 1e-12
            dist = [h / total for h in hist]
            entropy_val = -sum(p * math.log2(p + 1e-12) for p in dist)
        return {"items": len(needs), "active": len(resources),
                "confidence": round(avg_cov, 4), "accuracy": round(ema, 4),
                "cycles": cycles, "entropy": round(entropy_val, 4),
                "pending": sum(v.get("unmet", 0) for v in plan.values()),
                "bottleneck": self._bottleneck}

    # ── AUTONOMOUS DAEMON ────────────────────────────────────────────────────
    def _auto_loop(self) -> None:
        while True:
            try:
                time.sleep(60)
                needs = self._load_needs()
                if needs:
                    self.optimize("max_people_helped")
                    self.impact_score()
            except Exception:
                pass

    def auto_cycle(self) -> dict[str, Any]:
        """Returns result of one forced optimization + impact scoring cycle."""
        opt = self.optimize("max_people_helped")
        imp = self.impact_score()
        return {"optimization": opt, "impact": imp, "cycles": self._cycles}

# Usage: obj = ResourceOptimizer() | result = obj.add_resource("water", 10000, "liters")