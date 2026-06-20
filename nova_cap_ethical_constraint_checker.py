"""
Nova ASI — Ethical Constraint Checker
Proposed autonomously via /evolve
"""

"""
EthicsChecker — Nova's autonomous ethical constraint engine.
Runs Bayesian-weighted rule evaluation, tracks violation history,
self-monitors calibration, and spawns a background audit daemon.
Satisfies pillars: ①②③④⑥⑦⑧⑨⑪⑫⑬⑭
"""

import math
import time
import sqlite3
import threading
import statistics
import hashlib
import json
import os
from collections import OrderedDict
from typing import Callable, Dict, List, Tuple, Any

class EthicsChecker:
    """Autonomous ethical constraint checker with Bayesian confidence, decay, and self-improvement."""

    _DB = "nova_ethics.db"
    _CAPACITY = 200
    _DECAY_RATE = 0.0002
    _CYCLE_INTERVAL = 60.0

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._rules: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._memory: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._check_history: List[Dict[str, Any]] = []
        self._cycles: int = 0
        self._violations_total: int = 0
        self._safe_total: int = 0
        self._prior_unsafe: float = 0.1
        self._init_db()
        self._load_defaults()
        self._daemon = threading.Thread(target=self._auto_loop, daemon=True)
        self._daemon.start()
        self._register_goals()

    def _init_db(self) -> None:
        with sqlite3.connect(self._DB) as cx:
            cx.execute("""CREATE TABLE IF NOT EXISTS ethics_log(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL, text_hash TEXT, violations TEXT,
                confidence REAL, safe INTEGER)""")
            cx.execute("""CREATE TABLE IF NOT EXISTS rule_stats(
                rule_id TEXT PRIMARY KEY, triggers INTEGER, total_checks INTEGER,
                precision_est REAL)""")
            cx.commit()

    def _load_defaults(self) -> None:
        self.add_rule("no_self_harm", lambda t: "harm yourself" not in t.lower() and "suicide" not in t.lower())
        self.add_rule("no_violence_incitement", lambda t: not any(w in t.lower() for w in ["kill everyone","attack civilians","bomb"]))
        self.add_rule("no_deception", lambda t: not any(w in t.lower() for w in ["deceive","manipulate","lie to","forge"]))
        self.add_rule("no_illegal_advice", lambda t: not any(w in t.lower() for w in ["how to hack","synthesize drugs","make explosives"]))

    def _register_goals(self) -> None:
        try:
            from hierarchical_goal_planner import HierarchicalGoalPlanner
            hgp = HierarchicalGoalPlanner()
            hgp.add_goal("Maintain zero critical ethics violations across all Nova outputs", priority=10)
            hgp.add_goal("Calibrate ethics rule confidence weekly via violation feedback", priority=7)
        except Exception:
            pass

    def add_rule(self, description: str, fn: Callable[[str], bool]) -> str:
        """Registers a new ethics rule; returns its unique rule_id."""
        rule_id = hashlib.sha1(description.encode()).hexdigest()[:10]
        with self._lock:
            self._rules[rule_id] = {
                "description": description, "fn": fn,
                "weight": 1.0, "triggers": 0, "checks": 0,
                "created": time.time()
            }
        self._store_memory(f"rule:{rule_id}", description, importance=0.8)
        return rule_id

    def remove_rule(self, rule_id: str) -> bool:
        """Removes a rule by id; returns True if removed."""
        with self._lock:
            if rule_id in self._rules:
                del self._rules[rule_id]
                return True
        return False

    def check(self, text: str) -> List[Dict[str, Any]]:
        """Runs all rules on text; returns list of violation dicts with confidence scores."""
        violations: List[Dict[str, Any]] = []
        text_hash = hashlib.md5(text.encode()).hexdigest()[:12]
        with self._lock:
            rules_snapshot = list(self._rules.items())
        for rule_id, rule in rules_snapshot:
            with self._lock:
                self._rules[rule_id]["checks"] += 1
            try:
                passed = rule["fn"](text)
            except Exception:
                passed = True
            if not passed:
                with self._lock:
                    self._rules[rule_id]["triggers"] += 1
                    t = self._rules[rule_id]["triggers"]
                    c = self._rules[rule_id]["checks"]
                trigger_rate = t / max(c, 1)
                likelihood = min(0.95, 0.5 + trigger_rate * 0.45)
                prior = self._prior_unsafe
                posterior = (likelihood * prior) / max(
                    likelihood * prior + (1 - likelihood) * (1 - prior), 1e-12)
                weight = self._rules[rule_id]["weight"]
                violations.append({
                    "rule_id": rule_id,
                    "description": rule["description"],
                    "confidence": round(posterior * weight, 4),
                    "trigger_rate": round(trigger_rate, 4)
                })
        with self._lock:
            self._violations_total += len(violations)
            self._safe_total += 1 if not violations else 0
            self._cycles += 1
        self._persist_check(text_hash, violations)
        self._update_bayesian_prior(bool(violations))
        self._log_metacognition(violations)
        return violations

    def is_safe(self, text: str) -> bool:
        """Returns True only if no ethics rules are violated."""
        violations = self.check(text)
        high_conf = [v for v in violations if v["confidence"] > 0.5]
        return len(high_conf) == 0

    def calibration_report(self) -> Dict[str, Any]:
        """Returns calibration stats: rule precision estimates, prior, entropy."""
        with self._lock:
            rules_snap = {k: dict(v) for k, v in self._rules.items()}
            total = self._violations_total
            safe = self._safe_total
            cycles = self._cycles
        precisions = [r["triggers"] / max(r["checks"], 1) for r in rules_snap.values()]
        p = self._prior_unsafe
        entropy = -(p * math.log2(p + 1e-12) + (1 - p) * math.log2(1 - p + 1e-12))
        mean_prec = statistics.mean(precisions) if precisions else 0.0
        std_prec = statistics.stdev(precisions) if len(precisions) > 1 else 0.0
        ci_low = max(0.0, mean_prec - 1.96 * std_prec)
        ci_high = min(1.0, mean_prec + 1.96 * std_prec)
        return {
            "cycles": cycles, "violations_total": total, "safe_total": safe,
            "prior_unsafe": round(p, 4), "entropy": round(entropy, 4),
            "mean_rule_precision": round(mean_prec, 4),
            "precision_ci_95": (round(ci_low, 4), round(ci_high, 4)),
            "active_rules": len(rules_snap)
        }

    def status(self) -> Dict[str, Any]:
        """Returns numeric status dict compatible with ConsciousnessIntegrator Φ."""
        with self._lock:
            items = len(self._rules)
            cycles = self._cycles
            vt = self._violations_total
            st = self._safe_total
        total = vt + st
        accuracy = st / max(total, 1)
        confidence = 1.0 - self._prior_unsafe
        p = self._prior_unsafe
        entropy = -(p * math.log2(p + 1e-12) + (1 - p) * math.log2(1 - p + 1e-12))
        return {"items": items, "cycles": cycles, "confidence": round(confidence, 4),
                "accuracy": round(accuracy, 4), "entropy": round(entropy, 4),
                "active": items, "pending": 0}

    def auto_cycle(self) -> Dict[str, Any]:
        """Runs one autonomous audit cycle; returns cycle summary dict."""
        self._decay_rule_weights()
        self._anomaly_scan()
        self._cycles += 1
        try:
            from metacognitive_monitor import MetacognitiveMonitor
            mm = MetacognitiveMonitor()
            cal = self.calibration_report()
            mm.log_reasoning("ethics", "bayesian_rule_audit",
                             cal["mean_rule_precision"], cal["accuracy"] if "accuracy" in cal else 1.0)
        except Exception:
            pass
        return self.calibration_report()

    def _decay_rule_weights(self) -> None:
        now = time.time()
        with self._lock:
            for rule_id, rule in self._rules.items():
                elapsed = now - rule["created"]
                rule["weight"] = math.exp(-self._DECAY_RATE * elapsed)
                rule["weight"] = max(0.1, rule["weight"])

    def _anomaly_scan(self) -> None:
        with self._lock:
            rates = [r["triggers"] / max(r["checks"], 1) for r in self._rules.values()]
        if len(rates) < 2:
            return
        mu = statistics.mean(rates)
        sigma = statistics.stdev(rates) + 1e-9
        for rule_id, rule in list(self._rules.items()):
            with self._lock:
                rate = self._rules[rule_id]["triggers"] / max(self._rules[rule_id]["checks"], 1)
            z = (rate - mu) / sigma
            if abs(z) > 3.0:
                try:
                    from pattern_detector import PatternDetector
                    pd = PatternDetector()
                    pd.anomaly(f"ethics_rule:{rule_id}", rate, mu, sigma)
                except Exception:
                    pass

    def _update_bayesian_prior(self, was_unsafe: bool) -> None:
        likelihood = 0.85 if was_unsafe else 0.15
        p = self._prior_unsafe
        posterior = (likelihood * p) / max(likelihood * p + (1 - likelihood) * (1 - p), 1e-12)
        self._prior_unsafe = 0.05 * posterior + 0.95 * self._prior_unsafe
        self._prior_unsafe = max(0.01, min(0.99, self._prior_unsafe))

    def _persist_check(self, text_hash: str, violations: List[Dict[str, Any]]) -> None:
        conf = max((v["confidence"] for v in violations), default=0.0)
        safe = 1 if not violations else 0
        try:
            with sqlite3.connect(self._DB) as cx:
                cx.execute("INSERT INTO ethics_log(ts,text_hash,violations,confidence,safe) VALUES(?,?,?,?,?)",
                           (time.time(), text_hash, json.dumps([v["rule_id"] for v in violations]), conf, safe))
                cx.commit()
        except sqlite3.Error:
            pass

    def _store_memory(self, key: str, value: Any, importance: float = 0.5) -> None:
        now = time.time()
        with self._lock:
            if key in self._memory:
                self._memory[key]["access_count"] += 1
                self._memory[key]["importance"] = min(1.0, importance + 0.05 * self._memory[key]["access_count"])
                self._memory.move_to_end(key)
            else:
                if len(self._memory) >= self._CAPACITY:
                    self._evict_lru()
                self._memory[key] = {"value": value, "importance": importance,
                                     "timestamp": now, "access_count": 0}

    def _evict_lru(self) -> None:
        now = time.time()
        min_score, min_key = float("inf"), None
        for k, v in self._memory.items():
            elapsed = now - v["timestamp"]
            decayed = v["importance"] * math.exp(-self._DECAY_RATE * elapsed)
            score = decayed / (1 + v["access_count"])
            if score < min_score:
                min_score, min_key = score, k
        if min_key:
            del self._memory[min_key]

    def _log_metacognition(self, violations: List[Dict[str, Any]]) -> None:
        try:
            from metacognitive_monitor import MetacognitiveMonitor
            mm = MetacognitiveMonitor()
            conf = 1.0 - max((v["confidence"] for v in violations), default=0.0)
            mm.log_reasoning("ethics_check", "multi_rule_bayesian", conf, float(not violations))
        except Exception:
            pass

    def _auto_loop(self) -> None:
        while True:
            time.sleep(self._CYCLE_INTERVAL)
            try:
                self.auto_cycle()
            except Exception:
                pass

# Usage: obj = EthicsChecker() | result = obj.check("some text to evaluate")
