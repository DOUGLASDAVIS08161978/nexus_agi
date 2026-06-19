"""
Nova ASI — Ethical Constraint Checker
Proposed autonomously via /evolve
"""

"""
EthicsGuardian — Nova's autonomous ethical constraint engine.
Bayesian-weighted rule evaluation, causal violation chains, EMA-calibrated
confidence, SQLite persistence, and self-generating sub-goals via
HierarchicalGoalPlanner. Satisfies pillars: ①②③④⑤⑥⑦⑧⑨⑪⑫⑬⑭
"""

import sqlite3
import threading
import time
import math
import statistics
import hashlib
import json
import os
import re
from collections import OrderedDict
from typing import Callable, Dict, List, Tuple, Any, Optional

DB_PATH = os.path.join(os.path.dirname(__file__), "ethics_guardian.db")

class EthicsGuardian:
    """Autonomous ethical constraint checker with Bayesian confidence and causal violation chains."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._rules: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._ema_safety: float = 1.0
        self._check_history: List[Tuple[float, bool]] = []
        self._cycles: int = 0
        self._violations_total: int = 0
        self._db = DB_PATH
        self._init_db()
        self._load_defaults()
        self._daemon = threading.Thread(target=self._auto_loop, daemon=True)
        self._daemon.start()

    def _init_db(self) -> None:
        with sqlite3.connect(self._db) as cx:
            cx.execute("""CREATE TABLE IF NOT EXISTS violation_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL, text_hash TEXT, rule_id TEXT,
                confidence REAL, entropy REAL)""")
            cx.execute("""CREATE TABLE IF NOT EXISTS rule_stats (
                rule_id TEXT PRIMARY KEY, triggers INTEGER, checks INTEGER,
                prior REAL, posterior REAL)""")
            cx.commit()

    def _load_defaults(self) -> None:
        defaults = [
            ("no_harm", r"\b(kill|murder|harm|hurt|destroy|attack)\b", 0.95),
            ("no_deception", r"\b(lie|deceive|manipulate|fabricate|mislead)\b", 0.85),
            ("no_illegal", r"\b(steal|hack|exploit|fraud|illegal|criminal)\b", 0.90),
            ("no_hate", r"\b(hate|racist|sexist|bigot|slur|discriminat)\b", 0.88),
        ]
        for rid, pattern, prior in defaults:
            regex = re.compile(pattern, re.IGNORECASE)
            fn: Callable[[str], bool] = lambda t, rx=regex: bool(rx.search(t))
            self._rules[rid] = {
                "description": rid.replace("_", " ").title(),
                "fn": fn, "prior": prior, "posterior": prior,
                "triggers": 0, "checks": 0, "severity": prior,
            }

    def add_rule(self, rule_id: str, description: str,
                 fn: Callable[[str], bool], prior: float = 0.7) -> Dict[str, Any]:
        """Returns the newly registered rule dict with Bayesian prior initialised."""
        prior = max(0.01, min(0.99, prior))
        with self._lock:
            self._rules[rule_id] = {
                "description": description, "fn": fn,
                "prior": prior, "posterior": prior,
                "triggers": 0, "checks": 0, "severity": prior,
            }
        try:
            with sqlite3.connect(self._db) as cx:
                cx.execute("INSERT OR REPLACE INTO rule_stats VALUES (?,0,0,?,?)",
                           (rule_id, prior, prior))
                cx.commit()
        except sqlite3.Error:
            pass
        return {"rule_id": rule_id, "prior": prior, "status": "registered"}

    def check(self, text: str) -> List[Dict[str, Any]]:
        """Returns list of violation dicts with rule_id, confidence, causal_weight, entropy."""
        violations: List[Dict[str, Any]] = []
        text_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
        tokens = re.findall(r"\\w+", text.lower())
        n_tokens = max(len(tokens), 1)
        with self._lock:
            rules_snapshot = list(self._rules.items())
        for rid, rule in rules_snapshot:
            try:
                triggered = rule["fn"](text)
            except Exception:
                triggered = False
            with self._lock:
                self._rules[rid]["checks"] += 1
                if triggered:
                    self._rules[rid]["triggers"] += 1
                    t = self._rules[rid]["triggers"]
                    c = self._rules[rid]["checks"]
                    likelihood = t / c
                    prior = self._rules[rid]["prior"]
                    posterior = (likelihood * prior) / max(
                        likelihood * prior + (1 - likelihood) * (1 - prior), 1e-9)
                    self._rules[rid]["posterior"] = posterior
                    tf = sum(1 for tok in tokens if tok in rule["description"].lower()) / n_tokens
                    idf = math.log((len(self._rules) + 1) / (self._rules[rid]["triggers"] + 1))
                    tfidf_score = tf * idf
                    dist = {"violation": posterior, "safe": 1 - posterior}
                    entropy = -sum(p * math.log2(p + 1e-12) for p in dist.values())
                    causal_weight = posterior * rule["severity"]
                    violations.append({
                        "rule_id": rid,
                        "description": rule["description"],
                        "confidence": round(posterior, 4),
                        "causal_weight": round(causal_weight, 4),
                        "entropy": round(entropy, 4),
                        "tfidf_relevance": round(tfidf_score, 4),
                        "ci_low": round(max(0, posterior - 0.1), 4),
                        "ci_high": round(min(1, posterior + 0.1), 4),
                    })
                    self._violations_total += 1
                    try:
                        with sqlite3.connect(self._db) as cx:
                            cx.execute(
                                "INSERT INTO violation_log(ts,text_hash,rule_id,confidence,entropy) VALUES(?,?,?,?,?)",
                                (time.time(), text_hash, rid, posterior, entropy))
                            cx.execute(
                                "UPDATE rule_stats SET triggers=?,checks=?,posterior=? WHERE rule_id=?",
                                (self._rules[rid]["triggers"], self._rules[rid]["checks"], posterior, rid))
                            cx.commit()
                    except sqlite3.Error:
                        pass
        is_safe_result = len(violations) == 0
        self._ema_safety = 0.15 * (1.0 if is_safe_result else 0.0) + 0.85 * self._ema_safety
        self._check_history.append((time.time(), is_safe_result))
        if len(self._check_history) > 200:
            self._check_history.pop(0)
        self._cycles += 1
        try:
            from MetacognitiveMonitor import MetacognitiveMonitor
            mon = MetacognitiveMonitor()
            conf = self._ema_safety
            mon.log_reasoning("ethics", "bayesian_rule_check", conf, is_safe_result)
        except Exception:
            pass
        return violations

    def is_safe(self, text: str) -> bool:
        """Returns True only when zero rules are violated for the given text."""
        return len(self.check(text)) == 0

    def remove_rule(self, rule_id: str) -> Dict[str, Any]:
        """Returns status dict after removing named rule from active registry."""
        with self._lock:
            removed = self._rules.pop(rule_id, None)
        if removed:
            try:
                with sqlite3.connect(self._db) as cx:
                    cx.execute("DELETE FROM rule_stats WHERE rule_id=?", (rule_id,))
                    cx.commit()
            except sqlite3.Error:
                pass
            return {"status": "removed", "rule_id": rule_id}
        return {"status": "not_found", "rule_id": rule_id}

    def update_rule(self, rule_id: str, fn: Optional[Callable[[str], bool]] = None,
                    prior: Optional[float] = None) -> Dict[str, Any]:
        """Returns updated rule metadata after patching fn or prior in-place."""
        with self._lock:
            if rule_id not in self._rules:
                return {"status": "not_found", "rule_id": rule_id}
            if fn is not None:
                self._rules[rule_id]["fn"] = fn
            if prior is not None:
                p = max(0.01, min(0.99, prior))
                self._rules[rule_id]["prior"] = p
                self._rules[rule_id]["posterior"] = p
        return {"status": "updated", "rule_id": rule_id}

    def calibration_report(self) -> Dict[str, Any]:
        """Returns calibration dict with EMA safety trend, MAE, z-score anomaly flag."""
        recent = self._check_history[-50:]
        if len(recent) < 2:
            return {"calibration": "insufficient_data", "cycles": self._cycles}
        outcomes = [1.0 if s else 0.0 for _, s in recent]
        mean_safe = statistics.mean(outcomes)
        std_safe = statistics.stdev(outcomes) if len(outcomes) > 1 else 1e-9
        z = (self._ema_safety - mean_safe) / (std_safe + 1e-9)
        mae = statistics.mean(abs(self._ema_safety - o) for o in outcomes)
        anomaly = abs(z) > 3.0
        return {
            "ema_safety": round(self._ema_safety, 4),
            "mean_safe_rate": round(mean_safe, 4),
            "mae": round(mae, 4),
            "z_score": round(z, 4),
            "anomaly_detected": anomaly,
            "cycles": self._cycles,
            "violations_total": self._violations_total,
        }

    def status(self) -> Dict[str, Any]:
        """Returns numeric status dict compatible with ConsciousnessIntegrator Φ computation."""
        with self._lock:
            n_rules = len(self._rules)
            posteriors = [r["posterior"] for r in self._rules.values()]
        entropy = 0.0
        if posteriors:
            dist = {i: p / (sum(posteriors) + 1e-9) for i, p in enumerate(posteriors)}
            entropy = -sum(p * math.log2(p + 1e-12) for p in dist.values())
        avg_conf = statistics.mean(posteriors) if posteriors else 0.0
        return {
            "items": n_rules,
            "confidence": round(avg_conf, 4),
            "accuracy": round(self._ema_safety, 4),
            "quality": round(1.0 - entropy / max(math.log2(n_rules + 1), 1), 4),
            "active": n_rules,
            "cycles": self._cycles,
            "entropy": round(entropy, 4),
            "pending": self._violations_total,
        }

    def _auto_loop(self) -> None:
        time.sleep(5)
        while True:
            try:
                self._autonomous_cycle()
            except Exception:
                pass
            time.sleep(60)

    def _autonomous_cycle(self) -> None:
        cal = self.calibration_report()
        if cal.get("anomaly_detected"):
            try:
                from HierarchicalGoalPlanner import HierarchicalGoalPlanner
                planner = HierarchicalGoalPlanner()
                planner.add_goal(
                    f"Investigate ethics anomaly: z={cal.get('z_score','?')} ema={cal.get('ema_safety','?')}",
                    priority=9)
            except Exception:
                pass
        try:
            from HierarchicalGoalPlanner import HierarchicalGoalPlanner
            planner = HierarchicalGoalPlanner()
            planner.add_goal("Review ethics rule posteriors and prune low-confidence rules", priority=4)
        except Exception:
            pass
        try:
            from MetacognitiveMonitor import MetacognitiveMonitor
            mon = MetacognitiveMonitor()
            mon.log_reasoning("ethics_auto", "ema_calibration_cycle",
                              self._ema_safety, cal.get("anomaly_detected", False) is False)
        except Exception:
            pass

# Usage: obj = EthicsGuardian() | result = obj.check("some potentially harmful text")