"""
AdaptiveMetaStrategist — Meta-level strategy selector with causal chain reasoning.
Implements Thompson sampling over strategy space and forward-chaining confidence
propagation. Satisfies pillars: ①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭.
"""
import math
import time
import random
import sqlite3
import threading
import statistics
import json
import os
from collections import OrderedDict
from typing import Any

class AdaptiveMetaStrategist:
    """Thompson-sampling meta-strategist with multiplicative causal chain inference."""

    def __init__(self) -> None:
        self._alpha: float = 0.15
        self._q_values: dict[str, float] = {}
        self._alpha_counts: dict[str, float] = {}
        self._beta_counts: dict[str, float] = {}
        self._domain_strategy_map: dict[str, dict[str, float]] = {}
        self._last_selected: dict[str, str] = {}
        self._facts: dict[str, float] = {}
        self._rules: list[tuple[str, str, float]] = []
        self._inference_cache: dict[str, tuple[float, list[str]]] = {}
        self._contradiction_log: list[tuple[str, str, float, float]] = []
        self._chain_weights: list[float] = []
        self._observation_count: int = 0
        self._lock = threading.Lock()
        self._db_path = os.path.join(os.path.expanduser("~"), "nova_meta_strategist.db")
        self._init_db()
        self._load_state()
        self._default_strategies = ["deductive", "inductive", "abductive", "analogical", "causal"]
        for s in self._default_strategies:
            if s not in self._q_values:
                self._q_values[s] = 0.5
                self._alpha_counts[s] = 1.0
                self._beta_counts[s] = 1.0
        self._cycle_count: int = 0
        self._daemon = threading.Thread(target=self._auto_loop, daemon=True)
        self._daemon.start()

    def _init_db(self) -> None:
        conn = sqlite3.connect(self._db_path)
        c = conn.cursor()
        c.execute("CREATE TABLE IF NOT EXISTS facts (claim TEXT PRIMARY KEY, confidence REAL)")
        c.execute("CREATE TABLE IF NOT EXISTS rules (premise TEXT, conclusion TEXT, weight REAL)")
        c.execute("CREATE TABLE IF NOT EXISTS q_values (strategy TEXT PRIMARY KEY, q REAL, alpha_c REAL, beta_c REAL)")
        conn.commit()
        conn.close()

    def _load_state(self) -> None:
        try:
            conn = sqlite3.connect(self._db_path)
            c = conn.cursor()
            for row in c.execute("SELECT claim, confidence FROM facts"):
                self._facts[row[0]] = row[1]
            for row in c.execute("SELECT premise, conclusion, weight FROM rules"):
                self._rules.append((row[0], row[1], row[2]))
            for row in c.execute("SELECT strategy, q, alpha_c, beta_c FROM q_values"):
                self._q_values[row[0]] = row[1]
                self._alpha_counts[row[0]] = row[2]
                self._beta_counts[row[0]] = row[3]
            conn.close()
        except sqlite3.Error:
            pass

    def _persist_q(self, strategy: str) -> None:
        try:
            conn = sqlite3.connect(self._db_path)
            c = conn.cursor()
            c.execute("INSERT OR REPLACE INTO q_values VALUES (?,?,?,?)",
                      (strategy, self._q_values.get(strategy, 0.5),
                       self._alpha_counts.get(strategy, 1.0),
                       self._beta_counts.get(strategy, 1.0)))
            conn.commit()
            conn.close()
        except sqlite3.Error:
            pass

    def _auto_loop(self) -> None:
        while True:
            time.sleep(60)
            try:
                self._autonomous_cycle()
            except Exception:
                pass

    def _autonomous_cycle(self) -> None:
        with self._lock:
            self._cycle_count += 1
            if self._facts:
                claims = list(self._facts.keys())
                sample = random.choice(claims)
                self.infer(sample)
            if self._q_values:
                domain = f"auto_domain_{self._cycle_count % 5}"
                self.select_strategy({"domain": domain, "complexity": random.random()})
            try:
                from HierarchicalGoalPlanner import HierarchicalGoalPlanner as HGP
                hgp = HGP()
                best = max(self._q_values, key=lambda s: self._q_values[s]) if self._q_values else "deductive"
                hgp.add_goal(f"Reinforce strategy '{best}' across high-complexity domains", priority=2)
            except Exception:
                pass
            try:
                from MetacognitiveMonitor import MetacognitiveMonitor as MCM
                mcm = MCM()
                avg_q = statistics.mean(self._q_values.values()) if self._q_values else 0.5
                mcm.log_reasoning("meta_strategy", "thompson_sampling", avg_q, avg_q > 0.5)
            except Exception:
                pass

    def select_strategy(self, problem: dict) -> str:
        """Returns the Thompson-sampled argmax strategy name for the problem's domain."""
        with self._lock:
            domain = problem.get("domain", "general")
            if domain not in self._domain_strategy_map:
                self._domain_strategy_map[domain] = {s: 0.5 for s in self._q_values}
            sampled: dict[str, float] = {}
            for s in self._q_values:
                a = self._alpha_counts.get(s, 1.0)
                b = self._beta_counts.get(s, 1.0)
                sampled[s] = random.betavariate(max(a, 1e-9), max(b, 1e-9))
            chosen = max(sampled, key=lambda s: sampled[s])
            self._last_selected[domain] = chosen
            return chosen

    def evaluate(self, result: dict) -> float:
        """Returns reward = accuracy * (1 / (1 + elapsed_seconds)) in (0, 1]."""
        accuracy = float(result.get("accuracy", 0.5))
        elapsed = float(result.get("elapsed_seconds", 1.0))
        reward = accuracy * (1.0 / (1.0 + elapsed))
        return max(0.0, min(1.0, reward))

    def update_policy(self, strategy: str, outcome: float) -> None:
        """Applies EMA Q-update and Beta count update; increments observation_count."""
        with self._lock:
            if strategy not in self._q_values:
                self._q_values[strategy] = 0.5
                self._alpha_counts[strategy] = 1.0
                self._beta_counts[strategy] = 1.0
            self._q_values[strategy] = (1 - self._alpha) * self._q_values[strategy] + self._alpha * outcome
            if outcome >= 0.5:
                self._alpha_counts[strategy] += outcome
            else:
                self._beta_counts[strategy] += (1.0 - outcome)
            self._observation_count += 1
            self._persist_q(strategy)

    def best_strategy(self, domain: str) -> str:
        """Returns argmax Q-value strategy for domain without sampling noise."""
        with self._lock:
            domain_map = self._domain_strategy_map.get(domain, self._q_values)
            if not domain_map:
                return "deductive"
            return max(domain_map, key=lambda s: self._q_values.get(s, 0.5))

    def add_fact(self, claim: str, confidence: float) -> None:
        """Stores claim with confidence, detects contradictions, invalidates inference cache."""
        with self._lock:
            confidence = max(0.0, min(1.0, confidence))
            negation = f"not_{claim}" if not claim.startswith("not_") else claim[4:]
            if negation in self._facts:
                c2 = self._facts[negation]
                if confidence + c2 > 1.05:
                    self._contradiction_log.append((claim, negation, confidence, c2))
            self._facts[claim] = confidence
            self._inference_cache.clear()
            try:
                conn = sqlite3.connect(self._db_path)
                conn.execute("INSERT OR REPLACE INTO facts VALUES (?,?)", (claim, confidence))
                conn.commit()
                conn.close()
            except sqlite3.Error:
                pass

    def infer(self, hypothesis: str) -> tuple[float, list[str]]:
        """Runs forward-chaining BFS; returns (chain_product_confidence, path) or (0.0, [])."""
        with self._lock:
            if hypothesis in self._inference_cache:
                return self._inference_cache[hypothesis]
            if hypothesis in self._facts:
                result = (self._facts[hypothesis], [hypothesis])
                self._inference_cache[hypothesis] = result
                return result
            visited: set[str] = set()
            queue: list[tuple[str, float, list[str]]] = []
            for fact, conf in self._facts.items():
                queue.append((fact, conf, [fact]))
            best_conf = 0.0
            best_path: list[str] = []
            while queue:
                current, conf, path = queue.pop(0)
                if current in visited:
                    continue
                visited.add(current)
                if current == hypothesis:
                    if conf > best_conf:
                        best_conf = conf
                        best_path = path
                    continue
                for premise, conclusion, weight in self._rules:
                    if premise == current and conclusion not in visited:
                        new_conf = conf * weight
                        if new_conf >= 0.05:
                            queue.append((conclusion, new_conf, path + [f"{premise}->{conclusion}({weight:.2f})", conclusion]))
            if best_conf > 0.0:
                self._chain_weights.append(best_conf)
                if len(self._chain_weights) > 200:
                    self._chain_weights = self._chain_weights[-200:]
            result = (round(best_conf, 6), best_path)
            self._inference_cache[hypothesis] = result
            return result

    def explain(self, conclusion: str) -> str:
        """Returns human-readable reasoning path string for the given conclusion."""
        conf, path = self.infer(conclusion)
        if not path:
            return f"No inference path found for '{conclusion}'"
        steps = " -> ".join(path)
        return f"{steps} => confidence={conf:.4f}"

    def chain_strength(self) -> float:
        """Returns mean of all computed chain confidences, or 0.0 if none exist."""
        with self._lock:
            if not self._chain_weights:
                return 0.0
            return round(statistics.mean(self._chain_weights), 6)

    def contradictions(self) -> list[tuple[str, str, float, float]]:
        """Returns full contradiction log as list of (claim_a, claim_b, conf_a, conf_b)."""
        with self._lock:
            return list(self._contradiction_log)

    def status(self) -> dict[str, Any]:
        """Returns numeric status dict compatible with ConsciousnessIntegrator Φ scoring."""
        with self._lock:
            avg_q = round(statistics.mean(self._q_values.values()), 4) if self._q_values else 0.5
            entropy_vals = [v for v in self._q_values.values() if v > 0]
            entropy = round(-sum(p * math.log2(p + 1e-12) for p in entropy_vals), 4) if entropy_vals else 0.0
            return {
                "items": len(self._facts),
                "confidence": avg_q,
                "accuracy": avg_q,
                "active": len(self._q_values),
                "pending": len(self._inference_cache),
                "cycles": self._cycle_count,
                "entropy": entropy,
                "observations": self._observation_count,
                "contradictions": len(self._contradiction_log),
                "chain_strength": self.chain_strength(),
                "strategies": list(self._q_values.keys()),
                "facts_loaded": len(self._facts),
                "rules_loaded": len(self._rules),
            }

# Usage: obj = AdaptiveMetaStrategist() | result = obj.select_strategy({"domain": "math", "complexity": 0.8})
