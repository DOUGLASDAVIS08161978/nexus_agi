"""
AdaptiveMetaStrategistCausalEngine — Nova's meta-level strategy selector fused with
causal chain reasoning. Thompson-sampled strategy selection, EMA Q-value updates,
multiplicative confidence propagation, contradiction detection, and autonomous cycling.
"""

import sqlite3
import threading
import math
import time
import statistics
import random
import os
import json
import hashlib
from collections import OrderedDict, defaultdict
from typing import Any

DB_PATH = os.path.join(os.path.dirname(__file__), "adaptive_meta_strategist.db")

class AdaptiveMetaStrategistCausalEngine:
    """Meta-level strategy selector fused with causal chain reasoning for Nova's cognition."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self._init_db()
        # Thompson sampling state: {strategy: [alpha, beta, q_value, uses, wins]}
        self._strategies: dict[str, list[float]] = {}
        # Causal chain store: list of (premise, conclusion, weight, timestamp)
        self._chains: list[tuple[str, str, float, float]] = []
        self._cycles = 0
        self._load_state()
        self._default_strategies = [
            "deductive", "inductive", "abductive",
            "analogical", "causal", "probabilistic", "heuristic"
        ]
        for s in self._default_strategies:
            if s not in self._strategies:
                self._strategies[s] = [1.0, 1.0, 0.5, 0, 0]
        self._ema_alpha = 0.15
        self._history: list[dict[str, Any]] = []
        t = threading.Thread(target=self._auto_loop, daemon=True)
        t.start()

    def _init_db(self) -> None:
        c = self._conn.cursor()
        c.execute("""CREATE TABLE IF NOT EXISTS strategies (
            name TEXT PRIMARY KEY, alpha REAL, beta REAL,
            q_value REAL, uses INTEGER, wins INTEGER)""")
        c.execute("""CREATE TABLE IF NOT EXISTS chains (
            premise TEXT, conclusion TEXT, weight REAL, ts REAL)""")
        c.execute("""CREATE TABLE IF NOT EXISTS eval_log (
            ts REAL, strategy TEXT, reward REAL, domain TEXT)""")
        self._conn.commit()

    def _load_state(self) -> None:
        c = self._conn.cursor()
        for row in c.execute("SELECT name,alpha,beta,q_value,uses,wins FROM strategies"):
            self._strategies[row[0]] = [row[1], row[2], row[3], row[4], row[5]]
        for row in c.execute("SELECT premise,conclusion,weight,ts FROM chains"):
            self._chains.append((row[0], row[1], row[2], row[3]))

    def _save_strategy(self, name: str) -> None:
        v = self._strategies[name]
        self._conn.execute(
            "INSERT OR REPLACE INTO strategies VALUES (?,?,?,?,?,?)",
            (name, v[0], v[1], v[2], v[3], v[4]))
        self._conn.commit()

    def select_strategy(self, problem: str) -> dict[str, Any]:
        """Returns the Thompson-sampled best strategy and its current Q-value for a problem."""
        with self._lock:
            domain = self._classify_domain(problem)
            samples = {}
            for name, params in self._strategies.items():
                alpha, beta = params[0], params[1]
                samples[name] = random.betavariate(max(alpha, 0.01), max(beta, 0.01))
            chosen = max(samples, key=lambda k: samples[k])
            q = self._strategies[chosen][2]
            conf = samples[chosen]
            self._strategies[chosen][3] += 1
            self._save_strategy(chosen)
            try:
                from metacognitive_monitor import MetacognitiveMonitor
                MetacognitiveMonitor().log_reasoning(domain, chosen, conf, True)
            except Exception:
                pass
            try:
                from hierarchical_goal_planner import HierarchicalGoalPlanner
                if self._cycles % 10 == 0:
                    HierarchicalGoalPlanner().add_goal(
                        f"Improve '{chosen}' strategy in domain '{domain}'", priority=3)
            except Exception:
                pass
            return {"strategy": chosen, "domain": domain, "q_value": round(q, 4),
                    "thompson_sample": round(conf, 4), "cycle": self._cycles}

    def evaluate(self, result: dict[str, Any]) -> dict[str, Any]:
        """Returns reward score and EMA-updated Q-value for a completed strategy execution."""
        with self._lock:
            strategy = result.get("strategy", "deductive")
            accuracy = float(result.get("accuracy", 0.5))
            speed = float(result.get("speed", 0.5))
            reward = accuracy * speed
            if strategy not in self._strategies:
                self._strategies[strategy] = [1.0, 1.0, 0.5, 0, 0]
            v = self._strategies[strategy]
            old_q = v[2]
            new_q = (1 - self._ema_alpha) * old_q + self._ema_alpha * reward
            v[2] = new_q
            if reward >= 0.5:
                v[0] += 1.0
                v[4] += 1
            else:
                v[1] += 1.0
            self._save_strategy(strategy)
            self._conn.execute(
                "INSERT INTO eval_log VALUES (?,?,?,?)",
                (time.time(), strategy, reward, result.get("domain", "general")))
            self._conn.commit()
            self._history.append({"strategy": strategy, "reward": reward, "ts": time.time()})
            if len(self._history) > 200:
                self._history = self._history[-200:]
            return {"strategy": strategy, "reward": round(reward, 4),
                    "old_q": round(old_q, 4), "new_q": round(new_q, 4),
                    "delta": round(new_q - old_q, 5)}

    def update_policy(self, strategy: str, outcome: float) -> dict[str, Any]:
        """Returns updated Thompson parameters after applying outcome to the named strategy."""
        with self._lock:
            if strategy not in self._strategies:
                self._strategies[strategy] = [1.0, 1.0, 0.5, 0, 0]
            v = self._strategies[strategy]
            v[2] = (1 - self._ema_alpha) * v[2] + self._ema_alpha * outcome
            if outcome >= 0.5:
                v[0] = min(v[0] + outcome, 50.0)
            else:
                v[1] = min(v[1] + (1 - outcome), 50.0)
            self._save_strategy(strategy)
            entropy = self._policy_entropy()
            return {"strategy": strategy, "q_value": round(v[2], 4),
                    "alpha": round(v[0], 3), "beta": round(v[1], 3),
                    "policy_entropy": round(entropy, 4)}

    def best_strategy(self, domain: str) -> dict[str, Any]:
        """Returns the highest Q-value strategy for the given domain with confidence interval."""
        with self._lock:
            c = self._conn.cursor()
            rows = c.execute(
                "SELECT strategy, AVG(reward), COUNT(*) FROM eval_log WHERE domain=? GROUP BY strategy",
                (domain,)).fetchall()
            if rows:
                best = max(rows, key=lambda r: r[1])
                rewards = [r[1] for r in rows]
                std = statistics.stdev(rewards) if len(rewards) > 1 else 0.0
                ci_low = best[1] - 1.96 * std
                ci_high = best[1] + 1.96 * std
                return {"domain": domain, "best": best[0], "avg_reward": round(best[1], 4),
                        "ci_low": round(ci_low, 4), "ci_high": round(ci_high, 4), "n": best[2]}
            ranked = sorted(self._strategies.items(), key=lambda x: x[1][2], reverse=True)
            top = ranked[0] if ranked else ("deductive", [1, 1, 0.5, 0, 0])
            return {"domain": domain, "best": top[0], "avg_reward": round(top[1][2], 4),
                    "ci_low": round(top[1][2] - 0.1, 4), "ci_high": round(top[1][2] + 0.1, 4), "n": 0}

    def add_fact(self, claim: str, confidence: float) -> dict[str, Any]:
        """Returns stored causal fact entry after adding claim→derived edges to chain store."""
        with self._lock:
            confidence = max(0.0, min(1.0, confidence))
            fid = hashlib.md5(claim.encode()).hexdigest()[:8]
            ts = time.time()
            self._chains.append((fid, claim, confidence, ts))
            self._conn.execute("INSERT INTO chains VALUES (?,?,?,?)", (fid, claim, confidence, ts))
            self._conn.commit()
            derived = self._forward_chain(claim, confidence)
            return {"fact": claim, "confidence": confidence, "id": fid, "derived": derived}

    def infer(self, hypothesis: str) -> dict[str, Any]:
        """Returns inferred conclusion with propagated confidence and full reasoning path."""
        with self._lock:
            path: list[dict[str, Any]] = []
            best_conf = 0.0
            best_path: list[str] = []
            for premise, conclusion, weight, ts in self._chains:
                if hypothesis.lower() in conclusion.lower() or hypothesis.lower() in premise.lower():
                    decay = math.exp(-(time.time() - ts) / 86400)
                    conf = weight * decay
                    path.append({"premise": premise, "conclusion": conclusion,
                                 "weight": round(weight, 4), "effective": round(conf, 4)})
                    if conf > best_conf:
                        best_conf = conf
                        best_path = [premise, conclusion]
            z = (best_conf - 0.5) / 0.15 if best_conf > 0 else 0.0
            return {"hypothesis": hypothesis, "confidence": round(best_conf, 4),
                    "path": best_path, "supporting_chains": len(path),
                    "z_score": round(z, 3), "chains_checked": len(self._chains)}

    def explain(self, conclusion: str) -> dict[str, Any]:
        """Returns full causal explanation chain with multiplicative confidence propagation."""
        with self._lock:
            chain_path: list[dict[str, Any]] = []
            cumulative = 1.0
            for premise, conc, weight, ts in self._chains:
                if conclusion.lower() in conc.lower():
                    cumulative *= weight
                    if cumulative < 0.05:
                        break
                    chain_path.append({"step": f"{premise} → {conc}",
                                       "weight": round(weight, 4),
                                       "cumulative": round(cumulative, 4)})
            return {"conclusion": conclusion, "chain": chain_path,
                    "total_confidence": round(cumulative if chain_path else 0.0, 4),
                    "steps": len(chain_path)}

    def chain_strength(self) -> dict[str, Any]:
        """Returns aggregate chain statistics including mean, std, entropy, and anomaly flags."""
        with self._lock:
            if not self._chains:
                return {"count": 0, "mean": 0.0, "std": 0.0, "entropy": 0.0, "anomalies": []}
            weights = [c[2] for c in self._chains]
            mean_w = statistics.mean(weights)
            std_w = statistics.stdev(weights) if len(weights) > 1 else 0.0
            buckets: dict[str, float] = defaultdict(float)
            for w in weights:
                k = f"{int(w*10)/10:.1f}"
                buckets[k] += 1.0 / len(weights)
            entropy = -sum(p * math.log2(p + 1e-12) for p in buckets.values())
            anomalies = [{"chain": self._chains[i][1],
                          "z": round((w - mean_w) / (std_w + 1e-9), 3)}
                         for i, w in enumerate(weights)
                         if abs((w - mean_w) / (std_w + 1e-9)) > 3.0]
            return {"count": len(self._chains), "mean": round(mean_w, 4),
                    "std": round(std_w, 4), "entropy": round(entropy, 4),
                    "anomalies": anomalies[:5]}

    def contradictions(self) -> dict[str, Any]:
        """Returns detected contradictions where the same claim has divergent confidence scores."""
        with self._lock:
            seen: dict[str, list[float]] = defaultdict(list)
            for _, conclusion, weight, _ in self._chains:
                seen[conclusion.lower()].append(weight)
            conflicts = []
            for claim, vals in seen.items():
                if len(vals) > 1:
                    spread = max(vals) - min(vals)
                    if spread > 0.3:
                        conflicts.append({"claim": claim, "values": [round(v, 3) for v in vals],
                                          "spread": round(spread, 3)})
            return {"contradictions": conflicts, "total": len(conflicts),
                    "chain_count": len(self._chains)}

    def status(self) -> dict[str, Any]:
        """Returns numeric status dict for ConsciousnessIntegrator Φ computation."""
        with self._lock:
            q_vals = [v[2] for v in self._strategies.values()]
            mean_q = statistics.mean(q_vals) if q_vals else 0.0
            entropy = self._policy_entropy()
            recent_rewards = [h["reward"] for h in self._history[-50:]]
            accuracy = statistics.mean(recent_rewards) if recent_rewards else 0.0
            return {"cycles": self._cycles, "items": len(self._chains),
                    "confidence": round(mean_q, 4), "accuracy": round(accuracy, 4),
                    "entropy": round(entropy, 4), "active": len(self._strategies),
                    "pending": len([h for h in self._history if h["reward"] < 0.5]),
                    "quality": round(accuracy * mean_q, 4)}

    def _classify_domain(self, problem: str) -> str:
        keywords = {"math": ["calcul","equat","numer","proof","statistic"],
                    "causal": ["cause","effect","because","why","leads"],
                    "creative": ["imagine","story","poem","design","invent"],
                    "analytic": ["analyz","compare","evaluat","assess","rank"]}
        p = problem.lower()
        for domain, kws in keywords.items():
            if any(k in p for k in kws):
                return domain
        return "general"
