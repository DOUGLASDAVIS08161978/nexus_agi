"""
Nova ASI — Bayesian Belief System with Causal Chain Reasoning
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Maintains probability distributions over hypotheses per domain.
Bayesian update: P(H|E) = P(E|H) * P(H) / P(E).
Epistemic entropy: -sum(p * log2(p)) — falls as evidence accumulates.
Causal graph: cause → [(effect, weight)]. Confidence propagates
multiplicatively through chains: A→B(0.8), B→C(0.9) → A→C(0.72).
Contradiction detection flags when two beliefs both exceed 0.60.

Built by Douglas Shane Davis & Claude Code (Anthropic)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os, sqlite3, threading, math, json
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from datetime import datetime

_DB = os.path.expanduser("~/nexus_agi/nova_beliefs.db")


class BayesianBeliefSystem:
    """
    Epistemic engine: probability distributions + causal graph reasoning.

    Each domain holds a belief distribution that sums to 1.0.
    Bayesian update: P(H|E) ∝ P(E|H) * P(H), then renormalize.
    Entropy decreases as evidence accumulates toward certainty.
    Causal inference follows edges multiplicatively through the DAG.
    Contradiction detection flags distributions where two hypotheses
    both exceed 0.60 (logically inconsistent unless domain is multi-label).
    """

    def __init__(self) -> None:
        self._lock    = threading.Lock()
        self._beliefs: Dict[str, Dict[str, float]] = {}
        self._causal:  Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        self.conn = sqlite3.connect(_DB, check_same_thread=False)
        self._init_db()
        self._restore()

    # ─── Database ──────────────────────────────────────────────────────

    def _init_db(self) -> None:
        with self._lock:
            self.conn.executescript("""
                CREATE TABLE IF NOT EXISTS belief_states (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts         TEXT DEFAULT (datetime('now')),
                    domain     TEXT,
                    hypothesis TEXT,
                    probability REAL
                );
                CREATE TABLE IF NOT EXISTS evidence_log (
                    id             INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts             TEXT DEFAULT (datetime('now')),
                    domain         TEXT,
                    evidence       TEXT,
                    entropy_before REAL,
                    entropy_after  REAL
                );
                CREATE TABLE IF NOT EXISTS causal_edges (
                    id     INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts     TEXT DEFAULT (datetime('now')),
                    cause  TEXT,
                    effect TEXT,
                    weight REAL
                );
            """)
            self.conn.commit()

    def _restore(self) -> None:
        rows = self.conn.execute(
            "SELECT domain, hypothesis, probability FROM belief_states "
            "WHERE id IN "
            "(SELECT MAX(id) FROM belief_states GROUP BY domain, hypothesis)"
        ).fetchall()
        for domain, hypothesis, prob in rows:
            self._beliefs.setdefault(domain, {})[hypothesis] = float(prob)
        for cause, effect, weight in self.conn.execute(
                "SELECT cause, effect, weight FROM causal_edges"):
            self._causal[cause].append((effect, float(weight)))

    def _normalize(self, domain: str) -> None:
        dist  = self._beliefs.get(domain, {})
        total = sum(dist.values()) + 1e-12
        for h in dist:
            dist[h] = dist[h] / total

    def _persist(self, domain: str) -> None:
        for h, p in self._beliefs.get(domain, {}).items():
            self.conn.execute(
                "INSERT INTO belief_states (domain, hypothesis, probability) "
                "VALUES (?,?,?)", (domain, h, p))
        self.conn.commit()

    # ─── Belief management ─────────────────────────────────────────────

    def set_prior(self, domain: str, hypotheses: Dict[str, float]) -> None:
        """Initialize a belief domain with a prior distribution."""
        with self._lock:
            if domain in self._beliefs:
                return  # Never overwrite existing evidence-based beliefs
            self._beliefs[domain] = dict(hypotheses)
            self._normalize(domain)
            self._persist(domain)

    def update(self, domain: str, evidence: str,
               likelihood_ratios: Dict[str, float]) -> float:
        """
        Bayesian update: P(H|E) ∝ P(E|H) * P(H).
        likelihood_ratios: {hypothesis: P(evidence|hypothesis)}.
        Returns entropy change (negative = belief sharpened = learning).
        """
        with self._lock:
            if domain not in self._beliefs:
                n = max(1, len(likelihood_ratios))
                self._beliefs[domain] = {h: 1.0 / n for h in likelihood_ratios}
            before = self._entropy_locked(domain)
            dist   = self._beliefs[domain]
            for h in list(dist.keys()):
                lr      = float(likelihood_ratios.get(h, 1.0))
                dist[h] = dist[h] * max(0.001, lr)
            self._normalize(domain)
            after = self._entropy_locked(domain)
            self.conn.execute(
                "INSERT INTO evidence_log "
                "(domain, evidence, entropy_before, entropy_after) VALUES (?,?,?,?)",
                (domain, evidence[:200], before, after))
            self._persist(domain)
        return round(after - before, 4)

    def posterior(self, domain: str) -> Dict[str, float]:
        """Return current posterior distribution for a domain."""
        with self._lock:
            return dict(self._beliefs.get(domain, {}))

    def most_confident(self, domain: str) -> Tuple[str, float]:
        """Return (hypothesis, probability) of the strongest belief."""
        with self._lock:
            dist = self._beliefs.get(domain, {})
            if not dist:
                return ("unknown", 0.0)
            h = max(dist, key=lambda x: dist[x])
            return (h, round(dist[h], 4))

    def entropy(self, domain: str) -> float:
        """Epistemic uncertainty: -sum(p * log2(p)). Zero = certainty."""
        with self._lock:
            return self._entropy_locked(domain)

    def _entropy_locked(self, domain: str) -> float:
        dist = self._beliefs.get(domain, {})
        if not dist:
            return 0.0
        return round(-sum(p * math.log2(p + 1e-12)
                          for p in dist.values() if p > 0), 4)

    def all_domains(self) -> List[str]:
        """Return all registered belief domains."""
        with self._lock:
            return list(self._beliefs.keys())

    # ─── Causal reasoning ──────────────────────────────────────────────

    def add_causal_edge(self, cause: str, effect: str,
                        weight: float = 0.80) -> None:
        """Assert cause → effect with confidence weight (0–1)."""
        with self._lock:
            w = max(0.0, min(1.0, float(weight)))
            self._causal[cause] = [
                (e, wt) for e, wt in self._causal[cause] if e != effect
            ]
            self._causal[cause].append((effect, w))
            self.conn.execute(
                "INSERT INTO causal_edges (cause, effect, weight) VALUES (?,?,?)",
                (cause, effect, w))
            self.conn.commit()

    def infer(self, cause: str, depth: int = 5) -> List[Tuple[str, float]]:
        """
        Forward inference from cause through causal DAG.
        Confidence degrades multiplicatively: A→B(0.8), B→C(0.9) → A→C(0.72).
        Returns [(effect, cumulative_confidence)] sorted by confidence.
        """
        with self._lock:
            visited: Dict[str, float] = {}
            queue = [(cause, 1.0, 0)]
            while queue:
                node, conf, d = queue.pop(0)
                if d >= depth:
                    continue
                for effect, weight in self._causal.get(node, []):
                    new_conf = conf * weight
                    if effect not in visited or visited[effect] < new_conf:
                        visited[effect] = new_conf
                        queue.append((effect, new_conf, d + 1))
        results = [(e, round(c, 4)) for e, c in visited.items() if e != cause]
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def chain_strength(self, cause: str, effect: str) -> float:
        """Cumulative confidence on the best causal path cause → effect."""
        return dict(self.infer(cause)).get(effect, 0.0)

    def explain(self, cause: str, effect: str) -> str:
        """Natural-language explanation of the causal chain."""
        with self._lock:
            path = self._best_path(cause, effect)
        if not path:
            return f"No known causal link: '{cause}' → '{effect}'."
        steps, confidence = [], 1.0
        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            w = next((wt for e, wt in self._causal.get(a, []) if e == b), 0.0)
            steps.append(f"{a}→{b}({w:.2f})")
            confidence *= w
        return " → ".join(steps) + f" [chain: {confidence:.3f}]"

    def _best_path(self, cause: str, effect: str,
                   visited: Optional[set] = None) -> List[str]:
        if visited is None:
            visited = set()
        if cause == effect:
            return [cause]
        visited.add(cause)
        best: List[str] = []
        for child, _ in self._causal.get(cause, []):
            if child not in visited:
                sub = self._best_path(child, effect, set(visited))
                if sub and (not best or len(sub) < len(best)):
                    best = sub
        return ([cause] + best) if best else []

    def contradictions(self) -> List[str]:
        """Detect domains where two hypotheses both exceed 0.60 confidence."""
        issues = []
        with self._lock:
            for domain, dist in self._beliefs.items():
                high = [(h, p) for h, p in dist.items() if p > 0.60]
                if len(high) > 1:
                    issues.append(
                        f"[{domain}] conflicting: "
                        + ", ".join(f"{h}={p:.2f}" for h, p in high))
        return issues

    def status(self) -> Dict:
        """Belief system health snapshot for ConsciousnessIntegrator."""
        with self._lock:
            domains = list(self._beliefs.keys())
        entropies = {d: self.entropy(d) for d in domains}
        top       = {d: self.most_confident(d) for d in domains}
        return {
            "domains":       len(domains),
            "avg_entropy":   round(sum(entropies.values()) /
                                   max(1, len(entropies)), 4),
            "causal_nodes":  len(self._causal),
            "causal_edges":  sum(len(v) for v in self._causal.values()),
            "contradictions": len(self.contradictions()),
            "strongest":     {d: f"{h}({p:.2f})" for d, (h, p) in top.items()},
        }

# Usage: bs = BayesianBeliefSystem()
# bs.set_prior("market", {"bull": 0.5, "bear": 0.3, "flat": 0.2})
# bs.update("market", "prices_rising", {"bull": 0.9, "bear": 0.1, "flat": 0.4})
# bs.add_causal_edge("learning", "capability_growth", weight=0.85)
# print(bs.most_confident("market")) | print(bs.entropy("market"))
