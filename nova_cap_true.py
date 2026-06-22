"""
Nova ASI — TRUE
Built autonomously via /build + APIHunter
No API credentials required.
"""

"""
NovaTruthVerificationEngine — autonomous truth verification, calibration, and belief auditing module.
Implements Bayesian belief updates, causal confidence propagation, EMA-based accuracy tracking,
z-score anomaly detection, TF-IDF claim scoring, and self-generating goals for epistemic integrity.
Pillar coverage: ①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭
"""

import sqlite3, threading, math, statistics, time, re, json, hashlib, os
from collections import OrderedDict
from datetime import datetime

class NovaTruthVerificationEngine:
    """Autonomous truth verification engine with Bayesian calibration and causal confidence propagation."""

    DB = "nova_truth_verification.db"

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._ema_accuracy: float = 0.5
        self._cycle_count: int = 0
        self._history: list[tuple[float, float]] = []
        self._conn = sqlite3.connect(self.DB, check_same_thread=False)
        self._init_db()
        self._daemon = threading.Thread(target=self._auto_loop, daemon=True)
        self._daemon.start()

    def _init_db(self) -> None:
        c = self._conn.cursor()
        c.executescript("""
            CREATE TABLE IF NOT EXISTS claims (
                id TEXT PRIMARY KEY, text TEXT, confidence REAL,
                verified INTEGER, timestamp REAL, accesses INTEGER DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS verdicts (
                id TEXT PRIMARY KEY, claim_id TEXT, verdict TEXT,
                prior REAL, posterior REAL, evidence TEXT, timestamp REAL
            );
            CREATE TABLE IF NOT EXISTS anomalies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                claim_id TEXT, z_score REAL, flagged_at REAL
            );
        """)
        self._conn.commit()

    def _tfidf_score(self, claim: str, corpus: list[str]) -> float:
        tokens = re.findall(r'\\w+', claim.lower())
        N = max(len(corpus), 1)
        score = 0.0
        for t in set(tokens):
            tf = tokens.count(t) / max(len(tokens), 1)
            df = sum(1 for doc in corpus if t in doc.lower())
            idf = math.log(N / (df + 1))
            score += tf * idf
        return round(score, 6)

    def _bayesian_update(self, prior: float, likelihood: float) -> float:
        posterior = (likelihood * prior) / max(
            likelihood * prior + (1 - likelihood) * (1 - prior), 1e-12)
        return round(min(max(posterior, 0.0), 1.0), 6)

    def _claim_id(self, text: str) -> str:
        return hashlib.sha1(text.strip().lower().encode()).hexdigest()[:12]

    def verify(self, claim: str, evidence_strength: float = 0.7, prior: float = 0.5) -> dict:
        """Returns Bayesian posterior confidence and TF-IDF salience for a truth claim."""
        with self._lock:
            cid = self._claim_id(claim)
            c = self._conn.cursor()
            c.execute("SELECT text FROM claims", )
            corpus = [r[0] for r in c.fetchall()]
            tfidf = self._tfidf_score(claim, corpus)
            posterior = self._bayesian_update(prior, evidence_strength)
            entropy = -sum(p * math.log2(p + 1e-12) for p in [posterior, 1 - posterior])
            ci_low = max(0.0, posterior - 1.96 * math.sqrt(posterior * (1 - posterior) / 10))
            ci_high = min(1.0, posterior + 1.96 * math.sqrt(posterior * (1 - posterior) / 10))
            now = time.time()
            c.execute("INSERT OR REPLACE INTO claims VALUES (?,?,?,?,?,?)",
                      (cid, claim, posterior, 0, now, 0))
            c.execute("INSERT OR REPLACE INTO verdicts VALUES (?,?,?,?,?,?,?)",
                      (cid + "_v", cid, "pending", prior, posterior,
                       json.dumps({"evidence_strength": evidence_strength}), now))
            self._conn.commit()
            self._history.append((posterior, 0.5))
            self._ema_accuracy = 0.15 * posterior + 0.85 * self._ema_accuracy
            return {"claim_id": cid, "claim": claim[:80], "prior": prior,
                    "posterior": posterior, "entropy": round(entropy, 4),
                    "tfidf_salience": tfidf, "ci": [round(ci_low, 4), round(ci_high, 4)]}

    def record_outcome(self, claim_id: str, was_true: bool) -> dict:
        """Returns updated EMA accuracy after recording ground-truth outcome for a claim."""
        with self._lock:
            actual = 1.0 if was_true else 0.0
            c = self._conn.cursor()
            c.execute("SELECT confidence FROM claims WHERE id=?", (claim_id,))
            row = c.fetchone()
            if not row:
                return {"error": "claim_not_found"}
            predicted = row[0]
            c.execute("UPDATE claims SET verified=? WHERE id=?", (int(was_true), claim_id))
            self._conn.commit()
            self._history.append((predicted, actual))
            self._ema_accuracy = 0.15 * actual + 0.85 * self._ema_accuracy
            mae = statistics.mean(abs(p - a) for p, a in self._history[-50:])
            return {"claim_id": claim_id, "predicted": predicted, "actual": actual,
                    "ema_accuracy": round(self._ema_accuracy, 4), "mae_50": round(mae, 4)}

    def anomaly_scan(self) -> list[dict]:
        """Returns list of claims whose confidence deviates > 3 z-score from rolling mean."""
        with self._lock:
            c = self._conn.cursor()
            c.execute("SELECT id, confidence FROM claims ORDER BY timestamp DESC LIMIT 100")
            rows = c.fetchall()
            if len(rows) < 3:
                return []
            confs = [r[1] for r in rows]
            mu = statistics.mean(confs)
            sigma = statistics.stdev(confs) if len(confs) > 1 else 1e-9
            flagged = []
            for cid, conf in rows:
                z = (conf - mu) / (sigma + 1e-9)
                if abs(z) > 3.0:
                    c.execute("INSERT INTO anomalies (claim_id, z_score, flagged_at) VALUES (?,?,?)",
                              (cid, round(z, 4), time.time()))
                    flagged.append({"claim_id": cid, "confidence": conf, "z_score": round(z, 4)})
            self._conn.commit()
            return flagged

    def causal_chain_confidence(self, chain: list[float]) -> dict:
        """Returns multiplicative causal confidence for a chain, pruning links below 0.05."""
        pruned = [c for c in chain if c >= 0.05]
        if not pruned:
            return {"chain_confidence": 0.0, "pruned": len(chain), "links": []}
        product = 1.0
        for c in pruned:
            product *= c
        entropy = -sum(c * math.log2(c + 1e-12) for c in pruned) / max(len(pruned), 1)
        return {"chain_confidence": round(product, 6), "links": len(pruned),
                "pruned": len(chain) - len(pruned), "chain_entropy": round(entropy, 4)}

    def calibration_report(self) -> dict:
        """Returns calibration error, EMA accuracy, and epistemic entropy across all verdicts."""
        with self._lock:
            if len(self._history) < 2:
                return {"calibration_error": None, "ema_accuracy": self._ema_accuracy,
                        "samples": len(self._history)}
            preds = [p for p, _ in self._history[-100:]]
            actuals = [a for _, a in self._history[-100:]]
            mae = statistics.mean(abs(p - a) for p, a in zip(preds, actuals))
            stdev = statistics.stdev(preds) if len(preds) > 1 else 0.0
            entropy = -sum(p * math.log2(p + 1e-12) + (1 - p) * math.log2(1 - p + 1e-12)
                           for p in preds) / max(len(preds), 1)
            return {"calibration_error": round(mae, 4), "ema_accuracy": round(self._ema_accuracy, 4),
                    "stdev": round(stdev, 4), "entropy": round(entropy, 4),
                    "samples": len(self._history), "cycles": self._cycle_count}

    def top_uncertain_claims(self, n: int = 5) -> list[dict]:
        """Returns top-n unverified claims ranked by maximum entropy (uncertainty ~ 0.5 posterior)."""
        with self._lock:
            c = self._conn.cursor()
            c.execute("SELECT id, text, confidence FROM claims WHERE verified=0 ORDER BY timestamp DESC LIMIT 200")
            rows = c.fetchall()
            scored = []
            for cid, text, conf in rows:
                entropy = -(conf * math.log2(conf + 1e-12) + (1 - conf) * math.log2(1 - conf + 1e-12))
                scored.append({"claim_id": cid, "text": text[:60], "confidence": conf,
                               "entropy": round(entropy, 4)})
            scored.sort(key=lambda x: x["entropy"], reverse=True)
            return scored[:n]

    def status(self) -> dict:
        """Returns numeric status dict compatible with ConsciousnessIntegrator Φ computation."""
        with self._lock:
            c = self._conn.cursor()
            c.execute("SELECT COUNT(*) FROM claims")
            items = c.fetchone()[0]
            c.execute("SELECT COUNT(*) FROM claims WHERE verified=1")
            verified = c.fetchone()[0]
            c.execute("SELECT COUNT(*) FROM anomalies")
            anomalies = c.fetchone()[0]
            accuracy = round(self._ema_accuracy, 4)
            cal = self.calibration_report()
            return {"items": items, "confidence": accuracy, "accuracy": accuracy,
                    "verified": verified, "anomalies": anomalies,
                    "cycles": self._cycle_count, "entropy": cal.get("entropy", 0.0),
                    "active": 1, "pending": max(0, items - verified)}

    def auto_cycle(self) -> dict:
        """Returns cycle summary after running anomaly scan, calibration, and goal generation."""
        with self._lock:
            self._cycle_count += 1
        anomalies = self.anomaly_scan()
        cal = self.calibration_report()
        try:
            from tools import HierarchicalGoalPlanner, MetacognitiveMonitor
            planner = HierarchicalGoalPlanner()
            if cal.get("calibration_error") and cal["calibration_error"] > 0.2:
                planner.add_goal("Reduce truth-verification calibration error below 0.15", priority=8)
            if anomalies:
                planner.add_goal(f"Investigate {len(anomalies)} anomalous belief confidence scores", priority=7)
            planner.add_goal("Expand verified claim corpus for epistemic grounding", priority=5)
            monitor = MetacognitiveMonitor()
            monitor.log_reasoning("truth_verification", "bayesian+tfidf+causal",
                                  self._ema_accuracy, cal.get("calibration_error", 1.0) < 0.2)
        except (ImportError, AttributeError, Exception):
            pass
        return {"cycle": self._cycle_count, "anomalies_found": len(anomalies),
                "calibration_error": cal.get("calibration_error"), "ema_accuracy": self._ema_accuracy}

    def _auto_loop(self) -> None:
        while True:
            time.sleep(90)
            try:
                self.auto_cycle()
            except Exception:
                pass

# Usage: obj = NovaTruthVerificationEngine() | result = obj.verify("The sky is blue", evidence_strength=0.9)