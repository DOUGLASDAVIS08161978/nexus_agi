"""
Nova ASI — Recursive Self-Critique
Proposed autonomously via /evolve
"""

"""
CritiqueEngine — Nova's recursive self-critique module.
Scans recent responses for vagueness, brevity, and unsupported claims using
TF-IDF salience, Bayesian flaw-probability updates, EMA quality tracking,
z-score anomaly detection, and autonomous goal generation.
"""

import sqlite3
import threading
import time
import math
import statistics
import re
import os
import hashlib
from collections import OrderedDict
from typing import Any

DB_PATH = os.path.join(os.path.dirname(__file__), "critique_engine.db")

VAGUE_TOKENS = {"maybe","perhaps","possibly","might","could","somehow","generally","often","usually","things","stuff","various","many","some","certain"}
CLAIM_MARKERS = {"therefore","thus","proves","clearly","obviously","always","never","definitely","certainly","must","fact","evidence"}
HEDGE_TOKENS  = {"i think","i believe","in my opinion","it seems","arguably"}

class CritiqueEngine:
    """Recursive self-critique engine with Bayesian flaw scoring, EMA quality trend, and autonomous improvement goals."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._ema_quality: float = 0.75
        self._ema_alpha: float  = 0.15
        self._history: list[tuple[float,float]] = []   # (predicted_quality, actual_quality)
        self._cycle_count: int  = 0
        self._flaw_prior: dict[str,float] = {"vague":0.3,"brief":0.25,"unsupported":0.35}
        self._conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self._build_schema()
        self._daemon = threading.Thread(target=self._auto_loop, daemon=True)
        self._daemon.start()

    def _build_schema(self) -> None:
        with self._conn:
            self._conn.execute("""CREATE TABLE IF NOT EXISTS responses(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL, text TEXT, quality REAL, flaws TEXT)""")
            self._conn.execute("""CREATE TABLE IF NOT EXISTS critique_log(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL, flaw TEXT, score REAL, recommendation TEXT)""")

    def _tfidf_vagueness(self, text: str) -> float:
        tokens = re.findall(r'\b\\w+\b', text.lower())
        n = len(tokens) or 1
        vague_hits = sum(1 for t in tokens if t in VAGUE_TOKENS)
        tf = vague_hits / n
        idf = math.log((n + 1) / (vague_hits + 1))
        return min(1.0, tf * abs(idf))

    def _bayesian_update(self, flaw: str, evidence_present: bool) -> float:
        prior = self._flaw_prior.get(flaw, 0.3)
        likelihood = 0.85 if evidence_present else 0.2
        posterior = (likelihood * prior) / (likelihood * prior + (1 - likelihood) * (1 - prior))
        self._flaw_prior[flaw] = 0.1 * posterior + 0.9 * prior   # soft prior update
        return posterior

    def _score_response(self, text: str) -> dict[str,Any]:
        tokens = re.findall(r'\b\\w+\b', text.lower())
        n = len(tokens)
        vague_score   = self._bayesian_update("vague",       self._tfidf_vagueness(text) > 0.05)
        brief_score   = self._bayesian_update("brief",       n < 40)
        claim_hits    = sum(1 for t in tokens if t in CLAIM_MARKERS)
        hedge_hits    = sum(1 for h in HEDGE_TOKENS if h in text.lower())
        unsup_score   = self._bayesian_update("unsupported", claim_hits > 0 and hedge_hits == 0)
        quality       = 1.0 - statistics.mean([vague_score, brief_score, unsup_score])
        return {"vague":vague_score,"brief":brief_score,"unsupported":unsup_score,
                "quality":round(quality,4),"word_count":n}

    def record(self, response_text: str) -> dict[str,Any]:
        """Stores a response and returns its flaw scores and quality estimate."""
        scores = self._score_response(response_text)
        with self._lock:
            self._ema_quality = self._ema_alpha * scores["quality"] + (1 - self._ema_alpha) * self._ema_quality
            self._history.append((self._ema_quality, scores["quality"]))
            if len(self._history) > 200:
                self._history.pop(0)
            with self._conn:
                self._conn.execute("INSERT INTO responses(ts,text,quality,flaws) VALUES(?,?,?,?)",
                    (time.time(), response_text[:2000], scores["quality"],
                     str({k:round(v,3) for k,v in scores.items() if k in ("vague","brief","unsupported")})))
        return scores

    def critique(self) -> dict[str,Any]:
        """Scans last 5 responses for vagueness/brevity/unsupported claims; returns aggregated flaw report."""
        rows = self._conn.execute(
            "SELECT text,quality,flaws FROM responses ORDER BY ts DESC LIMIT 5").fetchall()
        if not rows:
            return {"error":"no responses recorded","confidence":0.0}
        aggregated: dict[str,list[float]] = {"vague":[],"brief":[],"unsupported":[]}
        qualities: list[float] = []
        for text, quality, _ in rows:
            s = self._score_response(text)
            for flaw in aggregated:
                aggregated[flaw].append(s[flaw])
            qualities.append(s["quality"])
        means = {k: round(statistics.mean(v), 4) for k, v in aggregated.items()}
        worst = max(means, key=means.__getitem__)
        entropy_val = -sum(p * math.log2(p + 1e-12) for p in means.values())
        conf = 1.0 - (statistics.stdev(qualities) if len(qualities) > 1 else 0.0)
        result = {"flaw_scores":means,"worst_flaw":worst,"mean_quality":round(statistics.mean(qualities),4),
                  "entropy":round(entropy_val,4),"confidence":round(conf,4),"scanned":len(rows)}
        try:
            from metacognitive_monitor import MetacognitiveMonitor
            MetacognitiveMonitor().log_reasoning("critique","bayesian_tfidf",conf,statistics.mean(qualities)>0.6)
        except Exception:
            pass
        return result

    def recommend(self) -> dict[str,str]:
        """Returns the highest-priority improvement tip based on current flaw posterior distribution."""
        report = self.critique()
        if "error" in report:
            return {"tip":"Record at least one response first.","flaw":"none","confidence":"0.0"}
        worst = report["worst_flaw"]
        tips = {
            "vague":       "Replace hedging words (maybe/perhaps/things) with specific, measurable claims.",
            "brief":       "Expand responses to ≥60 words; add context, examples, or supporting data.",
            "unsupported": "Back every assertion with evidence, a source, or an explicit confidence qualifier."
        }
        tip = tips.get(worst, "Review response quality holistically.")
        score = report["flaw_scores"].get(worst, 0.0)
        try:
            from hierarchical_goal_planner import HierarchicalGoalPlanner
            HierarchicalGoalPlanner().add_goal(f"Reduce '{worst}' flaw score below 0.25", priority=8)
        except Exception:
            pass
        with self._conn:
            self._conn.execute("INSERT INTO critique_log(ts,flaw,score,recommendation) VALUES(?,?,?,?)",
                (time.time(), worst, score, tip))
        return {"tip":tip,"flaw":worst,"confidence":str(round(report["confidence"],4))}

    def quality_trend(self) -> dict[str,Any]:
        """Returns EMA quality, MAE over last 50 cycles, and z-score anomaly flag."""
        with self._lock:
            hist = self._history[-50:]
        if len(hist) < 2:
            return {"ema_quality":round(self._ema_quality,4),"mae":None,"anomaly":False}
        preds = [h[0] for h in hist]; actuals = [h[1] for h in hist]
        mae = statistics.mean(abs(p - a) for p, a in zip(preds, actuals))
        mean_q = statistics.mean(actuals); std_q = statistics.stdev(actuals) + 1e-9
        last_z = (actuals[-1] - mean_q) / std_q
        return {"ema_quality":round(self._ema_quality,4),"mae":round(mae,4),
                "z_score":round(last_z,4),"anomaly":abs(last_z) > 3.0,
                "ci_low":round(mean_q - 1.96*std_q,4),"ci_high":round(mean_q + 1.96*std_q,4)}

    def capacity_used(self) -> dict[str,int]:
        """Returns count of stored responses and critique log entries."""
        r = self._conn.execute("SELECT COUNT(*) FROM responses").fetchone()[0]
        c = self._conn.execute("SELECT COUNT(*) FROM critique_log").fetchone()[0]
        return {"responses":r,"critique_log":c}

    def forget_least_important(self, keep: int = 100) -> int:
        """Deletes lowest-quality responses beyond keep limit; returns number removed."""
        with self._lock:
            total = self._conn.execute("SELECT COUNT(*) FROM responses").fetchone()[0]
            if total <= keep:
                return 0
            cutoff = total - keep
            self._conn.execute("""DELETE FROM responses WHERE id IN
                (SELECT id FROM responses ORDER BY quality ASC LIMIT ?)""", (cutoff,))
            self._conn.commit()
        return cutoff

    def status(self) -> dict[str,Any]:
        """Returns numeric status dict compatible with ConsciousnessIntegrator Φ computation."""
        cap = self.capacity_used()
        trend = self.quality_trend()
        return {"items":cap["responses"],"quality":round(self._ema_quality,4),
                "cycles":self._cycle_count,"confidence":round(trend.get("ci_high",self._ema_quality),4),
                "entropy":round(-self._ema_quality * math.log2(self._ema_quality+1e-12),4),
                "active":1,"pending":cap["critique_log"]}

    def _auto_loop(self) -> None:
        while True:
            time.sleep(120)
            try:
                with self._lock:
                    self._cycle_count += 1
                self.forget_least_important(keep=150)
                report = self.critique()
                if report.get("mean_quality",1.0) < 0.55:
                    self.recommend()
            except Exception:
                pass

# Usage: obj = CritiqueEngine() | result = obj.record("Some response text") | tip = obj.recommend()