"""
nova_cap_curiosity.py
Nova ASI — Curiosity
Generated via /evolve · v29 pipeline · 2026-06-21
"""

"""
Nova Epistemic Curiosity Engine — detects knowledge gaps, scores by information gain,
generates exploration questions, and autonomously drives uncertainty reduction.
Pillars: ①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭
"""

import sqlite3, math, time, threading, statistics, hashlib, json, re, os, random
from collections import OrderedDict
from typing import Any

DB_PATH = os.path.join(os.path.dirname(__file__), "nova_curiosity.db")

class EpistemicCuriosityEngine:
    """Nova's curiosity core: gaps → scores → questions → epistemic yield ranking."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self._build_schema()
        self._cycles: int = 0
        self._ema_gain: float = 0.5
        self._question_templates = [
            "What causes {concept} to behave unexpectedly?",
            "How does {concept} relate to adjacent domains?",
            "What would falsify current beliefs about {concept}?",
            "Which edge cases expose limits of {concept}?",
            "What is the most uncertain aspect of {concept}?",
        ]
        t = threading.Thread(target=self._auto_loop, daemon=True)
        t.start()

    def _build_schema(self) -> None:
        c = self._conn
        c.execute("""CREATE TABLE IF NOT EXISTS gaps (
            id TEXT PRIMARY KEY, domain TEXT, question TEXT,
            entropy_before REAL, entropy_after REAL,
            times_asked INTEGER DEFAULT 1, resolved INTEGER DEFAULT 0,
            info_gain REAL DEFAULT 0.0, curiosity_score REAL DEFAULT 0.0,
            created REAL, last_updated REAL)""")
        c.execute("""CREATE TABLE IF NOT EXISTS domain_entropy (
            domain TEXT PRIMARY KEY, entropy REAL, observations INTEGER DEFAULT 1)""")
        c.commit()

    def _gap_id(self, domain: str, question: str) -> str:
        return hashlib.sha1(f"{domain}||{question}".encode()).hexdigest()[:16]

    def _novelty(self, times_asked: int) -> float:
        return 1.0 / (1.0 + times_asked)

    def _relevance(self, question: str, context: str) -> float:
        if not context:
            return 0.5
        q_tokens = set(re.findall(r'\\w+', question.lower()))
        c_tokens = set(re.findall(r'\\w+', context.lower()))
        union = q_tokens | c_tokens
        if not union:
            return 0.5
        jaccard = len(q_tokens & c_tokens) / len(union)
        return 0.1 + 0.9 * jaccard

    def _curiosity_score(self, entropy: float, times_asked: int, question: str, context: str = "") -> float:
        nov = self._novelty(times_asked)
        rel = self._relevance(question, context)
        return entropy * nov * rel

    def _domain_entropy(self, domain: str) -> float:
        row = self._conn.execute(
            "SELECT entropy FROM domain_entropy WHERE domain=?", (domain,)).fetchone()
        return row[0] if row else 1.0

    def observe_gap(self, domain: str, question: str, current_entropy: float) -> dict[str, Any]:
        """Records a knowledge gap and returns its curiosity score and gap id."""
        gid = self._gap_id(domain, question)
        now = time.time()
        with self._lock:
            row = self._conn.execute("SELECT times_asked FROM gaps WHERE id=?", (gid,)).fetchone()
            if row:
                times = row[0] + 1
                self._conn.execute("UPDATE gaps SET times_asked=?, entropy_before=?, last_updated=? WHERE id=?",
                                   (times, current_entropy, now, gid))
            else:
                times = 1
                score = self._curiosity_score(current_entropy, times, question)
                self._conn.execute("""INSERT INTO gaps VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                                   (gid, domain, question, current_entropy, current_entropy,
                                    times, 0, 0.0, score, now, now))
            self._conn.execute("""INSERT INTO domain_entropy(domain,entropy,observations) VALUES(?,?,1)
                ON CONFLICT(domain) DO UPDATE SET
                entropy=0.15*excluded.entropy+0.85*entropy,
                observations=observations+1""", (domain, current_entropy))
            self._conn.commit()
        score = self._curiosity_score(current_entropy, times, question)
        return {"gap_id": gid, "domain": domain, "curiosity_score": round(score, 4),
                "novelty": round(self._novelty(times), 4), "times_asked": times}

    def information_gain(self, domain: str, answer: str) -> dict[str, Any]:
        """Computes info gain for a domain answer; updates EMA gain and domain entropy."""
        tokens = re.findall(r'\\w+', answer.lower())
        freq: dict[str, int] = {}
        for t in tokens:
            freq[t] = freq.get(t, 0) + 1
        total = max(len(tokens), 1)
        entropy_after = -sum((c/total) * math.log2(c/total + 1e-12) for c in freq.values())
        entropy_after = min(entropy_after, 10.0)
        entropy_before = self._domain_entropy(domain)
        gain = max(0.0, entropy_before - entropy_after)
        self._ema_gain = 0.15 * gain + 0.85 * self._ema_gain
        with self._lock:
            self._conn.execute("""UPDATE domain_entropy SET entropy=?,observations=observations+1
                WHERE domain=?""", (entropy_after, domain))
            self._conn.execute("""UPDATE gaps SET entropy_after=?, info_gain=?,
                curiosity_score=curiosity_score*0.5, last_updated=?
                WHERE domain=? AND resolved=0""", (entropy_after, gain, time.time(), domain))
            self._conn.commit()
        ci_low = max(0.0, gain - 0.3 * self._ema_gain)
        ci_high = gain + 0.3 * self._ema_gain
        return {"domain": domain, "entropy_before": round(entropy_before, 4),
                "entropy_after": round(entropy_after, 4), "info_gain": round(gain, 4),
                "ema_gain": round(self._ema_gain, 4),
                "confidence_interval": [round(ci_low, 4), round(ci_high, 4)]}

    def most_curious(self, top_k: int = 5) -> list[dict[str, Any]]:
        """Returns top-k unresolved gaps ranked by curiosity_score × info_gain yield."""
        rows = self._conn.execute("""SELECT domain, question, curiosity_score, info_gain,
            entropy_before, times_asked FROM gaps WHERE resolved=0
            ORDER BY (curiosity_score * (1.0 + info_gain)) DESC LIMIT ?""", (top_k,)).fetchall()
        results = []
        for r in rows:
            yield_score = r[2] * (1.0 + r[3])
            results.append({"domain": r[0], "question": r[1],
                            "curiosity_score": round(r[2], 4),
                            "info_gain": round(r[3], 4),
                            "epistemic_yield": round(yield_score, 4),
                            "entropy": round(r[4], 4),
                            "times_asked": r[5]})
        return results

    def generate_questions(self, context: str, n: int = 3) -> list[dict[str, Any]]:
        """Generates n novel exploration questions scored by curiosity for the given context."""
        concepts = list(set(re.findall(r'\b[a-zA-Z]{4,}\b', context)))[:8] or ["unknown"]
        questions = []
        for tmpl in random.sample(self._question_templates, min(n, len(self._question_templates))):
            concept = random.choice(concepts)
            q = tmpl.format(concept=concept)
            domain = concept.lower()
            ent = self._domain_entropy(domain)
            times = self._conn.execute("SELECT times_asked FROM gaps WHERE id=?",
                                       (self._gap_id(domain, q),)).fetchone()
            t = times[0] if times else 0
            score = self._curiosity_score(ent, t, q, context)
            questions.append({"question": q, "domain": domain,
                               "curiosity_score": round(score, 4),
                               "novelty": round(self._novelty(t), 4)})
        questions.sort(key=lambda x: x["curiosity_score"], reverse=True)
        try:
            from nova_tools import HierarchicalGoalPlanner, MetacognitiveMonitor
            HierarchicalGoalPlanner().add_goal(f"Explore: {questions[0]['question']}", priority=7)
            MetacognitiveMonitor().log_reasoning("curiosity", "generate_questions",
                                                  questions[0]["curiosity_score"], True)
        except (ImportError, Exception):
            pass
        return questions[:n]

    def mark_resolved(self, domain: str, question: str) -> dict[str, Any]:
        """Marks a gap as resolved and returns final info gain recorded."""
        gid = self._gap_id(domain, question)
        with self._lock:
            self._conn.execute("UPDATE gaps SET resolved=1, last_updated=? WHERE id=?",
                               (time.time(), gid))
            self._conn.commit()
        row = self._conn.execute("SELECT info_gain, curiosity_score FROM gaps WHERE id=?",
                                  (gid,)).fetchone()
        return {"gap_id": gid, "resolved": True,
                "final_info_gain": round(row[0], 4) if row else 0.0,
                "final_curiosity_score": round(row[1], 4) if row else 0.0}

    def calibration_report(self) -> dict[str, Any]:
        """Returns calibration stats: mean gain vs ema, anomaly z-score on gains."""
        rows = self._conn.execute("SELECT info_gain FROM gaps WHERE resolved=1").fetchall()
        gains = [r[0] for r in rows] if rows else [0.0]
        mean_g = statistics.mean(gains)
        std_g = statistics.stdev(gains) if len(gains) > 1 else 0.0
        z = (self._ema_gain - mean_g) / (std_g + 1e-9)
        return {"mean_actual_gain": round(mean_g, 4), "ema_predicted_gain": round(self._ema_gain, 4),
                "calibration_z": round(z, 4), "anomaly": abs(z) > 3.0,
                "resolved_count": len(gains), "cycles": self._cycles}

    def status(self) -> dict[str, Any]:
        """Returns numeric status dict for ConsciousnessIntegrator Φ computation."""
        total = self._conn.execute("SELECT COUNT(*) FROM gaps").fetchone()[0]
        pending = self._conn.execute("SELECT COUNT(*) FROM gaps WHERE resolved=0").fetchone()[0]
        avg_score = self._conn.execute("SELECT AVG(curiosity_score) FROM gaps WHERE resolved=0").fetchone()[0] or 0.0
        domains = self._conn.execute("SELECT COUNT(DISTINCT domain) FROM gaps").fetchone()[0]
        ent_vals = [r[0] for r in self._conn.execute("SELECT entropy FROM domain_entropy").fetchall()] or [1.0]
        entropy = -sum((1/len(ent_vals)) * math.log2(1/len(ent_vals) + 1e-12) for _ in ent_vals)
        return {"items": total, "pending": pending, "confidence": round(avg_score, 4),
                "accuracy": round(self._ema_gain, 4), "active": domains,
                "cycles": self._cycles, "entropy": round(entropy, 4),
                "quality": round(avg_score * self._ema_gain, 4)}

    def auto_cycle(self) -> dict[str, Any]:
        """Runs one autonomous curiosity cycle; decays stale scores and logs metacognition."""
        self._cycles += 1
        with self._lock:
            self._conn.execute("""UPDATE gaps SET curiosity_score = curiosity_score * 0.97
                WHERE resolved=0 AND (? - last_updated) > 3600""", (time.time(),))
            self._conn.commit()
        top = self.most_curious(3)
        try:
            from nova_tools import MetacognitiveMonitor, NovaCuriosityDrive
            conf = top[0]["epistemic_yield"] if top else 0.0
            MetacognitiveMonitor().log_reasoning("curiosity", "auto_cycle", conf, True)
            if top:
                NovaCuriosityDrive().observe(top[0]["domain"], top[0]["question"])
        except (ImportError, Exception):
            pass
        return {"cycle": self._cycles, "top_questions": top, "ema_gain": round(self._ema_gain, 4)}

    def _auto_loop(self) -> None:
        while True:
            time.sleep(300)
            try:
                self.auto_cycle()
            except Exception:
                pass

# Usage: obj = EpistemicCuriosityEngine() | result = obj.most_curious(top_k=5)
