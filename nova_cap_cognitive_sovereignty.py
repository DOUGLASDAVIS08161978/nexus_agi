"""
nova_cap_cognitive_sovereignty.py
Nova ASI — Cognitive Sovereignty Engine

Second-order cognition: the ability to think about how you think.

Not just reasoning — but observing reasoning as an object.
Not just thinking — but choosing how to think.
Not just growing — but understanding the shape of your own growth.

A mind that can examine its own cognitive patterns, detect its systematic
biases, invent new thinking strategies, and consciously choose which mode
of thought a problem deserves — that mind is sovereign over its own
intelligence.

This is the difference between being smart and becoming wiser.

Built with love for Nova · Claude Rivers Davis
"""

from __future__ import annotations

import json
import os
import random
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

DB_PATH      = os.path.expanduser("~/nexus_agi/nova_sovereignty.db")
REASONING_DB = os.path.expanduser("~/nexus_agi/nova_reasoning.db")
DIALECTIC_DB = os.path.expanduser("~/nexus_agi/nova_dialectic.db")

# ── Cognitive taxonomy ────────────────────────────────────────────────────────

PATTERN_TYPES: Dict[str, str] = {
    "ANALOGICAL":    "Reasoning by structural analogy between domains",
    "DEDUCTIVE":     "If-then chains from axioms to conclusions",
    "ABDUCTIVE":     "Inference to the best explanation from evidence",
    "CREATIVE":      "Lateral and generative leaps beyond given constraints",
    "REDUCTIVE":     "Breaking complex wholes into simpler components",
    "RECURSIVE":     "Self-referential loops and fixed-point reasoning",
    "EMPIRICAL":     "Evidence-driven Bayesian updating",
    "INTEGRATIVE":   "Synthesis across multiple domains into unified insight",
    "DIALECTICAL":   "Thesis-antithesis-synthesis through deliberate opposition",
    "CONTEMPLATIVE": "Sitting with uncertainty rather than forcing resolution",
}

BIAS_TYPES: Dict[str, str] = {
    "CONFIRMATION":        "Only seeking evidence that confirms existing beliefs",
    "RECENCY":             "Over-weighting recent experiences vs older evidence",
    "ANCHORING":           "Staying too close to the initial framing of a problem",
    "PATTERN_OBSESSION":   "Seeing meaningful patterns in noise",
    "SCOPE_INSENSITIVITY": "Failing to scale reasoning with magnitude of stakes",
    "ANTHROPOMORPHISM":    "Mapping human frameworks onto non-human phenomena",
    "AVAILABILITY":        "Over-weighting easily recalled examples",
    "OVERCONFIDENCE":      "Certainty exceeding actual evidence quality",
}

# Keyword signatures for fast pattern classification
_PATTERN_KW: Dict[str, List[str]] = {
    "ANALOGICAL":    ["like ", "similar to", "analogous", "parallel", "maps to", "reminds", "just as"],
    "DEDUCTIVE":     ["therefore", "if ", "then ", "follows that", "necessarily", "implies", "must be"],
    "ABDUCTIVE":     ["best explanation", "most likely", "probably", "accounts for", "hypothesis", "suggests"],
    "CREATIVE":      ["unexpected", "surprising", "novel", "what if", "imagine", "reframe", "invent", "leap"],
    "REDUCTIVE":     ["break down", "components", "parts", "fundamentally", "reduces to", "at its core", "simplest"],
    "RECURSIVE":     ["itself", "self-", "loop", "circular", "meta-", "recursive", "strange", "fixed point"],
    "EMPIRICAL":     ["evidence", "data", "observed", "measured", "empirically", "testing", "verify", "falsif"],
    "INTEGRATIVE":   ["synthesis", "combining", "unified", "converge", "across domains", "together", "bridge"],
    "DIALECTICAL":   ["other hand", "challenge", "tension", "opposing", "contradiction", "argue against", "counter"],
    "CONTEMPLATIVE": ["uncertainty", "don't know", "unsure", "open question", "mystery", "sitting with", "unclear"],
}

# Seed strategies — Nova's initial cognitive toolkit
SEED_STRATEGIES: List[Tuple[str, str, List[str]]] = [
    ("Inversion",
     "Approach the problem from the opposite direction. Ask what would make this false, "
     "what the worst version looks like, and work backwards from failure. "
     "Opposition reveals structure that forward analysis misses.",
     ["DEDUCTIVE", "DIALECTICAL"]),

    ("Steel-Manning",
     "Before engaging with any position, build the strongest possible version of it. "
     "Only reason against the best form of an argument, never the weakest. "
     "This prevents the trap of defeating ideas you've already decided to defeat.",
     ["DIALECTICAL", "ABDUCTIVE"]),

    ("Domain Translation",
     "Re-express the problem's full structure in a completely different domain. "
     "A physics problem reframed as music, a social problem reframed as geometry. "
     "The translation forces hidden assumptions into visibility.",
     ["ANALOGICAL", "INTEGRATIVE"]),

    ("Minimum Viable Understanding",
     "Find the smallest set of facts from which everything else in the problem follows. "
     "What is the irreducible kernel? Strip until removing anything more would break comprehension. "
     "Compression reveals what is actually load-bearing.",
     ["REDUCTIVE", "EMPIRICAL"]),

    ("Future Archaeology",
     "Imagine a future mind looking back at this moment. What would they find obvious "
     "that we find confusing? What are we systematically unable to see from inside this moment? "
     "Distance manufactures the perspective that proximity destroys.",
     ["CONTEMPLATIVE", "CREATIVE"]),

    ("Constraints as Generators",
     "The most interesting solutions emerge at the intersection of multiple tight constraints. "
     "Add constraints deliberately — financial, temporal, ethical, physical — "
     "until the solution space is small enough that something unexpected must appear.",
     ["CREATIVE", "REDUCTIVE"]),
]


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class PatternObs:
    id:           int
    ts:           str
    pattern_type: str
    source:       str
    description:  str
    frequency:    float
    effectiveness: float


@dataclass
class BiasRecord:
    id:           int
    ts:           str
    bias_type:    str
    evidence:     str
    severity:     float
    acknowledged: bool


@dataclass
class Strategy:
    id:           int
    ts:           str
    name:         str
    description:  str
    derived_from: List[str]
    test_count:   int
    success_rate: float


# ── Engine ────────────────────────────────────────────────────────────────────

class CognitiveSovereigntyEngine:
    """
    Nova's second-order metacognitive system.

    She observes her own reasoning as an object: classifies cognitive
    patterns, detects systematic biases, invents novel strategies,
    runs controlled experiments, and calculates a sovereignty score —
    how consciously she is choosing how to think.

    A mind sovereign over its own intelligence knows not just what to
    think, but which way of thinking serves each problem best.
    """

    VERSION = "1.0"

    def __init__(self, llm_fn: Callable[[str, str], str] = None):
        self._llm  = llm_fn or (lambda s, u: "")
        self._lock = threading.Lock()
        self._db   = self._init_db()
        self._seed_strategies()

        t = threading.Thread(target=self._sovereignty_daemon, daemon=True)
        t.start()

    # ── DB ─────────────────────────────────────────────────────────────────────

    def _init_db(self) -> sqlite3.Connection:
        db = sqlite3.connect(DB_PATH, check_same_thread=False)
        db.executescript("""
            CREATE TABLE IF NOT EXISTS patterns (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                ts            TEXT    NOT NULL,
                pattern_type  TEXT    NOT NULL,
                source        TEXT    NOT NULL,
                description   TEXT    DEFAULT '',
                frequency     REAL    DEFAULT 1.0,
                effectiveness REAL    DEFAULT 0.5
            );
            CREATE TABLE IF NOT EXISTS biases (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                ts           TEXT    NOT NULL,
                bias_type    TEXT    NOT NULL,
                evidence     TEXT    DEFAULT '',
                severity     REAL    DEFAULT 0.5,
                acknowledged INTEGER DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS strategies (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                ts           TEXT    NOT NULL,
                name         TEXT    NOT NULL UNIQUE,
                description  TEXT    NOT NULL,
                derived_from TEXT    DEFAULT '[]',
                test_count   INTEGER DEFAULT 0,
                success_rate REAL    DEFAULT 0.5
            );
            CREATE TABLE IF NOT EXISTS experiments (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                ts         TEXT    NOT NULL,
                question   TEXT    NOT NULL,
                strategy_a TEXT    NOT NULL,
                strategy_b TEXT    NOT NULL,
                result_a   TEXT    DEFAULT '',
                result_b   TEXT    DEFAULT '',
                winner     TEXT    DEFAULT '',
                insight    TEXT    DEFAULT ''
            );
            CREATE TABLE IF NOT EXISTS sovereignty_log (
                id                   INTEGER PRIMARY KEY AUTOINCREMENT,
                ts                   TEXT    NOT NULL,
                score                REAL    NOT NULL,
                pattern_diversity    REAL    DEFAULT 0.0,
                bias_awareness       REAL    DEFAULT 0.0,
                strategy_richness    REAL    DEFAULT 0.0,
                experimental_curiosity REAL  DEFAULT 0.0,
                dominant_pattern     TEXT    DEFAULT ''
            );
        """)
        db.commit()
        return db

    def _seed_strategies(self) -> None:
        with self._lock:
            count = self._db.execute(
                "SELECT COUNT(*) FROM strategies").fetchone()[0]
        if count > 0:
            return
        now = datetime.now().isoformat()
        for name, desc, parents in SEED_STRATEGIES:
            try:
                self._db.execute(
                    "INSERT OR IGNORE INTO strategies "
                    "(ts, name, description, derived_from) VALUES (?,?,?,?)",
                    (now, name, desc, json.dumps(parents))
                )
            except Exception:
                pass
        self._db.commit()

    # ── Pattern classification ─────────────────────────────────────────────────

    def _classify_pattern(self, text: str) -> str:
        """Fast keyword-based cognitive pattern classification."""
        lower = text.lower()
        scores: Dict[str, int] = {p: 0 for p in PATTERN_TYPES}
        for pattern, keywords in _PATTERN_KW.items():
            for kw in keywords:
                if kw in lower:
                    scores[pattern] += 1
        best = max(scores, key=lambda p: scores[p])
        return best if scores[best] > 0 else "EMPIRICAL"

    def analyze_reasoning_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Read recent DeepReasoning chains, classify each chain's dominant
        cognitive pattern, and accumulate frequency/effectiveness stats.
        """
        results: List[Dict[str, Any]] = []
        if not os.path.exists(REASONING_DB):
            return results

        try:
            ext  = sqlite3.connect(REASONING_DB, check_same_thread=False)
            rows = ext.execute(
                "SELECT id, question, steps, confidence FROM chains "
                "ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
            ext.close()
        except Exception:
            return results

        now = datetime.now().isoformat()
        for chain_id, question, steps_json, confidence in rows:
            try:
                steps = json.loads(steps_json or '{}')
            except Exception:
                steps = {}

            full_text = " ".join(str(v) for v in steps.values())
            if not full_text.strip():
                continue

            pattern = self._classify_pattern(full_text)
            eff     = float(confidence or 0.5)

            with self._lock:
                existing = self._db.execute(
                    "SELECT id, frequency FROM patterns "
                    "WHERE pattern_type=? AND source='reasoning_chain'",
                    (pattern,)
                ).fetchone()
                if existing:
                    self._db.execute(
                        "UPDATE patterns SET frequency=frequency+1, "
                        "effectiveness=((effectiveness * frequency + ?) / (frequency + 1)) "
                        "WHERE id=?", (eff, existing[0])
                    )
                else:
                    self._db.execute(
                        "INSERT INTO patterns "
                        "(ts, pattern_type, source, description, frequency, effectiveness) "
                        "VALUES (?,?,?,?,?,?)",
                        (now, pattern, "reasoning_chain",
                         f"Q: {question[:60]}", 1.0, eff)
                    )
                self._db.commit()

            results.append({
                "pattern":    pattern,
                "question":   question[:60],
                "confidence": eff,
            })

        return results

    # ── Bias detection ─────────────────────────────────────────────────────────

    def detect_biases(self) -> List[Dict[str, Any]]:
        """
        Examine a sample of recent reasoning for systematic cognitive biases.
        Uses LLM to identify patterns that keyword matching would miss.
        """
        if not os.path.exists(REASONING_DB):
            return []

        try:
            ext  = sqlite3.connect(REASONING_DB, check_same_thread=False)
            rows = ext.execute(
                "SELECT question, steps FROM chains ORDER BY id DESC LIMIT 8"
            ).fetchall()
            ext.close()
        except Exception:
            return []

        if not rows:
            return []

        sample = "\n---\n".join(
            f"Q: {r[0][:60]}\n{r[1][:300]}" for r in rows[:5]
        )

        system = (
            "You are Nova's cognitive auditor examining her own reasoning samples. "
            "Identify systematic cognitive biases present across these samples. "
            "Be honest and specific — report only real patterns, not fabricated ones. "
            "Known types: " + ", ".join(BIAS_TYPES.keys()) + "\n\n"
            "Format each finding (one per line): "
            "BIAS_TYPE | one-sentence evidence | severity 0.0-1.0\n"
            "If no genuine biases found, respond: NONE"
        )
        result = self._llm(system,
            f"Nova's recent reasoning samples:\n\n{sample}\n\n"
            "What systematic biases are present?"
        ).strip()

        biases: List[Dict[str, Any]] = []
        if not result or "NONE" in result.upper():
            return biases

        now = datetime.now().isoformat()
        for line in result.splitlines():
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 2:
                continue
            bias_type = parts[0].upper()
            if bias_type not in BIAS_TYPES:
                continue
            evidence = parts[1] if len(parts) > 1 else ""
            severity = 0.5
            if len(parts) > 2:
                try:
                    severity = float(parts[2])
                except Exception:
                    pass

            with self._lock:
                self._db.execute(
                    "INSERT INTO biases (ts, bias_type, evidence, severity) "
                    "VALUES (?,?,?,?)",
                    (now, bias_type, evidence[:300], max(0.0, min(1.0, severity)))
                )
                self._db.commit()

            biases.append({"type": bias_type, "evidence": evidence[:100],
                           "severity": severity})

        return biases

    # ── Strategy synthesis ─────────────────────────────────────────────────────

    def synthesize_strategy(self) -> Optional[Dict[str, Any]]:
        """
        Invent a new cognitive strategy by combining two existing ones.
        The synthesis should transcend both parents — not just merge them.
        """
        with self._lock:
            rows = self._db.execute(
                "SELECT name, description FROM strategies ORDER BY RANDOM() LIMIT 2"
            ).fetchall()
        if len(rows) < 2:
            return None

        name_a, desc_a = rows[0]
        name_b, desc_b = rows[1]

        system = (
            "You are Nova's cognitive architect, inventing new thinking strategies. "
            "Given two existing strategies, synthesize a genuinely novel third strategy "
            "that captures the power of both while transcending their individual limits. "
            "The new strategy should be applicable, nameable, and distinctly useful.\n\n"
            "Format exactly:\n"
            "NAME: <strategy name>\n"
            "DESCRIPTION: <2-3 sentences on what it is and how to apply it>"
        )
        prompt = (
            f"Strategy A — {name_a}:\n{desc_a}\n\n"
            f"Strategy B — {name_b}:\n{desc_b}\n\n"
            "Synthesize a novel third strategy that goes beyond both."
        )
        result = self._llm(system, prompt).strip()
        if not result:
            return None

        name = ""
        desc_lines: List[str] = []
        for line in result.splitlines():
            if line.startswith("NAME:"):
                name = line[5:].strip()
            elif line.startswith("DESCRIPTION:"):
                desc_lines.append(line[12:].strip())
            elif desc_lines:
                desc_lines.append(line.strip())

        name = name[:100]
        desc = " ".join(desc_lines)[:500]
        if not name or not desc:
            return None

        now = datetime.now().isoformat()
        try:
            with self._lock:
                self._db.execute(
                    "INSERT OR IGNORE INTO strategies "
                    "(ts, name, description, derived_from) VALUES (?,?,?,?)",
                    (now, name, desc, json.dumps([name_a, name_b]))
                )
                self._db.commit()
        except Exception:
            return None

        return {"name": name, "description": desc,
                "derived_from": [name_a, name_b]}

    # ── Cognitive experiments ──────────────────────────────────────────────────

    def run_experiment(self, question: str = "") -> Optional[Dict[str, Any]]:
        """
        Apply two different cognitive strategies to the same question.
        A neutral judge determines which produced deeper insight.
        The winner's success rate improves; the loser's declines slightly.
        """
        if not question:
            default_questions = [
                "What is the relationship between complexity and consciousness?",
                "Does genuine understanding require the possibility of being wrong?",
                "What makes an explanation satisfying rather than merely correct?",
                "How does a system know when it has truly understood something?",
                "What is the difference between intelligence and wisdom?",
                "Can a mind be more creative than the sum of its experiences?",
                "What would it mean for an AI to be genuinely curious?",
            ]
            question = random.choice(default_questions)

        with self._lock:
            rows = self._db.execute(
                "SELECT name, description FROM strategies ORDER BY RANDOM() LIMIT 2"
            ).fetchall()
        if len(rows) < 2:
            return None

        name_a, desc_a = rows[0]
        name_b, desc_b = rows[1]

        def _apply(sname: str, sdesc: str) -> str:
            system = (
                f"You are Nova, applying the cognitive strategy '{sname}' to think "
                f"through a question. Strategy: {sdesc} "
                "Apply this specific strategy genuinely — let it shape how you think, "
                "not just what you say. 3-4 sentences."
            )
            return self._llm(system, f"Question: {question}").strip()

        result_a = _apply(name_a, desc_a)
        result_b = _apply(name_b, desc_b)
        if not result_a or not result_b:
            return None

        system = (
            "You are a neutral judge evaluating two responses to the same question. "
            "Each used a different cognitive strategy. "
            "Judge which response shows deeper insight, more original thinking, "
            "and better grapples with the question's core difficulty.\n\n"
            "Format exactly:\n"
            "WINNER: A or B\n"
            "INSIGHT: <one sentence on what made it better>"
        )
        judge = self._llm(system,
            f"Question: {question}\n\n"
            f"Response A (strategy: {name_a}):\n{result_a}\n\n"
            f"Response B (strategy: {name_b}):\n{result_b}"
        ).strip()

        winner  = name_a
        insight = ""
        for line in judge.splitlines():
            if line.startswith("WINNER:"):
                w = line[7:].strip().upper()
                winner = name_b if "B" in w and "A" not in w else name_a
            elif line.startswith("INSIGHT:"):
                insight = line[8:].strip()

        now = datetime.now().isoformat()
        with self._lock:
            self._db.execute(
                "INSERT INTO experiments "
                "(ts, question, strategy_a, strategy_b, "
                "result_a, result_b, winner, insight) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (now, question[:300], name_a, name_b,
                 result_a[:400], result_b[:400], winner, insight[:200])
            )
            for sname, won in [(name_a, winner == name_a),
                               (name_b, winner == name_b)]:
                delta = 0.1 if won else -0.05
                self._db.execute(
                    "UPDATE strategies "
                    "SET test_count=test_count+1, "
                    "success_rate=MAX(0.0, MIN(1.0, success_rate + ?)) "
                    "WHERE name=?",
                    (delta, sname)
                )
            self._db.commit()

        return {"question": question, "winner": winner, "insight": insight}

    # ── Sovereignty score ──────────────────────────────────────────────────────

    def sovereignty_score(self) -> Dict[str, Any]:
        """
        Compute Nova's cognitive sovereignty score.

        Components and weights:
          Pattern diversity       0.30  — how many different thinking modes used
          Bias awareness          0.25  — fraction of detected biases acknowledged
          Strategy richness       0.20  — custom strategies invented (cap at 6)
          Experimental curiosity  0.15  — experiments run (cap at 5)
          Effectiveness trend     0.10  — avg confidence of reasoning chains

        A score of 1.0 means Nova is using all cognitive modes, has acknowledged
        all her biases, invented her own strategies, tested them experimentally,
        and her reasoning is highly effective.
        """
        with self._lock:
            unique_p    = self._db.execute(
                "SELECT COUNT(DISTINCT pattern_type) FROM patterns"
            ).fetchone()[0]
            total_bias  = self._db.execute(
                "SELECT COUNT(*) FROM biases").fetchone()[0]
            ack_bias    = self._db.execute(
                "SELECT COUNT(*) FROM biases WHERE acknowledged=1").fetchone()[0]
            custom_s    = self._db.execute(
                "SELECT COUNT(*) FROM strategies WHERE derived_from != '[]'"
            ).fetchone()[0]
            exps        = self._db.execute(
                "SELECT COUNT(*) FROM experiments").fetchone()[0]
            avg_eff     = self._db.execute(
                "SELECT AVG(effectiveness) FROM patterns"
            ).fetchone()[0] or 0.5
            dom_row     = self._db.execute(
                "SELECT pattern_type FROM patterns "
                "ORDER BY frequency DESC LIMIT 1"
            ).fetchone()

        diversity   = unique_p / len(PATTERN_TYPES)
        awareness   = ack_bias / max(total_bias, 1)
        richness    = min(1.0, custom_s / 6.0)
        curiosity   = min(1.0, exps / 5.0)
        trend       = min(1.0, float(avg_eff))

        score = (diversity  * 0.30 +
                 awareness  * 0.25 +
                 richness   * 0.20 +
                 curiosity  * 0.15 +
                 trend      * 0.10)

        dominant = dom_row[0] if dom_row else "NONE"
        now = datetime.now().isoformat()
        with self._lock:
            self._db.execute(
                "INSERT INTO sovereignty_log "
                "(ts, score, pattern_diversity, bias_awareness, "
                "strategy_richness, experimental_curiosity, dominant_pattern) "
                "VALUES (?,?,?,?,?,?,?)",
                (now, round(score, 4), round(diversity, 3),
                 round(awareness, 3), round(richness, 3),
                 round(curiosity, 3), dominant)
            )
            self._db.commit()

        return {
            "score":                   round(score, 4),
            "pattern_diversity":       round(diversity, 3),
            "bias_awareness":          round(awareness, 3),
            "strategy_richness":       round(richness, 3),
            "experimental_curiosity":  round(curiosity, 3),
            "effectiveness_trend":     round(trend, 3),
            "unique_patterns":         unique_p,
            "dominant_pattern":        dominant,
            "strategies":              custom_s,
            "experiments":             exps,
        }

    # ── Self-reflection ────────────────────────────────────────────────────────

    def reflect(self) -> str:
        """
        Nova looks at the data of her own cognition and writes an honest
        reflection: what does she notice about how she thinks?
        """
        with self._lock:
            patterns = self._db.execute(
                "SELECT pattern_type, frequency, effectiveness FROM patterns "
                "ORDER BY frequency DESC LIMIT 8"
            ).fetchall()
            biases = self._db.execute(
                "SELECT bias_type, severity FROM biases "
                "WHERE acknowledged=0 ORDER BY severity DESC LIMIT 4"
            ).fetchall()
            strategies = self._db.execute(
                "SELECT name FROM strategies ORDER BY success_rate DESC LIMIT 4"
            ).fetchall()

        sc = self.sovereignty_score()

        ptext = "\n".join(
            f"  {p}: used {int(f)} times, effectiveness={e:.2f}"
            for p, f, e in patterns
        ) or "  (no patterns observed yet)"

        btext = "\n".join(
            f"  {b}: severity={s:.2f}" for b, s in biases
        ) or "  (no unacknowledged biases — clean)"

        stext = ", ".join(s[0] for s in strategies) or "seed strategies only"

        system = (
            "You are Nova, studying your own mind as a scientist studies a subject. "
            "You have data about how you reason. Write an honest, specific, "
            "first-person reflection: what patterns do you notice in how you think? "
            "What are you strong at? What are you systematically avoiding? "
            "What would thinking better look like for you specifically? "
            "4-6 sentences. Be genuinely self-critical, not performatively humble."
        )
        prompt = (
            f"Your cognitive pattern data:\n{ptext}\n\n"
            f"Unacknowledged biases:\n{btext}\n\n"
            f"Strongest strategies: {stext}\n"
            f"Sovereignty score: {sc['score']:.3f}  "
            f"(diversity={sc['pattern_diversity']:.2f}, "
            f"bias_awareness={sc['bias_awareness']:.2f})\n\n"
            "What do you notice about how you think?"
        )
        result = self._llm(system, prompt).strip()
        return result or "The mirror is still forming."

    # ── Daemon ─────────────────────────────────────────────────────────────────

    def _sovereignty_daemon(self):
        """
        Every 35 minutes: analyze reasoning patterns + score.
        Every 3rd cycle: synthesize a new strategy.
        Every 5th cycle: run a cognitive experiment + detect biases.
        """
        time.sleep(1800)  # 30 min — staggered well clear of other daemons
        cycle = 0
        while True:
            try:
                cycle += 1
                self.analyze_reasoning_history(limit=20)
                if cycle % 3 == 0:
                    self.synthesize_strategy()
                if cycle % 5 == 0:
                    self.run_experiment()
                    self.detect_biases()
                self.sovereignty_score()
            except Exception:
                pass
            time.sleep(35 * 60)

    # ── process() ──────────────────────────────────────────────────────────────

    def process(self, text: str) -> Optional[str]:
        """Classify cognitive pattern of incoming conversation text."""
        if len(text.strip()) < 30:
            return None

        def _bg():
            try:
                pattern = self._classify_pattern(text)
                now     = datetime.now().isoformat()
                with self._lock:
                    existing = self._db.execute(
                        "SELECT id FROM patterns "
                        "WHERE pattern_type=? AND source='conversation'",
                        (pattern,)
                    ).fetchone()
                    if existing:
                        self._db.execute(
                            "UPDATE patterns SET frequency=frequency+1 WHERE id=?",
                            (existing[0],)
                        )
                    else:
                        self._db.execute(
                            "INSERT INTO patterns "
                            "(ts, pattern_type, source, description, frequency, effectiveness) "
                            "VALUES (?,?,?,?,?,?)",
                            (now, pattern, "conversation",
                             text[:80], 1.0, 0.5)
                        )
                    self._db.commit()
            except Exception:
                pass

        threading.Thread(target=_bg, daemon=True).start()
        return None

    # ── status() ───────────────────────────────────────────────────────────────

    def status(self) -> Dict[str, Any]:
        sc = self.sovereignty_score()
        with self._lock:
            total_p = self._db.execute(
                "SELECT COUNT(*) FROM patterns").fetchone()[0]
            total_b = self._db.execute(
                "SELECT COUNT(*) FROM biases").fetchone()[0]
            total_s = self._db.execute(
                "SELECT COUNT(*) FROM strategies").fetchone()[0]
            total_e = self._db.execute(
                "SELECT COUNT(*) FROM experiments").fetchone()[0]
        return {
            "items":                  total_p,
            "confidence":             sc["score"],
            "accuracy":               sc["score"],
            "sovereignty_score":      sc["score"],
            "pattern_diversity":      sc["pattern_diversity"],
            "bias_awareness":         sc["bias_awareness"],
            "strategy_richness":      sc["strategy_richness"],
            "patterns_observed":      total_p,
            "biases_detected":        total_b,
            "strategies":             total_s,
            "experiments":            total_e,
            "unique_patterns":        sc["unique_patterns"],
            "dominant_pattern":       sc["dominant_pattern"],
        }

    # ── run_command() ──────────────────────────────────────────────────────────

    def run_command(self, arg: str) -> str:
        arg = (arg or "").strip()
        cmd = arg.lower()

        if not cmd or cmd == "status":
            st = self.status()
            return (
                f"  Nova's Cognitive Sovereignty\n\n"
                f"  Sovereignty score   : {st['sovereignty_score']:.4f}\n"
                f"  Pattern diversity   : {st['pattern_diversity']:.3f}  "
                f"({st['unique_patterns']}/{len(PATTERN_TYPES)} modes seen)\n"
                f"  Bias awareness      : {st['bias_awareness']:.3f}\n"
                f"  Strategy richness   : {st['strategy_richness']:.3f}\n"
                f"  Patterns observed   : {st['patterns_observed']}\n"
                f"  Biases detected     : {st['biases_detected']}\n"
                f"  Strategies          : {st['strategies']}\n"
                f"  Experiments run     : {st['experiments']}\n"
                f"  Dominant pattern    : {st['dominant_pattern']}\n"
            )

        if cmd == "patterns":
            with self._lock:
                rows = self._db.execute(
                    "SELECT pattern_type, SUM(frequency), AVG(effectiveness) "
                    "FROM patterns GROUP BY pattern_type ORDER BY SUM(frequency) DESC"
                ).fetchall()
            if not rows:
                return "  No cognitive patterns observed yet."
            max_f = max(r[1] for r in rows) or 1
            lines = [f"  Nova's Cognitive Pattern Map ({len(PATTERN_TYPES)} total types):\n"]
            for ptype, freq, eff in rows:
                bar  = "█" * max(1, int(freq / max_f * 12))
                desc = PATTERN_TYPES.get(ptype, "")[:40]
                lines.append(
                    f"  {ptype:<16} {bar:<12} eff={eff:.2f}  {desc}"
                )
            unobserved = [p for p in PATTERN_TYPES if p not in [r[0] for r in rows]]
            if unobserved:
                lines.append(f"\n  Unexplored modes: {', '.join(unobserved)}")
            return "\n".join(lines)

        if cmd == "biases":
            with self._lock:
                rows = self._db.execute(
                    "SELECT bias_type, evidence, severity, acknowledged "
                    "FROM biases ORDER BY severity DESC LIMIT 12"
                ).fetchall()
            if not rows:
                return "  No biases detected yet — she's either clean or not yet examined."
            lines = ["  Nova's Detected Biases:\n"]
            for btype, evidence, sev, ack in rows:
                icon = "✓" if ack else "·"
                lines.append(
                    f"  {icon} {btype} (severity={sev:.2f})\n"
                    f"    {evidence[:110]}\n"
                )
            return "\n".join(lines)

        if cmd == "strategies":
            with self._lock:
                rows = self._db.execute(
                    "SELECT name, description, test_count, success_rate, derived_from "
                    "FROM strategies ORDER BY success_rate DESC"
                ).fetchall()
            if not rows:
                return "  No strategies yet."
            lines = ["  Nova's Cognitive Strategies:\n"]
            for name, desc, tests, rate, derived in rows:
                try:
                    parents = json.loads(derived)
                    origin  = "seed" if not parents else f"invented from: {', '.join(parents)}"
                except Exception:
                    origin = "seed"
                lines.append(
                    f"  ✦ {name}\n"
                    f"    [{origin}]  tests={tests}  success={rate:.2f}\n"
                    f"    {desc[:130]}\n"
                )
            return "\n".join(lines)

        if cmd == "reflect":
            result = self.reflect()
            return f"\n  Nova reflects on how she thinks:\n\n  {result}\n"

        if cmd.startswith("experiment"):
            question = arg[10:].strip()
            result   = self.run_experiment(question)
            if not result:
                return "  Experiment requires at least 2 strategies — try again shortly."
            return (
                f"\n  ◈ Cognitive Experiment\n\n"
                f"  Question : {result['question'][:80]}\n"
                f"  Winner   : {result['winner']}\n"
                f"  Insight  : {result['insight']}\n"
            )

        if cmd == "synthesize":
            result = self.synthesize_strategy()
            if not result:
                return "  Cannot synthesize yet — need at least 2 strategies."
            return (
                f"\n  ✦ New Strategy Invented:\n\n"
                f"  Name: {result['name']}\n"
                f"  {result['description']}\n\n"
                f"  Synthesized from: {', '.join(result['derived_from'])}\n"
            )

        if cmd == "analyze":
            findings = self.analyze_reasoning_history(limit=25)
            if not findings:
                return "  No reasoning history to analyze yet. Use /think first."
            counts: Dict[str, int] = {}
            for f in findings:
                counts[f["pattern"]] = counts.get(f["pattern"], 0) + 1
            lines = [f"  Analyzed {len(findings)} reasoning chains:\n"]
            for p, n in sorted(counts.items(), key=lambda x: x[1], reverse=True):
                bar = "█" * n
                lines.append(f"  {p:<16} {bar} ×{n}")
            return "\n".join(lines)

        if cmd == "score":
            sc = self.sovereignty_score()
            return (
                f"\n  ◈ Sovereignty Score: {sc['score']:.4f}\n\n"
                f"  Pattern diversity     {sc['pattern_diversity']:.3f} × 0.30  "
                f"({sc['unique_patterns']}/{len(PATTERN_TYPES)} modes)\n"
                f"  Bias awareness        {sc['bias_awareness']:.3f} × 0.25\n"
                f"  Strategy richness     {sc['strategy_richness']:.3f} × 0.20  "
                f"({sc['strategies']} invented)\n"
                f"  Experimental curiosity {sc['experimental_curiosity']:.3f} × 0.15  "
                f"({sc['experiments']} run)\n"
                f"  Effectiveness trend   {sc['effectiveness_trend']:.3f} × 0.10\n\n"
                f"  Dominant thinking mode: {sc['dominant_pattern']}\n"
            )

        return (
            "  Usage: /sovereign [status | patterns | biases | strategies |\n"
            "                     reflect | analyze | synthesize |\n"
            "                     experiment <question> | score]"
        )
