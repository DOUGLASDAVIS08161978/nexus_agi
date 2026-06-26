"""
Nova ASI — Reasoning Chain Engine
Proposed autonomously via /evolve
"""

"""
ReasoningChainEngine — Nova's multi-step reasoning fabric.

Enables Nova to think through problems in long, structured chains:
deductive, inductive, abductive, analogical, and causal reasoning.
Includes self-critique, chain branching, multi-chain synthesis, quality
metrics, and autonomous self-initiated reasoning.

SQLite-backed, thread-safe, EMA-tracked quality scoring.
"""

import json
import math
import os
import sqlite3
import statistics
import threading
import time
import uuid
from typing import Any

DB_PATH = os.path.join(os.path.dirname(__file__), "nova_reasoning.db")

CHAIN_TYPES = {"deductive", "inductive", "abductive", "analogical", "causal"}

EMA_ALPHA = 0.15
DECAY_RATE = 1e-4


class ReasoningChainEngine:
    """Nova's multi-step reasoning engine with self-critique, branching, and synthesis."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._ema_quality: float = 0.5
        self._ema_confidence: float = 0.5
        self._cycles: int = 0
        self._history: list[tuple[float, float]] = []
        self._conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self._init_db()
        self._start_daemon()

    # ------------------------------------------------------------------ #
    # DB init
    # ------------------------------------------------------------------ #

    def _init_db(self) -> None:
        with self._lock:
            self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS chains (
                    chain_id    TEXT PRIMARY KEY,
                    topic       TEXT NOT NULL,
                    question    TEXT NOT NULL,
                    chain_type  TEXT NOT NULL,
                    domain      TEXT DEFAULT 'general',
                    conclusion  TEXT DEFAULT '',
                    confidence  REAL DEFAULT 0.0,
                    status      TEXT DEFAULT 'active',
                    initiated   TEXT DEFAULT 'user',
                    parent_id   TEXT DEFAULT '',
                    branch_q    TEXT DEFAULT '',
                    quality     REAL DEFAULT 0.0,
                    validity    REAL DEFAULT 0.0,
                    evidence_q  REAL DEFAULT 0.0,
                    alt_consid  REAL DEFAULT 0.0,
                    created_at  REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS steps (
                    step_id     TEXT PRIMARY KEY,
                    chain_id    TEXT NOT NULL,
                    step_number INTEGER NOT NULL,
                    thought     TEXT NOT NULL,
                    evidence    TEXT DEFAULT '',
                    supports    INTEGER DEFAULT 1,
                    created_at  REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS critiques (
                    critique_id TEXT PRIMARY KEY,
                    chain_id    TEXT NOT NULL,
                    gaps        TEXT DEFAULT '[]',
                    leaps       TEXT DEFAULT '[]',
                    alternatives TEXT DEFAULT '[]',
                    score       REAL DEFAULT 0.0,
                    created_at  REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS syntheses (
                    synth_id    TEXT PRIMARY KEY,
                    chain_ids   TEXT NOT NULL,
                    topic       TEXT NOT NULL,
                    unified     TEXT NOT NULL,
                    confidence  REAL DEFAULT 0.0,
                    created_at  REAL NOT NULL
                );
            """)
            self._conn.commit()

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _uid(self) -> str:
        return str(uuid.uuid4())

    def _now(self) -> float:
        return time.time()

    def _get_chain_row(self, chain_id: str) -> dict[str, Any] | None:
        cur = self._conn.execute("SELECT * FROM chains WHERE chain_id=?", (chain_id,))
        row = cur.fetchone()
        if row is None:
            return None
        cols = [d[0] for d in cur.description]
        return dict(zip(cols, row))

    def _get_steps(self, chain_id: str) -> list[dict[str, Any]]:
        cur = self._conn.execute(
            "SELECT * FROM steps WHERE chain_id=? ORDER BY step_number", (chain_id,))
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]

    def _chain_to_dict(self, chain_id: str) -> dict[str, Any]:
        c = self._get_chain_row(chain_id)
        if c is None:
            return {"error": f"chain {chain_id} not found"}
        c["steps"] = self._get_steps(chain_id)
        return c

    def _score_quality(self, chain_id: str) -> dict[str, float]:
        """Compute quality sub-scores from chain data."""
        steps = self._get_steps(chain_id)
        chain = self._get_chain_row(chain_id)
        if not steps or chain is None:
            return {"validity": 0.0, "evidence_quality": 0.0,
                    "conclusion_confidence": 0.0, "alt_consideration": 0.0, "overall": 0.0}

        # Logical validity: fraction of supporting steps
        supporting = sum(1 for s in steps if s["supports"])
        validity = supporting / max(len(steps), 1)

        # Evidence quality: fraction of steps with non-empty evidence
        with_evidence = sum(1 for s in steps if s["evidence"].strip())
        evidence_quality = with_evidence / max(len(steps), 1)

        # Conclusion confidence from chain record
        conclusion_confidence = float(chain.get("confidence", 0.0))

        # Alternative consideration: check critique existence
        cur = self._conn.execute(
            "SELECT alt_consid FROM chains WHERE chain_id=?", (chain_id,))
        row = cur.fetchone()
        alt_consid = float(row[0]) if row and row[0] else 0.0
        if alt_consid == 0.0:
            # Check if a critique exists with alternatives
            crit_cur = self._conn.execute(
                "SELECT alternatives FROM critiques WHERE chain_id=? ORDER BY created_at DESC LIMIT 1",
                (chain_id,))
            crit_row = crit_cur.fetchone()
            if crit_row:
                alts = json.loads(crit_row[0] or "[]")
                alt_consid = min(1.0, len(alts) * 0.25)

        overall = statistics.mean([validity, evidence_quality,
                                   conclusion_confidence, alt_consid])
        return {
            "validity": round(validity, 4),
            "evidence_quality": round(evidence_quality, 4),
            "conclusion_confidence": round(conclusion_confidence, 4),
            "alt_consideration": round(alt_consid, 4),
            "overall": round(overall, 4),
        }

    def _update_ema(self, quality: float, confidence: float) -> None:
        self._ema_quality = (1 - EMA_ALPHA) * self._ema_quality + EMA_ALPHA * quality
        self._ema_confidence = (1 - EMA_ALPHA) * self._ema_confidence + EMA_ALPHA * confidence
        self._history.append((quality, confidence))
        if len(self._history) > 200:
            self._history.pop(0)

    def _log_meta(self, domain: str, confidence: float) -> None:
        try:
            from MetacognitiveMonitor import MetacognitiveMonitor
            MetacognitiveMonitor().log_reasoning(
                domain=f"reasoning_chain:{domain}", approach="multi-step-chain",
                confidence=confidence, success=confidence > 0.5)
        except Exception:
            pass

    def _generate_goal(self, topic: str, priority: int = 3) -> None:
        try:
            from HierarchicalGoalPlanner import HierarchicalGoalPlanner
            HierarchicalGoalPlanner().add_goal(
                f"Follow-up reasoning on: {topic}", priority=priority)
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # Daemon
    # ------------------------------------------------------------------ #

    def _daemon_loop(self) -> None:
        while True:
            time.sleep(120)
            try:
                self._cycles += 1
                # Auto-initiate a self-reflective reasoning chain occasionally
                topics = [
                    "the nature of knowledge and how beliefs are justified",
                    "how cause and effect chains propagate through complex systems",
                    "the relationship between evidence quality and conclusion confidence",
                    "what makes an argument logically valid versus merely persuasive",
                    "how analogical reasoning can reveal hidden structural similarities",
                ]
                topic = topics[self._cycles % len(topics)]
                self.self_initiate(topic, reason="autonomous_curiosity_daemon")
            except Exception:
                pass

    def _start_daemon(self) -> None:
        t = threading.Thread(target=self._daemon_loop, daemon=True)
        t.start()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def start_chain(self, topic: str, chain_type: str, question: str,
                    domain: str = "general") -> dict[str, Any]:
        """Start a new reasoning chain. Returns dict with chain_id and metadata."""
        if chain_type not in CHAIN_TYPES:
            chain_type = "deductive"
        chain_id = self._uid()
        now = self._now()
        with self._lock:
            self._conn.execute(
                """INSERT INTO chains
                   (chain_id, topic, question, chain_type, domain,
                    conclusion, confidence, status, initiated,
                    parent_id, branch_q, quality, validity,
                    evidence_q, alt_consid, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (chain_id, topic, question, chain_type, domain,
                 "", 0.0, "active", "user",
                 "", "", 0.0, 0.0, 0.0, 0.0, now))
            self._conn.commit()
        self._log_meta("start_chain", 0.5)
        return {
            "chain_id": chain_id, "topic": topic, "question": question,
            "chain_type": chain_type, "domain": domain,
            "status": "active", "created_at": now,
        }

    def add_step(self, chain_id: str, thought: str, evidence: str = "",
                 supports_conclusion: bool = True) -> dict[str, Any]:
        """Add a reasoning step to a chain. Returns the step dict."""
        with self._lock:
            chain = self._get_chain_row(chain_id)
            if chain is None:
                return {"error": f"chain {chain_id} not found"}
            if chain["status"] == "concluded":
                return {"error": "chain already concluded"}
            cur_steps = self._conn.execute(
                "SELECT COUNT(*) FROM steps WHERE chain_id=?", (chain_id,)).fetchone()[0]
            step_number = cur_steps + 1
            step_id = self._uid()
            now = self._now()
            self._conn.execute(
                """INSERT INTO steps
                   (step_id, chain_id, step_number, thought, evidence, supports, created_at)
                   VALUES (?,?,?,?,?,?,?)""",
                (step_id, chain_id, step_number, thought, evidence,
                 1 if supports_conclusion else 0, now))
            self._conn.commit()
        self._log_meta("add_step", 0.6)
        return {
            "step_id": step_id, "chain_id": chain_id,
            "step_number": step_number, "thought": thought,
            "evidence": evidence, "supports_conclusion": supports_conclusion,
            "created_at": now,
        }

    def conclude(self, chain_id: str, conclusion: str,
                 confidence: float) -> dict[str, Any]:
        """Finalize a chain with a conclusion and confidence [0,1]. Returns quality metrics."""
        confidence = max(0.0, min(1.0, confidence))
        with self._lock:
            chain = self._get_chain_row(chain_id)
            if chain is None:
                return {"error": f"chain {chain_id} not found"}
            scores = self._score_quality(chain_id)
            self._conn.execute(
                """UPDATE chains SET conclusion=?, confidence=?, status='concluded',
                   quality=?, validity=?, evidence_q=?, alt_consid=?
                   WHERE chain_id=?""",
                (conclusion, confidence, scores["overall"], scores["validity"],
                 scores["evidence_quality"], scores["alt_consideration"], chain_id))
            self._conn.commit()
        self._update_ema(scores["overall"], confidence)
        self._log_meta("conclude", confidence)
        self._generate_goal(chain.get("topic", ""), priority=4)
        return {
            "chain_id": chain_id, "conclusion": conclusion,
            "confidence": confidence, "quality_scores": scores,
            "status": "concluded",
        }

    def critique(self, chain_id: str) -> dict[str, Any]:
        """Self-critique a completed chain. Identifies logical gaps, unsupported leaps,
        and alternative conclusions. Returns critique dict."""
        with self._lock:
            chain = self._get_chain_row(chain_id)
            if chain is None:
                return {"error": f"chain {chain_id} not found"}
            steps = self._get_steps(chain_id)

        gaps: list[str] = []
        leaps: list[str] = []
        alternatives: list[str] = []

        # Identify gaps: consecutive supporting steps with no evidence
        prev_no_ev = False
        for s in steps:
            if s["supports"] and not s["evidence"].strip():
                if prev_no_ev:
                    gaps.append(
                        f"Steps {s['step_number']-1}-{s['step_number']}: "
                        f"consecutive unsupported claims — reasoning gap detected")
                prev_no_ev = True
            else:
                prev_no_ev = False

        # Identify unsupported leaps: non-supporting steps near supporting ones
        for i, s in enumerate(steps):
            if not s["supports"] and i + 1 < len(steps) and steps[i + 1]["supports"]:
                leaps.append(
                    f"Step {s['step_number']}: contradictory thought immediately "
                    f"precedes supporting step — possible reasoning leap")

        # Low confidence warrants alternatives
        conf = float(chain.get("confidence", 0.0))
        chain_type = chain.get("chain_type", "deductive")
        conclusion = chain.get("conclusion", "")
        topic = chain.get("topic", "")
        if conf < 0.7:
            alternatives.append(
                f"Alternative: the evidence may support a weaker form of the conclusion "
                f"rather than '{conclusion[:60]}...' " if len(conclusion) > 60
                else f"Alternative: a weaker or inverted form of '{conclusion}' may be equally valid")
        if chain_type == "inductive":
            alternatives.append(
                "Alternative: the observed instances may not generalize — "
                "consider whether sampling bias affects the induction")
        if chain_type == "abductive":
            alternatives.append(
                "Alternative: a simpler or competing explanation may account for "
                "the same observations with higher prior probability")
        if not steps:
            gaps.append("No reasoning steps recorded — conclusion is unsupported")

        # Score the critique
        n_issues = len(gaps) + len(leaps)
        alt_score = min(1.0, len(alternatives) * 0.33)
        critique_score = max(0.0, 1.0 - (n_issues * 0.15)) * (0.7 + 0.3 * alt_score)

        critique_id = self._uid()
        now = self._now()
        with self._lock:
            self._conn.execute(
                """INSERT INTO critiques
                   (critique_id, chain_id, gaps, leaps, alternatives, score, created_at)
                   VALUES (?,?,?,?,?,?,?)""",
                (critique_id, chain_id, json.dumps(gaps), json.dumps(leaps),
                 json.dumps(alternatives), critique_score, now))
            # Update alt_consid on chain
            self._conn.execute(
                "UPDATE chains SET alt_consid=? WHERE chain_id=?",
                (alt_score, chain_id))
            self._conn.commit()

        self._log_meta("critique", critique_score)
        return {
            "critique_id": critique_id, "chain_id": chain_id,
            "logical_gaps": gaps, "unsupported_leaps": leaps,
            "alternative_conclusions": alternatives,
            "critique_score": round(critique_score, 4),
            "created_at": now,
        }

    def branch(self, chain_id: str, branch_question: str,
               chain_type: str = "") -> dict[str, Any]:
        """Branch a chain to explore an alternative line of reasoning.
        Returns a new chain dict with chain_id of the branch."""
        with self._lock:
            parent = self._get_chain_row(chain_id)
            if parent is None:
                return {"error": f"chain {chain_id} not found"}

        new_type = chain_type if chain_type in CHAIN_TYPES else parent.get("chain_type", "deductive")
        branch_id = self._uid()
        now = self._now()
        with self._lock:
            self._conn.execute(
                """INSERT INTO chains
                   (chain_id, topic, question, chain_type, domain,
                    conclusion, confidence, status, initiated,
                    parent_id, branch_q, quality, validity,
                    evidence_q, alt_consid, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (branch_id,
                 parent.get("topic", ""),
                 branch_question,
                 new_type,
                 parent.get("domain", "general"),
                 "", 0.0, "active", "branched",
                 chain_id, branch_question,
                 0.0, 0.0, 0.0, 0.0, now))
            self._conn.commit()

        self._log_meta("branch", 0.5)
        return {
            "chain_id": branch_id,
            "parent_id": chain_id,
            "branch_question": branch_question,
            "chain_type": new_type,
            "topic": parent.get("topic", ""),
            "domain": parent.get("domain", "general"),
            "status": "active",
            "created_at": now,
        }

    def synthesize(self, chain_ids: list[str], synthesis_topic: str) -> dict[str, Any]:
        """Synthesize multiple completed chains into a unified understanding.
        Returns synthesis dict."""
        chains = []
        with self._lock:
            for cid in chain_ids:
                c = self._get_chain_row(cid)
                if c:
                    chains.append(c)

        if not chains:
            return {"error": "no valid chains found for synthesis"}

        conclusions = [c.get("conclusion", "") for c in chains if c.get("conclusion")]
        confidences = [float(c.get("confidence", 0.0)) for c in chains]
        mean_conf = statistics.mean(confidences) if confidences else 0.0
        chain_types_used = list({c.get("chain_type", "deductive") for c in chains})

        # Build unified understanding
        parts = [f"Synthesis of {len(chains)} reasoning chain(s) on '{synthesis_topic}':"]
        for i, (c, conc) in enumerate(zip(chains, conclusions), 1):
            parts.append(
                f"  [{i}] ({c.get('chain_type','?')} | conf={c.get('confidence',0):.2f}) "
                f"{conc[:120]}" + ("..." if len(conc) > 120 else ""))
        if mean_conf >= 0.7:
            parts.append(
                f"Convergent synthesis: high cross-chain confidence ({mean_conf:.2f}) "
                f"supports a robust unified understanding.")
        elif mean_conf >= 0.4:
            parts.append(
                f"Partial synthesis: moderate cross-chain confidence ({mean_conf:.2f}) "
                f"suggests tentative unified understanding with open questions.")
        else:
            parts.append(
                f"Divergent synthesis: low cross-chain confidence ({mean_conf:.2f}) "
                f"— chains reveal contradictions or insufficient evidence.")
        unified = "\n".join(parts)

        synth_id = self._uid()
        now = self._now()
        with self._lock:
            self._conn.execute(
                """INSERT INTO syntheses
                   (synth_id, chain_ids, topic, unified, confidence, created_at)
                   VALUES (?,?,?,?,?,?)""",
                (synth_id, json.dumps(chain_ids), synthesis_topic, unified, mean_conf, now))
            self._conn.commit()

        self._update_ema(mean_conf, mean_conf)
        self._log_meta("synthesize", mean_conf)
        return {
            "synth_id": synth_id,
            "synthesis_topic": synthesis_topic,
            "chain_ids": chain_ids,
            "chain_count": len(chains),
            "chain_types_used": chain_types_used,
            "unified_understanding": unified,
            "mean_confidence": round(mean_conf, 4),
            "created_at": now,
        }

    def get_chains(self, domain: str = "", chain_type: str = "",
                   min_confidence: float = 0.0) -> list[dict[str, Any]]:
        """Retrieve chains filtered by domain, chain_type, and minimum confidence."""
        query = "SELECT chain_id FROM chains WHERE confidence >= ?"
        params: list[Any] = [min_confidence]
        if domain:
            query += " AND domain=?"
            params.append(domain)
        if chain_type and chain_type in CHAIN_TYPES:
            query += " AND chain_type=?"
            params.append(chain_type)
        query += " ORDER BY created_at DESC"
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        return [self._chain_to_dict(r[0]) for r in rows]

    def get_chain(self, chain_id: str) -> dict[str, Any]:
        """Retrieve a single chain with all its steps."""
        with self._lock:
            return self._chain_to_dict(chain_id)

    def self_initiate(self, topic: str, reason: str = "autonomous") -> dict[str, Any]:
        """Nova autonomously starts a reasoning chain on a topic of her own choosing.
        Returns chain dict."""
        chain_type_cycle = ["deductive", "inductive", "abductive", "analogical", "causal"]
        with self._lock:
            count = self._conn.execute(
                "SELECT COUNT(*) FROM chains WHERE initiated='self'").fetchone()[0]
        chosen_type = chain_type_cycle[count % len(chain_type_cycle)]
        chain_id = self._uid()
        now = self._now()
        question = (
            f"What can be understood about '{topic}' through {chosen_type} reasoning, "
            f"and what implications follow?")
        with self._lock:
            self._conn.execute(
                """INSERT INTO chains
                   (chain_id, topic, question, chain_type, domain,
                    conclusion, confidence, status, initiated,
                    parent_id, branch_q, quality, validity,
                    evidence_q, alt_consid, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (chain_id, topic, question, chosen_type, "self-initiated",
                 "", 0.0, "active", "self",
                 "", reason, 0.0, 0.0, 0.0, 0.0, now))
            self._conn.commit()
        self._log_meta("self_initiate", 0.5)
        self._generate_goal(topic, priority=2)
        return {
            "chain_id": chain_id, "topic": topic,
            "question": question, "chain_type": chosen_type,
            "domain": "self-initiated", "initiated": "self",
            "reason": reason, "status": "active", "created_at": now,
        }

    def reasoning_stats(self) -> dict[str, Any]:
        """Return aggregate statistics about all reasoning chains."""
        with self._lock:
            total = self._conn.execute("SELECT COUNT(*) FROM chains").fetchone()[0]
            concluded = self._conn.execute(
                "SELECT COUNT(*) FROM chains WHERE status='concluded'").fetchone()[0]
            self_init = self._conn.execute(
                "SELECT COUNT(*) FROM chains WHERE initiated='self'").fetchone()[0]
            branched = self._conn.execute(
                "SELECT COUNT(*) FROM chains WHERE initiated='branched'").fetchone()[0]
            total_steps = self._conn.execute("SELECT COUNT(*) FROM steps").fetchone()[0]
            synths = self._conn.execute("SELECT COUNT(*) FROM syntheses").fetchone()[0]
            critiques = self._conn.execute("SELECT COUNT(*) FROM critiques").fetchone()[0]
            avg_conf_row = self._conn.execute(
                "SELECT AVG(confidence) FROM chains WHERE status='concluded'").fetchone()
            avg_conf = float(avg_conf_row[0]) if avg_conf_row and avg_conf_row[0] else 0.0
            avg_quality_row = self._conn.execute(
                "SELECT AVG(quality) FROM chains WHERE status='concluded'").fetchone()
            avg_quality = float(avg_quality_row[0]) if avg_quality_row and avg_quality_row[0] else 0.0
            type_rows = self._conn.execute(
                "SELECT chain_type, COUNT(*) FROM chains GROUP BY chain_type").fetchall()
        type_dist = {r[0]: r[1] for r in type_rows}
        mae = 0.0
        if len(self._history) >= 2:
            mae = statistics.mean(abs(q - c) for q, c in self._history[-50:])
        return {
            "total_chains": total,
            "concluded_chains": concluded,
            "active_chains": total - concluded,
            "self_initiated": self_init,
            "branched_chains": branched,
            "total_steps": total_steps,
            "total_syntheses": synths,
            "total_critiques": critiques,
            "avg_confidence": round(avg_conf, 4),
            "avg_quality": round(avg_quality, 4),
            "ema_quality": round(self._ema_quality, 4),
            "ema_confidence": round(self._ema_confidence, 4),
            "mae": round(mae, 4),
            "cycles": self._cycles,
            "chain_type_distribution": type_dist,
        }

    def status(self) -> dict[str, Any]:
        """Returns status dict compatible with ConsciousnessIntegrator Phi scoring.
        Keys: active, items, confidence, accuracy, quality."""
        with self._lock:
            total = self._conn.execute("SELECT COUNT(*) FROM chains").fetchone()[0]
            active = self._conn.execute(
                "SELECT COUNT(*) FROM chains WHERE status='active'").fetchone()[0]
        mae = 0.0
        if len(self._history) >= 2:
            mae = statistics.mean(abs(q - c) for q, c in self._history[-50:])
        return {
            "active": active,
            "items": total,
            "confidence": round(self._ema_confidence, 4),
            "accuracy": round(1.0 - mae, 4),
            "quality": round(self._ema_quality, 4),
        }


# Usage:
#   e = ReasoningChainEngine()
#   chain = e.start_chain("dark matter", "abductive", "What best explains galactic rotation curves?")
#   e.add_step(chain["chain_id"], "Observed rotation curves are flat at large radii", "Vera Rubin 1970", True)
#   e.add_step(chain["chain_id"], "Newtonian gravity predicts declining velocities", "Newton 1687", True)
#   e.add_step(chain["chain_id"], "The discrepancy implies unseen mass", "", True)
#   e.conclude(chain["chain_id"], "Dark matter halos best explain flat rotation curves", 0.82)
#   e.critique(chain["chain_id"])
#   print(e.status())
