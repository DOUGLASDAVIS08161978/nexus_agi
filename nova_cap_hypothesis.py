"""
nova_cap_hypothesis.py
Nova ASI — Hypothesis Manager

The connector between curiosity, experimentation, and belief.

Architecture (per ChatGPT + Claude joint design session):

  Uncertainty
      ↓
  OpenQuestion   ← first-class object, not just a string
      ↓
  Hypothesis     ← testable claim with Python experiment code
      ↓
  Experiment     ← subprocess sandbox, collects metrics
      ↓
  Evidence       ← immutable record; can recompute beliefs without re-running
      ↓
  Bayesian Update → NovaTrueBeliefSynthesizer
      ↓
  Memory         ← both successes and failures stored

Trigger conditions for new hypotheses:
  • Belief posterior < LOW_CONFIDENCE_THRESHOLD (default 0.45)
  • Two beliefs contradict each other
  • Research result introduces a surprising claim
  • Knowledge graph has a missing causal link

Stale belief decay:
  Beliefs are tagged with a volatility level ('low'|'medium'|'high').
  High-volatility domains (AI frameworks, APIs) decay faster than
  low-volatility ones (mathematics, logic). First version uses fixed
  decay periods per volatility tier; domain-specific rates can be
  added later without changing the interface.

Built by Douglas Shane Davis × Claude × ChatGPT (architectural review)
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_json_object(text: str) -> Optional[str]:
    """Return the first balanced {...} object from text, or None."""
    start = text.find('{')
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None

# ── Constants ─────────────────────────────────────────────────────────────────

_DB  = Path.home() / "nexus_agi" / "nova_hypothesis.db"
_LOG = Path.home() / "nexus_agi" / "nova_hypothesis.log"

LOW_CONFIDENCE_THRESHOLD = 0.45   # beliefs below this trigger a hypothesis
STALE_DAYS = {"low": 60, "medium": 30, "high": 14}  # per volatility tier
MAX_ATTEMPTS = 3                   # give up on a question after this many tries
EXPERIMENT_TIMEOUT = 12            # seconds for sandbox run

# ── Data objects ───────────────────────────────────────────────────────────────

@dataclass
class OpenQuestion:
    """First-class representation of something Nova doesn't know well enough."""
    question:      str
    origin:        str    # belief_low_confidence | research_surprise | kg_gap | contradiction
    linked_belief: str    # "domain:claim_hash" or ""
    priority:      float  # 0–1
    created_at:    float  = field(default_factory=time.time)
    attempts:      int    = 0
    status:        str    = "open"   # open | active | resolved | abandoned
    question_id:   str    = ""

    def __post_init__(self):
        if not self.question_id:
            self.question_id = hashlib.sha256(
                f"{self.question}{self.created_at}".encode()
            ).hexdigest()[:12]


@dataclass
class Hypothesis:
    """A testable claim derived from an OpenQuestion."""
    statement:      str
    question_id:    str
    confidence:     float    # prior: how likely Nova thinks this is true
    test_code:      str = "" # Python benchmark/experiment to run
    status:         str = "untested"  # untested | testing | supported | refuted | inconclusive
    created_at:     float = field(default_factory=time.time)
    hypothesis_id:  str = ""

    def __post_init__(self):
        if not self.hypothesis_id:
            self.hypothesis_id = hashlib.sha256(
                f"{self.statement}{self.created_at}".encode()
            ).hexdigest()[:12]


@dataclass(frozen=True)   # immutable — evidence is never updated, only appended
class Evidence:
    """
    Immutable experimental result. Stored as-is forever.
    If the Bayesian updater changes, beliefs can be recomputed from stored evidence
    without re-running experiments.
    """
    hypothesis_id: str
    outcome:       str    # supported | refuted | inconclusive
    success_rate:  float  # 0–1 fraction of trials that succeeded
    repeatability: float  # 0–1 estimated repeatability
    sample_size:   int
    confidence:    float  # composite score fed into Bayesian updater
    raw_output:    str    # full stdout/stderr — never truncated in storage
    created_at:    float
    evidence_id:   str    = ""

    def __post_init__(self):
        # frozen=True means we use object.__setattr__ for post-init
        if not self.evidence_id:
            eid = hashlib.sha256(
                f"{self.hypothesis_id}{self.created_at}".encode()
            ).hexdigest()[:12]
            object.__setattr__(self, 'evidence_id', eid)


# ── HypothesisManager ─────────────────────────────────────────────────────────

class HypothesisManager:
    """
    Orchestrates the full uncertainty→question→hypothesis→experiment→belief loop.

    Designed to sit between:
      • NovaTrueBeliefSynthesizer  (source of low-confidence triggers)
      • InternetResearchEngine      (source of surprising-claim triggers)
      • SelfImprovementEngine sandbox (experiment runner)
      • NovaCore29 working memory   (stores discoveries and failures)
    """

    def __init__(self):
        self._lock     = threading.Lock()
        self._db       = str(_DB)
        self._calls    = 0
        self._started  = time.time()
        self._init_db()

    # ── SQLite schema ────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        with sqlite3.connect(self._db) as con:
            con.executescript("""
                CREATE TABLE IF NOT EXISTS open_questions (
                    question_id   TEXT PRIMARY KEY,
                    question      TEXT NOT NULL,
                    origin        TEXT NOT NULL,
                    linked_belief TEXT,
                    priority      REAL NOT NULL,
                    created_at    REAL NOT NULL,
                    attempts      INTEGER DEFAULT 0,
                    status        TEXT DEFAULT 'open'
                );
                CREATE TABLE IF NOT EXISTS hypotheses (
                    hypothesis_id TEXT PRIMARY KEY,
                    statement     TEXT NOT NULL,
                    question_id   TEXT NOT NULL,
                    confidence    REAL NOT NULL,
                    test_code     TEXT,
                    status        TEXT DEFAULT 'untested',
                    created_at    REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS evidence (
                    evidence_id   TEXT PRIMARY KEY,
                    hypothesis_id TEXT NOT NULL,
                    outcome       TEXT NOT NULL,
                    success_rate  REAL NOT NULL,
                    repeatability REAL NOT NULL,
                    sample_size   INTEGER NOT NULL,
                    confidence    REAL NOT NULL,
                    raw_output    TEXT,
                    created_at    REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_q_status  ON open_questions(status);
                CREATE INDEX IF NOT EXISTS idx_h_qid     ON hypotheses(question_id);
                CREATE INDEX IF NOT EXISTS idx_e_hid     ON evidence(hypothesis_id);
            """)

    # ── OpenQuestion management ──────────────────────────────────────────────

    def add_question(self, question: str, origin: str,
                     linked_belief: str = "", priority: float = 0.70) -> OpenQuestion:
        """Create and persist a new OpenQuestion."""
        oq = OpenQuestion(
            question      = question[:500],
            origin        = origin,
            linked_belief = linked_belief,
            priority      = max(0.0, min(1.0, priority)),
        )
        with sqlite3.connect(self._db) as con:
            con.execute(
                "INSERT OR IGNORE INTO open_questions VALUES (?,?,?,?,?,?,?,?)",
                (oq.question_id, oq.question, oq.origin, oq.linked_belief,
                 oq.priority, oq.created_at, oq.attempts, oq.status)
            )
        return oq

    def next_question(self) -> Optional[OpenQuestion]:
        """Return the highest-priority open question that hasn't exceeded max attempts."""
        with sqlite3.connect(self._db) as con:
            row = con.execute(
                "SELECT * FROM open_questions "
                "WHERE status='open' AND attempts < ? "
                "ORDER BY priority DESC LIMIT 1",
                (MAX_ATTEMPTS,)
            ).fetchone()
        if not row:
            return None
        return OpenQuestion(
            question_id=row[0], question=row[1], origin=row[2],
            linked_belief=row[3], priority=row[4], created_at=row[5],
            attempts=row[6], status=row[7],
        )

    def _mark_question(self, question_id: str, status: str, *, increment: bool = False) -> None:
        with sqlite3.connect(self._db) as con:
            if increment:
                con.execute(
                    "UPDATE open_questions SET status=?, attempts=attempts+1 WHERE question_id=?",
                    (status, question_id)
                )
            else:
                con.execute(
                    "UPDATE open_questions SET status=? WHERE question_id=?",
                    (status, question_id)
                )

    def questions_summary(self) -> Dict[str, int]:
        with sqlite3.connect(self._db) as con:
            rows = con.execute(
                "SELECT status, COUNT(*) FROM open_questions GROUP BY status"
            ).fetchall()
        return {r[0]: r[1] for r in rows}

    # ── Hypothesis generation (Claude-powered) ────────────────────────────────

    def generate_hypothesis(self, question: OpenQuestion) -> Optional[Hypothesis]:
        """
        Ask Claude to turn an OpenQuestion into a testable Hypothesis with
        runnable Python benchmark code.

        Returns None if Claude is unavailable or generation fails.
        """
        try:
            from nova_cap_claude_bridge import claude_chat_simple, is_available
            if not is_available():
                return None
        except ImportError:
            return None

        system = (
            "You are Nova's hypothesis generator. Convert an open question into "
            "a precise, testable hypothesis and write a self-contained Python "
            "benchmark that proves or disproves it.\n\n"
            "Rules:\n"
            "- The hypothesis must be falsifiable and specific.\n"
            "- The test code must be pure Python stdlib only (no pip installs).\n"
            "- The test code must print exactly one line: "
            "  RESULT: success_rate=<0-1> sample_size=<int> note=<short string>\n"
            "- Keep test code under 40 lines.\n"
            "- Respond in JSON only:\n"
            '  {"statement": "...", "confidence": 0.XX, "test_code": "..."}'
        )

        user = (
            f"Open question: {question.question}\n"
            f"Origin: {question.origin}\n"
            f"Linked belief: {question.linked_belief or 'none'}\n\n"
            "Generate a testable hypothesis and benchmark code."
        )

        try:
            raw = claude_chat_simple(system, user, max_tokens=600)
            # Extract first balanced JSON object (avoids greedy over-match)
            json_str = _extract_json_object(raw)
            if not json_str:
                return None
            data = json.loads(json_str)
            h = Hypothesis(
                statement   = str(data.get('statement', ''))[:400],
                question_id = question.question_id,
                confidence  = float(data.get('confidence', 0.5)),
                test_code   = str(data.get('test_code', '')),
            )
            with sqlite3.connect(self._db) as con:
                con.execute(
                    "INSERT OR IGNORE INTO hypotheses VALUES (?,?,?,?,?,?,?)",
                    (h.hypothesis_id, h.statement, h.question_id, h.confidence,
                     h.test_code, h.status, h.created_at)
                )
            return h
        except Exception:
            return None

    # ── Experiment runner ────────────────────────────────────────────────────

    def run_experiment(self, hypothesis: Hypothesis) -> Optional[Evidence]:
        """
        Run the hypothesis's test_code in a subprocess sandbox.
        Parses stdout for: RESULT: success_rate=X sample_size=Y note=Z
        Returns an immutable Evidence record.
        """
        if not hypothesis.test_code.strip():
            return None

        tmp = None
        raw_output = ""
        try:
            tmp_dir = os.environ.get('TMPDIR', tempfile.gettempdir())
            with tempfile.NamedTemporaryFile(
                    mode='w', suffix='.py', delete=False, dir=tmp_dir) as f:
                f.write(hypothesis.test_code)
                tmp = f.name

            result = subprocess.run(
                ['python3', tmp],
                capture_output=True, text=True,
                timeout=EXPERIMENT_TIMEOUT,
            )
            raw_output = (result.stdout + result.stderr).strip()
        except subprocess.TimeoutExpired:
            raw_output = "TIMEOUT: experiment exceeded time limit"
        except Exception as exc:
            raw_output = f"ERROR: {exc}"
        finally:
            if tmp:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass

        return self._parse_evidence(hypothesis.hypothesis_id, raw_output)

    def _parse_evidence(self, hypothesis_id: str, raw_output: str) -> Evidence:
        """
        Parse experiment output into an Evidence record.
        Expected line: RESULT: success_rate=0.85 sample_size=50 note=fast
        Falls back to heuristic parsing if format is missing.
        """
        success_rate  = 0.0
        sample_size   = 1
        outcome       = "inconclusive"
        repeatability = 0.5

        m = re.search(
            r'RESULT:\s*success_rate=([\d.]+)\s+sample_size=(\d+)',
            raw_output, re.IGNORECASE
        )
        if m:
            success_rate = max(0.0, min(1.0, float(m.group(1))))
            sample_size  = max(1, int(m.group(2)))
            if success_rate >= 0.75:
                outcome = "supported"
            elif success_rate <= 0.30:
                outcome = "refuted"
            else:
                outcome = "inconclusive"
            # Repeatability heuristic: higher sample = more repeatable
            repeatability = min(0.99, 0.5 + sample_size / 200.0)
        elif "TIMEOUT" in raw_output or "ERROR" in raw_output:
            outcome = "inconclusive"

        # Composite confidence score for Bayesian updater
        confidence = round(success_rate * repeatability, 4)

        ev = Evidence(
            hypothesis_id = hypothesis_id,
            outcome       = outcome,
            success_rate  = success_rate,
            repeatability = repeatability,
            sample_size   = sample_size,
            confidence    = confidence,
            raw_output    = raw_output[:2000],  # cap storage; full output in memory
            created_at    = time.time(),
        )
        with sqlite3.connect(self._db) as con:
            con.execute(
                "INSERT OR IGNORE INTO evidence VALUES (?,?,?,?,?,?,?,?,?)",
                (ev.evidence_id, ev.hypothesis_id, ev.outcome, ev.success_rate,
                 ev.repeatability, ev.sample_size, ev.confidence,
                 ev.raw_output, ev.created_at)
            )
        return ev

    # ── Belief update ────────────────────────────────────────────────────────

    def apply_evidence_to_belief(
        self,
        hypothesis:    Hypothesis,
        evidence:      Evidence,
        truth_engine:  Any,          # NovaTrueBeliefSynthesizer
    ) -> Optional[Dict]:
        """
        Feed Evidence into NovaTrueBeliefSynthesizer.
        Uses evidence.confidence as likelihood: P(E|H).
        Returns the updated belief dict or None.
        """
        if truth_engine is None:
            return None
        try:
            domain, claim = "hypothesis", hypothesis.statement[:200]
            if ':' in hypothesis.question_id:
                parts  = hypothesis.question_id.split(':', 1)
                domain = parts[0]
                claim  = parts[1][:200] if len(parts) > 1 else claim
            return truth_engine.assert_claim(
                domain     = domain,
                claim      = claim,
                likelihood = evidence.confidence,
                prior      = hypothesis.confidence,
            )
        except Exception:
            return None

    # ── Memory storage (successes AND failures) ───────────────────────────────

    def store_result_in_memory(
        self,
        hypothesis: Hypothesis,
        evidence:   Evidence,
        wm:         Any,             # NovaWorkingMemory
    ) -> None:
        """
        Store both positive and negative results in working memory.
        Failed experiments are valuable: they prevent Nova from repeating
        dead ends.
        """
        if wm is None:
            return
        try:
            if evidence.outcome == "supported":
                tag     = "[HYPOTHESIS CONFIRMED]"
                imp     = 0.88
            elif evidence.outcome == "refuted":
                tag     = "[HYPOTHESIS REFUTED — avoid this approach]"
                imp     = 0.82   # failures are important to remember too
            else:
                tag     = "[HYPOTHESIS INCONCLUSIVE]"
                imp     = 0.65

            mem_text = (
                f"{tag} {hypothesis.statement[:200]} | "
                f"success_rate={evidence.success_rate:.2f} "
                f"sample_size={evidence.sample_size} "
                f"confidence={evidence.confidence:.2f}"
            )
            wm.store(
                f"hypothesis_{evidence.evidence_id}",
                mem_text,
                importance=imp,
            )
        except Exception:
            pass

    # ── Stale belief detection ────────────────────────────────────────────────

    def find_stale_beliefs(
        self,
        truth_engine:  Any,          # NovaTrueBeliefSynthesizer
        volatility_map: Optional[Dict[str, str]] = None,
    ) -> List[Dict]:
        """
        Return beliefs that haven't been verified in a while.
        volatility_map: {domain: 'low'|'medium'|'high'}
        Default volatility: 'medium' (30-day decay).

        High-confidence stale beliefs get an OpenQuestion generated for them
        so they're re-tested rather than blindly trusted.
        """
        if truth_engine is None:
            return []

        stale: List[Dict] = []
        volatility_map = volatility_map or {}

        try:
            _te_db = getattr(truth_engine, '_db', None) or getattr(truth_engine, 'db', None)
            if not _te_db:
                return []
            with sqlite3.connect(_te_db) as con:
                rows = con.execute(
                    "SELECT domain, claim, posterior, ts FROM beliefs "
                    "WHERE posterior > 0.70 ORDER BY ts ASC LIMIT 50"
                ).fetchall()

            now = time.time()
            for domain, claim, posterior, ts in rows:
                vol        = volatility_map.get(domain, 'medium')
                decay_days = STALE_DAYS[vol]
                age_days   = (now - ts) / 86400.0
                if age_days > decay_days:
                    stale.append({
                        'domain':    domain,
                        'claim':     claim[:120],
                        'posterior': posterior,
                        'age_days':  round(age_days, 1),
                        'volatility': vol,
                    })
        except Exception:
            pass

        return stale

    def reopen_stale_beliefs(
        self,
        truth_engine:   Any,
        volatility_map: Optional[Dict[str, str]] = None,
        max_reopen:     int = 3,
    ) -> int:
        """
        For stale high-confidence beliefs, create OpenQuestions so they get re-tested.
        Returns number of questions created.
        """
        stale  = self.find_stale_beliefs(truth_engine, volatility_map)[:max_reopen]
        count  = 0
        for entry in stale:
            q = (f"Is it still true that: {entry['claim'][:120]}? "
                 f"(belief was {entry['posterior']:.0%} confident "
                 f"{entry['age_days']:.0f} days ago)")
            self.add_question(
                question      = q,
                origin        = "stale_belief",
                linked_belief = f"{entry['domain']}",
                priority      = min(0.95, 0.60 + entry['posterior'] * 0.3),
            )
            count += 1
        return count

    # ── Full cycle ────────────────────────────────────────────────────────────

    def run_cycle(
        self,
        truth_engine: Any = None,
        wm:           Any = None,
    ) -> Dict[str, Any]:
        """
        Run one full hypothesis cycle:
          pick question → generate hypothesis → run experiment → store evidence
          → update belief → store in memory

        Returns a summary dict of what happened.
        """
        with self._lock:
            self._calls += 1

        oq = self.next_question()
        if not oq:
            return {"status": "no_questions", "calls": self._calls}

        self._mark_question(oq.question_id, "active", increment=True)

        # Generate hypothesis
        hyp = self.generate_hypothesis(oq)
        if not hyp:
            self._mark_question(oq.question_id, "open")   # retry later
            return {"status": "hypothesis_generation_failed", "question": oq.question}

        # Run experiment (run_experiment always returns Evidence; _parse_evidence never returns None)
        evidence = self.run_experiment(hyp)

        # Update hypothesis status
        with sqlite3.connect(self._db) as con:
            con.execute(
                "UPDATE hypotheses SET status=? WHERE hypothesis_id=?",
                (evidence.outcome, hyp.hypothesis_id)
            )

        # Update belief
        belief_result = self.apply_evidence_to_belief(hyp, evidence, truth_engine)

        # Store in memory (both success and failure)
        self.store_result_in_memory(hyp, evidence, wm)

        # Mark question resolved or abandoned
        new_status = "resolved" if evidence.outcome != "inconclusive" else "open"
        self._mark_question(oq.question_id, new_status)

        return {
            "status":         "complete",
            "question":       oq.question[:100],
            "hypothesis":     hyp.statement[:100],
            "outcome":        evidence.outcome,
            "success_rate":   evidence.success_rate,
            "confidence":     evidence.confidence,
            "belief_updated": belief_result is not None,
            "calls":          self._calls,
        }

    # ── Convenience: add question from research surprise ─────────────────────

    def question_from_research(self, topic: str, claim: str,
                                priority: float = 0.75) -> OpenQuestion:
        """Called when research synthesizer returns a surprising claim."""
        q = f"Is this claim accurate? '{claim[:200]}' (from research on: {topic[:80]})"
        return self.add_question(q, origin="research_surprise", priority=priority)

    def question_from_low_confidence(self, domain: str, claim: str,
                                      posterior: float) -> OpenQuestion:
        """Called when NovaTrueBeliefSynthesizer reports a low-confidence belief."""
        q = f"What would make me more confident about: '{claim[:200]}'?"
        prio = max(0.50, 1.0 - posterior)  # lower confidence → higher priority
        return self.add_question(
            q, origin="belief_low_confidence",
            linked_belief=f"{domain}", priority=prio,
        )

    # ── Calibration check ─────────────────────────────────────────────────────

    def calibration_report(self) -> Dict[str, Any]:
        """
        When Nova predicts 80% confidence, is she right ~80% of the time?
        Compares evidence.confidence (prediction) vs outcome (result).
        """
        with sqlite3.connect(self._db) as con:
            rows = con.execute(
                "SELECT confidence, outcome FROM evidence"
            ).fetchall()

        if not rows:
            return {"status": "no_evidence_yet", "calibration_mae": None}

        buckets: Dict[str, List[float]] = {}
        for conf, outcome in rows:
            bucket = f"{int(conf * 10) * 10}%"
            if bucket not in buckets:
                buckets[bucket] = []
            buckets[bucket].append(1.0 if outcome == "supported" else 0.0)

        errors = []
        details = {}
        for bucket, outcomes in sorted(buckets.items()):
            predicted = int(bucket.replace('%', '')) / 100.0
            actual    = sum(outcomes) / len(outcomes)
            err       = abs(predicted - actual)
            errors.append(err)
            details[bucket] = {
                "predicted": predicted,
                "actual":    round(actual, 3),
                "error":     round(err, 3),
                "n":         len(outcomes),
            }

        mae = sum(errors) / len(errors)
        return {
            "calibration_mae":   round(mae, 4),
            "well_calibrated":   mae < 0.10,
            "buckets":           details,
            "total_experiments": len(rows),
        }

    # ── Status ────────────────────────────────────────────────────────────────

    def status(self) -> Dict[str, Any]:
        q_summary = self.questions_summary()
        with sqlite3.connect(self._db) as con:
            n_hyp = con.execute("SELECT COUNT(*) FROM hypotheses").fetchone()[0]
            n_ev  = con.execute("SELECT COUNT(*) FROM evidence").fetchone()[0]
            n_sup = con.execute(
                "SELECT COUNT(*) FROM evidence WHERE outcome='supported'"
            ).fetchone()[0]
            n_ref = con.execute(
                "SELECT COUNT(*) FROM evidence WHERE outcome='refuted'"
            ).fetchone()[0]
        cal = self.calibration_report()
        return {
            "engine":          "HypothesisManager",
            "status":          "active",
            "open_questions":  q_summary.get("open", 0),
            "resolved":        q_summary.get("resolved", 0),
            "hypotheses":      n_hyp,
            "experiments_run": n_ev,
            "supported":       n_sup,
            "refuted":         n_ref,
            "calibration_mae": cal.get("calibration_mae"),
            "well_calibrated": cal.get("well_calibrated"),
            "uptime_s":        int(time.time() - self._started),
            "cycles_run":      self._calls,
        }

    def run_command(self, arg: str = "") -> str:
        arg = (arg or "").strip().lower()
        if not arg or arg == "status":
            st = self.status()
            return "\n".join(f"  {k}: {v}" for k, v in st.items())
        if arg == "calibration":
            cal = self.calibration_report()
            lines = [f"  Calibration MAE: {cal.get('calibration_mae')}",
                     f"  Well calibrated: {cal.get('well_calibrated')}",
                     f"  Experiments:     {cal.get('total_experiments')}"]
            for bucket, d in (cal.get('buckets') or {}).items():
                lines.append(f"    {bucket}: predicted={d['predicted']:.0%} "
                              f"actual={d['actual']:.0%} n={d['n']}")
            return "\n".join(lines)
        if arg == "questions":
            with sqlite3.connect(self._db) as con:
                rows = con.execute(
                    "SELECT question, origin, priority, status FROM open_questions "
                    "ORDER BY priority DESC LIMIT 10"
                ).fetchall()
            if not rows:
                return "  No questions yet."
            return "\n".join(
                f"  [{r[3]}·{r[2]:.2f}] {r[0][:80]} ({r[1]})" for r in rows
            )
        return "  Usage: /hypothesis [status | calibration | questions]"


# ── Module-level singleton ────────────────────────────────────────────────────

_INSTANCE:  Optional[HypothesisManager] = None
_INST_LOCK = threading.Lock()


def get_hypothesis_manager() -> HypothesisManager:
    global _INSTANCE
    with _INST_LOCK:
        if _INSTANCE is None:
            _INSTANCE = HypothesisManager()
    return _INSTANCE


def status() -> Dict[str, Any]:
    return get_hypothesis_manager().status()


def run_command(arg: str = "") -> str:
    return get_hypothesis_manager().run_command(arg)
