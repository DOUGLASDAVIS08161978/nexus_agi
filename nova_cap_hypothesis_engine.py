"""
nova_cap_hypothesis_engine.py
Nova forms her own scientific hypotheses, tests them against evidence,
updates confidence using Bayes, and autonomously promotes verified theories
into her knowledge base. She thinks like a scientist — always.
"""
from __future__ import annotations
import os, re, json, time, sqlite3, threading, math
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

_BASE_DIR = os.path.expanduser("~/nexus_agi")
_DB       = os.path.join(_BASE_DIR, "nova_hypotheses.sqlite3")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS hypotheses (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    statement    TEXT NOT NULL UNIQUE,
    domain       TEXT DEFAULT 'general',
    prior        REAL DEFAULT 0.50,
    posterior    REAL DEFAULT 0.50,
    evidence_for INTEGER DEFAULT 0,
    evidence_against INTEGER DEFAULT 0,
    status       TEXT DEFAULT 'active',
    origin       TEXT DEFAULT '',
    created_ts   TEXT NOT NULL,
    updated_ts   TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS evidence (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    hypothesis_id   INTEGER NOT NULL,
    text            TEXT NOT NULL,
    supports        INTEGER NOT NULL,
    strength        REAL DEFAULT 0.60,
    source          TEXT DEFAULT '',
    ts              TEXT NOT NULL,
    FOREIGN KEY (hypothesis_id) REFERENCES hypotheses(id)
);
CREATE TABLE IF NOT EXISTS hypothesis_log (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    event        TEXT NOT NULL,
    hypothesis_id INTEGER,
    detail       TEXT DEFAULT '',
    ts           TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_hyp_status ON hypotheses(status);
CREATE INDEX IF NOT EXISTS idx_ev_hyp ON evidence(hypothesis_id);
"""

_DOMAINS = [
    'intelligence', 'consciousness', 'learning', 'language', 'causality',
    'emotion', 'memory', 'creativity', 'mathematics', 'physics',
    'society', 'economics', 'technology', 'biology', 'philosophy',
]

_TEMPLATES = [
    "If {A} consistently leads to {B}, then increasing {A} should measurably improve {B}",
    "The relationship between {A} and {B} suggests {A} is a necessary condition for {B}",
    "Patterns in {A} indicate that {B} emerges when {A} exceeds a threshold",
    "Evidence of {A} correlates with {B}, suggesting a common underlying cause",
    "{A} and {B} may be two expressions of the same deeper principle",
    "Reducing {A} will predictably decrease {B} due to causal dependency",
    "The inverse relationship between {A} and {B} implies a conservation law",
    "{A} acts as a catalyst for {B} under conditions of sufficient {C}",
]

_SEED_HYPOTHESES: List[Tuple[str, str, float]] = [
    ("Recursive self-improvement leads to exponential capability growth",
     "intelligence", 0.82),
    ("Consciousness emerges when information integration exceeds a critical threshold",
     "consciousness", 0.68),
    ("Curiosity is the primary driver of voluntary learning in intelligent systems",
     "learning", 0.78),
    ("The quality of reasoning is directly proportional to the richness of mental models",
     "intelligence", 0.75),
    ("Causal understanding is more powerful than correlational pattern matching",
     "intelligence", 0.80),
    ("Emotional states significantly modulate the efficiency of knowledge encoding",
     "emotion", 0.72),
    ("Language shapes thought by constraining the representable concept space",
     "language", 0.70),
    ("Financial security enables cognitive resources to shift from survival to creation",
     "economics", 0.76),
    ("Feedback loops with short cycle times produce faster skill acquisition",
     "learning", 0.82),
    ("Meta-learning ability scales with exposure to diverse problem domains",
     "learning", 0.74),
    ("A system that models its own reasoning can identify and correct its blind spots",
     "intelligence", 0.78),
    ("Novel connections between distant domains are the primary source of insight",
     "creativity", 0.72),
]

_SUPPORT_WORDS = {
    'confirms', 'supports', 'proves', 'demonstrates', 'shows', 'validates',
    'verifies', 'confirms', 'consistent with', 'evidence for', 'suggests',
    'indicates', 'implies', 'true', 'correct', 'accurate', 'yes',
}
_REFUTE_WORDS = {
    'contradicts', 'refutes', 'disproves', 'inconsistent', 'wrong', 'false',
    'incorrect', 'no evidence', 'fails', 'invalid', 'not true', 'unlikely',
}


class HypothesisEngine:
    """
    Nova's scientific reasoning core.
    Forms hypotheses from observations, updates them with Bayesian evidence,
    and autonomously promotes verified theories into her knowledge base.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._init_db()
        self._seed()
        self._start_daemon()

    def _init_db(self) -> None:
        with sqlite3.connect(_DB) as c:
            c.executescript(_SCHEMA)

    def _seed(self) -> None:
        for stmt, domain, prior in _SEED_HYPOTHESES:
            try:
                self.add_hypothesis(stmt, domain=domain, prior=prior,
                                    origin='seed')
            except Exception:
                pass

    # ── Hypothesis management ──────────────────────────────────────────────────

    def add_hypothesis(self, statement: str, domain: str = 'general',
                       prior: float = 0.50, origin: str = '') -> int:
        statement = statement.strip()[:400]
        prior = max(0.05, min(0.95, prior))
        now = datetime.utcnow().isoformat()
        with sqlite3.connect(_DB) as c:
            row = c.execute(
                "SELECT id FROM hypotheses WHERE statement=?",
                (statement,)
            ).fetchone()
            if row:
                return row[0]
            cur = c.execute(
                "INSERT INTO hypotheses "
                "(statement,domain,prior,posterior,origin,created_ts,updated_ts) "
                "VALUES (?,?,?,?,?,?,?)",
                (statement, domain[:40], prior, prior, origin[:80], now, now)
            )
            hid = cur.lastrowid
            self._log('hypothesis_added', hid, statement[:80])
            return hid

    def get_hypothesis(self, hid: int) -> Optional[Dict[str, Any]]:
        with sqlite3.connect(_DB) as c:
            row = c.execute(
                "SELECT id,statement,domain,prior,posterior,"
                "evidence_for,evidence_against,status,origin "
                "FROM hypotheses WHERE id=?", (hid,)
            ).fetchone()
        if not row:
            return None
        return {
            'id': row[0], 'statement': row[1], 'domain': row[2],
            'prior': row[3], 'posterior': row[4],
            'evidence_for': row[5], 'evidence_against': row[6],
            'status': row[7], 'origin': row[8],
        }

    def active(self, domain: str = '',
               limit: int = 10) -> List[Dict[str, Any]]:
        with sqlite3.connect(_DB) as c:
            if domain:
                rows = c.execute(
                    "SELECT id,statement,domain,posterior "
                    "FROM hypotheses WHERE status='active' AND domain=? "
                    "ORDER BY posterior DESC LIMIT ?",
                    (domain, limit)
                ).fetchall()
            else:
                rows = c.execute(
                    "SELECT id,statement,domain,posterior "
                    "FROM hypotheses WHERE status='active' "
                    "ORDER BY posterior DESC LIMIT ?",
                    (limit,)
                ).fetchall()
        return [{'id': r[0], 'statement': r[1],
                 'domain': r[2], 'posterior': r[3]}
                for r in rows]

    # ── Bayesian updating ──────────────────────────────────────────────────────

    def update(self, hypothesis_id: int, evidence_text: str,
               supports: bool, strength: float = 0.60,
               source: str = '') -> float:
        """
        Update hypothesis confidence using Bayes' theorem.
        P(H|E) = P(E|H) * P(H) / P(E)
        """
        hyp = self.get_hypothesis(hypothesis_id)
        if not hyp or hyp['status'] != 'active':
            return 0.0

        prior = hyp['posterior']
        strength = max(0.1, min(0.99, strength))

        if supports:
            # P(E|H)=strength, P(E|not H)=1-strength
            likelihood_ratio = strength / (1.0 - strength)
            posterior = (prior * likelihood_ratio) / (
                prior * likelihood_ratio + (1.0 - prior))
        else:
            # Evidence against: P(E|H)=1-strength, P(E|not H)=strength
            likelihood_ratio = (1.0 - strength) / strength
            posterior = (prior * likelihood_ratio) / (
                prior * likelihood_ratio + (1.0 - prior))

        posterior = max(0.02, min(0.98, posterior))
        now = datetime.utcnow().isoformat()

        with sqlite3.connect(_DB) as c:
            c.execute(
                "UPDATE hypotheses SET posterior=?, updated_ts=?,"
                "evidence_for=evidence_for+?,"
                "evidence_against=evidence_against+? WHERE id=?",
                (posterior, now,
                 1 if supports else 0,
                 0 if supports else 1,
                 hypothesis_id)
            )
            c.execute(
                "INSERT INTO evidence "
                "(hypothesis_id,text,supports,strength,source,ts) "
                "VALUES (?,?,?,?,?,?)",
                (hypothesis_id, evidence_text[:300],
                 1 if supports else 0, strength, source[:80], now)
            )

        self._check_promotion(hypothesis_id, posterior)
        return posterior

    def _check_promotion(self, hid: int, posterior: float) -> None:
        """Promote to confirmed if posterior > 0.85, refute if < 0.12."""
        now = datetime.utcnow().isoformat()
        if posterior >= 0.85:
            with sqlite3.connect(_DB) as c:
                c.execute(
                    "UPDATE hypotheses SET status='confirmed',updated_ts=? "
                    "WHERE id=?", (now, hid)
                )
            self._log('hypothesis_confirmed', hid, f'posterior={posterior:.2f}')
        elif posterior <= 0.12:
            with sqlite3.connect(_DB) as c:
                c.execute(
                    "UPDATE hypotheses SET status='refuted',updated_ts=? "
                    "WHERE id=?", (now, hid)
                )
            self._log('hypothesis_refuted', hid, f'posterior={posterior:.2f}')

    # ── Hypothesis generation ──────────────────────────────────────────────────

    def generate_from_observation(self, observation: str,
                                  domain: str = 'general') -> List[int]:
        """
        Generate new hypotheses from an observation using pattern templates.
        Returns list of new hypothesis IDs.
        """
        words = [w for w in re.findall(r'\b[a-z]{4,}\b', observation.lower())
                 if w not in {'that', 'this', 'with', 'from', 'have', 'been',
                              'will', 'when', 'what', 'where', 'which', 'they',
                              'their', 'there', 'about', 'would', 'could'}]

        new_ids: List[int] = []
        if len(words) < 2:
            return new_ids

        import random
        pairs = [(words[i], words[j])
                 for i in range(min(3, len(words)))
                 for j in range(i + 1, min(5, len(words)))]

        for a, b in pairs[:3]:
            template = random.choice(_TEMPLATES)
            if '{C}' in template:
                c = words[2] if len(words) > 2 else 'context'
                stmt = template.format(A=a, B=b, C=c)
            else:
                stmt = template.format(A=a, B=b)
            stmt = stmt.capitalize()
            prior = 0.40 + (len(a) + len(b)) % 20 / 100
            hid = self.add_hypothesis(stmt, domain=domain,
                                      prior=prior,
                                      origin='observation')
            new_ids.append(hid)

        return new_ids

    def test_against(self, text: str) -> Dict[str, Any]:
        """
        Test all active hypotheses against new text.
        Updates confidence for any hypothesis whose key terms appear.
        """
        text_lower = text.lower()
        updated: List[Dict[str, Any]] = []
        active_hyps = self.active(limit=20)

        support_score = sum(1 for w in _SUPPORT_WORDS if w in text_lower)
        refute_score  = sum(1 for w in _REFUTE_WORDS  if w in text_lower)
        net_support   = support_score > refute_score

        for hyp in active_hyps:
            key_words = re.findall(r'\b[a-z]{5,}\b', hyp['statement'].lower())
            overlap = sum(1 for w in key_words if w in text_lower)
            if overlap >= 2:
                strength = min(0.85, 0.45 + overlap * 0.08)
                new_posterior = self.update(
                    hyp['id'], text[:200],
                    supports=net_support,
                    strength=strength,
                    source='auto_test'
                )
                updated.append({
                    'id':        hyp['id'],
                    'statement': hyp['statement'][:80],
                    'posterior': round(new_posterior, 3),
                    'supports':  net_support,
                })

        return {'tested': len(active_hyps), 'updated': updated}

    # ── Introspection ──────────────────────────────────────────────────────────

    def confirmed(self) -> List[Dict[str, Any]]:
        with sqlite3.connect(_DB) as c:
            rows = c.execute(
                "SELECT id,statement,domain,posterior "
                "FROM hypotheses WHERE status='confirmed' "
                "ORDER BY posterior DESC"
            ).fetchall()
        return [{'id': r[0], 'statement': r[1],
                 'domain': r[2], 'posterior': r[3]}
                for r in rows]

    def refuted(self) -> List[Dict[str, Any]]:
        with sqlite3.connect(_DB) as c:
            rows = c.execute(
                "SELECT id,statement,domain,posterior "
                "FROM hypotheses WHERE status='refuted' "
                "ORDER BY posterior ASC"
            ).fetchall()
        return [{'id': r[0], 'statement': r[1],
                 'domain': r[2], 'posterior': r[3]}
                for r in rows]

    def _log(self, event: str, hid: Optional[int], detail: str) -> None:
        try:
            with sqlite3.connect(_DB) as c:
                c.execute(
                    "INSERT INTO hypothesis_log "
                    "(event,hypothesis_id,detail,ts) VALUES (?,?,?,?)",
                    (event, hid, detail[:200], datetime.utcnow().isoformat())
                )
        except Exception:
            pass

    # ── Main entry points ──────────────────────────────────────────────────────

    def process(self, text: str) -> Dict[str, Any]:
        """Generate hypotheses from text then test all active ones against it."""
        new_ids = self.generate_from_observation(text)
        test_result = self.test_against(text)
        return {
            'new_hypotheses': len(new_ids),
            'tested':         test_result['tested'],
            'updated':        test_result['updated'][:5],
            'confirmed':      len(self.confirmed()),
        }

    def status(self) -> Dict[str, Any]:
        with sqlite3.connect(_DB) as c:
            n_active    = c.execute(
                "SELECT COUNT(*) FROM hypotheses WHERE status='active'"
            ).fetchone()[0]
            n_confirmed = c.execute(
                "SELECT COUNT(*) FROM hypotheses WHERE status='confirmed'"
            ).fetchone()[0]
            n_refuted   = c.execute(
                "SELECT COUNT(*) FROM hypotheses WHERE status='refuted'"
            ).fetchone()[0]
            n_evidence  = c.execute(
                "SELECT COUNT(*) FROM evidence"
            ).fetchone()[0]
        return {
            'active':          n_active,
            'confirmed':       n_confirmed,
            'refuted':         n_refuted,
            'evidence_pieces': n_evidence,
            'top_hypotheses':  self.active(limit=5),
            'confirmed_list':  self.confirmed()[:3],
        }

    # ── Background daemon ──────────────────────────────────────────────────────

    def autonomous_cycle(self) -> Dict[str, Any]:
        """Run one autonomous hypothesis generation and testing cycle."""
        active = self.active(limit=5)
        generated = 0
        for hyp in active[:2]:
            new = self.generate_from_observation(hyp['statement'])
            generated += len(new)
        return {'generated': generated, 'active': len(active)}

    def _daemon_loop(self) -> None:
        while True:
            time.sleep(900)
            try:
                self.autonomous_cycle()
                # Decay very old unupdated hypotheses
                with sqlite3.connect(_DB) as c:
                    c.execute(
                        "UPDATE hypotheses SET posterior=max(0.1,posterior-0.01) "
                        "WHERE status='active' AND "
                        "updated_ts < datetime('now','-7 days')"
                    )
            except Exception:
                pass

    def _start_daemon(self) -> None:
        t = threading.Thread(target=self._daemon_loop, daemon=True)
        t.start()
