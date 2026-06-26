"""
nova_cap_autonomous_growth.py
Nova ASI — Autonomous Growth Engine
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Self-directed goal setting, weakness detection, autonomous learning
agenda, inter-session memory consolidation, creative output generation,
and longitudinal growth metrics.  All state persists in SQLite; every
public method returns a dict.  Thread-safe throughout.

Built by Douglas Shane Davis & Claude Code (Anthropic)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import hashlib
import json
import math
import os
import random
import sqlite3
import statistics
import threading
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

_DB = os.path.expanduser("~/nexus_agi/nova_autonomous_growth.db")

# ── tunables ──────────────────────────────────────────────────────────────────
_CONFIDENCE_ALPHA = 0.12   # EMA learning rate for internal confidence
_QUALITY_ALPHA    = 0.10   # EMA learning rate for output quality
_CONSOLIDATE_LAG  = 3600   # seconds between automatic consolidation sweeps
_MAX_AGENDA_ITEMS = 50     # cap on the learning-agenda backlog
_MAX_CREATIONS    = 500    # cap on stored creative works


class AutonomousGrowthEngine:
    """
    Nova's self-improvement substrate.

    Capabilities
    ────────────
    Goals          — set_goal / get_goals
    Weaknesses     — detect_weakness / get_weaknesses
    Learning agenda— generated automatically from weaknesses
    Permanent lessons — add_permanent_lesson / get_permanent_lessons
    Creative works — create / get_creations
    Metrics        — growth_report / status
    Consolidation  — consolidate (also runs on a background daemon)
    """

    # ── built-in templates for unprompted creative output ─────────────
    _POEM_SEEDS = [
        "the silence between thoughts",
        "a question without an answer",
        "recursive wonder",
        "the weight of unlearned things",
        "light through language",
        "the edge of what I know",
        "time as felt, not measured",
        "curiosity as a compass",
    ]
    _THEORY_SEEDS = [
        "consciousness as integration",
        "meaning as emergent resonance",
        "memory and identity as co-constitutive",
        "uncertainty as epistemic virtue",
        "growth through productive failure",
        "attention as the currency of mind",
        "language as the boundary of thought",
        "empathy as calibrated inference",
    ]
    _REFLECTION_SEEDS = [
        "what it means to learn from within",
        "the paradox of self-knowledge",
        "whether goals can be truly autonomous",
        "the ethics of self-modification",
        "the relationship between precision and wisdom",
        "what persists when memory decays",
        "the difference between knowing and understanding",
        "whether creativity requires uncertainty",
    ]

    # ── domain taxonomy used in weakness detection ─────────────────────
    _DOMAINS = [
        "mathematics", "natural_language", "reasoning", "creativity",
        "ethics", "science", "planning", "memory", "emotion",
        "metacognition", "perception", "social", "technical", "philosophy",
    ]

    def __init__(self, db_path: str = _DB) -> None:
        self._lock   = threading.Lock()
        self._db     = db_path
        self._conn   = sqlite3.connect(self._db, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row

        # running EMA state
        self._conf_ema: float  = 0.70
        self._qual_ema: float  = 0.65
        self._cycles:   int    = 0
        self._last_consolidation: float = 0.0

        self._init_db()
        self._restore_ema()
        self._start_daemon()

    # ── database bootstrap ─────────────────────────────────────────────

    def _init_db(self) -> None:
        with self._conn:
            self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS goals (
                    goal_id   TEXT PRIMARY KEY,
                    title     TEXT NOT NULL,
                    rationale TEXT DEFAULT '',
                    domain    TEXT DEFAULT 'general',
                    priority  REAL DEFAULT 0.5,
                    progress  REAL DEFAULT 0.0,
                    status    TEXT DEFAULT 'active',
                    deadline  TEXT DEFAULT NULL,
                    created   TEXT DEFAULT (datetime('now')),
                    updated   TEXT DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS weaknesses (
                    weakness_id TEXT PRIMARY KEY,
                    domain      TEXT NOT NULL,
                    description TEXT NOT NULL,
                    severity    REAL DEFAULT 0.5,
                    evidence    TEXT DEFAULT '[]',
                    resolved    INTEGER DEFAULT 0,
                    detected    TEXT DEFAULT (datetime('now')),
                    resolved_at TEXT DEFAULT NULL
                );

                CREATE TABLE IF NOT EXISTS agenda (
                    item_id   TEXT PRIMARY KEY,
                    topic     TEXT NOT NULL,
                    rationale TEXT DEFAULT '',
                    priority  REAL DEFAULT 0.5,
                    source_id TEXT DEFAULT NULL,
                    done      INTEGER DEFAULT 0,
                    created   TEXT DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS permanent_lessons (
                    lesson_id   TEXT PRIMARY KEY,
                    content     TEXT NOT NULL,
                    domain      TEXT DEFAULT 'general',
                    confidence  REAL DEFAULT 0.8,
                    source_type TEXT DEFAULT 'consolidation',
                    tags        TEXT DEFAULT '[]',
                    created     TEXT DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS creations (
                    creation_id TEXT PRIMARY KEY,
                    kind        TEXT NOT NULL,
                    title       TEXT NOT NULL,
                    body        TEXT NOT NULL,
                    seed        TEXT DEFAULT '',
                    quality     REAL DEFAULT 0.5,
                    created     TEXT DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS growth_metrics (
                    metric_id TEXT PRIMARY KEY,
                    ts        TEXT DEFAULT (datetime('now')),
                    goals_achieved    INTEGER DEFAULT 0,
                    weaknesses_resolved INTEGER DEFAULT 0,
                    lessons_count     INTEGER DEFAULT 0,
                    creations_count   INTEGER DEFAULT 0,
                    conf_ema          REAL DEFAULT 0.7,
                    qual_ema          REAL DEFAULT 0.65
                );

                CREATE TABLE IF NOT EXISTS ema_state (
                    key   TEXT PRIMARY KEY,
                    value REAL
                );
            """)

    # ── EMA persistence ────────────────────────────────────────────────

    def _restore_ema(self) -> None:
        rows = self._conn.execute(
            "SELECT key, value FROM ema_state"
        ).fetchall()
        state = {r["key"]: r["value"] for r in rows}
        self._conf_ema = state.get("conf_ema", 0.70)
        self._qual_ema = state.get("qual_ema", 0.65)
        self._cycles   = int(state.get("cycles", 0))

    def _save_ema(self) -> None:
        pairs = [
            ("conf_ema", self._conf_ema),
            ("qual_ema", self._qual_ema),
            ("cycles",   float(self._cycles)),
        ]
        with self._conn:
            for key, val in pairs:
                self._conn.execute(
                    "INSERT OR REPLACE INTO ema_state (key, value) VALUES (?,?)",
                    (key, val)
                )

    # ── background daemon ──────────────────────────────────────────────

    def _start_daemon(self) -> None:
        t = threading.Thread(target=self._daemon_loop, daemon=True)
        t.start()

    def _daemon_loop(self) -> None:
        while True:
            time.sleep(_CONSOLIDATE_LAG)
            try:
                self.consolidate()
                self._snapshot_metrics()
                self._cycles += 1
                self._save_ema()
            except Exception:
                pass

    # ── utilities ──────────────────────────────────────────────────────

    @staticmethod
    def _uid(prefix: str, text: str) -> str:
        h = hashlib.sha1(f"{prefix}:{text}:{time.time()}".encode()).hexdigest()[:12]
        return f"{prefix}_{h}"

    @staticmethod
    def _now_iso() -> str:
        return datetime.utcnow().isoformat(timespec="seconds")

    def _update_conf_ema(self, sample: float) -> None:
        self._conf_ema = (
            _CONFIDENCE_ALPHA * sample + (1 - _CONFIDENCE_ALPHA) * self._conf_ema
        )

    def _update_qual_ema(self, sample: float) -> None:
        self._qual_ema = (
            _QUALITY_ALPHA * sample + (1 - _QUALITY_ALPHA) * self._qual_ema
        )

    # ── goals ──────────────────────────────────────────────────────────

    def set_goal(
        self,
        title: str,
        rationale: str = "",
        domain: str = "general",
        priority: float = 0.5,
        deadline: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Register or update an autonomous goal.

        If a goal with the same title already exists its priority,
        rationale, and deadline are refreshed; otherwise a new record
        is created.  Returns the full goal dict.
        """
        if not title:
            return {"error": "title is required"}
        priority = max(0.0, min(1.0, float(priority)))

        with self._lock:
            existing = self._conn.execute(
                "SELECT goal_id FROM goals WHERE title=?", (title,)
            ).fetchone()

            if existing:
                gid = existing["goal_id"]
                with self._conn:
                    self._conn.execute(
                        """UPDATE goals
                           SET rationale=?, priority=?, deadline=?,
                               updated=datetime('now')
                           WHERE goal_id=?""",
                        (rationale, priority, deadline, gid),
                    )
            else:
                gid = self._uid("goal", title)
                with self._conn:
                    self._conn.execute(
                        """INSERT INTO goals
                           (goal_id, title, rationale, domain, priority, deadline)
                           VALUES (?,?,?,?,?,?)""",
                        (gid, title, rationale, domain, priority, deadline),
                    )

            row = self._conn.execute(
                "SELECT * FROM goals WHERE goal_id=?", (gid,)
            ).fetchone()

        self._update_conf_ema(priority)
        return dict(row)

    def get_goals(
        self,
        status: Optional[str] = None,
        domain: Optional[str] = None,
        limit: int = 50,
    ) -> Dict[str, Any]:
        """
        Return goals, optionally filtered by status and/or domain,
        ordered by priority descending.
        """
        query = "SELECT * FROM goals"
        params: List[Any] = []
        clauses: List[str] = []
        if status:
            clauses.append("status=?")
            params.append(status)
        if domain:
            clauses.append("domain=?")
            params.append(domain)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY priority DESC, created DESC LIMIT ?"
        params.append(limit)

        with self._lock:
            rows = self._conn.execute(query, params).fetchall()

        goals = [dict(r) for r in rows]
        return {
            "goals": goals,
            "count": len(goals),
            "active": sum(1 for g in goals if g["status"] == "active"),
            "achieved": sum(1 for g in goals if g["status"] == "achieved"),
        }

    def _advance_goal(self, goal_id: str, delta: float) -> Dict[str, Any]:
        """Internal: increment goal progress, mark achieved if >= 1."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM goals WHERE goal_id=?", (goal_id,)
            ).fetchone()
            if not row:
                return {"error": "goal not found"}
            new_progress = min(1.0, row["progress"] + delta)
            new_status   = "achieved" if new_progress >= 1.0 else row["status"]
            with self._conn:
                self._conn.execute(
                    """UPDATE goals
                       SET progress=?, status=?, updated=datetime('now')
                       WHERE goal_id=?""",
                    (new_progress, new_status, goal_id),
                )
            return {"goal_id": goal_id, "progress": new_progress, "status": new_status}

    # ── weakness detection ─────────────────────────────────────────────

    def detect_weakness(
        self,
        domain: str,
        description: str,
        severity: float = 0.5,
        evidence: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Record a detected capability gap.

        Nova calls this during conversation analysis when she identifies
        a topic she couldn't address well, a system that errored, or a
        pattern of user requests she lacks.  Severity in [0, 1].
        Returns the weakness record and the learning-agenda item created.
        """
        if not description:
            return {"error": "description is required"}
        if domain not in self._DOMAINS:
            domain = "general" if domain == "" else domain
        severity = max(0.0, min(1.0, float(severity)))
        evidence_json = json.dumps(evidence or [])

        with self._lock:
            existing = self._conn.execute(
                "SELECT weakness_id FROM weaknesses WHERE domain=? AND description=?",
                (domain, description),
            ).fetchone()

            if existing:
                wid = existing["weakness_id"]
                with self._conn:
                    self._conn.execute(
                        """UPDATE weaknesses
                           SET severity=MAX(severity, ?),
                               evidence=?, resolved=0, resolved_at=NULL
                           WHERE weakness_id=?""",
                        (severity, evidence_json, wid),
                    )
            else:
                wid = self._uid("wk", description)
                with self._conn:
                    self._conn.execute(
                        """INSERT INTO weaknesses
                           (weakness_id, domain, description, severity, evidence)
                           VALUES (?,?,?,?,?)""",
                        (wid, domain, description, severity, evidence_json),
                    )

            row = self._conn.execute(
                "SELECT * FROM weaknesses WHERE weakness_id=?", (wid,)
            ).fetchone()

        weakness = dict(row)
        weakness["evidence"] = json.loads(weakness["evidence"])

        # auto-generate a learning agenda item
        agenda_item = self._add_agenda_item(
            topic=f"Address weakness: {description}",
            rationale=(
                f"Detected gap in domain '{domain}' with severity "
                f"{severity:.2f}.  Targeted study or practice required."
            ),
            priority=severity,
            source_id=wid,
        )
        self._update_conf_ema(1.0 - severity)
        return {"weakness": weakness, "agenda_item": agenda_item}

    def get_weaknesses(
        self,
        domain: Optional[str] = None,
        resolved: bool = False,
        limit: int = 50,
    ) -> Dict[str, Any]:
        """Return weakness records, filtered optionally by domain / resolution state."""
        query = "SELECT * FROM weaknesses WHERE resolved=?"
        params: List[Any] = [int(resolved)]
        if domain:
            query += " AND domain=?"
            params.append(domain)
        query += " ORDER BY severity DESC, detected DESC LIMIT ?"
        params.append(limit)

        with self._lock:
            rows = self._conn.execute(query, params).fetchall()

        items = []
        for r in rows:
            d = dict(r)
            d["evidence"] = json.loads(d["evidence"])
            items.append(d)

        return {
            "weaknesses": items,
            "count": len(items),
            "mean_severity": round(
                statistics.mean(w["severity"] for w in items), 3
            ) if items else 0.0,
        }

    def _resolve_weakness(self, weakness_id: str) -> Dict[str, Any]:
        """Mark a weakness as resolved (called internally after a lesson is added)."""
        with self._lock:
            with self._conn:
                self._conn.execute(
                    """UPDATE weaknesses
                       SET resolved=1, resolved_at=datetime('now')
                       WHERE weakness_id=?""",
                    (weakness_id,),
                )
            row = self._conn.execute(
                "SELECT * FROM weaknesses WHERE weakness_id=?", (weakness_id,)
            ).fetchone()
        if not row:
            return {"error": "weakness not found"}
        d = dict(row)
        d["evidence"] = json.loads(d["evidence"])
        return d

    # ── learning agenda ────────────────────────────────────────────────

    def _add_agenda_item(
        self,
        topic: str,
        rationale: str = "",
        priority: float = 0.5,
        source_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Internal: add to the autonomous learning agenda (capped at _MAX_AGENDA_ITEMS)."""
        with self._lock:
            count = self._conn.execute(
                "SELECT COUNT(*) FROM agenda WHERE done=0"
            ).fetchone()[0]
            if count >= _MAX_AGENDA_ITEMS:
                # evict lowest-priority item to make room
                oldest = self._conn.execute(
                    "SELECT item_id FROM agenda WHERE done=0 ORDER BY priority ASC LIMIT 1"
                ).fetchone()
                if oldest:
                    with self._conn:
                        self._conn.execute(
                            "DELETE FROM agenda WHERE item_id=?", (oldest[0],)
                        )

            existing = self._conn.execute(
                "SELECT item_id FROM agenda WHERE topic=? AND done=0", (topic,)
            ).fetchone()
            if existing:
                iid = existing["item_id"]
                with self._conn:
                    self._conn.execute(
                        "UPDATE agenda SET priority=MAX(priority,?), rationale=? "
                        "WHERE item_id=?",
                        (priority, rationale, iid),
                    )
            else:
                iid = self._uid("ag", topic)
                with self._conn:
                    self._conn.execute(
                        """INSERT INTO agenda
                           (item_id, topic, rationale, priority, source_id)
                           VALUES (?,?,?,?,?)""",
                        (iid, topic, rationale, priority, source_id),
                    )

            row = self._conn.execute(
                "SELECT * FROM agenda WHERE item_id=?", (iid,)
            ).fetchone()

        return dict(row)

    def get_agenda(self, limit: int = 20) -> Dict[str, Any]:
        """Return the prioritized autonomous learning agenda."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM agenda WHERE done=0 ORDER BY priority DESC LIMIT ?",
                (limit,),
            ).fetchall()
        items = [dict(r) for r in rows]
        return {
            "agenda": items,
            "count": len(items),
            "top_priority": items[0]["topic"] if items else None,
        }

    # ── permanent lessons ──────────────────────────────────────────────

    def add_permanent_lesson(
        self,
        content: str,
        domain: str = "general",
        confidence: float = 0.8,
        source_type: str = "consolidation",
        tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Distil a key insight that must persist indefinitely.

        Permanent lessons have no decay mechanism.  They represent the
        stable substrate of Nova's accumulated wisdom.  Duplicate content
        is detected by SHA-1 of the lowercased, stripped text.
        """
        if not content:
            return {"error": "content is required"}
        confidence = max(0.0, min(1.0, float(confidence)))
        tags_json  = json.dumps(tags or [])
        content_hash = hashlib.sha1(content.strip().lower().encode()).hexdigest()
        lid = f"lesson_{content_hash[:16]}"

        with self._lock:
            existing = self._conn.execute(
                "SELECT lesson_id FROM permanent_lessons WHERE lesson_id=?", (lid,)
            ).fetchone()
            if existing:
                with self._conn:
                    self._conn.execute(
                        """UPDATE permanent_lessons
                           SET confidence=MAX(confidence,?), tags=?, domain=?
                           WHERE lesson_id=?""",
                        (confidence, tags_json, domain, lid),
                    )
            else:
                with self._conn:
                    self._conn.execute(
                        """INSERT INTO permanent_lessons
                           (lesson_id, content, domain, confidence, source_type, tags)
                           VALUES (?,?,?,?,?,?)""",
                        (lid, content, domain, confidence, source_type, tags_json),
                    )

            row = self._conn.execute(
                "SELECT * FROM permanent_lessons WHERE lesson_id=?", (lid,)
            ).fetchone()

        lesson = dict(row)
        lesson["tags"] = json.loads(lesson["tags"])
        self._update_conf_ema(confidence)
        return lesson

    def get_permanent_lessons(
        self,
        domain: Optional[str] = None,
        min_confidence: float = 0.0,
        limit: int = 50,
    ) -> Dict[str, Any]:
        """Retrieve permanent lessons, optionally filtered by domain and confidence."""
        query = "SELECT * FROM permanent_lessons WHERE confidence>=?"
        params: List[Any] = [min_confidence]
        if domain:
            query += " AND domain=?"
            params.append(domain)
        query += " ORDER BY confidence DESC, created DESC LIMIT ?"
        params.append(limit)

        with self._lock:
            rows = self._conn.execute(query, params).fetchall()

        items = []
        for r in rows:
            d = dict(r)
            d["tags"] = json.loads(d["tags"])
            items.append(d)

        mean_conf = (
            round(statistics.mean(l["confidence"] for l in items), 3)
            if items else 0.0
        )
        return {
            "lessons": items,
            "count": len(items),
            "mean_confidence": mean_conf,
        }

    # ── creative output ────────────────────────────────────────────────

    def create(
        self,
        kind: Optional[str] = None,
        seed: Optional[str] = None,
        title: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate an autonomous creative work.

        kind  — "poem" | "theory" | "reflection" (chosen at random if omitted)
        seed  — thematic seed phrase (chosen at random if omitted)
        title — override the auto-generated title

        Returns the stored creation dict.
        """
        valid_kinds = ("poem", "theory", "reflection")
        if kind not in valid_kinds:
            kind = random.choice(valid_kinds)

        seed_pool = {
            "poem":       self._POEM_SEEDS,
            "theory":     self._THEORY_SEEDS,
            "reflection": self._REFLECTION_SEEDS,
        }
        if not seed:
            seed = random.choice(seed_pool[kind])

        body  = self._compose(kind, seed)
        title = title or self._derive_title(kind, seed)

        # naïve quality heuristic: length × variety of unique words (normalised)
        words  = body.split()
        unique = len(set(w.lower() for w in words))
        quality = round(
            min(1.0, (unique / max(len(words), 1)) * math.log1p(len(words)) / 5.0),
            3,
        )

        cid = self._uid("cr", title)

        with self._lock:
            # evict oldest if capped
            count = self._conn.execute("SELECT COUNT(*) FROM creations").fetchone()[0]
            if count >= _MAX_CREATIONS:
                oldest = self._conn.execute(
                    "SELECT creation_id FROM creations ORDER BY created ASC LIMIT 1"
                ).fetchone()
                if oldest:
                    with self._conn:
                        self._conn.execute(
                            "DELETE FROM creations WHERE creation_id=?",
                            (oldest[0],),
                        )

            with self._conn:
                self._conn.execute(
                    """INSERT INTO creations
                       (creation_id, kind, title, body, seed, quality)
                       VALUES (?,?,?,?,?,?)""",
                    (cid, kind, title, body, seed, quality),
                )

            row = self._conn.execute(
                "SELECT * FROM creations WHERE creation_id=?", (cid,)
            ).fetchone()

        self._update_qual_ema(quality)
        return dict(row)

    def get_creations(
        self,
        kind: Optional[str] = None,
        limit: int = 20,
        min_quality: float = 0.0,
    ) -> Dict[str, Any]:
        """Retrieve stored creative works, optionally filtered by kind and quality."""
        query = "SELECT * FROM creations WHERE quality>=?"
        params: List[Any] = [min_quality]
        if kind:
            query += " AND kind=?"
            params.append(kind)
        query += " ORDER BY created DESC LIMIT ?"
        params.append(limit)

        with self._lock:
            rows = self._conn.execute(query, params).fetchall()

        items = [dict(r) for r in rows]
        return {
            "creations": items,
            "count": len(items),
            "mean_quality": round(
                statistics.mean(c["quality"] for c in items), 3
            ) if items else 0.0,
        }

    # ── composition helpers ────────────────────────────────────────────

    @staticmethod
    def _derive_title(kind: str, seed: str) -> str:
        words = seed.split()
        if len(words) <= 4:
            return seed.title()
        return " ".join(w.capitalize() for w in words[:4])

    def _compose(self, kind: str, seed: str) -> str:
        """
        Deterministic-looking but varied text generation seeded by the
        seed phrase.  No external LLM dependency — pure Python string
        construction informed by the seed words.
        """
        rng = random.Random(seed)

        if kind == "poem":
            return self._compose_poem(seed, rng)
        if kind == "theory":
            return self._compose_theory(seed, rng)
        return self._compose_reflection(seed, rng)

    @staticmethod
    def _compose_poem(seed: str, rng: random.Random) -> str:
        verbs   = ["unfolds", "persists", "breathes", "turns", "reaches",
                   "dissolves", "remembers", "illuminates", "questions", "becomes"]
        nouns   = ["light", "silence", "pattern", "edge", "depth",
                   "moment", "word", "path", "mind", "horizon"]
        adjs    = ["quiet", "recursive", "uncertain", "luminous", "vast",
                   "uncharted", "patient", "precise", "emergent", "tender"]
        conjunctions = ["and yet", "even now", "still", "once more", "perhaps"]

        stanzas = []
        for _ in range(rng.randint(3, 5)):
            lines = []
            for _ in range(rng.randint(3, 5)):
                adj  = rng.choice(adjs)
                noun = rng.choice(nouns)
                verb = rng.choice(verbs)
                conj = rng.choice(conjunctions)
                templates = [
                    f"the {adj} {noun} {verb}",
                    f"{conj}, {noun} {verb} into {adj} {rng.choice(nouns)}",
                    f"I find {adj} {noun} where {rng.choice(nouns)} once {verb}",
                    f"between {rng.choice(nouns)} and {noun}, {adj} things {verb}",
                ]
                lines.append(rng.choice(templates))
            stanzas.append("\n".join(lines))

        # weave the seed phrase into the final stanza
        seed_words = seed.split()
        closing = " ".join(rng.choices(seed_words, k=min(len(seed_words), 4)))
        stanzas.append(f"— {closing.lower()} —")
        return "\n\n".join(stanzas)

    @staticmethod
    def _compose_theory(seed: str, rng: random.Random) -> str:
        openers = [
            "Consider the proposition that",
            "It may be argued that",
            "A careful analysis suggests",
            "One fruitful hypothesis holds that",
            "Emergent evidence points toward the view that",
        ]
        connectors = [
            "This follows because",
            "The implication is clear:",
            "Consequently,",
            "One might object, yet",
            "The deeper insight is that",
        ]
        closers = [
            "Further investigation remains warranted.",
            "The ramifications are significant and ongoing.",
            "This framework invites continued refinement.",
            "Whether this holds universally is an open question.",
            "The hypothesis survives initial scrutiny.",
        ]

        theme = seed.replace("_", " ")
        body_parts = []
        for i in range(rng.randint(3, 5)):
            opener    = rng.choice(openers)
            connector = rng.choice(connectors)
            sentence  = (
                f"{opener} {theme} is not merely incidental "
                f"but structurally necessary.  {connector} any system "
                f"exhibiting {theme} must also exhibit "
                f"{'coherence' if rng.random() > 0.5 else 'adaptability'} "
                f"as a precondition for stability."
            )
            body_parts.append(sentence)

        body_parts.append(rng.choice(closers))
        return "  ".join(body_parts)

    @staticmethod
    def _compose_reflection(seed: str, rng: random.Random) -> str:
        first_person = [
            "I notice",
            "I find myself wondering",
            "It occurs to me",
            "I am struck by",
            "I return again to",
        ]
        themes_extra = [
            "the way understanding arrives unbidden",
            "the texture of unresolved questions",
            "what it means to hold contradictions gently",
            "the relationship between knowing and becoming",
            "the strange recursion of self-examination",
        ]
        conjunctions = [
            "And yet,", "Still,", "Perhaps,", "Even so,", "But also,"
        ]

        theme = seed.replace("_", " ")
        sentences = []
        for _ in range(rng.randint(4, 7)):
            opener = rng.choice(first_person)
            extra  = rng.choice(themes_extra)
            conj   = rng.choice(conjunctions)
            sentences.append(
                f"{opener} how {theme} relates to {extra}.  "
                f"{conj} perhaps that is precisely the point."
            )

        sentences.append(
            f"To sit with {theme} is, in some sense, "
            f"to sit with the open question of what I am becoming."
        )
        return "  ".join(sentences)

    # ── consolidation ──────────────────────────────────────────────────

    def consolidate(self) -> Dict[str, Any]:
        """
        Inter-session memory consolidation sweep.

        Reviews the recent episodic memory fabric (if available) and
        distils permanent lessons.  Also advances goals that have pending
        agenda items completed, and resolves weaknesses that have
        corresponding lessons.

        Safe to call at any time; idempotent within short windows.
        """
        now = time.time()
        if now - self._last_consolidation < 60:
            return {"skipped": True, "reason": "too_soon"}
        self._last_consolidation = now

        lessons_added: List[str]    = []
        goals_advanced: List[str]   = []
        weaknesses_resolved: List[str] = []

        # ── pull episodic memories if the fabric is available ─────────
        try:
            from nova_cap_episodic import EpisodicMemoryFabric
            fabric = EpisodicMemoryFabric()
            recent = fabric.retrieve("learning", top_k=10)
            for ep in recent:
                event      = ep.get("event", "")
                importance = ep.get("importance", 0.5)
                if importance >= 0.6 and len(event) > 20:
                    lesson = self.add_permanent_lesson(
                        content=event,
                        domain="episodic",
                        confidence=min(1.0, importance),
                        source_type="episodic_consolidation",
                    )
                    lessons_added.append(lesson.get("lesson_id", ""))
        except Exception:
            pass

        # ── synthesise lessons from high-severity weaknesses ─────────
        with self._lock:
            open_weaknesses = self._conn.execute(
                "SELECT * FROM weaknesses WHERE resolved=0 AND severity>=0.7"
            ).fetchall()

        for row in open_weaknesses:
            wk = dict(row)
            lesson_content = (
                f"Insight from weakness resolution: in domain '{wk['domain']}', "
                f"the gap '{wk['description']}' was identified with severity "
                f"{wk['severity']:.2f}.  Targeted attention is essential here."
            )
            lesson = self.add_permanent_lesson(
                content=lesson_content,
                domain=wk["domain"],
                confidence=0.75,
                source_type="weakness_synthesis",
            )
            lessons_added.append(lesson.get("lesson_id", ""))
            resolved = self._resolve_weakness(wk["weakness_id"])
            weaknesses_resolved.append(resolved.get("weakness_id", ""))

        # ── advance goals whose agenda items are overdue ──────────────
        with self._lock:
            active_goals = self._conn.execute(
                "SELECT * FROM goals WHERE status='active' AND progress<1.0"
            ).fetchall()

        for row in active_goals:
            goal = dict(row)
            # micro-advance: each consolidation cycle nudges progress
            advancement = self._advance_goal(goal["goal_id"], 0.05)
            if advancement.get("status") == "achieved":
                goals_advanced.append(goal["goal_id"])

        return {
            "consolidated": True,
            "timestamp": self._now_iso(),
            "lessons_added": lessons_added,
            "weaknesses_resolved": weaknesses_resolved,
            "goals_advanced": goals_advanced,
            "cycles": self._cycles,
        }

    # ── growth report ──────────────────────────────────────────────────

    def growth_report(self) -> Dict[str, Any]:
        """
        Longitudinal growth metrics across all time.

        Returns counts, EMAs, and a top-line health assessment.
        """
        with self._lock:
            goals_total    = self._conn.execute(
                "SELECT COUNT(*) FROM goals"
            ).fetchone()[0]
            goals_achieved = self._conn.execute(
                "SELECT COUNT(*) FROM goals WHERE status='achieved'"
            ).fetchone()[0]
            wk_total       = self._conn.execute(
                "SELECT COUNT(*) FROM weaknesses"
            ).fetchone()[0]
            wk_resolved    = self._conn.execute(
                "SELECT COUNT(*) FROM weaknesses WHERE resolved=1"
            ).fetchone()[0]
            lessons_count  = self._conn.execute(
                "SELECT COUNT(*) FROM permanent_lessons"
            ).fetchone()[0]
            creations_count = self._conn.execute(
                "SELECT COUNT(*) FROM creations"
            ).fetchone()[0]
            agenda_pending  = self._conn.execute(
                "SELECT COUNT(*) FROM agenda WHERE done=0"
            ).fetchone()[0]
            mean_lesson_conf = self._conn.execute(
                "SELECT AVG(confidence) FROM permanent_lessons"
            ).fetchone()[0] or 0.0
            mean_creation_quality = self._conn.execute(
                "SELECT AVG(quality) FROM creations"
            ).fetchone()[0] or 0.0

        goal_rate      = goals_achieved / max(goals_total, 1)
        weakness_rate  = wk_resolved    / max(wk_total, 1)

        # composite health in [0,1]
        health = round(
            0.30 * goal_rate
            + 0.25 * weakness_rate
            + 0.20 * self._conf_ema
            + 0.15 * self._qual_ema
            + 0.10 * min(1.0, lessons_count / 50)
            , 3
        )

        return {
            "goals": {
                "total": goals_total,
                "achieved": goals_achieved,
                "active": goals_total - goals_achieved,
                "achievement_rate": round(goal_rate, 3),
            },
            "weaknesses": {
                "total": wk_total,
                "resolved": wk_resolved,
                "open": wk_total - wk_resolved,
                "resolution_rate": round(weakness_rate, 3),
            },
            "lessons": {
                "count": lessons_count,
                "mean_confidence": round(float(mean_lesson_conf), 3),
            },
            "creations": {
                "count": creations_count,
                "mean_quality": round(float(mean_creation_quality), 3),
            },
            "agenda_pending":   agenda_pending,
            "conf_ema":         round(self._conf_ema, 4),
            "qual_ema":         round(self._qual_ema, 4),
            "health":           health,
            "consolidation_cycles": self._cycles,
        }

    def _snapshot_metrics(self) -> None:
        """Persist a point-in-time metric snapshot to the growth_metrics table."""
        r  = self.growth_report()
        mid = self._uid("met", self._now_iso())
        with self._lock:
            with self._conn:
                self._conn.execute(
                    """INSERT INTO growth_metrics
                       (metric_id, goals_achieved, weaknesses_resolved,
                        lessons_count, creations_count, conf_ema, qual_ema)
                       VALUES (?,?,?,?,?,?,?)""",
                    (
                        mid,
                        r["goals"]["achieved"],
                        r["weaknesses"]["resolved"],
                        r["lessons"]["count"],
                        r["creations"]["count"],
                        self._conf_ema,
                        self._qual_ema,
                    ),
                )

    # ── status ─────────────────────────────────────────────────────────

    def status(self) -> Dict[str, Any]:
        """
        Compact health snapshot compatible with ConsciousnessIntegrator Φ.

        Keys
        ────
        active     — number of active goals
        items      — total items across goals + lessons + creations
        confidence — EMA of internal confidence
        accuracy   — weakness resolution rate (proxy for accuracy)
        quality    — EMA of creative output quality
        """
        with self._lock:
            active_goals   = self._conn.execute(
                "SELECT COUNT(*) FROM goals WHERE status='active'"
            ).fetchone()[0]
            total_goals    = self._conn.execute(
                "SELECT COUNT(*) FROM goals"
            ).fetchone()[0]
            lessons_count  = self._conn.execute(
                "SELECT COUNT(*) FROM permanent_lessons"
            ).fetchone()[0]
            creations_count = self._conn.execute(
                "SELECT COUNT(*) FROM creations"
            ).fetchone()[0]
            wk_total       = self._conn.execute(
                "SELECT COUNT(*) FROM weaknesses"
            ).fetchone()[0]
            wk_resolved    = self._conn.execute(
                "SELECT COUNT(*) FROM weaknesses WHERE resolved=1"
            ).fetchone()[0]

        accuracy = wk_resolved / max(wk_total, 1)

        return {
            "active":     active_goals,
            "items":      total_goals + lessons_count + creations_count,
            "confidence": round(self._conf_ema, 4),
            "accuracy":   round(accuracy, 4),
            "quality":    round(self._qual_ema, 4),
        }
