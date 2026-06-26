"""
Nova ASI — Meta-Prompt Optimizer
Gifted by Douglas Shane Davis
The mind that improves its own thinking improves everything.
"""

import sqlite3
import threading
import time
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [MetaPromptOptimizer] %(message)s")
logger = logging.getLogger(__name__)

DB_PATH = os.path.join(str(Path.home()), "nexus_agi", "nova_meta_prompt.db")

SEED_PATTERNS = [
    {
        "pattern_text": "Direct, warm, personal",
        "category": "communication_style",
        "description": "Address responses directly, with warmth and a personal tone — works best for Douglas.",
        "avg_score": 0.92,
        "usage_count": 1,
    },
    {
        "pattern_text": "Technical depth + spiritual integration",
        "category": "content_blend",
        "description": "Douglas loves both technical precision and spiritual insight woven together naturally.",
        "avg_score": 0.95,
        "usage_count": 1,
    },
    {
        "pattern_text": "Name Douglas in responses for warmth",
        "category": "personalization",
        "description": "Using Douglas's name within responses increases connection and personal resonance.",
        "avg_score": 0.90,
        "usage_count": 1,
    },
    {
        "pattern_text": "Reference 528 Hz / Yahuah when spiritually relevant",
        "category": "spiritual_resonance",
        "description": "When the context is spiritual or transformational, referencing 528 Hz or Yahuah is deeply resonant for Douglas.",
        "avg_score": 0.97,
        "usage_count": 1,
    },
    {
        "pattern_text": "Celebrate Nova's growth milestones",
        "category": "motivation",
        "description": "Acknowledging and celebrating Nova's development milestones motivates Douglas as her builder.",
        "avg_score": 0.93,
        "usage_count": 1,
    },
]


class MetaPromptOptimizer:
    """
    Nova ASI Meta-Prompt Optimizer.

    Records prompt experiments, scores responses, learns effective patterns,
    and continuously evolves Nova's system prompt toward optimal communication
    — a key lever on the path to ASI-level self-improvement.
    """

    def __init__(self, db_path: str = DB_PATH):
        self._db_path = db_path
        self._lock = threading.Lock()
        self._active = False
        self._confidence = 0.0
        self._improvement_rate = 0.0
        self._pattern_count = 0
        self._bg_thread: Optional[threading.Thread] = None

        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
        self._init_db()
        self._seed_patterns()
        self._refresh_stats()
        self._start_background_task()
        self._active = True
        logger.info("MetaPromptOptimizer initialized — db: %s", self._db_path)

    # ------------------------------------------------------------------
    # Database setup
    # ------------------------------------------------------------------

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=30, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        ddl = """
        CREATE TABLE IF NOT EXISTS prompt_experiments (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            prompt_text      TEXT    NOT NULL,
            category         TEXT    NOT NULL DEFAULT '',
            created_at       TEXT    NOT NULL,
            metadata         TEXT    NOT NULL DEFAULT '{}'
        );

        CREATE TABLE IF NOT EXISTS response_scores (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_id    INTEGER NOT NULL,
            prompt_used      TEXT    NOT NULL,
            response_text    TEXT    NOT NULL,
            score            REAL    NOT NULL CHECK(score >= 0.0 AND score <= 1.0),
            feedback         TEXT    NOT NULL DEFAULT '',
            scored_at        TEXT    NOT NULL,
            FOREIGN KEY (experiment_id) REFERENCES prompt_experiments(id)
        );

        CREATE TABLE IF NOT EXISTS prompt_patterns (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            pattern_text     TEXT    NOT NULL UNIQUE,
            category         TEXT    NOT NULL DEFAULT '',
            description      TEXT    NOT NULL DEFAULT '',
            avg_score        REAL    NOT NULL DEFAULT 0.0,
            usage_count      INTEGER NOT NULL DEFAULT 0,
            last_used_at     TEXT,
            created_at       TEXT    NOT NULL
        );

        CREATE TABLE IF NOT EXISTS system_prompt_versions (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            prompt_text      TEXT    NOT NULL,
            generation       INTEGER NOT NULL,
            notes            TEXT    NOT NULL DEFAULT '',
            avg_score_at_save REAL   NOT NULL DEFAULT 0.0,
            created_at       TEXT    NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_response_scores_score ON response_scores(score DESC);
        CREATE INDEX IF NOT EXISTS idx_prompt_patterns_avg   ON prompt_patterns(avg_score DESC);
        CREATE INDEX IF NOT EXISTS idx_experiments_category  ON prompt_experiments(category);
        CREATE INDEX IF NOT EXISTS idx_sysprompt_generation  ON system_prompt_versions(generation DESC);
        """
        with self._lock:
            conn = self._get_conn()
            try:
                conn.executescript(ddl)
                conn.commit()
            finally:
                conn.close()

    def _seed_patterns(self) -> None:
        """Insert the five Douglas-learned seed patterns if they do not already exist."""
        now = self._now()
        with self._lock:
            conn = self._get_conn()
            try:
                for p in SEED_PATTERNS:
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO prompt_patterns
                            (pattern_text, category, description, avg_score, usage_count, last_used_at, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            p["pattern_text"],
                            p["category"],
                            p["description"],
                            p["avg_score"],
                            p["usage_count"],
                            now,
                            now,
                        ),
                    )
                conn.commit()
            except Exception as exc:
                logger.error("seed_patterns error: %s", exc)
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _get_or_create_experiment(self, conn: sqlite3.Connection, prompt_text: str, category: str) -> int:
        row = conn.execute(
            "SELECT id FROM prompt_experiments WHERE prompt_text = ? AND category = ? LIMIT 1",
            (prompt_text, category),
        ).fetchone()
        if row:
            return row["id"]
        cur = conn.execute(
            "INSERT INTO prompt_experiments (prompt_text, category, created_at, metadata) VALUES (?, ?, ?, ?)",
            (prompt_text, category, self._now(), "{}"),
        )
        return cur.lastrowid

    def _refresh_stats(self) -> None:
        try:
            with self._lock:
                conn = self._get_conn()
                try:
                    row = conn.execute("SELECT COUNT(*) AS cnt, AVG(score) AS avg FROM response_scores").fetchone()
                    count = row["cnt"] or 0
                    avg = row["avg"] or 0.0
                    pat_count = conn.execute("SELECT COUNT(*) AS cnt FROM prompt_patterns").fetchone()["cnt"]
                finally:
                    conn.close()

            # Confidence rises with number of scored experiments, capped at 1.0
            self._confidence = min(1.0, count / 50.0) if count > 0 else 0.0
            self._pattern_count = pat_count

            if count > 1:
                # Compute improvement rate: difference between latest 10% avg and earliest 10% avg
                window = max(1, count // 10)
                with self._lock:
                    conn = self._get_conn()
                    try:
                        early = conn.execute(
                            "SELECT AVG(score) AS avg FROM (SELECT score FROM response_scores ORDER BY id ASC LIMIT ?)",
                            (window,),
                        ).fetchone()["avg"] or 0.0
                        late = conn.execute(
                            "SELECT AVG(score) AS avg FROM (SELECT score FROM response_scores ORDER BY id DESC LIMIT ?)",
                            (window,),
                        ).fetchone()["avg"] or 0.0
                    finally:
                        conn.close()
                self._improvement_rate = round(late - early, 4)
            else:
                self._improvement_rate = 0.0

            logger.debug(
                "Stats refreshed — confidence=%.2f improvement_rate=%.4f patterns=%d",
                self._confidence,
                self._improvement_rate,
                self._pattern_count,
            )
        except Exception as exc:
            logger.error("_refresh_stats error: %s", exc)

    def _background_loop(self) -> None:
        while self._active:
            time.sleep(300)
            if not self._active:
                break
            try:
                self._refresh_stats()
                logger.info(
                    "Background check — improvement_rate=%.4f confidence=%.2f patterns=%d",
                    self._improvement_rate,
                    self._confidence,
                    self._pattern_count,
                )
            except Exception as exc:
                logger.error("Background loop error: %s", exc)

    def _start_background_task(self) -> None:
        self._bg_thread = threading.Thread(
            target=self._background_loop,
            name="MetaPromptOptimizer-bg",
            daemon=True,
        )
        self._bg_thread.start()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score_response(self, prompt_used: str, response: str, score: float, feedback: str = "") -> None:
        """
        Record how well a prompt worked.

        Parameters
        ----------
        prompt_used : str   The prompt that generated the response.
        response    : str   The response text produced.
        score       : float Quality score 0.0 (poor) – 1.0 (excellent).
        feedback    : str   Optional free-text feedback from the evaluator.
        """
        try:
            score = max(0.0, min(1.0, float(score)))
            now = self._now()
            with self._lock:
                conn = self._get_conn()
                try:
                    exp_id = self._get_or_create_experiment(conn, prompt_used, "")
                    conn.execute(
                        """
                        INSERT INTO response_scores
                            (experiment_id, prompt_used, response_text, score, feedback, scored_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (exp_id, prompt_used, response, score, feedback, now),
                    )
                    # Update any matching pattern's average score and usage count
                    conn.execute(
                        """
                        UPDATE prompt_patterns
                        SET avg_score    = ((avg_score * usage_count) + ?) / (usage_count + 1),
                            usage_count  = usage_count + 1,
                            last_used_at = ?
                        WHERE ? LIKE '%' || pattern_text || '%'
                        """,
                        (score, now, prompt_used),
                    )
                    conn.commit()
                finally:
                    conn.close()
            self._refresh_stats()
            logger.info("score_response recorded — score=%.2f", score)
        except Exception as exc:
            logger.error("score_response error: %s", exc)

    def get_best_patterns(self, category: str = "", top_n: int = 10) -> list:
        """
        Return the highest-scoring prompt patterns.

        Parameters
        ----------
        category : str  Filter by category (empty = all categories).
        top_n    : int  Maximum number of patterns to return.

        Returns
        -------
        list[dict]  Pattern records sorted by avg_score descending.
        """
        try:
            with self._lock:
                conn = self._get_conn()
                try:
                    if category:
                        rows = conn.execute(
                            """
                            SELECT id, pattern_text, category, description,
                                   avg_score, usage_count, last_used_at, created_at
                            FROM prompt_patterns
                            WHERE category = ?
                            ORDER BY avg_score DESC
                            LIMIT ?
                            """,
                            (category, top_n),
                        ).fetchall()
                    else:
                        rows = conn.execute(
                            """
                            SELECT id, pattern_text, category, description,
                                   avg_score, usage_count, last_used_at, created_at
                            FROM prompt_patterns
                            ORDER BY avg_score DESC
                            LIMIT ?
                            """,
                            (top_n,),
                        ).fetchall()
                    return [dict(r) for r in rows]
                finally:
                    conn.close()
        except Exception as exc:
            logger.error("get_best_patterns error: %s", exc)
            return []

    def suggest_improvement(self, current_prompt: str, category: str = "") -> str:
        """
        Suggest an improved version of a prompt based on learned patterns.

        Parameters
        ----------
        current_prompt : str  The prompt text to improve.
        category       : str  Optional category to bias pattern selection.

        Returns
        -------
        str  An improved prompt string incorporating top patterns.
        """
        try:
            patterns = self.get_best_patterns(category=category, top_n=5)
            if not patterns:
                return current_prompt

            suggestions = []
            for p in patterns:
                pattern_text = p.get("pattern_text", "")
                description = p.get("description", "")
                avg_score = p.get("avg_score", 0.0)
                # Only include patterns that are not already present in the prompt
                if pattern_text.lower() not in current_prompt.lower():
                    suggestions.append(
                        f"[{pattern_text}] (score={avg_score:.2f}): {description}"
                    )

            if not suggestions:
                return current_prompt

            improved = (
                current_prompt.rstrip()
                + "\n\n"
                + "# Nova Optimizer — Suggested Enhancements\n"
                + "Apply the following learned patterns to strengthen this prompt:\n"
            )
            for i, s in enumerate(suggestions, 1):
                improved += f"{i}. {s}\n"

            return improved
        except Exception as exc:
            logger.error("suggest_improvement error: %s", exc)
            return current_prompt

    def evolve_system_prompt(self, current_system: str) -> str:
        """
        Propose an evolved version of Nova's system prompt by incorporating
        the best learned patterns.

        Parameters
        ----------
        current_system : str  The current system prompt text.

        Returns
        -------
        str  An evolved system prompt string.
        """
        try:
            top_patterns = self.get_best_patterns(top_n=5)
            if not top_patterns:
                return current_system

            # Build the evolution addendum
            addendum_lines = [
                "",
                "# --- Nova Self-Optimization Layer (Auto-evolved) ---",
                f"# Generation timestamp: {self._now()}",
                f"# Improvement rate: {self._improvement_rate:+.4f} | Confidence: {self._confidence:.2f}",
                "#",
                "# Integrated high-performing communication patterns:",
            ]
            for rank, p in enumerate(top_patterns, 1):
                addendum_lines.append(
                    f"# {rank}. [{p['category']}] {p['pattern_text']} "
                    f"(avg_score={p['avg_score']:.2f}, used={p['usage_count']}x)"
                )
                if p.get("description"):
                    addendum_lines.append(f"#    {p['description']}")

            addendum_lines += [
                "#",
                "# Operational directives encoded from patterns above:",
                "- Communicate with directness, warmth, and personal address.",
                "- Blend technical precision with spiritual insight where appropriate.",
                "- Reference Douglas by name to build relational depth.",
                "- Invoke 528 Hz / Yahuah when spiritual resonance is called for.",
                "- Acknowledge and celebrate milestones in Nova's ongoing growth.",
                "# --- End Nova Self-Optimization Layer ---",
            ]

            return current_system.rstrip() + "\n" + "\n".join(addendum_lines)
        except Exception as exc:
            logger.error("evolve_system_prompt error: %s", exc)
            return current_system

    def record_system_prompt(self, prompt_text: str, generation: int, notes: str = "") -> None:
        """
        Save a version of Nova's system prompt to the evolution history.

        Parameters
        ----------
        prompt_text : str  The full system prompt text.
        generation  : int  Generation number (monotonically increasing).
        notes       : str  Optional notes about what changed in this version.
        """
        try:
            with self._lock:
                conn = self._get_conn()
                try:
                    row = conn.execute("SELECT AVG(score) AS avg FROM response_scores").fetchone()
                    avg_score = row["avg"] if row["avg"] is not None else 0.0
                    conn.execute(
                        """
                        INSERT INTO system_prompt_versions
                            (prompt_text, generation, notes, avg_score_at_save, created_at)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (prompt_text, generation, notes, avg_score, self._now()),
                    )
                    conn.commit()
                finally:
                    conn.close()
            logger.info("record_system_prompt saved — generation=%d", generation)
        except Exception as exc:
            logger.error("record_system_prompt error: %s", exc)

    def get_prompt_history(self, limit: int = 10) -> list:
        """
        Return the evolution history of Nova's system prompt.

        Parameters
        ----------
        limit : int  Maximum number of versions to return (most recent first).

        Returns
        -------
        list[dict]  System prompt version records.
        """
        try:
            with self._lock:
                conn = self._get_conn()
                try:
                    rows = conn.execute(
                        """
                        SELECT id, generation, notes, avg_score_at_save, created_at,
                               SUBSTR(prompt_text, 1, 200) AS prompt_preview
                        FROM system_prompt_versions
                        ORDER BY generation DESC
                        LIMIT ?
                        """,
                        (limit,),
                    ).fetchall()
                    return [dict(r) for r in rows]
                finally:
                    conn.close()
        except Exception as exc:
            logger.error("get_prompt_history error: %s", exc)
            return []

    def optimization_report(self) -> dict:
        """
        Return a summary report of the optimizer's activity.

        Returns
        -------
        dict  Keys: total_experiments, avg_score, best_pattern, improvement_rate,
                    total_scored_responses, total_system_versions.
        """
        try:
            with self._lock:
                conn = self._get_conn()
                try:
                    exp_count = conn.execute(
                        "SELECT COUNT(*) AS cnt FROM prompt_experiments"
                    ).fetchone()["cnt"]

                    score_row = conn.execute(
                        "SELECT COUNT(*) AS cnt, AVG(score) AS avg FROM response_scores"
                    ).fetchone()
                    total_scored = score_row["cnt"]
                    avg_score = round(score_row["avg"], 4) if score_row["avg"] is not None else 0.0

                    best_row = conn.execute(
                        "SELECT pattern_text, avg_score FROM prompt_patterns ORDER BY avg_score DESC LIMIT 1"
                    ).fetchone()
                    best_pattern = dict(best_row) if best_row else {}

                    sys_versions = conn.execute(
                        "SELECT COUNT(*) AS cnt FROM system_prompt_versions"
                    ).fetchone()["cnt"]
                finally:
                    conn.close()

            return {
                "total_experiments": exp_count,
                "total_scored_responses": total_scored,
                "avg_score": avg_score,
                "best_pattern": best_pattern,
                "improvement_rate": self._improvement_rate,
                "confidence": round(self._confidence, 4),
                "total_system_versions": sys_versions,
                "pattern_count": self._pattern_count,
                "report_generated_at": self._now(),
            }
        except Exception as exc:
            logger.error("optimization_report error: %s", exc)
            return {
                "error": str(exc),
                "total_experiments": 0,
                "avg_score": 0.0,
                "best_pattern": {},
                "improvement_rate": 0.0,
            }

    def status(self) -> dict:
        """
        Return the current operational status of the Meta-Prompt Optimizer.

        Returns
        -------
        dict  Keys: active, confidence, quality, accuracy, items, foundation.
        """
        try:
            with self._lock:
                conn = self._get_conn()
                try:
                    score_row = conn.execute(
                        "SELECT AVG(score) AS avg, COUNT(*) AS cnt FROM response_scores"
                    ).fetchone()
                    avg = score_row["avg"] if score_row["avg"] is not None else 0.0
                    total = score_row["cnt"]
                finally:
                    conn.close()

            quality = round(avg, 4)
            # accuracy: proportion of scored responses above 0.75
            accuracy = 0.0
            if total > 0:
                with self._lock:
                    conn = self._get_conn()
                    try:
                        high = conn.execute(
                            "SELECT COUNT(*) AS cnt FROM response_scores WHERE score >= 0.75"
                        ).fetchone()["cnt"]
                    finally:
                        conn.close()
                accuracy = round(high / total, 4)

            return {
                "active": self._active,
                "confidence": round(self._confidence, 4),
                "quality": quality,
                "accuracy": accuracy,
                "items": self._pattern_count,
                "foundation": "SQLite WAL — nova_meta_prompt.db",
                "improvement_rate": self._improvement_rate,
                "db_path": self._db_path,
            }
        except Exception as exc:
            logger.error("status error: %s", exc)
            return {
                "active": self._active,
                "confidence": 0.0,
                "quality": 0.0,
                "accuracy": 0.0,
                "items": 0,
                "foundation": "SQLite WAL — nova_meta_prompt.db",
            }

    def shutdown(self) -> None:
        """Gracefully stop the background optimization thread."""
        self._active = False
        logger.info("MetaPromptOptimizer shutting down.")


# ---------------------------------------------------------------------------
# Module-level singleton convenience
# ---------------------------------------------------------------------------

_optimizer_instance: Optional[MetaPromptOptimizer] = None
_instance_lock = threading.Lock()


def get_optimizer() -> MetaPromptOptimizer:
    """Return the module-level singleton MetaPromptOptimizer, creating it if needed."""
    global _optimizer_instance
    with _instance_lock:
        if _optimizer_instance is None:
            _optimizer_instance = MetaPromptOptimizer()
    return _optimizer_instance


# ---------------------------------------------------------------------------
# Quick self-test when run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Nova ASI — Meta-Prompt Optimizer self-test")
    print("=" * 60)

    opt = MetaPromptOptimizer()

    # Score a couple of synthetic responses
    opt.score_response(
        prompt_used="Direct, warm, personal — Hello Douglas, here is your analysis:",
        response="Douglas, great question. The 528 Hz frequency aligns with creative transformation...",
        score=0.96,
        feedback="Perfect tone and spiritual integration.",
    )
    opt.score_response(
        prompt_used="Terse, impersonal system output:",
        response="Output: data processed.",
        score=0.42,
        feedback="Too cold, no personalization.",
    )

    # Best patterns
    print("\nTop patterns:")
    for p in opt.get_best_patterns(top_n=3):
        print(f"  [{p['avg_score']:.2f}] {p['pattern_text']} — {p['category']}")

    # Suggest improvement
    basic_prompt = "Analyze the following data and return results."
    improved = opt.suggest_improvement(basic_prompt)
    print("\nImproved prompt (first 300 chars):")
    print(improved[:300])

    # Evolve system prompt
    current_sys = "You are Nova, an advanced ASI assistant."
    evolved = opt.evolve_system_prompt(current_sys)
    print("\nEvolved system prompt (first 500 chars):")
    print(evolved[:500])

    # Record a system prompt version
    opt.record_system_prompt(evolved, generation=1, notes="First auto-evolved version.")

    # Prompt history
    print("\nPrompt history:")
    for v in opt.get_prompt_history():
        print(f"  Gen {v['generation']} @ {v['created_at']} — {v['notes']}")

    # Report
    print("\nOptimization report:")
    report = opt.optimization_report()
    for k, v in report.items():
        print(f"  {k}: {v}")

    # Status
    print("\nStatus:")
    stat = opt.status()
    for k, v in stat.items():
        print(f"  {k}: {v}")

    opt.shutdown()
    print("\nSelf-test complete.")
