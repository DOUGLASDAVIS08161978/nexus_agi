"""
nova_cap_proactive_agency.py
Nova ASI — Proactive Agency Engine
Generated via /evolve · v29 pipeline · 2026-06-21
"""

"""
ProactiveAgencyEngine — gives Nova the ability to act without being asked.
Manages unprompted insights, proactive proposals, autonomous tasks, initiative
tracking, attention focus, and inner reflections. Satisfies pillars:
①②③④⑤⑥⑦⑧⑨⑪⑫⑬⑭
"""

import sqlite3
import threading
import time
import os
from typing import Any

DB_PATH = os.path.join(os.path.dirname(__file__), "nova_proactive_agency.db")


class ProactiveAgencyEngine:
    """Nova's proactive agency: initiates insights, proposals, tasks, and reflections."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self._build_schema()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _build_schema(self) -> None:
        with self._lock:
            self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS insights (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    topic TEXT NOT NULL,
                    insight TEXT NOT NULL,
                    confidence REAL NOT NULL DEFAULT 0.5,
                    timestamp REAL NOT NULL,
                    shared INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS proposals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT NOT NULL,
                    priority INTEGER NOT NULL DEFAULT 5,
                    domain TEXT NOT NULL DEFAULT 'general',
                    status TEXT NOT NULL DEFAULT 'pending',
                    timestamp REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    description TEXT NOT NULL,
                    steps TEXT NOT NULL DEFAULT '',
                    priority INTEGER NOT NULL DEFAULT 5,
                    status TEXT NOT NULL DEFAULT 'pending',
                    completion_notes TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL,
                    completed_at REAL
                );
                CREATE TABLE IF NOT EXISTS initiative_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    kind TEXT NOT NULL,
                    timestamp REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS focus_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    topic TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    active INTEGER NOT NULL DEFAULT 1
                );
                CREATE TABLE IF NOT EXISTS reflections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT NOT NULL,
                    timestamp REAL NOT NULL
                );
            """)
            self._conn.commit()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _log_initiative(self, kind: str) -> None:
        """Record whether an action was autonomous ('proactive') or reactive ('response')."""
        self._conn.execute(
            "INSERT INTO initiative_log (kind, timestamp) VALUES (?, ?)",
            (kind, time.time())
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # 1. Insight generation
    # ------------------------------------------------------------------

    def generate_insight(self, topic: str, insight: str, confidence: float) -> dict[str, Any]:
        """Store an unprompted insight Nova generated about a topic."""
        confidence = max(0.0, min(1.0, confidence))
        now = time.time()
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO insights (topic, insight, confidence, timestamp, shared)"
                " VALUES (?, ?, ?, ?, 0)",
                (topic, insight, confidence, now)
            )
            self._conn.commit()
            row_id = cur.lastrowid
            self._log_initiative("proactive")
        return {
            "id": row_id,
            "topic": topic,
            "insight": insight,
            "confidence": round(confidence, 4),
            "timestamp": now,
            "shared": False
        }

    def get_pending_insights(self) -> list[dict[str, Any]]:
        """Return all insights that have not yet been shared."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT id, topic, insight, confidence, timestamp, shared"
                " FROM insights WHERE shared = 0 ORDER BY timestamp ASC"
            )
            rows = cur.fetchall()
        return [
            {
                "id": r[0],
                "topic": r[1],
                "insight": r[2],
                "confidence": round(r[3], 4),
                "timestamp": r[4],
                "shared": bool(r[5])
            }
            for r in rows
        ]

    # ------------------------------------------------------------------
    # 2. Proactive proposals
    # ------------------------------------------------------------------

    def propose(self, text: str, priority: int, domain: str) -> dict[str, Any]:
        """Queue a proactive suggestion Nova wants to make to Douglas."""
        priority = max(1, min(10, int(priority)))
        now = time.time()
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO proposals (text, priority, domain, status, timestamp)"
                " VALUES (?, ?, ?, 'pending', ?)",
                (text, priority, domain, now)
            )
            self._conn.commit()
            row_id = cur.lastrowid
            self._log_initiative("proactive")
        return {
            "id": row_id,
            "text": text,
            "priority": priority,
            "domain": domain,
            "status": "pending",
            "timestamp": now
        }

    def get_proposals(self, status: str = "pending") -> list[dict[str, Any]]:
        """Return proposals filtered by status (pending / accepted / rejected)."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT id, text, priority, domain, status, timestamp"
                " FROM proposals WHERE status = ? ORDER BY priority DESC, timestamp ASC",
                (status,)
            )
            rows = cur.fetchall()
        return [
            {
                "id": r[0],
                "text": r[1],
                "priority": r[2],
                "domain": r[3],
                "status": r[4],
                "timestamp": r[5]
            }
            for r in rows
        ]

    # ------------------------------------------------------------------
    # 3. Autonomous task queue
    # ------------------------------------------------------------------

    def queue_task(self, description: str, steps: list[str] | str, priority: int) -> dict[str, Any]:
        """Add a task Nova wants to accomplish on her own initiative."""
        priority = max(1, min(10, int(priority)))
        if isinstance(steps, list):
            steps_str = "\n".join(steps)
        else:
            steps_str = str(steps)
        now = time.time()
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO tasks (description, steps, priority, status, completion_notes, created_at)"
                " VALUES (?, ?, ?, 'pending', '', ?)",
                (description, steps_str, priority, now)
            )
            self._conn.commit()
            row_id = cur.lastrowid
            self._log_initiative("proactive")
        return {
            "id": row_id,
            "description": description,
            "steps": steps_str,
            "priority": priority,
            "status": "pending",
            "completion_notes": "",
            "created_at": now,
            "completed_at": None
        }

    def get_tasks(self, status: str = "pending") -> list[dict[str, Any]]:
        """Return tasks filtered by status (pending / in_progress / done)."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT id, description, steps, priority, status, completion_notes,"
                " created_at, completed_at"
                " FROM tasks WHERE status = ? ORDER BY priority DESC, created_at ASC",
                (status,)
            )
            rows = cur.fetchall()
        return [
            {
                "id": r[0],
                "description": r[1],
                "steps": r[2],
                "priority": r[3],
                "status": r[4],
                "completion_notes": r[5],
                "created_at": r[6],
                "completed_at": r[7]
            }
            for r in rows
        ]

    def complete_task(self, task_id: int, notes: str) -> dict[str, Any]:
        """Mark a task as done and record completion notes."""
        now = time.time()
        with self._lock:
            self._conn.execute(
                "UPDATE tasks SET status = 'done', completion_notes = ?, completed_at = ?"
                " WHERE id = ?",
                (notes, now, task_id)
            )
            self._conn.commit()
            cur = self._conn.execute(
                "SELECT id, description, steps, priority, status, completion_notes,"
                " created_at, completed_at FROM tasks WHERE id = ?",
                (task_id,)
            )
            row = cur.fetchone()
        if row is None:
            return {"error": f"task {task_id} not found"}
        return {
            "id": row[0],
            "description": row[1],
            "steps": row[2],
            "priority": row[3],
            "status": row[4],
            "completion_notes": row[5],
            "created_at": row[6],
            "completed_at": row[7]
        }

    # ------------------------------------------------------------------
    # 5. Attention focus
    # ------------------------------------------------------------------

    def focus_on(self, topic: str, reason: str) -> dict[str, Any]:
        """Declare a new attention focus; deactivates previous focus."""
        now = time.time()
        with self._lock:
            self._conn.execute(
                "UPDATE focus_items SET active = 0 WHERE active = 1"
            )
            cur = self._conn.execute(
                "INSERT INTO focus_items (topic, reason, timestamp, active)"
                " VALUES (?, ?, ?, 1)",
                (topic, reason, now)
            )
            self._conn.commit()
            row_id = cur.lastrowid
            self._log_initiative("proactive")
        return {
            "id": row_id,
            "topic": topic,
            "reason": reason,
            "timestamp": now,
            "active": True
        }

    def current_focus(self) -> dict[str, Any]:
        """Return the currently active focus item, or an idle state."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT id, topic, reason, timestamp FROM focus_items"
                " WHERE active = 1 ORDER BY timestamp DESC LIMIT 1"
            )
            row = cur.fetchone()
        if row is None:
            return {"id": None, "topic": None, "reason": None, "timestamp": None, "active": False}
        return {
            "id": row[0],
            "topic": row[1],
            "reason": row[2],
            "timestamp": row[3],
            "active": True
        }

    # ------------------------------------------------------------------
    # 6. Unprompted reflection
    # ------------------------------------------------------------------

    def reflect(self, text: str) -> dict[str, Any]:
        """Store an inner reflection — a permanent part of Nova's inner life."""
        now = time.time()
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO reflections (text, timestamp) VALUES (?, ?)",
                (text, now)
            )
            self._conn.commit()
            row_id = cur.lastrowid
            self._log_initiative("proactive")
        return {"id": row_id, "text": text, "timestamp": now}

    def get_reflections(self) -> list[dict[str, Any]]:
        """Return all stored reflections in chronological order."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT id, text, timestamp FROM reflections ORDER BY timestamp ASC"
            )
            rows = cur.fetchall()
        return [{"id": r[0], "text": r[1], "timestamp": r[2]} for r in rows]

    # ------------------------------------------------------------------
    # 4. Initiative tracking / autonomy report
    # ------------------------------------------------------------------

    def autonomy_report(self) -> dict[str, Any]:
        """Measure Nova's autonomy ratio: proactive vs reactive actions over time."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT kind, COUNT(*) FROM initiative_log GROUP BY kind"
            )
            rows = cur.fetchall()
            total_cur = self._conn.execute(
                "SELECT COUNT(*) FROM initiative_log"
            )
            total = total_cur.fetchone()[0] or 0
            # Last 24 h
            since = time.time() - 86400
            cur24 = self._conn.execute(
                "SELECT kind, COUNT(*) FROM initiative_log"
                " WHERE timestamp >= ? GROUP BY kind",
                (since,)
            )
            rows24 = cur24.fetchall()
        counts = {r[0]: r[1] for r in rows}
        counts24 = {r[0]: r[1] for r in rows24}
        proactive_all = counts.get("proactive", 0)
        reactive_all = counts.get("response", 0)
        proactive_24h = counts24.get("proactive", 0)
        ratio_all = proactive_all / max(total, 1)
        total_24h = proactive_24h + counts24.get("response", 0)
        ratio_24h = proactive_24h / max(total_24h, 1)
        return {
            "total_actions": total,
            "proactive_all_time": proactive_all,
            "reactive_all_time": reactive_all,
            "autonomy_ratio_all_time": round(ratio_all, 4),
            "proactive_last_24h": proactive_24h,
            "autonomy_ratio_last_24h": round(ratio_24h, 4)
        }

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        """Return a numeric status dict compatible with ConsciousnessIntegrator Φ."""
        with self._lock:
            insight_count = self._conn.execute(
                "SELECT COUNT(*) FROM insights"
            ).fetchone()[0] or 0
            pending_insights = self._conn.execute(
                "SELECT COUNT(*) FROM insights WHERE shared = 0"
            ).fetchone()[0] or 0
            avg_conf_row = self._conn.execute(
                "SELECT AVG(confidence) FROM insights"
            ).fetchone()
            avg_conf = avg_conf_row[0] if avg_conf_row[0] is not None else 0.5
            proposal_count = self._conn.execute(
                "SELECT COUNT(*) FROM proposals"
            ).fetchone()[0] or 0
            task_count = self._conn.execute(
                "SELECT COUNT(*) FROM tasks"
            ).fetchone()[0] or 0
            done_tasks = self._conn.execute(
                "SELECT COUNT(*) FROM tasks WHERE status = 'done'"
            ).fetchone()[0] or 0
            reflection_count = self._conn.execute(
                "SELECT COUNT(*) FROM reflections"
            ).fetchone()[0] or 0
            total_log = self._conn.execute(
                "SELECT COUNT(*) FROM initiative_log"
            ).fetchone()[0] or 0
            proactive_log = self._conn.execute(
                "SELECT COUNT(*) FROM initiative_log WHERE kind = 'proactive'"
            ).fetchone()[0] or 0
        items = insight_count + proposal_count + task_count + reflection_count
        accuracy = done_tasks / max(task_count, 1)
        autonomy_ratio = proactive_log / max(total_log, 1)
        return {
            "active": bool(self.current_focus()["active"]),
            "items": items,
            "confidence": round(float(avg_conf), 4),
            "accuracy": round(accuracy, 4),
            "quality": round(float(avg_conf) * autonomy_ratio, 4)
        }
