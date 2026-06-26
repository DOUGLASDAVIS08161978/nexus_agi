"""
Nova ASI — Hierarchical Goal Planner with Dependency Tracking
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Goals decompose into sub-goals and atomic tasks forming a tree.
Priority = base_priority × (1 - progress). Dependency-blocked goals
are excluded from next_action(). Topological sort respects ordering.
Failed goals reschedule as 'retry'. SQLite persists goals across sessions.
Background frontier check runs every 5 minutes.

Built by Douglas Shane Davis & Claude Code (Anthropic)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os, sqlite3, threading, time
from typing import Any, Dict, List, Optional
from datetime import datetime

_DB = os.path.expanduser("~/nexus_agi/nova_goals.db")


class HierarchicalGoalPlanner:
    """
    Goal-directed planning engine with hierarchical decomposition.

    Goals form a parent/child tree. Priority queue ensures the highest-
    priority unblocked leaf task executes next. Progress propagates
    upward when children complete. Failed tasks reschedule automatically.
    Topological sort respects declared dependencies between goals.
    """

    def __init__(self) -> None:
        self._lock    = threading.Lock()
        self._next_id = 1
        self.conn = sqlite3.connect(_DB, check_same_thread=False)
        self._init_db()
        self._next_id = self._max_id() + 1
        self._start_daemon()

    # ─── Database ──────────────────────────────────────────────────────

    def _init_db(self) -> None:
        with self._lock:
            self.conn.executescript("""
                CREATE TABLE IF NOT EXISTS goals (
                    id        INTEGER PRIMARY KEY,
                    desc      TEXT NOT NULL,
                    priority  REAL DEFAULT 1.0,
                    parent_id INTEGER DEFAULT NULL,
                    status    TEXT DEFAULT 'pending',
                    progress  REAL DEFAULT 0.0,
                    created   TEXT DEFAULT (datetime('now')),
                    updated   TEXT DEFAULT (datetime('now')),
                    outcome   TEXT DEFAULT ''
                );
                CREATE TABLE IF NOT EXISTS dependencies (
                    goal_id    INTEGER,
                    depends_on INTEGER,
                    PRIMARY KEY (goal_id, depends_on)
                );
                CREATE TABLE IF NOT EXISTS goal_log (
                    id      INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts      TEXT DEFAULT (datetime('now')),
                    goal_id INTEGER,
                    event   TEXT
                );
            """)
            self.conn.commit()

    def _max_id(self) -> int:
        row = self.conn.execute("SELECT MAX(id) FROM goals").fetchone()
        return int(row[0]) if row and row[0] else 0

    def _log(self, goal_id: int, event: str) -> None:
        try:
            self.conn.execute(
                "INSERT INTO goal_log (goal_id, event) VALUES (?,?)",
                (goal_id, event))
            self.conn.commit()
        except Exception:
            pass

    # ─── Goal creation ─────────────────────────────────────────────────

    def add_goal(self, desc: str, priority: float = 1.0,
                 parent_id: Optional[int] = None,
                 dependencies: Optional[List[int]] = None) -> int:
        """Create a goal and return its ID."""
        with self._lock:
            gid = self._next_id
            self._next_id += 1
            p = max(0.0, min(10.0, float(priority)))
            self.conn.execute(
                "INSERT INTO goals (id, desc, priority, parent_id) VALUES (?,?,?,?)",
                (gid, desc[:500], p, parent_id))
            for dep in (dependencies or []):
                self.conn.execute(
                    "INSERT OR IGNORE INTO dependencies VALUES (?,?)",
                    (gid, int(dep)))
            self.conn.commit()
        self._log(gid, "created")
        return gid

    def decompose(self, goal_id: int, steps: List[str]) -> List[int]:
        """Break a goal into ordered sub-steps; each step depends on the previous."""
        parent = self._get(goal_id)
        if not parent:
            return []
        ids, prev = [], None
        for i, step in enumerate(steps):
            p   = float(parent["priority"]) * (1.0 - i * 0.04)
            cid = self.add_goal(step, priority=p, parent_id=goal_id,
                                dependencies=[prev] if prev else None)
            ids.append(cid)
            prev = cid
        return ids

    def _get(self, goal_id: int) -> Optional[Dict]:
        row = self.conn.execute(
            "SELECT id, desc, priority, parent_id, status, progress, outcome "
            "FROM goals WHERE id=?", (goal_id,)).fetchone()
        if not row:
            return None
        return dict(zip(
            ["id", "desc", "priority", "parent_id", "status", "progress", "outcome"],
            row))

    # ─── Planning core ─────────────────────────────────────────────────

    def next_action(self) -> Optional[Dict]:
        """
        Highest-priority unblocked leaf goal.
        Score = priority × (1 - progress). Blocked goals excluded.
        """
        rows = self.conn.execute(
            "SELECT id, desc, priority, progress FROM goals "
            "WHERE status IN ('pending','retry')"
        ).fetchall()
        candidates = []
        for gid, desc, priority, progress in rows:
            if self._is_leaf(gid) and not self._is_blocked(gid):
                score = float(priority) * (1.0 - float(progress))
                candidates.append((score, gid, desc, float(priority), float(progress)))
        if not candidates:
            return None
        candidates.sort(reverse=True)
        _, gid, desc, priority, progress = candidates[0]
        return {"id": gid, "desc": desc, "priority": priority,
                "progress": progress, "blocked_by": []}

    def _is_leaf(self, goal_id: int) -> bool:
        count = self.conn.execute(
            "SELECT COUNT(*) FROM goals WHERE parent_id=? "
            "AND status NOT IN ('done','cancelled')", (goal_id,)).fetchone()[0]
        return count == 0

    def _is_blocked(self, goal_id: int) -> bool:
        deps = self.conn.execute(
            "SELECT depends_on FROM dependencies WHERE goal_id=?",
            (goal_id,)).fetchall()
        for (dep_id,) in deps:
            dep = self._get(dep_id)
            if dep and dep["status"] != "done":
                return True
        return False

    def blocked_by(self, goal_id: int) -> List[str]:
        """Human-readable reasons why a goal cannot execute yet."""
        reasons = []
        for (dep_id,) in self.conn.execute(
                "SELECT depends_on FROM dependencies WHERE goal_id=?",
                (goal_id,)).fetchall():
            dep = self._get(dep_id)
            if dep and dep["status"] != "done":
                reasons.append(
                    f"Waiting on #{dep_id}: '{dep['desc'][:40]}' [{dep['status']}]")
        return reasons

    def complete(self, goal_id: int, success: bool = True,
                 outcome: str = "") -> None:
        """Mark goal done or failed; propagate progress upward."""
        with self._lock:
            status = "done" if success else "failed"
            self.conn.execute(
                "UPDATE goals SET status=?, outcome=?, progress=?, "
                "updated=datetime('now') WHERE id=?",
                (status, outcome[:300], 1.0 if success else 0.0, goal_id))
            self.conn.commit()
        self._log(goal_id, status)
        if success:
            self._propagate(goal_id)
        else:
            self._reschedule(goal_id)

    def _propagate(self, goal_id: int) -> None:
        row = self.conn.execute(
            "SELECT parent_id FROM goals WHERE id=?", (goal_id,)).fetchone()
        if not row or not row[0]:
            return
        pid   = row[0]
        total = self.conn.execute(
            "SELECT COUNT(*) FROM goals WHERE parent_id=?", (pid,)).fetchone()[0]
        done  = self.conn.execute(
            "SELECT COUNT(*) FROM goals WHERE parent_id=? AND status='done'",
            (pid,)).fetchone()[0]
        prog  = done / max(1, total)
        new_s = "done" if prog >= 1.0 else "pending"
        self.conn.execute(
            "UPDATE goals SET progress=?, status=?, updated=datetime('now') WHERE id=?",
            (prog, new_s, pid))
        self.conn.commit()
        if new_s == "done":
            self._propagate(pid)

    def _reschedule(self, goal_id: int) -> None:
        self.conn.execute(
            "UPDATE goals SET status='retry', updated=datetime('now') WHERE id=?",
            (goal_id,))
        self.conn.commit()
        self._log(goal_id, "rescheduled_after_failure")

    def schedule(self) -> List[Dict]:
        """All active goals in dependency-topological order."""
        rows = self.conn.execute(
            "SELECT id, desc, priority, progress, status FROM goals "
            "WHERE status IN ('pending','retry') ORDER BY priority DESC"
        ).fetchall()
        goals = [{"id": r[0], "desc": r[1], "priority": r[2],
                  "progress": r[3], "status": r[4]} for r in rows]
        ordered, seen = [], set()
        def visit(g: Dict) -> None:
            if g["id"] in seen:
                return
            seen.add(g["id"])
            for (dep_id,) in self.conn.execute(
                    "SELECT depends_on FROM dependencies WHERE goal_id=?",
                    (g["id"],)).fetchall():
                dep = next((x for x in goals if x["id"] == dep_id), None)
                if dep:
                    visit(dep)
            ordered.append(g)
        for g in goals:
            visit(g)
        return ordered

    def eta_for_all(self, minutes_per_task: float = 15.0) -> str:
        """Estimate completion time for all pending goals."""
        pending = self.conn.execute(
            "SELECT COUNT(*) FROM goals WHERE status IN ('pending','retry')"
        ).fetchone()[0]
        mins    = int(pending * minutes_per_task)
        h, m    = divmod(mins, 60)
        return f"{pending} goals pending — ETA {h}h {m}m at {minutes_per_task:.0f}min/task"

    def domain_performance(self) -> Dict[str, Any]:
        """Goal completion stats for ConsciousnessIntegrator."""
        total   = self.conn.execute("SELECT COUNT(*) FROM goals").fetchone()[0]
        done    = self.conn.execute(
            "SELECT COUNT(*) FROM goals WHERE status='done'").fetchone()[0]
        pending = self.conn.execute(
            "SELECT COUNT(*) FROM goals WHERE status IN ('pending','retry')"
        ).fetchone()[0]
        return {
            "total_goals":      total,
            "completed":        done,
            "pending":          pending,
            "completion_rate":  round(done / max(1, total), 3),
        }

    # ─── Background daemon ─────────────────────────────────────────────

    def _start_daemon(self) -> None:
        def _loop() -> None:
            while True:
                time.sleep(300)
                try:
                    action = self.next_action()
                    if action:
                        self._log(action["id"], "frontier_check")
                except Exception:
                    pass
        threading.Thread(target=_loop, daemon=True).start()

# Usage: gp = HierarchicalGoalPlanner()
# top  = gp.add_goal("Achieve superintelligence", priority=9.0)
# subs = gp.decompose(top, ["Build belief system", "Build causal engine", "Integrate"])
# print(gp.next_action()) | print(gp.eta_for_all())
