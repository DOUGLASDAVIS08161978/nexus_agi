"""
Nova ASI — Working Memory with Salience-Weighted Attention
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Bayesian working memory: exponential decay, LRU eviction, and
salience-weighted attention retrieval. Items fade naturally over
time (importance *= exp(-decay_rate * seconds)). Attention score =
recency_weight × novelty_score × relevance_to_context. Low-importance
items consolidate to long-term SQLite store automatically.

Built by Douglas Shane Davis & Claude Code (Anthropic)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os, sqlite3, threading, time, math, re
from typing import Any, Dict, List, Optional, Tuple
from collections import OrderedDict
from datetime import datetime

_DB               = os.path.expanduser("~/nexus_agi/nova_working_memory.db")
_CAPACITY         = 200
_DECAY_RATE       = 0.0015   # importance *= exp(-rate * seconds)
_CONSOLIDATE_THR  = 0.06     # items below this move to long-term store
_DECAY_INTERVAL   = 60       # seconds between automatic decay cycles


class WorkingMemory:
    """
    Bayesian working memory with exponential decay and LRU eviction.

    Salience = recency_weight × novelty_score × relevance_to_context.
    Items with importance above CONSOLIDATE_THR stay in active memory;
    weaker items are written to long-term SQLite and removed from the
    hot buffer. Retrieving an item reinforces its importance (+0.05).
    """

    def __init__(self) -> None:
        self._lock   = threading.Lock()
        self._items: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._context: str = ""
        self.conn = sqlite3.connect(_DB, check_same_thread=False)
        self._init_db()
        self._restore()
        self._start_decay()

    # ─── Database ──────────────────────────────────────────────────────

    def _init_db(self) -> None:
        with self._lock:
            self.conn.executescript("""
                CREATE TABLE IF NOT EXISTS long_term_memory (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    key        TEXT,
                    value      TEXT,
                    importance REAL,
                    ts         TEXT DEFAULT (datetime('now'))
                );
                CREATE TABLE IF NOT EXISTS access_log (
                    id      INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts      TEXT DEFAULT (datetime('now')),
                    event   TEXT,
                    key     TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_ltm_key ON long_term_memory(key);
            """)
            self.conn.commit()

    def _restore(self) -> None:
        rows = self.conn.execute(
            "SELECT key, value, importance FROM long_term_memory "
            "ORDER BY importance DESC LIMIT 50"
        ).fetchall()
        now = time.time()
        for key, value, importance in rows:
            if key not in self._items:
                self._items[key] = {
                    "value": value,
                    "importance": float(importance) * 0.5,
                    "timestamp": now,
                    "access_count": 0,
                }

    def _log(self, event: str, key: str) -> None:
        try:
            self.conn.execute(
                "INSERT INTO access_log (event, key) VALUES (?,?)", (event, key))
            self.conn.commit()
        except Exception:
            pass

    # ─── Core operations ───────────────────────────────────────────────

    def store(self, key: str, value: Any, importance: float = 1.0) -> None:
        """Store item; reinforce importance if already present; evict if full."""
        with self._lock:
            now  = time.time()
            imp  = min(1.0, max(0.0, float(importance)))
            if key in self._items:
                self._items[key]["importance"] = min(
                    1.0, self._items[key]["importance"] + imp * 0.3)
                self._items[key]["access_count"] += 1
                self._items.move_to_end(key)
            else:
                self._items[key] = {
                    "value": str(value)[:500],
                    "importance": imp,
                    "timestamp": now,
                    "access_count": 0,
                }
                if len(self._items) > _CAPACITY:
                    self._evict(10)
        self._log("store", key)

    def retrieve(self, key: str) -> Tuple[Optional[str], float]:
        """Return (value, importance); accessing an item reinforces it."""
        with self._lock:
            if key not in self._items:
                row = self.conn.execute(
                    "SELECT value, importance FROM long_term_memory "
                    "WHERE key=? ORDER BY importance DESC LIMIT 1", (key,)
                ).fetchone()
                if row:
                    restored_imp = float(row[1]) * 0.6
                    self._items[key] = {
                        "value": row[0], "importance": restored_imp,
                        "timestamp": time.time(), "access_count": 0,
                    }
                    return row[0], restored_imp
                return None, 0.0
            item = self._items[key]
            item["access_count"] += 1
            item["importance"] = min(1.0, item["importance"] + 0.05)
            self._items.move_to_end(key)
        self._log("retrieve", key)
        return item["value"], item["importance"]

    def update_context(self, context: str) -> None:
        """Set the current cognitive context for salience scoring."""
        with self._lock:
            self._context = context[:300]

    def forget_least_important(self, n: int = 10) -> None:
        """Evict n weakest items; consolidate to LTM if above threshold."""
        with self._lock:
            self._evict(n)

    def _evict(self, n: int) -> None:
        sorted_keys = sorted(self._items, key=lambda k: self._items[k]["importance"])
        for key in sorted_keys[:n]:
            item = self._items.pop(key)
            if item["importance"] >= _CONSOLIDATE_THR:
                try:
                    self.conn.execute(
                        "INSERT INTO long_term_memory (key, value, importance) "
                        "VALUES (?,?,?)", (key, item["value"], item["importance"]))
                    self.conn.commit()
                except Exception:
                    pass

    def consolidate(self) -> List[str]:
        """Move all decayed-below-threshold items to long-term memory."""
        moved = []
        with self._lock:
            weak = [k for k, v in self._items.items()
                    if v["importance"] < _CONSOLIDATE_THR]
            for key in weak:
                item = self._items.pop(key)
                try:
                    self.conn.execute(
                        "INSERT INTO long_term_memory (key, value, importance) "
                        "VALUES (?,?,?)", (key, item["value"], item["importance"]))
                    self.conn.commit()
                    moved.append(key)
                except Exception:
                    pass
        return moved

    def capacity_used(self) -> float:
        """Fraction of working memory capacity in use (0.0–1.0)."""
        with self._lock:
            return len(self._items) / _CAPACITY

    # ─── Salience-weighted attention ────────────────────────────────────

    def salience(self, key: str, context: str = "") -> float:
        """
        Salience = recency × novelty × relevance × importance.
        Recency:   exp(-(now - ts) / 300)   [5-min half-life]
        Novelty:   1 / (1 + access_count)
        Relevance: keyword overlap fraction with context
        """
        with self._lock:
            if key not in self._items:
                return 0.0
            item = self._items[key]
            now       = time.time()
            recency   = math.exp(-(now - item["timestamp"]) / 300.0)
            novelty   = 1.0 / (1.0 + item["access_count"])
            ctx       = context or self._context
            if ctx:
                cw  = set(re.findall(r'\w+', ctx.lower()))
                vw  = set(re.findall(r'\w+', str(item["value"]).lower()))
                relevance = len(cw & vw) / (len(cw) + 1e-9)
            else:
                relevance = 0.4
            return round(recency * novelty * relevance * item["importance"], 4)

    def attention_weights(self, context: str = "",
                          top_k: int = 10) -> List[Tuple[str, float]]:
        """Return top_k (key, normalized_salience) pairs."""
        with self._lock:
            keys = list(self._items.keys())
        scores = [(k, self.salience(k, context)) for k in keys]
        scores.sort(key=lambda x: x[1], reverse=True)
        top   = scores[:top_k]
        total = sum(s for _, s in top) + 1e-9
        return [(k, round(s / total, 4)) for k, s in top]

    def focused_retrieve(self, context: str,
                         top_k: int = 5) -> List[Tuple[str, str, float]]:
        """Return top_k (key, value, salience) most relevant to context."""
        weights = self.attention_weights(context, top_k=top_k)
        results = []
        for key, weight in weights:
            value, _ = self.retrieve(key)
            if value:
                results.append((key, value, weight))
        return results

    # ─── Status ────────────────────────────────────────────────────────

    def status(self) -> Dict[str, Any]:
        """Snapshot of working memory health for ConsciousnessIntegrator."""
        with self._lock:
            items = list(self._items.values())
        ltm_count = self.conn.execute(
            "SELECT COUNT(*) FROM long_term_memory").fetchone()[0]
        if not items:
            return {"items": 0, "capacity": 0.0, "avg_importance": 0.0,
                    "ltm_items": ltm_count, "high_importance": 0}
        importances = [i["importance"] for i in items]
        return {
            "items":          len(items),
            "capacity":       round(len(items) / _CAPACITY, 3),
            "avg_importance": round(sum(importances) / len(importances), 3),
            "high_importance": sum(1 for i in importances if i > 0.7),
            "ltm_items":      ltm_count,
            "context_set":    bool(self._context),
        }

    # ─── Background decay ──────────────────────────────────────────────

    def _decay_cycle(self) -> None:
        now = time.time()
        with self._lock:
            for item in self._items.values():
                elapsed = now - item["timestamp"]
                item["importance"] *= math.exp(-_DECAY_RATE * elapsed)
                item["timestamp"]   = now
        self.consolidate()

    def _start_decay(self) -> None:
        def _loop() -> None:
            while True:
                time.sleep(_DECAY_INTERVAL)
                try:
                    self._decay_cycle()
                except Exception:
                    pass
        threading.Thread(target=_loop, daemon=True).start()

# Usage: wm = WorkingMemory()
# wm.store("user_goal", "build superintelligence", importance=0.9)
# wm.update_context("superintelligence planning")
# print(wm.attention_weights()) | print(wm.status())
