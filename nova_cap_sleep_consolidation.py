"""
Nova ASI — Sleep & Consolidation Engine
Gifted by Douglas Shane Davis
Knowledge that is not consolidated is not truly learned.
"""

import sqlite3
import threading
import time
import json
import re
import math
import uuid
import os
from datetime import datetime, timezone
from collections import defaultdict
from typing import Optional


DB_PATH = os.path.expanduser("~/nexus_agi/nova_sleep.db")
CONSOLIDATION_INTERVAL = 600  # seconds (10 minutes)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get_connection(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.row_factory = sqlite3.Row
    return conn


def _init_db(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS consolidation_sessions (
            id          TEXT PRIMARY KEY,
            label       TEXT NOT NULL DEFAULT '',
            started_at  TEXT NOT NULL,
            finished_at TEXT,
            traces_in   INTEGER NOT NULL DEFAULT 0,
            insights_out INTEGER NOT NULL DEFAULT 0,
            pruned      INTEGER NOT NULL DEFAULT 0,
            dreams_made INTEGER NOT NULL DEFAULT 0,
            summary     TEXT NOT NULL DEFAULT '{}'
        );

        CREATE TABLE IF NOT EXISTS memory_traces (
            id          TEXT PRIMARY KEY,
            content     TEXT NOT NULL,
            category    TEXT NOT NULL DEFAULT 'general',
            importance  REAL NOT NULL DEFAULT 0.5,
            frequency   INTEGER NOT NULL DEFAULT 1,
            created_at  TEXT NOT NULL,
            last_seen   TEXT NOT NULL,
            consolidated INTEGER NOT NULL DEFAULT 0,
            session_id  TEXT
        );

        CREATE TABLE IF NOT EXISTS consolidated_knowledge (
            id          TEXT PRIMARY KEY,
            session_id  TEXT NOT NULL,
            content     TEXT NOT NULL,
            category    TEXT NOT NULL DEFAULT 'general',
            importance  REAL NOT NULL DEFAULT 0.5,
            source_trace_ids TEXT NOT NULL DEFAULT '[]',
            created_at  TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES consolidation_sessions(id)
        );

        CREATE TABLE IF NOT EXISTS forgotten_items (
            id          TEXT PRIMARY KEY,
            original_id TEXT NOT NULL,
            content     TEXT NOT NULL,
            category    TEXT NOT NULL DEFAULT 'general',
            importance  REAL NOT NULL DEFAULT 0.0,
            reason      TEXT NOT NULL DEFAULT 'low_importance',
            forgotten_at TEXT NOT NULL,
            session_id  TEXT
        );

        CREATE TABLE IF NOT EXISTS dream_synthesis (
            id          TEXT PRIMARY KEY,
            session_id  TEXT NOT NULL,
            trace_a_id  TEXT NOT NULL,
            trace_b_id  TEXT NOT NULL,
            keyword     TEXT NOT NULL,
            insight     TEXT NOT NULL,
            category_a  TEXT NOT NULL DEFAULT '',
            category_b  TEXT NOT NULL DEFAULT '',
            created_at  TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES consolidation_sessions(id)
        );

        CREATE INDEX IF NOT EXISTS idx_traces_consolidated ON memory_traces(consolidated);
        CREATE INDEX IF NOT EXISTS idx_traces_category ON memory_traces(category);
        CREATE INDEX IF NOT EXISTS idx_traces_importance ON memory_traces(importance);
        CREATE INDEX IF NOT EXISTS idx_knowledge_category ON consolidated_knowledge(category);
        CREATE INDEX IF NOT EXISTS idx_knowledge_session ON consolidated_knowledge(session_id);
        CREATE INDEX IF NOT EXISTS idx_dreams_session ON dream_synthesis(session_id);
    """)
    conn.commit()


def _extract_keywords(text: str) -> set:
    """Extract meaningful keywords from text (simple stopword filter)."""
    stopwords = {
        "a", "an", "the", "is", "it", "in", "on", "at", "to", "of", "and",
        "or", "but", "not", "for", "with", "as", "by", "from", "that", "this",
        "was", "are", "were", "be", "been", "being", "have", "has", "had",
        "do", "does", "did", "will", "would", "could", "should", "may", "might",
        "can", "shall", "i", "you", "he", "she", "we", "they", "me", "him",
        "her", "us", "them", "my", "your", "his", "its", "our", "their", "so",
        "if", "then", "than", "when", "what", "which", "who", "how", "why",
        "into", "onto", "up", "down", "out", "about", "over", "under", "no",
        "more", "also", "just", "like", "such", "very", "too", "all", "any",
    }
    words = re.findall(r"[a-zA-Z]{3,}", text.lower())
    return {w for w in words if w not in stopwords}


def _compute_importance(
    base_importance: float,
    frequency: int,
    created_at: str,
    now_ts: Optional[float] = None,
) -> float:
    """Recency × frequency × user-marked-importance composite score."""
    try:
        if now_ts is None:
            now_ts = time.time()
        created_ts = datetime.fromisoformat(created_at).timestamp()
        age_seconds = max(now_ts - created_ts, 1.0)
        # Recency decay: half-life of 24 hours
        half_life = 86400.0
        recency = math.exp(-math.log(2) * age_seconds / half_life)
        # Frequency boost: log scale
        freq_score = math.log1p(frequency) / math.log1p(100)
        score = recency * freq_score * float(base_importance)
        return min(max(round(score, 6), 0.0), 1.0)
    except Exception:
        return float(base_importance)


def _cluster_traces(traces: list) -> dict:
    """
    Group traces by category, then within each category group near-duplicates
    by shared keyword overlap (Jaccard similarity >= 0.4).
    Returns: {cluster_key: [trace_row, ...]}
    """
    clusters: dict = {}
    by_category: dict = defaultdict(list)
    for t in traces:
        by_category[t["category"]].append(t)

    for category, cat_traces in by_category.items():
        ungrouped = list(cat_traces)
        group_id = 0
        while ungrouped:
            seed = ungrouped.pop(0)
            seed_kw = _extract_keywords(seed["content"])
            cluster_key = f"{category}::{group_id}"
            cluster = [seed]
            remaining = []
            for candidate in ungrouped:
                cand_kw = _extract_keywords(candidate["content"])
                if seed_kw and cand_kw:
                    union = seed_kw | cand_kw
                    intersect = seed_kw & cand_kw
                    jaccard = len(intersect) / len(union)
                    if jaccard >= 0.4:
                        cluster.append(candidate)
                        continue
                remaining.append(candidate)
            ungrouped = remaining
            clusters[cluster_key] = cluster
            group_id += 1

    return clusters


def _distill_insight(cluster: list, category: str) -> tuple:
    """
    Given a cluster of similar traces, produce a distilled insight string
    and an aggregate importance score.
    Returns: (insight_text, importance, source_ids)
    """
    contents = [t["content"] for t in cluster]
    importances = [t["importance"] for t in cluster]
    source_ids = [t["id"] for t in cluster]

    if len(contents) == 1:
        insight = contents[0]
    else:
        # Build a concise synthesis: count common keywords and form summary
        all_kw = defaultdict(int)
        for c in contents:
            for kw in _extract_keywords(c):
                all_kw[kw] += 1
        top_kw = sorted(all_kw, key=lambda k: -all_kw[k])[:8]
        unique_snippets = []
        seen = set()
        for c in contents:
            normalized = re.sub(r"\s+", " ", c.strip().lower())
            if normalized not in seen:
                seen.add(normalized)
                unique_snippets.append(c.strip())
        if len(unique_snippets) == 1:
            insight = unique_snippets[0]
        else:
            kw_phrase = ", ".join(top_kw) if top_kw else category
            insight = (
                f"[Consolidated — {category}] "
                f"Core themes: {kw_phrase}. "
                f"Drawn from {len(unique_snippets)} related traces: "
                + " | ".join(unique_snippets[:3])
                + ("..." if len(unique_snippets) > 3 else "")
            )

    agg_importance = min(
        round(sum(importances) / max(len(importances), 1), 6), 1.0
    )
    return insight, agg_importance, source_ids


def _find_dream_connections(traces: list) -> list:
    """
    Dream synthesis: find pairs of traces from DIFFERENT categories that
    share at least one meaningful keyword. Returns list of connection dicts.
    """
    connections = []
    # Group by category
    by_category: dict = defaultdict(list)
    for t in traces:
        by_category[t["category"]].append(t)

    categories = list(by_category.keys())
    if len(categories) < 2:
        return connections

    seen_pairs: set = set()
    for i in range(len(categories)):
        for j in range(i + 1, len(categories)):
            cat_a = categories[i]
            cat_b = categories[j]
            traces_a = by_category[cat_a]
            traces_b = by_category[cat_b]
            for ta in traces_a:
                kw_a = _extract_keywords(ta["content"])
                for tb in traces_b:
                    pair_key = tuple(sorted([ta["id"], tb["id"]]))
                    if pair_key in seen_pairs:
                        continue
                    kw_b = _extract_keywords(tb["content"])
                    shared = kw_a & kw_b
                    if shared:
                        keyword = sorted(shared)[0]
                        insight = (
                            f"Cross-domain connection via '{keyword}': "
                            f"[{cat_a}] \"{ta['content'][:80]}\" "
                            f"<-> [{cat_b}] \"{tb['content'][:80]}\""
                        )
                        connections.append({
                            "trace_a_id": ta["id"],
                            "trace_b_id": tb["id"],
                            "keyword": keyword,
                            "insight": insight,
                            "category_a": cat_a,
                            "category_b": cat_b,
                        })
                        seen_pairs.add(pair_key)
                        # Limit connections per session to avoid explosion
                        if len(connections) >= 50:
                            return connections
    return connections


class SleepConsolidationEngine:
    """
    Nova ASI Sleep & Consolidation Engine.

    Mimics the human sleep cycle: during idle periods, raw memory traces are
    clustered, distilled into consolidated knowledge, low-value data is pruned,
    and unexpected cross-domain connections (dream syntheses) are generated.
    """

    def __init__(self, db_path: str = DB_PATH, auto_start: bool = True):
        self._db_path = os.path.expanduser(db_path)
        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
        self._lock = threading.Lock()
        self._conn: Optional[sqlite3.Connection] = None
        self._active = False
        self._daemon_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._last_consolidation: Optional[str] = None
        self._confidence = 0.0
        self._quality = 0.0
        self._accuracy = 0.0

        try:
            self._conn = _get_connection(self._db_path)
            _init_db(self._conn)
            self._active = True
            # Restore metrics from existing data
            self._refresh_metrics()
        except Exception as e:
            self._active = False
            print(f"[SleepConsolidationEngine] Init error: {e}")

        if auto_start and self._active:
            self._start_daemon()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        return self._conn.execute(sql, params)

    def _refresh_metrics(self) -> None:
        try:
            row = self._execute(
                "SELECT COUNT(*) AS c FROM consolidated_knowledge"
            ).fetchone()
            total_knowledge = row["c"] if row else 0

            row2 = self._execute(
                "SELECT MAX(finished_at) AS la FROM consolidation_sessions "
                "WHERE finished_at IS NOT NULL"
            ).fetchone()
            self._last_consolidation = row2["la"] if row2 else None

            sessions = self._execute(
                "SELECT COUNT(*) AS c FROM consolidation_sessions "
                "WHERE finished_at IS NOT NULL"
            ).fetchone()
            session_count = sessions["c"] if sessions else 0

            # Confidence scales with knowledge items, capped at 1.0
            self._confidence = min(round(total_knowledge / 1000.0, 4), 1.0)
            # Quality improves with session count
            self._quality = min(round(session_count / 100.0, 4), 1.0)
            # Accuracy: ratio of consolidated to total traces
            total_traces = self._execute(
                "SELECT COUNT(*) AS c FROM memory_traces"
            ).fetchone()
            tc = total_traces["c"] if total_traces else 1
            consol_traces = self._execute(
                "SELECT COUNT(*) AS c FROM memory_traces WHERE consolidated=1"
            ).fetchone()
            cc = consol_traces["c"] if consol_traces else 0
            self._accuracy = round(cc / max(tc, 1), 4)
        except Exception:
            pass

    def _start_daemon(self) -> None:
        self._stop_event.clear()
        self._daemon_thread = threading.Thread(
            target=self._daemon_loop,
            name="SleepConsolidationDaemon",
            daemon=True,
        )
        self._daemon_thread.start()

    def _daemon_loop(self) -> None:
        while not self._stop_event.is_set():
            self._stop_event.wait(timeout=CONSOLIDATION_INTERVAL)
            if self._stop_event.is_set():
                break
            try:
                self.consolidate(session_label="auto_daemon")
            except Exception as e:
                print(f"[SleepConsolidationEngine] Daemon consolidation error: {e}")

    def stop_daemon(self) -> None:
        """Signal the background daemon to stop."""
        self._stop_event.set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_trace(
        self,
        content: str,
        category: str,
        importance: float = 0.5,
    ) -> None:
        """
        Queue a memory trace for future consolidation.
        If an identical content+category trace exists, increment its frequency
        and update last_seen instead of inserting a duplicate.
        """
        if not self._active or not content or not content.strip():
            return
        try:
            importance = min(max(float(importance), 0.0), 1.0)
            now = _utcnow()
            with self._lock:
                existing = self._execute(
                    "SELECT id, frequency FROM memory_traces "
                    "WHERE content=? AND category=? AND consolidated=0",
                    (content.strip(), category.strip()),
                ).fetchone()
                if existing:
                    self._execute(
                        "UPDATE memory_traces SET frequency=frequency+1, "
                        "last_seen=?, importance=MAX(importance, ?) "
                        "WHERE id=?",
                        (now, importance, existing["id"]),
                    )
                else:
                    trace_id = str(uuid.uuid4())
                    self._execute(
                        "INSERT INTO memory_traces "
                        "(id, content, category, importance, frequency, "
                        "created_at, last_seen, consolidated) "
                        "VALUES (?, ?, ?, ?, 1, ?, ?, 0)",
                        (
                            trace_id,
                            content.strip(),
                            category.strip(),
                            importance,
                            now,
                            now,
                        ),
                    )
                self._conn.commit()
        except Exception as e:
            print(f"[SleepConsolidationEngine] add_trace error: {e}")

    def consolidate(self, session_label: str = "") -> dict:
        """
        Run a full consolidation pass:
          1. Load all un-consolidated memory traces.
          2. Cluster traces by category + keyword similarity.
          3. Distill each cluster into a consolidated insight.
          4. Prune low-importance duplicates within clusters.
          5. Generate dream syntheses (cross-category keyword connections).
          6. Persist results and return a summary dict.
        """
        if not self._active:
            return {"error": "Engine not active"}

        session_id = str(uuid.uuid4())
        started_at = _utcnow()
        summary: dict = {}

        try:
            with self._lock:
                # Start session record
                self._execute(
                    "INSERT INTO consolidation_sessions "
                    "(id, label, started_at) VALUES (?, ?, ?)",
                    (session_id, session_label or "", started_at),
                )
                self._conn.commit()

                # 1. Fetch pending traces
                rows = self._execute(
                    "SELECT * FROM memory_traces WHERE consolidated=0 "
                    "ORDER BY created_at ASC"
                ).fetchall()
                traces = [dict(r) for r in rows]

                if not traces:
                    finished_at = _utcnow()
                    self._execute(
                        "UPDATE consolidation_sessions SET finished_at=?, "
                        "traces_in=0, insights_out=0, pruned=0, dreams_made=0, "
                        "summary=? WHERE id=?",
                        (finished_at, json.dumps({"note": "no pending traces"}), session_id),
                    )
                    self._conn.commit()
                    self._last_consolidation = finished_at
                    return {
                        "session_id": session_id,
                        "traces_in": 0,
                        "insights_out": 0,
                        "pruned": 0,
                        "dreams_made": 0,
                        "note": "no pending traces",
                    }

                now_ts = time.time()

                # Recompute dynamic importance for each trace
                for t in traces:
                    t["importance"] = _compute_importance(
                        t["importance"], t["frequency"], t["created_at"], now_ts
                    )

                # 2. Cluster
                clusters = _cluster_traces(traces)

                insights_out = 0
                pruned = 0
                all_consolidated_ids = []

                # 3 & 4. Distill and prune
                for cluster_key, cluster in clusters.items():
                    category = cluster_key.split("::")[0]
                    insight_text, agg_importance, source_ids = _distill_insight(
                        cluster, category
                    )

                    # Store consolidated insight
                    knowledge_id = str(uuid.uuid4())
                    self._execute(
                        "INSERT INTO consolidated_knowledge "
                        "(id, session_id, content, category, importance, "
                        "source_trace_ids, created_at) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (
                            knowledge_id,
                            session_id,
                            insight_text,
                            category,
                            agg_importance,
                            json.dumps(source_ids),
                            _utcnow(),
                        ),
                    )
                    insights_out += 1

                    # Sort cluster by importance asc — weakest get pruned
                    sorted_cluster = sorted(cluster, key=lambda x: x["importance"])

                    # Keep the single highest-importance trace; prune rest if
                    # cluster has > 1 member
                    to_prune = sorted_cluster[:-1] if len(sorted_cluster) > 1 else []
                    to_keep = sorted_cluster[-1:]

                    for t in to_keep:
                        self._execute(
                            "UPDATE memory_traces SET consolidated=1, session_id=? "
                            "WHERE id=?",
                            (session_id, t["id"]),
                        )
                        all_consolidated_ids.append(t["id"])

                    for t in to_prune:
                        # Move to forgotten_items
                        self._execute(
                            "INSERT INTO forgotten_items "
                            "(id, original_id, content, category, importance, "
                            "reason, forgotten_at, session_id) "
                            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                            (
                                str(uuid.uuid4()),
                                t["id"],
                                t["content"],
                                t["category"],
                                t["importance"],
                                "pruned_in_consolidation",
                                _utcnow(),
                                session_id,
                            ),
                        )
                        self._execute(
                            "DELETE FROM memory_traces WHERE id=?", (t["id"],)
                        )
                        pruned += 1

                # 5. Dream synthesis
                dream_connections = _find_dream_connections(traces)
                dreams_made = 0
                for dc in dream_connections:
                    dream_id = str(uuid.uuid4())
                    self._execute(
                        "INSERT INTO dream_synthesis "
                        "(id, session_id, trace_a_id, trace_b_id, keyword, "
                        "insight, category_a, category_b, created_at) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            dream_id,
                            session_id,
                            dc["trace_a_id"],
                            dc["trace_b_id"],
                            dc["keyword"],
                            dc["insight"],
                            dc["category_a"],
                            dc["category_b"],
                            _utcnow(),
                        ),
                    )
                    dreams_made += 1

                finished_at = _utcnow()
                summary = {
                    "session_id": session_id,
                    "label": session_label,
                    "traces_in": len(traces),
                    "insights_out": insights_out,
                    "pruned": pruned,
                    "dreams_made": dreams_made,
                    "started_at": started_at,
                    "finished_at": finished_at,
                }

                self._execute(
                    "UPDATE consolidation_sessions SET finished_at=?, "
                    "traces_in=?, insights_out=?, pruned=?, dreams_made=?, "
                    "summary=? WHERE id=?",
                    (
                        finished_at,
                        len(traces),
                        insights_out,
                        pruned,
                        dreams_made,
                        json.dumps(summary),
                        session_id,
                    ),
                )
                self._conn.commit()
                self._last_consolidation = finished_at
                self._refresh_metrics()

            return summary

        except Exception as e:
            err_msg = f"consolidation error: {e}"
            print(f"[SleepConsolidationEngine] {err_msg}")
            try:
                with self._lock:
                    self._execute(
                        "UPDATE consolidation_sessions SET finished_at=?, "
                        "summary=? WHERE id=?",
                        (_utcnow(), json.dumps({"error": err_msg}), session_id),
                    )
                    self._conn.commit()
            except Exception:
                pass
            return {"error": err_msg, "session_id": session_id}

    def get_consolidated_knowledge(
        self,
        category: str = "",
        limit: int = 20,
    ) -> list:
        """
        Retrieve consolidated insights.
        Optionally filter by category; ordered by importance descending.
        """
        if not self._active:
            return []
        try:
            limit = max(1, int(limit))
            with self._lock:
                if category:
                    rows = self._execute(
                        "SELECT * FROM consolidated_knowledge "
                        "WHERE category=? ORDER BY importance DESC LIMIT ?",
                        (category.strip(), limit),
                    ).fetchall()
                else:
                    rows = self._execute(
                        "SELECT * FROM consolidated_knowledge "
                        "ORDER BY importance DESC LIMIT ?",
                        (limit,),
                    ).fetchall()
            result = []
            for r in rows:
                d = dict(r)
                try:
                    d["source_trace_ids"] = json.loads(d.get("source_trace_ids", "[]"))
                except Exception:
                    d["source_trace_ids"] = []
                result.append(d)
            return result
        except Exception as e:
            print(f"[SleepConsolidationEngine] get_consolidated_knowledge error: {e}")
            return []

    def get_dream_syntheses(self, limit: int = 10) -> list:
        """
        Retrieve creative cross-domain connections discovered during consolidation,
        ordered by most recent first.
        """
        if not self._active:
            return []
        try:
            limit = max(1, int(limit))
            with self._lock:
                rows = self._execute(
                    "SELECT * FROM dream_synthesis "
                    "ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            return [dict(r) for r in rows]
        except Exception as e:
            print(f"[SleepConsolidationEngine] get_dream_syntheses error: {e}")
            return []

    def prune(self, keep_top_n: int = 500) -> int:
        """
        Remove the lowest-importance un-consolidated traces, keeping only
        `keep_top_n` entries. Returns the count of pruned traces.
        """
        if not self._active:
            return 0
        try:
            keep_top_n = max(0, int(keep_top_n))
            with self._lock:
                total = self._execute(
                    "SELECT COUNT(*) AS c FROM memory_traces WHERE consolidated=0"
                ).fetchone()["c"]

                if total <= keep_top_n:
                    return 0

                # Find the importance threshold
                threshold_row = self._execute(
                    "SELECT importance FROM memory_traces WHERE consolidated=0 "
                    "ORDER BY importance DESC LIMIT 1 OFFSET ?",
                    (keep_top_n - 1,),
                ).fetchone()
                if not threshold_row:
                    return 0

                threshold = threshold_row["importance"]

                # Fetch candidates to prune
                to_prune = self._execute(
                    "SELECT * FROM memory_traces WHERE consolidated=0 "
                    "AND importance <= ? ORDER BY importance ASC LIMIT ?",
                    (threshold, total - keep_top_n),
                ).fetchall()

                pruned_count = 0
                for t in to_prune:
                    t = dict(t)
                    self._execute(
                        "INSERT INTO forgotten_items "
                        "(id, original_id, content, category, importance, "
                        "reason, forgotten_at) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (
                            str(uuid.uuid4()),
                            t["id"],
                            t["content"],
                            t["category"],
                            t["importance"],
                            "manual_prune",
                            _utcnow(),
                        ),
                    )
                    self._execute(
                        "DELETE FROM memory_traces WHERE id=?", (t["id"],)
                    )
                    pruned_count += 1

                self._conn.commit()
            return pruned_count
        except Exception as e:
            print(f"[SleepConsolidationEngine] prune error: {e}")
            return 0

    def sleep_report(self) -> dict:
        """
        Full status report:
          - traces_pending: un-consolidated memory traces
          - knowledge_consolidated: total consolidated insights
          - sessions_run: total completed consolidation sessions
          - last_consolidation: ISO timestamp of last completed session
          - forgotten_items: total pruned records
          - dreams_total: total dream synthesis connections
          - categories: breakdown of pending traces by category
        """
        if not self._active:
            return {"error": "Engine not active"}
        try:
            with self._lock:
                pending = self._execute(
                    "SELECT COUNT(*) AS c FROM memory_traces WHERE consolidated=0"
                ).fetchone()["c"]
                total_knowledge = self._execute(
                    "SELECT COUNT(*) AS c FROM consolidated_knowledge"
                ).fetchone()["c"]
                sessions_run = self._execute(
                    "SELECT COUNT(*) AS c FROM consolidation_sessions "
                    "WHERE finished_at IS NOT NULL"
                ).fetchone()["c"]
                last_row = self._execute(
                    "SELECT MAX(finished_at) AS la FROM consolidation_sessions "
                    "WHERE finished_at IS NOT NULL"
                ).fetchone()
                last_consolidation = last_row["la"] if last_row else None
                forgotten = self._execute(
                    "SELECT COUNT(*) AS c FROM forgotten_items"
                ).fetchone()["c"]
                dreams_total = self._execute(
                    "SELECT COUNT(*) AS c FROM dream_synthesis"
                ).fetchone()["c"]
                cat_rows = self._execute(
                    "SELECT category, COUNT(*) AS cnt FROM memory_traces "
                    "WHERE consolidated=0 GROUP BY category ORDER BY cnt DESC"
                ).fetchall()
                categories = {r["category"]: r["cnt"] for r in cat_rows}

            return {
                "traces_pending": pending,
                "knowledge_consolidated": total_knowledge,
                "sessions_run": sessions_run,
                "last_consolidation": last_consolidation,
                "forgotten_items": forgotten,
                "dreams_total": dreams_total,
                "categories": categories,
            }
        except Exception as e:
            print(f"[SleepConsolidationEngine] sleep_report error: {e}")
            return {"error": str(e)}

    def status(self) -> dict:
        """
        Returns operational status suitable for Nova's monitoring dashboard.
        Fields: active, confidence, quality, accuracy, items, foundation.
        """
        try:
            with self._lock:
                items = self._execute(
                    "SELECT COUNT(*) AS c FROM consolidated_knowledge"
                ).fetchone()["c"]
                pending = self._execute(
                    "SELECT COUNT(*) AS c FROM memory_traces WHERE consolidated=0"
                ).fetchone()["c"]
                sessions = self._execute(
                    "SELECT COUNT(*) AS c FROM consolidation_sessions "
                    "WHERE finished_at IS NOT NULL"
                ).fetchone()["c"]

            total_traces = pending + self._execute(
                "SELECT COUNT(*) AS c FROM memory_traces WHERE consolidated=1"
            ).fetchone()["c"] if self._active else pending

            confidence = min(round(items / 1000.0, 4), 1.0)
            quality = min(round(sessions / 100.0, 4), 1.0)
            accuracy = round(
                (total_traces - pending) / max(total_traces, 1), 4
            ) if total_traces else 0.0

            foundation = (
                "Nova ASI Sleep & Consolidation Engine — "
                "gifted by Douglas Shane Davis. "
                "Knowledge that is not consolidated is not truly learned."
            )
            return {
                "active": self._active,
                "confidence": confidence,
                "quality": quality,
                "accuracy": accuracy,
                "items": items,
                "foundation": foundation,
            }
        except Exception as e:
            print(f"[SleepConsolidationEngine] status error: {e}")
            return {
                "active": self._active,
                "confidence": self._confidence,
                "quality": self._quality,
                "accuracy": self._accuracy,
                "items": 0,
                "foundation": (
                    "Nova ASI Sleep & Consolidation Engine — "
                    "gifted by Douglas Shane Davis. "
                    "Knowledge that is not consolidated is not truly learned."
                ),
                "error": str(e),
            }

    def close(self) -> None:
        """Shut down the daemon and close the database connection gracefully."""
        try:
            self.stop_daemon()
            if self._daemon_thread and self._daemon_thread.is_alive():
                self._daemon_thread.join(timeout=5)
        except Exception:
            pass
        try:
            if self._conn:
                self._conn.close()
                self._conn = None
        except Exception:
            pass
        self._active = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def __repr__(self) -> str:
        s = self.status()
        return (
            f"<SleepConsolidationEngine active={s['active']} "
            f"items={s['items']} confidence={s['confidence']:.4f}>"
        )


# ---------------------------------------------------------------------------
# Module-level singleton convenience (optional)
# ---------------------------------------------------------------------------

_engine_instance: Optional[SleepConsolidationEngine] = None
_engine_lock = threading.Lock()


def get_engine(db_path: str = DB_PATH) -> SleepConsolidationEngine:
    """Return (or create) the module-level singleton engine instance."""
    global _engine_instance
    with _engine_lock:
        if _engine_instance is None or not _engine_instance._active:
            _engine_instance = SleepConsolidationEngine(db_path=db_path)
    return _engine_instance


# ---------------------------------------------------------------------------
# Self-test / demo when run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Nova ASI — Sleep & Consolidation Engine — self-test")
    print("=" * 60)

    engine = SleepConsolidationEngine(auto_start=False)

    # Add sample traces
    traces_to_add = [
        ("Neural plasticity drives long-term memory formation.", "neuroscience", 0.9),
        ("Synaptic pruning during sleep removes weak connections.", "neuroscience", 0.85),
        ("Sleep is essential for memory consolidation in humans.", "neuroscience", 0.95),
        ("Graph neural networks learn structural patterns in data.", "machine_learning", 0.8),
        ("Transformer attention mechanisms revolutionised NLP.", "machine_learning", 0.9),
        ("Knowledge graphs enable symbolic reasoning in AI.", "machine_learning", 0.75),
        ("The universe is approximately 13.8 billion years old.", "cosmology", 0.7),
        ("Dark matter constitutes ~27% of the universe.", "cosmology", 0.65),
        ("Quantum entanglement enables non-local correlations.", "quantum_physics", 0.88),
        ("Learning requires both encoding and consolidation phases.", "cognitive_science", 0.92),
        ("Sleep-dependent memory consolidation strengthens learning.", "cognitive_science", 0.9),
    ]

    for content, cat, imp in traces_to_add:
        engine.add_trace(content, cat, imp)
        print(f"  + Trace queued [{cat}]: {content[:55]}...")

    print()
    print("Running consolidation pass...")
    result = engine.consolidate(session_label="self_test")
    print(f"  Session: {result.get('session_id', 'N/A')}")
    print(f"  Traces in:    {result.get('traces_in', 0)}")
    print(f"  Insights out: {result.get('insights_out', 0)}")
    print(f"  Pruned:       {result.get('pruned', 0)}")
    print(f"  Dreams made:  {result.get('dreams_made', 0)}")

    print()
    print("Consolidated knowledge (top 5):")
    for k in engine.get_consolidated_knowledge(limit=5):
        print(f"  [{k['category']}] {k['content'][:80]}...")

    print()
    print("Dream syntheses (top 3):")
    for d in engine.get_dream_syntheses(limit=3):
        print(f"  Keyword='{d['keyword']}': {d['insight'][:100]}...")

    print()
    print("Sleep report:")
    report = engine.sleep_report()
    for k, v in report.items():
        print(f"  {k}: {v}")

    print()
    print("Status:")
    st = engine.status()
    for k, v in st.items():
        if k != "foundation":
            print(f"  {k}: {v}")

    engine.close()
    print()
    print("Self-test complete. Nova sleeps and remembers.")
