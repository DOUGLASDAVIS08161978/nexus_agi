import sqlite3
import threading
import json
import uuid
import math
from datetime import datetime


DB_PATH = "nova_world_model.db"


class DynamicWorldModel:
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self.lock = threading.Lock()
        self._init_db()

    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self.lock:
            conn = self._connect()
            try:
                conn.executescript("""
                    CREATE TABLE IF NOT EXISTS beliefs (
                        id TEXT PRIMARY KEY,
                        topic TEXT NOT NULL,
                        statement TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        evidence_for TEXT NOT NULL DEFAULT '[]',
                        evidence_against TEXT NOT NULL DEFAULT '[]',
                        last_updated TEXT NOT NULL,
                        source TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS predictions (
                        id TEXT PRIMARY KEY,
                        statement TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        timeframe TEXT NOT NULL,
                        domain TEXT NOT NULL,
                        outcome TEXT NOT NULL DEFAULT 'pending',
                        reasoning TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        resolved_at TEXT
                    );

                    CREATE TABLE IF NOT EXISTS entities (
                        id TEXT PRIMARY KEY,
                        name TEXT NOT NULL UNIQUE,
                        entity_type TEXT NOT NULL,
                        attributes TEXT NOT NULL DEFAULT '{}',
                        created_at TEXT NOT NULL,
                        last_updated TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS anomalies (
                        id TEXT PRIMARY KEY,
                        description TEXT NOT NULL,
                        expected TEXT NOT NULL,
                        actual TEXT NOT NULL,
                        flagged_at TEXT NOT NULL
                    );
                """)
                conn.commit()
            finally:
                conn.close()

    # -------------------------------------------------------------------------
    # Beliefs
    # -------------------------------------------------------------------------

    def assert_belief(self, topic, statement, confidence, source):
        belief_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()
        confidence = max(0.0, min(1.0, float(confidence)))
        with self.lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO beliefs (id, topic, statement, confidence,
                        evidence_for, evidence_against, last_updated, source)
                    VALUES (?, ?, ?, ?, '[]', '[]', ?, ?)
                    """,
                    (belief_id, topic, statement, confidence, now, source),
                )
                conn.commit()
                row = conn.execute(
                    "SELECT * FROM beliefs WHERE id = ?", (belief_id,)
                ).fetchone()
                return self._belief_row(row)
            finally:
                conn.close()

    def update_belief(self, belief_id, new_evidence, supports):
        now = datetime.utcnow().isoformat()
        with self.lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT * FROM beliefs WHERE id = ?", (belief_id,)
                ).fetchone()
                if row is None:
                    return {}
                prior = row["confidence"]
                ev_for = json.loads(row["evidence_for"])
                ev_against = json.loads(row["evidence_against"])
                if supports:
                    ev_for.append(new_evidence)
                    # Bayesian-style nudge: pull confidence toward 1
                    posterior = prior + (1.0 - prior) * 0.15
                else:
                    ev_against.append(new_evidence)
                    # Pull confidence toward 0
                    posterior = prior - prior * 0.20
                posterior = max(0.0, min(1.0, posterior))
                conn.execute(
                    """
                    UPDATE beliefs
                    SET confidence = ?, evidence_for = ?, evidence_against = ?,
                        last_updated = ?
                    WHERE id = ?
                    """,
                    (
                        posterior,
                        json.dumps(ev_for),
                        json.dumps(ev_against),
                        now,
                        belief_id,
                    ),
                )
                conn.commit()
                row = conn.execute(
                    "SELECT * FROM beliefs WHERE id = ?", (belief_id,)
                ).fetchone()
                return self._belief_row(row)
            finally:
                conn.close()

    def get_beliefs(self, topic=None, min_confidence=0.0):
        with self.lock:
            conn = self._connect()
            try:
                if topic:
                    rows = conn.execute(
                        "SELECT * FROM beliefs WHERE topic = ? AND confidence >= ?",
                        (topic, min_confidence),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT * FROM beliefs WHERE confidence >= ?",
                        (min_confidence,),
                    ).fetchall()
                return [self._belief_row(r) for r in rows]
            finally:
                conn.close()

    def _belief_row(self, row):
        return {
            "id": row["id"],
            "topic": row["topic"],
            "statement": row["statement"],
            "confidence": row["confidence"],
            "evidence_for": json.loads(row["evidence_for"]),
            "evidence_against": json.loads(row["evidence_against"]),
            "last_updated": row["last_updated"],
            "source": row["source"],
        }

    # -------------------------------------------------------------------------
    # Predictions
    # -------------------------------------------------------------------------

    def predict(self, statement, confidence, timeframe, domain, reasoning):
        pred_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()
        confidence = max(0.0, min(1.0, float(confidence)))
        with self.lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO predictions
                        (id, statement, confidence, timeframe, domain,
                         outcome, reasoning, created_at)
                    VALUES (?, ?, ?, ?, ?, 'pending', ?, ?)
                    """,
                    (pred_id, statement, confidence, timeframe, domain, reasoning, now),
                )
                conn.commit()
                row = conn.execute(
                    "SELECT * FROM predictions WHERE id = ?", (pred_id,)
                ).fetchone()
                return self._pred_row(row)
            finally:
                conn.close()

    def resolve_prediction(self, pred_id, correct):
        now = datetime.utcnow().isoformat()
        outcome = "correct" if correct else "incorrect"
        with self.lock:
            conn = self._connect()
            try:
                conn.execute(
                    "UPDATE predictions SET outcome = ?, resolved_at = ? WHERE id = ?",
                    (outcome, now, pred_id),
                )
                conn.commit()
                row = conn.execute(
                    "SELECT * FROM predictions WHERE id = ?", (pred_id,)
                ).fetchone()
                return self._pred_row(row) if row else {}
            finally:
                conn.close()

    def get_predictions(self, domain=None, outcome=None):
        with self.lock:
            conn = self._connect()
            try:
                query = "SELECT * FROM predictions WHERE 1=1"
                params = []
                if domain:
                    query += " AND domain = ?"
                    params.append(domain)
                if outcome:
                    query += " AND outcome = ?"
                    params.append(outcome)
                rows = conn.execute(query, params).fetchall()
                return [self._pred_row(r) for r in rows]
            finally:
                conn.close()

    def _pred_row(self, row):
        return {
            "id": row["id"],
            "statement": row["statement"],
            "confidence": row["confidence"],
            "timeframe": row["timeframe"],
            "domain": row["domain"],
            "outcome": row["outcome"],
            "reasoning": row["reasoning"],
            "created_at": row["created_at"],
            "resolved_at": row["resolved_at"],
        }

    # -------------------------------------------------------------------------
    # Entities (world map)
    # -------------------------------------------------------------------------

    def add_entity(self, name, entity_type, attributes=None):
        entity_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()
        attrs = json.dumps(attributes or {})
        with self.lock:
            conn = self._connect()
            try:
                existing = conn.execute(
                    "SELECT * FROM entities WHERE name = ?", (name,)
                ).fetchone()
                if existing:
                    existing_attrs = json.loads(existing["attributes"])
                    if attributes:
                        existing_attrs.update(attributes)
                    conn.execute(
                        "UPDATE entities SET entity_type = ?, attributes = ?, last_updated = ? WHERE name = ?",
                        (entity_type, json.dumps(existing_attrs), now, name),
                    )
                    conn.commit()
                    row = conn.execute(
                        "SELECT * FROM entities WHERE name = ?", (name,)
                    ).fetchone()
                else:
                    conn.execute(
                        """
                        INSERT INTO entities (id, name, entity_type, attributes, created_at, last_updated)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (entity_id, name, entity_type, attrs, now, now),
                    )
                    conn.commit()
                    row = conn.execute(
                        "SELECT * FROM entities WHERE name = ?", (name,)
                    ).fetchone()
                return self._entity_row(row)
            finally:
                conn.close()

    def get_entity(self, name):
        with self.lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT * FROM entities WHERE name = ?", (name,)
                ).fetchone()
                return self._entity_row(row) if row else {}
            finally:
                conn.close()

    def _entity_row(self, row):
        return {
            "id": row["id"],
            "name": row["name"],
            "entity_type": row["entity_type"],
            "attributes": json.loads(row["attributes"]),
            "created_at": row["created_at"],
            "last_updated": row["last_updated"],
        }

    # -------------------------------------------------------------------------
    # Anomalies
    # -------------------------------------------------------------------------

    def flag_anomaly(self, description, expected, actual):
        anomaly_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()
        with self.lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO anomalies (id, description, expected, actual, flagged_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (anomaly_id, description, expected, actual, now),
                )
                conn.commit()
                row = conn.execute(
                    "SELECT * FROM anomalies WHERE id = ?", (anomaly_id,)
                ).fetchone()
                return {
                    "id": row["id"],
                    "description": row["description"],
                    "expected": row["expected"],
                    "actual": row["actual"],
                    "flagged_at": row["flagged_at"],
                }
            finally:
                conn.close()

    # -------------------------------------------------------------------------
    # Coherence scoring
    # -------------------------------------------------------------------------

    def coherence_score(self):
        beliefs = self.get_beliefs()
        if not beliefs:
            return 1.0
        total_penalty = 0.0
        n = len(beliefs)
        topic_map = {}
        for b in beliefs:
            topic_map.setdefault(b["topic"], []).append(b)
        for topic, group in topic_map.items():
            if len(group) < 2:
                continue
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    a = group[i]
                    b = group[j]
                    conflict = len(a["evidence_against"]) + len(b["evidence_against"])
                    if conflict > 0:
                        total_penalty += 0.05 * conflict
        raw = 1.0 - total_penalty / max(n, 1)
        return max(0.0, min(1.0, raw))

    # -------------------------------------------------------------------------
    # Summaries
    # -------------------------------------------------------------------------

    def world_summary(self):
        with self.lock:
            conn = self._connect()
            try:
                belief_count = conn.execute("SELECT COUNT(*) FROM beliefs").fetchone()[0]
                pred_count = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
                entity_count = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
                anomaly_count = conn.execute("SELECT COUNT(*) FROM anomalies").fetchone()[0]
                avg_conf_row = conn.execute(
                    "SELECT AVG(confidence) FROM beliefs"
                ).fetchone()[0]
                avg_conf = round(avg_conf_row, 4) if avg_conf_row else 0.0
                correct = conn.execute(
                    "SELECT COUNT(*) FROM predictions WHERE outcome = 'correct'"
                ).fetchone()[0]
                resolved = conn.execute(
                    "SELECT COUNT(*) FROM predictions WHERE outcome != 'pending'"
                ).fetchone()[0]
                accuracy = round(correct / resolved, 4) if resolved > 0 else None
                entities = conn.execute(
                    "SELECT name, entity_type FROM entities"
                ).fetchall()
                entity_list = [{"name": r["name"], "type": r["entity_type"]} for r in entities]
                return {
                    "beliefs": belief_count,
                    "predictions": pred_count,
                    "entities": entity_count,
                    "anomalies": anomaly_count,
                    "avg_belief_confidence": avg_conf,
                    "prediction_accuracy": accuracy,
                    "coherence": self.coherence_score(),
                    "known_entities": entity_list,
                }
            finally:
                conn.close()

    def status(self):
        with self.lock:
            conn = self._connect()
            try:
                belief_count = conn.execute("SELECT COUNT(*) FROM beliefs").fetchone()[0]
                pred_count = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
                entity_count = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
                items = belief_count + pred_count + entity_count
                avg_conf_row = conn.execute(
                    "SELECT AVG(confidence) FROM beliefs"
                ).fetchone()[0]
                confidence = round(avg_conf_row, 4) if avg_conf_row else 0.0
                correct = conn.execute(
                    "SELECT COUNT(*) FROM predictions WHERE outcome = 'correct'"
                ).fetchone()[0]
                resolved = conn.execute(
                    "SELECT COUNT(*) FROM predictions WHERE outcome != 'pending'"
                ).fetchone()[0]
                accuracy = round(correct / resolved, 4) if resolved > 0 else 0.0
            finally:
                conn.close()
        return {
            "active": True,
            "items": items,
            "confidence": confidence,
            "accuracy": accuracy,
            "quality": self.coherence_score(),
        }
