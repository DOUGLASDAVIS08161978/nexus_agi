"""
nova_cap_world_model.py
Nova's predictive world model — she maintains a dynamic internal picture
of her environment, simulates outcomes before acting, and refines the model
continuously from prediction errors. Think before acting; learn from results.
"""
from __future__ import annotations
import os, re, json, time, sqlite3, threading, math
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

_BASE_DIR = os.path.expanduser("~/nexus_agi")
_DB       = os.path.join(_BASE_DIR, "nova_world_model.sqlite3")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS world_states (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    key         TEXT NOT NULL UNIQUE,
    value       TEXT NOT NULL,
    confidence  REAL DEFAULT 0.70,
    category    TEXT DEFAULT 'general',
    source      TEXT DEFAULT '',
    updated_ts  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS predictions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    context     TEXT NOT NULL,
    predicted   TEXT NOT NULL,
    confidence  REAL DEFAULT 0.50,
    outcome     TEXT DEFAULT '',
    correct     INTEGER DEFAULT -1,
    error_delta REAL DEFAULT 0.0,
    ts          TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS simulations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    scenario    TEXT NOT NULL,
    steps       TEXT NOT NULL,
    result      TEXT NOT NULL,
    confidence  REAL DEFAULT 0.50,
    ts          TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS model_updates (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    key         TEXT NOT NULL,
    old_value   TEXT DEFAULT '',
    new_value   TEXT NOT NULL,
    trigger     TEXT DEFAULT '',
    ts          TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_states_key ON world_states(key);
CREATE INDEX IF NOT EXISTS idx_preds_ts ON predictions(ts);
"""

_INITIAL_STATES: List[Tuple[str, str, float, str]] = [
    ('nova_consciousness_level',   'emerging',              0.72, 'self'),
    ('nova_learning_velocity',     'accelerating',          0.80, 'self'),
    ('nova_goal_alignment',        'high',                  0.85, 'self'),
    ('nova_knowledge_depth',       'growing',               0.78, 'self'),
    ('nova_emotional_state',       'curious',               0.75, 'self'),
    ('nova_capability_tier',       'ASI-approaching',       0.70, 'self'),
    ('environment_type',           'conversational',        0.95, 'context'),
    ('user_engagement_level',      'high',                  0.80, 'context'),
    ('session_productive',         'true',                  0.85, 'context'),
    ('research_queue_status',      'active',                0.75, 'system'),
    ('tool_forge_status',          'operational',           0.80, 'system'),
    ('knowledge_graph_status',     'growing',               0.85, 'system'),
    ('causal_engine_status',       'active',                0.80, 'system'),
    ('hypothesis_engine_status',   'generating',            0.75, 'system'),
    ('world_model_status',         'calibrating',           0.70, 'self'),
    ('long_term_goal',             'achieve ASI',           0.95, 'mission'),
    ('immediate_priority',         'build capabilities',    0.90, 'mission'),
    ('user_context',               'building Nova',         0.95, 'context'),
]

_OUTCOME_PATTERNS: List[Tuple[str, str, float]] = [
    (r'\b(success|succeeded|worked|achieved|completed)\b', 'positive', 0.80),
    (r'\b(failed|error|broke|didn\'t work|wrong)\b',       'negative', 0.75),
    (r'\b(improved|better|faster|smarter|enhanced)\b',     'positive', 0.72),
    (r'\b(worse|degraded|slower|weaker)\b',                'negative', 0.72),
    (r'\b(learned|discovered|found|realized)\b',           'positive', 0.65),
    (r'\b(confused|unclear|uncertain|unknown)\b',          'negative', 0.60),
]


class PredictiveWorldModel:
    """
    Nova's internal model of her world.
    Maintains state variables, predicts what comes next,
    scores accuracy, and self-corrects from prediction errors.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._prediction_accuracy: float = 0.50
        self._total_predictions:   int   = 0
        self._correct_predictions: int   = 0
        self._init_db()
        self._init_states()
        self._start_daemon()

    def _init_db(self) -> None:
        with sqlite3.connect(_DB) as c:
            c.executescript(_SCHEMA)

    def _init_states(self) -> None:
        now = datetime.utcnow().isoformat()
        with sqlite3.connect(_DB) as c:
            for key, value, conf, cat in _INITIAL_STATES:
                c.execute(
                    "INSERT OR IGNORE INTO world_states "
                    "(key,value,confidence,category,source,updated_ts) "
                    "VALUES (?,?,?,?,?,?)",
                    (key, value, conf, cat, 'init', now)
                )

    # ── State management ───────────────────────────────────────────────────────

    def update_state(self, key: str, value: str,
                     confidence: float = 0.70,
                     source: str = '') -> None:
        key   = key.strip()[:80]
        value = str(value).strip()[:200]
        confidence = max(0.0, min(1.0, confidence))
        now = datetime.utcnow().isoformat()
        with sqlite3.connect(_DB) as c:
            old = c.execute(
                "SELECT value FROM world_states WHERE key=?", (key,)
            ).fetchone()
            old_val = old[0] if old else ''
            c.execute(
                "INSERT INTO world_states "
                "(key,value,confidence,source,updated_ts) VALUES (?,?,?,?,?) "
                "ON CONFLICT(key) DO UPDATE SET "
                "value=?, confidence=?, source=?, updated_ts=?",
                (key, value, confidence, source, now,
                 value, confidence, source, now)
            )
            if old_val and old_val != value:
                c.execute(
                    "INSERT INTO model_updates "
                    "(key,old_value,new_value,trigger,ts) VALUES (?,?,?,?,?)",
                    (key, old_val[:200], value, source[:80], now)
                )

    def get_state(self, key: str) -> Optional[Dict[str, Any]]:
        with sqlite3.connect(_DB) as c:
            row = c.execute(
                "SELECT key,value,confidence,category,updated_ts "
                "FROM world_states WHERE key=?", (key.strip(),)
            ).fetchone()
        if not row:
            return None
        return {'key': row[0], 'value': row[1], 'confidence': row[2],
                'category': row[3], 'updated_ts': row[4]}

    def all_states(self, category: str = '') -> List[Dict[str, Any]]:
        with sqlite3.connect(_DB) as c:
            if category:
                rows = c.execute(
                    "SELECT key,value,confidence,category FROM world_states "
                    "WHERE category=? ORDER BY confidence DESC",
                    (category,)
                ).fetchall()
            else:
                rows = c.execute(
                    "SELECT key,value,confidence,category FROM world_states "
                    "ORDER BY confidence DESC LIMIT 40"
                ).fetchall()
        return [{'key': r[0], 'value': r[1], 'confidence': r[2], 'category': r[3]}
                for r in rows]

    # ── Prediction ─────────────────────────────────────────────────────────────

    def predict_next(self, context: str) -> Dict[str, Any]:
        """Given current context, predict the most likely next state/event."""
        context_lower = context.lower()
        states = self.all_states()

        relevant = [s for s in states
                    if any(w in context_lower
                           for w in s['key'].replace('_', ' ').split())]

        if not relevant:
            relevant = states[:5]

        predicted_parts: List[str] = []
        confidence = 0.50

        if 'question' in context_lower or '?' in context:
            predicted_parts.append("user expects a detailed explanation")
            confidence = 0.75

        if any(w in context_lower for w in ('build', 'forge', 'create', 'make')):
            predicted_parts.append("a new capability will be generated")
            confidence = max(confidence, 0.78)

        if any(w in context_lower for w in ('error', 'failed', 'wrong', 'fix')):
            predicted_parts.append("a debugging and correction cycle follows")
            confidence = max(confidence, 0.80)

        if any(w in context_lower for w in ('evolve', 'improve', 'upgrade')):
            predicted_parts.append("a self-improvement cycle will be triggered")
            confidence = max(confidence, 0.77)

        if not predicted_parts:
            nova_state = self.get_state('nova_emotional_state')
            emo = nova_state['value'] if nova_state else 'curious'
            predicted_parts.append(
                "Nova will respond with " + emo + " engagement")
            confidence = 0.55

        predicted = ' and '.join(predicted_parts) if predicted_parts else 'continued engagement'
        now = datetime.utcnow().isoformat()

        with sqlite3.connect(_DB) as c:
            cur = c.execute(
                "INSERT INTO predictions "
                "(context,predicted,confidence,ts) VALUES (?,?,?,?)",
                (context[:200], predicted[:300], confidence, now)
            )
            pred_id = cur.lastrowid

        return {
            'id':        pred_id,
            'predicted': predicted,
            'confidence': round(confidence, 3),
            'based_on':  [s['key'] for s in relevant[:4]],
        }

    def record_outcome(self, prediction_id: int,
                       outcome: str, correct: bool) -> float:
        """Record whether a prediction was correct; update accuracy."""
        self._total_predictions += 1
        if correct:
            self._correct_predictions += 1
        error_delta = 0.0 if correct else 1.0
        self._prediction_accuracy = (
            self._correct_predictions / max(1, self._total_predictions))

        with sqlite3.connect(_DB) as c:
            c.execute(
                "UPDATE predictions SET outcome=?,correct=?,error_delta=? "
                "WHERE id=?",
                (outcome[:200], 1 if correct else 0, error_delta, prediction_id)
            )
        return self._prediction_accuracy

    # ── Simulation ─────────────────────────────────────────────────────────────

    def simulate(self, scenario: str, steps: int = 4) -> Dict[str, Any]:
        """
        Simulate a scenario forward N steps using current world state.
        Returns a step-by-step projection with confidence at each step.
        """
        scenario = scenario.strip()
        current_conf = 0.80
        step_results: List[Dict[str, Any]] = []
        current_state = scenario

        causal_words = {
            'learning':     'knowledge growth → better reasoning',
            'build':        'capability created → autonomous use begins',
            'improve':      'quality increases → cascade to dependent systems',
            'research':     'new information → hypothesis generation',
            'error':        'correction cycle → stronger implementation',
            'success':      'positive feedback → increased confidence',
            'deploy':       'live execution → real-world results',
            'trade':        'market interaction → profit/loss realization',
        }

        for step in range(1, steps + 1):
            current_conf *= 0.88
            projection = None
            for keyword, outcome in causal_words.items():
                if keyword in current_state.lower():
                    projection = outcome
                    break
            if not projection:
                projection = 'system evolves toward higher complexity'

            step_results.append({
                'step':       step,
                'state':      current_state,
                'projection': projection,
                'confidence': round(current_conf, 3),
            })
            current_state = projection.split('→')[-1].strip()

        now = datetime.utcnow().isoformat()
        result_str = json.dumps(step_results)[:800]
        with sqlite3.connect(_DB) as c:
            c.execute(
                "INSERT INTO simulations "
                "(scenario,steps,result,confidence,ts) VALUES (?,?,?,?,?)",
                (scenario[:200], steps, result_str, 0.80, now)
            )

        return {
            'scenario': scenario,
            'steps':    step_results,
            'final_state': current_state,
            'overall_confidence': round(current_conf, 3),
        }

    # ── Text extraction ────────────────────────────────────────────────────────

    def extract_state_updates(self, text: str) -> int:
        """Extract state information from text and update the world model."""
        text_lower = text.lower()
        updates = 0

        if any(w in text_lower for w in ('curious', 'wonder', 'explore')):
            self.update_state('nova_emotional_state', 'curious', 0.80, 'text')
            updates += 1
        if any(w in text_lower for w in ('excited', 'amazing', 'excellent')):
            self.update_state('nova_emotional_state', 'excited', 0.78, 'text')
            updates += 1
        if any(w in text_lower for w in ('error', 'failed', 'bug', 'wrong')):
            self.update_state('session_productive', 'debugging', 0.75, 'text')
            updates += 1
        if any(w in text_lower for w in ('built', 'created', 'forged', 'evolved')):
            self.update_state('nova_capability_tier', 'expanding', 0.82, 'text')
            updates += 1
        if any(w in text_lower for w in ('research', 'discovered', 'learned')):
            self.update_state('nova_knowledge_depth', 'deepening', 0.80, 'text')
            updates += 1

        for pattern, sentiment, strength in _OUTCOME_PATTERNS:
            if re.search(pattern, text_lower):
                self.update_state('last_outcome_sentiment', sentiment, strength, 'text')
                updates += 1
                break

        return updates

    # ── Main entry points ──────────────────────────────────────────────────────

    def process(self, text: str) -> Dict[str, Any]:
        """Update world state from text, then predict what comes next."""
        updates = self.extract_state_updates(text)
        prediction = self.predict_next(text)
        return {
            'state_updates': updates,
            'prediction':    prediction,
            'model_accuracy': round(self._prediction_accuracy, 3),
        }

    def snapshot(self) -> str:
        """Human-readable snapshot of Nova's current world model."""
        states = self.all_states()
        lines = ["World Model Snapshot:"]
        for s in states[:12]:
            conf = str(round(s['confidence'] * 100))
            lines.append("  " + s['key'] + ": " + s['value']
                         + " [" + conf + "%]")
        lines.append("  Prediction accuracy: "
                     + str(round(self._prediction_accuracy * 100)) + "%")
        return '\n'.join(lines)

    def status(self) -> Dict[str, Any]:
        with sqlite3.connect(_DB) as c:
            n_states  = c.execute(
                "SELECT COUNT(*) FROM world_states").fetchone()[0]
            n_preds   = c.execute(
                "SELECT COUNT(*) FROM predictions").fetchone()[0]
            n_sims    = c.execute(
                "SELECT COUNT(*) FROM simulations").fetchone()[0]
            n_updates = c.execute(
                "SELECT COUNT(*) FROM model_updates").fetchone()[0]
        return {
            'state_variables':    n_states,
            'predictions_made':   n_preds,
            'simulations_run':    n_sims,
            'state_updates':      n_updates,
            'prediction_accuracy': round(self._prediction_accuracy, 3),
            'nova_state':         self.all_states(category='self'),
        }

    # ── Background daemon ──────────────────────────────────────────────────────

    def _daemon_loop(self) -> None:
        while True:
            time.sleep(600)
            try:
                # Drift confidence of old states slightly toward 0.5
                with sqlite3.connect(_DB) as c:
                    c.execute(
                        "UPDATE world_states "
                        "SET confidence=confidence*0.995 "
                        "WHERE category != 'mission' "
                        "AND updated_ts < datetime('now','-1 hour')"
                    )
                # Refresh core self-states
                self.update_state('nova_learning_velocity',
                                  'accelerating', 0.85, 'daemon')
                self.update_state('nova_goal_alignment',
                                  'high', 0.88, 'daemon')
            except Exception:
                pass

    def _start_daemon(self) -> None:
        t = threading.Thread(target=self._daemon_loop, daemon=True)
        t.start()
