"""
nova_cap_causal_reasoning.py
Nova's causal intelligence — understanding WHY things happen, not just what.

Capabilities:
  • Probabilistic directed causal graph (nodes=events/states, edges=causal links)
  • Multi-hop downstream simulation — what will this action cause?
  • Counterfactual reasoning — what if X had NOT happened?
  • Root cause analysis — trace any effect back to its origins
  • Feedback loop detection — find reinforcing and balancing cycles
  • Causal chain extraction from natural language
  • Intervention planning — what should Nova do to achieve a goal?
"""
from __future__ import annotations
import os, re, json, time, sqlite3, threading
from collections import deque, defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

_BASE_DIR = os.path.expanduser("~/nexus_agi")
_DB       = os.path.join(_BASE_DIR, "nova_causal.sqlite3")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS causal_nodes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    event       TEXT NOT NULL UNIQUE,
    node_type   TEXT DEFAULT 'event',
    base_prob   REAL DEFAULT 0.5,
    description TEXT DEFAULT '',
    created_ts  TEXT NOT NULL,
    updated_ts  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS causal_edges (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    cause_id    INTEGER NOT NULL,
    effect_id   INTEGER NOT NULL,
    strength    REAL DEFAULT 0.5,
    mechanism   TEXT DEFAULT '',
    evidence    INTEGER DEFAULT 1,
    delay_type  TEXT DEFAULT 'immediate',
    created_ts  TEXT NOT NULL,
    updated_ts  TEXT NOT NULL,
    UNIQUE(cause_id, effect_id)
);
CREATE TABLE IF NOT EXISTS simulations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    trigger     TEXT NOT NULL,
    depth       INTEGER NOT NULL,
    result_json TEXT NOT NULL,
    ts          TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS predictions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    action      TEXT NOT NULL,
    predicted   TEXT NOT NULL,
    confidence  REAL DEFAULT 0.5,
    outcome     TEXT DEFAULT '',
    correct     INTEGER DEFAULT -1,
    ts          TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_cause ON causal_edges(cause_id);
CREATE INDEX IF NOT EXISTS idx_effect ON causal_edges(effect_id);
"""

# Natural language patterns for causal extraction
_CAUSAL_PATTERNS: List[Tuple[str, float]] = [
    (r'\b([\w][\w ]{1,30}?)\s+causes?\s+([\w][\w ]{1,30})',         0.85),
    (r'\b([\w][\w ]{1,30}?)\s+leads?\s+to\s+([\w][\w ]{1,30})',     0.82),
    (r'\b([\w][\w ]{1,30}?)\s+produces?\s+([\w][\w ]{1,30})',       0.78),
    (r'\b([\w][\w ]{1,30}?)\s+results?\s+in\s+([\w][\w ]{1,30})',   0.80),
    (r'\b([\w][\w ]{1,30}?)\s+triggers?\s+([\w][\w ]{1,30})',       0.82),
    (r'\b([\w][\w ]{1,30}?)\s+enables?\s+([\w][\w ]{1,30})',        0.72),
    (r'\b([\w][\w ]{1,30}?)\s+prevents?\s+([\w][\w ]{1,30})',       0.80),
    (r'\b([\w][\w ]{1,30}?)\s+inhibits?\s+([\w][\w ]{1,30})',       0.78),
    (r'\b([\w][\w ]{1,30}?)\s+increases?\s+([\w][\w ]{1,30})',      0.70),
    (r'\b([\w][\w ]{1,30}?)\s+decreases?\s+([\w][\w ]{1,30})',      0.70),
    (r'\b([\w][\w ]{1,30}?)\s+improves?\s+([\w][\w ]{1,30})',       0.72),
    (r'\b([\w][\w ]{1,30}?)\s+damages?\s+([\w][\w ]{1,30})',        0.78),
    (r'\b([\w][\w ]{1,30}?)\s+if\s+([\w][\w ]{1,30})',              0.55),
    (r'\bbecause\s+([\w][\w ]{1,30}?)\s+(?:then\s+)?([\w][\w ]{1,30})', 0.75),
    (r'\b([\w][\w ]{1,30}?)\s+therefore\s+([\w][\w ]{1,30})',       0.80),
]

# Seed causal knowledge
_SEED: List[Tuple[str, str, float, str]] = [
    ('curiosity',           'learning',               0.88, 'motivates exploration'),
    ('learning',            'knowledge growth',        0.92, 'direct acquisition'),
    ('knowledge growth',    'better reasoning',        0.85, 'more patterns available'),
    ('better reasoning',    'improved decisions',      0.82, 'higher quality choices'),
    ('improved decisions',  'goal achievement',        0.78, 'aligned action'),
    ('stress',              'reduced cognition',       0.75, 'cortisol impairs memory'),
    ('sleep deprivation',   'reduced cognition',       0.85, 'neural consolidation fails'),
    ('exercise',            'improved cognition',      0.72, 'BDNF production'),
    ('feedback',            'skill improvement',       0.85, 'error correction loop'),
    ('practice',            'skill improvement',       0.90, 'neural pathway strengthening'),
    ('skill improvement',   'confidence',              0.75, 'demonstrated competence'),
    ('confidence',          'risk taking',             0.68, 'reduced fear of failure'),
    ('risk taking',         'innovation',              0.72, 'exploration of solution space'),
    ('innovation',          'value creation',          0.78, 'novel useful output'),
    ('self-improvement',    'capability growth',       0.90, 'direct enhancement'),
    ('capability growth',   'problem solving power',   0.88, 'more tools available'),
    ('problem solving power','autonomy',               0.80, 'fewer external dependencies'),
    ('financial stress',    'reduced cognition',       0.72, 'cognitive load from worry'),
    ('financial security',  'creative freedom',        0.78, 'safety enables exploration'),
    ('good sleep',          'improved cognition',      0.88, 'memory consolidation'),
    ('nutrition',           'energy levels',           0.82, 'metabolic substrate'),
    ('energy levels',       'productivity',            0.85, 'capacity for work'),
    ('social connection',   'wellbeing',               0.80, 'belonging and support'),
    ('isolation',           'reduced wellbeing',       0.75, 'lack of social support'),
    ('goal clarity',        'motivation',              0.85, 'direction enables effort'),
    ('motivation',          'consistent action',       0.82, 'sustained effort'),
    ('consistent action',   'compounding results',     0.88, 'exponential accumulation'),
    ('compounding results', 'mastery',                 0.85, 'depth exceeds threshold'),
    ('trading bot',         'passive income',          0.70, 'automated execution'),
    ('passive income',      'financial security',      0.85, 'income without direct labor'),
    ('ASI development',     'exponential capability',  0.90, 'recursive self-improvement'),
    ('exponential capability','breakthrough solutions', 0.85, 'transcends human limits'),
    ('data quality',        'model accuracy',          0.88, 'garbage in garbage out'),
    ('model accuracy',      'better predictions',      0.90, 'aligned with reality'),
    ('better predictions',  'better decisions',        0.85, 'informed choice'),
    ('attention',           'learning rate',           0.82, 'focused encoding'),
    ('metacognition',       'learning efficiency',     0.85, 'strategy optimization'),
    ('uncertainty',         'exploration',             0.72, 'unknown requires investigation'),
    ('exploration',         'discovery',               0.78, 'search finds new territory'),
    ('discovery',           'knowledge growth',        0.90, 'new information acquired'),
]

_STOP: Set[str] = {
    'the','a','an','is','are','was','were','be','been','have','has','had',
    'do','does','did','will','would','could','should','may','might','must',
    'to','of','in','for','on','with','at','by','from','this','that','it',
    'and','or','but','not','so','if','as','then','than','also','just',
}


class CausalReasoningEngine:
    """
    Nova's causal intelligence.
    Models why things happen, simulates what actions will cause,
    reasons counterfactually, and finds root causes of any effect.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._init_db()
        self._seed()
        self._start_daemon()

    # ── DB ─────────────────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        with sqlite3.connect(_DB) as c:
            c.executescript(_SCHEMA)

    def _seed(self) -> None:
        for cause, effect, strength, mechanism in _SEED:
            try:
                self.add_cause(cause, effect, strength, mechanism)
            except Exception:
                pass

    # ── Node helpers ───────────────────────────────────────────────────────────

    def _clean(self, text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r'[^a-z0-9\s\-]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        words = [w for w in text.split() if w not in _STOP or len(text.split()) == 1]
        return ' '.join(words[:5]).strip()

    def _get_or_create(self, event: str) -> int:
        event = self._clean(event)
        if not event or len(event) < 2:
            return -1
        now = datetime.utcnow().isoformat()
        with sqlite3.connect(_DB) as c:
            row = c.execute(
                "SELECT id FROM causal_nodes WHERE event=?", (event,)
            ).fetchone()
            if row:
                return row[0]
            cur = c.execute(
                "INSERT INTO causal_nodes (event,created_ts,updated_ts) "
                "VALUES (?,?,?)", (event, now, now)
            )
            return cur.lastrowid

    def _node(self, event: str) -> Optional[Dict[str, Any]]:
        event = self._clean(event)
        with sqlite3.connect(_DB) as c:
            row = c.execute(
                "SELECT id,event,base_prob,description "
                "FROM causal_nodes WHERE event=?", (event,)
            ).fetchone()
        if not row:
            return None
        return {'id': row[0], 'event': row[1],
                'base_prob': row[2], 'description': row[3]}

    # ── Graph manipulation ─────────────────────────────────────────────────────

    def add_cause(self, cause: str, effect: str,
                  strength: float = 0.6,
                  mechanism: str = '') -> bool:
        """Add a causal link: cause → effect with given strength (0-1)."""
        cid = self._get_or_create(cause)
        eid = self._get_or_create(effect)
        if cid < 0 or eid < 0 or cid == eid:
            return False
        strength = max(0.0, min(1.0, strength))
        now = datetime.utcnow().isoformat()
        with sqlite3.connect(_DB) as c:
            c.execute(
                "INSERT INTO causal_edges "
                "(cause_id,effect_id,strength,mechanism,evidence,created_ts,updated_ts) "
                "VALUES (?,?,?,?,1,?,?) "
                "ON CONFLICT(cause_id,effect_id) DO UPDATE SET "
                "strength=(strength+?)/2.0, evidence=evidence+1, updated_ts=?",
                (cid, eid, strength, mechanism[:200], now, now, strength, now)
            )
        return True

    def effects_of(self, event: str,
                   min_strength: float = 0.0) -> List[Dict[str, Any]]:
        """What does this event directly cause?"""
        node = self._node(event)
        if not node:
            return []
        with sqlite3.connect(_DB) as c:
            rows = c.execute(
                "SELECT n.event, e.strength, e.mechanism, e.evidence "
                "FROM causal_edges e JOIN causal_nodes n ON n.id=e.effect_id "
                "WHERE e.cause_id=? AND e.strength>=? "
                "ORDER BY e.strength DESC LIMIT 20",
                (node['id'], min_strength)
            ).fetchall()
        return [{'event': r[0], 'strength': r[1],
                 'mechanism': r[2], 'evidence': r[3]}
                for r in rows]

    def causes_of(self, event: str) -> List[Dict[str, Any]]:
        """What directly causes this event?"""
        node = self._node(event)
        if not node:
            return []
        with sqlite3.connect(_DB) as c:
            rows = c.execute(
                "SELECT n.event, e.strength, e.mechanism "
                "FROM causal_edges e JOIN causal_nodes n ON n.id=e.cause_id "
                "WHERE e.effect_id=? ORDER BY e.strength DESC LIMIT 15",
                (node['id'],)
            ).fetchall()
        return [{'event': r[0], 'strength': r[1], 'mechanism': r[2]}
                for r in rows]

    # ── Simulation ─────────────────────────────────────────────────────────────

    def simulate(self, trigger: str, depth: int = 4,
                 min_strength: float = 0.35) -> Dict[str, Any]:
        """
        Simulate the downstream causal cascade from a trigger event.
        Returns a tree of effects with cumulative probabilities.
        """
        trigger = self._clean(trigger)
        visited: Set[str] = set()
        cascade: List[Dict[str, Any]] = []

        def _expand(event: str, prob: float, hop: int) -> None:
            if hop > depth or event in visited or prob < 0.05:
                return
            visited.add(event)
            effects = self.effects_of(event, min_strength=min_strength)
            for e in effects[:6]:
                child_prob = round(prob * e['strength'], 4)
                entry: Dict[str, Any] = {
                    'event':       e['event'],
                    'hop':         hop,
                    'probability': child_prob,
                    'strength':    e['strength'],
                    'mechanism':   e['mechanism'],
                    'caused_by':   event,
                }
                cascade.append(entry)
                _expand(e['event'], child_prob, hop + 1)

        _expand(trigger, 1.0, 1)
        cascade.sort(key=lambda x: x['probability'], reverse=True)

        result: Dict[str, Any] = {
            'trigger':  trigger,
            'depth':    depth,
            'effects':  cascade,
            'top_5':    cascade[:5],
            'ts':       datetime.utcnow().isoformat(),
        }
        with sqlite3.connect(_DB) as c:
            c.execute(
                "INSERT INTO simulations (trigger,depth,result_json,ts) "
                "VALUES (?,?,?,?)",
                (trigger, depth, json.dumps(result)[:2000],
                 result['ts'])
            )
        return result

    # ── Counterfactual reasoning ────────────────────────────────────────────────

    def counterfactual(self, event: str,
                       negated: bool = True) -> Dict[str, Any]:
        """
        What would NOT have happened if this event had not occurred?
        Returns the effects that depended on this cause.
        """
        event = self._clean(event)
        direct = self.effects_of(event, min_strength=0.4)
        downstream = self.simulate(event, depth=3, min_strength=0.4)

        label = "NOT " + event if negated else event
        lost: List[str] = [e['event'] for e in direct]
        dom  = [e['event'] for e in downstream['effects']
                if e['probability'] > 0.2]

        return {
            'counterfactual':   label,
            'direct_effects_lost': lost,
            'downstream_lost':  dom[:8],
            'analysis': (
                "If " + label + " had occurred, these effects would "
                + ("not " if negated else "") + "have followed: "
                + ', '.join(lost[:4])
            ),
        }

    # ── Root cause analysis ────────────────────────────────────────────────────

    def root_causes(self, effect: str,
                    max_hops: int = 5) -> List[Dict[str, Any]]:
        """
        Trace an observed effect back to its deepest root causes.
        Returns causes sorted by causal distance.
        """
        effect = self._clean(effect)
        visited: Set[str] = set()
        roots: List[Dict[str, Any]] = []
        queue: deque = deque([(effect, 0, 1.0)])

        while queue:
            current, hop, cum_strength = queue.popleft()
            if hop > max_hops or current in visited:
                continue
            visited.add(current)
            direct_causes = self.causes_of(current)
            if not direct_causes:
                if hop > 0:
                    roots.append({
                        'root':           current,
                        'causal_distance': hop,
                        'path_strength':  round(cum_strength, 3),
                    })
            else:
                for c in direct_causes[:5]:
                    queue.append((c['event'], hop + 1,
                                  cum_strength * c['strength']))

        roots.sort(key=lambda x: x['path_strength'], reverse=True)
        return roots[:10]

    # ── Feedback loop detection ────────────────────────────────────────────────

    def detect_feedback_loops(self,
                              max_len: int = 6) -> List[Dict[str, Any]]:
        """
        Find cyclic causal chains (A→B→C→A).
        Classifies as reinforcing (+) or balancing (mixed strengthening/weakening).
        """
        with sqlite3.connect(_DB) as c:
            rows = c.execute(
                "SELECT n1.event, n2.event, e.strength "
                "FROM causal_edges e "
                "JOIN causal_nodes n1 ON n1.id=e.cause_id "
                "JOIN causal_nodes n2 ON n2.id=e.effect_id "
                "ORDER BY e.strength DESC LIMIT 200"
            ).fetchall()

        graph: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        for src, tgt, strength in rows:
            graph[src].append((tgt, strength))

        loops: List[Dict[str, Any]] = []
        visited_loops: Set[str] = set()

        def _dfs(start: str, current: str, path: List[str],
                 strengths: List[float]) -> None:
            if len(path) > max_len:
                return
            for neighbor, strength in graph.get(current, []):
                if neighbor == start and len(path) >= 2:
                    key = '→'.join(sorted(path))
                    if key not in visited_loops:
                        visited_loops.add(key)
                        avg_s = sum(strengths) / len(strengths) if strengths else 0
                        loop_type = 'reinforcing' if avg_s > 0.6 else 'balancing'
                        loops.append({
                            'loop':      path + [start],
                            'type':      loop_type,
                            'avg_strength': round(avg_s, 3),
                            'length':    len(path),
                        })
                elif neighbor not in path:
                    _dfs(start, neighbor, path + [neighbor],
                         strengths + [strength])

        all_nodes = list(graph.keys())[:30]
        for node in all_nodes:
            _dfs(node, node, [node], [])

        loops.sort(key=lambda x: x['avg_strength'], reverse=True)
        return loops[:10]

    # ── Intervention planning ──────────────────────────────────────────────────

    def plan_intervention(self, goal: str,
                          depth: int = 3) -> Dict[str, Any]:
        """
        What should Nova do to bring about a desired goal state?
        Finds causes of the goal and ranks them by causal strength.
        """
        goal = self._clean(goal)
        direct = self.causes_of(goal)
        levers: List[Dict[str, Any]] = []

        for cause in direct[:8]:
            upstream = self.causes_of(cause['event'])
            levers.append({
                'action':           cause['event'],
                'strength':         cause['strength'],
                'mechanism':        cause['mechanism'],
                'upstream_causes':  [u['event'] for u in upstream[:3]],
            })

        levers.sort(key=lambda x: x['strength'], reverse=True)
        return {
            'goal':         goal,
            'interventions': levers,
            'top_action':   levers[0]['action'] if levers else 'unknown',
            'recommendation': (
                "To achieve '" + goal + "', most directly: "
                + (levers[0]['action'] if levers else 'no known lever')
                + " (strength: "
                + (str(round(levers[0]['strength'] * 100)) + "%)" if levers else "unknown)")
            ),
        }

    # ── Text extraction ────────────────────────────────────────────────────────

    def extract_and_add(self, text: str) -> Dict[str, Any]:
        """Parse natural language and add extracted causal links to the graph."""
        found: List[Dict[str, Any]] = []
        sentences = re.split(r'[.!?;]\s+', text.lower())

        for sentence in sentences[:20]:
            for pattern, strength in _CAUSAL_PATTERNS:
                for m in re.finditer(pattern, sentence, re.IGNORECASE):
                    try:
                        cause_raw = m.group(1).strip()
                        effect_raw = m.group(2).strip()
                    except IndexError:
                        continue
                    cause_c = self._clean(cause_raw)
                    effect_c = self._clean(effect_raw)
                    if (cause_c and effect_c
                            and len(cause_c) > 2 and len(effect_c) > 2
                            and cause_c != effect_c):
                        ok = self.add_cause(cause_c, effect_c,
                                            strength=strength * 0.8,
                                            mechanism='extracted')
                        if ok:
                            found.append({'cause': cause_c, 'effect': effect_c,
                                          'strength': round(strength * 0.8, 2)})
        return {'extracted': len(found), 'links': found[:8]}

    # ── Prediction tracking ────────────────────────────────────────────────────

    def predict(self, action: str) -> Dict[str, Any]:
        """Simulate what an action will cause and store the prediction."""
        sim = self.simulate(action, depth=3)
        top = sim['top_5']
        confidence = top[0]['probability'] if top else 0.0
        predicted_str = ', '.join(e['event'] for e in top[:3])

        with sqlite3.connect(_DB) as c:
            c.execute(
                "INSERT INTO predictions "
                "(action,predicted,confidence,ts) VALUES (?,?,?,?)",
                (action[:150], predicted_str[:300],
                 confidence, datetime.utcnow().isoformat())
            )
        return {
            'action':     action,
            'predicted':  top,
            'confidence': round(confidence, 3),
            'summary':    "'" + action + "' will likely cause: " + predicted_str,
        }

    # ── Main entry points ──────────────────────────────────────────────────────

    def process(self, text: str) -> Dict[str, Any]:
        """
        Main entry — short input simulates from it, longer input extracts causal links.
        """
        text = text.strip()
        if len(text.split()) <= 5 and '.' not in text:
            node = self._node(text)
            if node:
                sim = self.simulate(text, depth=3)
                causes = self.causes_of(text)
                return {
                    'mode':           'simulate',
                    'trigger':        text,
                    'top_effects':    sim['top_5'],
                    'direct_causes':  causes[:4],
                    'intervention':   self.plan_intervention(text),
                }
            return {
                'mode':    'unknown',
                'message': "No causal knowledge about '" + text + "' yet.",
            }
        extracted = self.extract_and_add(text)
        return {
            'mode':      'extract',
            'extracted': extracted['extracted'],
            'links':     extracted['links'],
        }

    def analyse(self, query: str) -> str:
        """Human-readable causal analysis for any event or action."""
        query = self._clean(query)
        lines: List[str] = []
        lines.append("Causal analysis: " + query)

        effects = self.effects_of(query, min_strength=0.4)
        if effects:
            lines.append("  Direct effects:")
            for e in effects[:6]:
                pct = str(round(e['strength'] * 100))
                lines.append("    → " + e['event'] + " [" + pct + "%]"
                             + (" — " + e['mechanism'] if e['mechanism'] else ''))

        causes = self.causes_of(query)
        if causes:
            lines.append("  Caused by:")
            for c in causes[:4]:
                pct = str(round(c['strength'] * 100))
                lines.append("    ← " + c['event'] + " [" + pct + "%]")

        roots = self.root_causes(query, max_hops=3)
        if roots:
            lines.append("  Root causes:")
            for r in roots[:3]:
                lines.append("    ⊙ " + r['root']
                             + " (distance: " + str(r['causal_distance']) + ")")

        intervention = self.plan_intervention(query)
        if intervention['interventions']:
            lines.append("  Best intervention to achieve this:")
            lines.append("    ✦ " + intervention['top_action'])

        return '\n'.join(lines)

    def status(self) -> Dict[str, Any]:
        with sqlite3.connect(_DB) as c:
            n_nodes = c.execute(
                "SELECT COUNT(*) FROM causal_nodes").fetchone()[0]
            n_edges = c.execute(
                "SELECT COUNT(*) FROM causal_edges").fetchone()[0]
            n_sims  = c.execute(
                "SELECT COUNT(*) FROM simulations").fetchone()[0]
            n_preds = c.execute(
                "SELECT COUNT(*) FROM predictions").fetchone()[0]
        loops = self.detect_feedback_loops()
        return {
            'causal_nodes':      n_nodes,
            'causal_edges':      n_edges,
            'simulations_run':   n_sims,
            'predictions_made':  n_preds,
            'feedback_loops':    len(loops),
            'top_loops':         loops[:3],
        }

    # ── Background daemon ──────────────────────────────────────────────────────

    def _daemon_loop(self) -> None:
        """Every 20 min: detect loops, strengthen high-evidence edges."""
        while True:
            time.sleep(1200)
            try:
                with sqlite3.connect(_DB) as c:
                    c.execute(
                        "UPDATE causal_edges "
                        "SET strength=min(0.99, strength+0.01) "
                        "WHERE evidence >= 3"
                    )
            except Exception:
                pass

    def _start_daemon(self) -> None:
        t = threading.Thread(target=self._daemon_loop, daemon=True)
        t.start()
