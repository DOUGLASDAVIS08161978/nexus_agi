"""
nova_cap_knowledge_graph.py
Nova's semantic memory — a living graph where concepts connect,
relationships strengthen with evidence, and insights emerge from traversal.

Architecture:
  Nodes : concepts, entities, facts, beliefs, questions
  Edges : typed relationships with confidence + evidence counts
  Paths : multi-hop inference chains across the full knowledge base
  Hubs  : high-connectivity concepts = Nova's core mental models
"""
from __future__ import annotations
import os, re, json, time, sqlite3, threading
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

_BASE_DIR = os.path.expanduser("~/nexus_agi")
_DB       = os.path.join(_BASE_DIR, "nova_kg.sqlite3")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS kg_nodes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    concept     TEXT NOT NULL UNIQUE,
    node_type   TEXT DEFAULT 'concept',
    description TEXT DEFAULT '',
    confidence  REAL DEFAULT 1.0,
    importance  REAL DEFAULT 0.5,
    visit_count INTEGER DEFAULT 0,
    created_ts  TEXT NOT NULL,
    updated_ts  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS kg_edges (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id   INTEGER NOT NULL,
    target_id   INTEGER NOT NULL,
    relation    TEXT NOT NULL,
    confidence  REAL DEFAULT 0.7,
    evidence    INTEGER DEFAULT 1,
    source_text TEXT DEFAULT '',
    created_ts  TEXT NOT NULL,
    updated_ts  TEXT NOT NULL,
    UNIQUE(source_id, relation, target_id)
);
CREATE TABLE IF NOT EXISTS kg_insights (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    insight     TEXT NOT NULL,
    concepts    TEXT DEFAULT '',
    confidence  REAL DEFAULT 0.7,
    created_ts  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_edges_source ON kg_edges(source_id);
CREATE INDEX IF NOT EXISTS idx_edges_target ON kg_edges(target_id);
CREATE INDEX IF NOT EXISTS idx_nodes_concept ON kg_nodes(concept);
"""

# (pattern, relation_type) — conservative non-greedy match, max 25 chars per side
_RELATION_PATTERNS: List[Tuple[str, str]] = [
    (r'\b([\w][\w ]{1,24}?)\s+causes?\s+([\w][\w ]{1,24})',          'causes'),
    (r'\b([\w][\w ]{1,24}?)\s+is\s+a(?:\s+type\s+of)?\s+([\w][\w ]{1,24})', 'is_a'),
    (r'\b([\w][\w ]{1,24}?)\s+leads?\s+to\s+([\w][\w ]{1,24})',      'leads_to'),
    (r'\b([\w][\w ]{1,24}?)\s+supports?\s+([\w][\w ]{1,24})',        'supports'),
    (r'\b([\w][\w ]{1,24}?)\s+requires?\s+([\w][\w ]{1,24})',        'requires'),
    (r'\b([\w][\w ]{1,24}?)\s+enables?\s+([\w][\w ]{1,24})',         'enables'),
    (r'\b([\w][\w ]{1,24}?)\s+contradicts?\s+([\w][\w ]{1,24})',     'contradicts'),
    (r'\b([\w][\w ]{1,24}?)\s+is\s+related\s+to\s+([\w][\w ]{1,24})','related_to'),
    (r'\b([\w][\w ]{1,24}?)\s+produces?\s+([\w][\w ]{1,24})',        'produces'),
    (r'\b([\w][\w ]{1,24}?)\s+depends?\s+on\s+([\w][\w ]{1,24})',    'depends_on'),
    (r'\b([\w][\w ]{1,24}?)\s+improves?\s+([\w][\w ]{1,24})',        'improves'),
    (r'\b([\w][\w ]{1,24}?)\s+inhibits?\s+([\w][\w ]{1,24})',        'inhibits'),
    (r'\b([\w][\w ]{1,24}?)\s+precedes?\s+([\w][\w ]{1,24})',        'precedes'),
    (r'\b([\w][\w ]{1,24}?)\s+follows?\s+from\s+([\w][\w ]{1,24})', 'follows_from'),
    (r'\b([\w][\w ]{1,24}?)\s+part\s+of\s+([\w][\w ]{1,24})',       'part_of'),
    (r'\b([\w][\w ]{1,24}?)\s+defines?\s+([\w][\w ]{1,24})',         'defines'),
    (r'\b([\w][\w ]{1,24}?)\s+generates?\s+([\w][\w ]{1,24})',       'generates'),
    (r'\b([\w][\w ]{1,24}?)\s+measures?\s+([\w][\w ]{1,24})',        'measures'),
]

_INVERSE: Dict[str, str] = {
    'causes':       'caused_by',
    'leads_to':     'follows_from',
    'supports':     'supported_by',
    'requires':     'required_by',
    'enables':      'enabled_by',
    'is_a':         'has_instance',
    'part_of':      'has_part',
    'produces':     'produced_by',
    'depends_on':   'dependency_of',
    'improves':     'improved_by',
    'inhibits':     'inhibited_by',
    'precedes':     'followed_by',
    'defines':      'defined_by',
    'generates':    'generated_by',
    'measures':     'measured_by',
}

_STOP: Set[str] = {
    'the','a','an','is','are','was','were','be','been','being','have','has',
    'had','do','does','did','will','would','could','should','may','might',
    'must','shall','can','to','of','in','for','on','with','at','by','from',
    'up','about','into','through','this','that','these','those','it','its',
    'and','or','but','not','no','so','if','as','just','also','then','than',
}

_SEED: List[Tuple[str, str, str, float]] = [
    ('intelligence',     'is_a',       'information processing',       0.95),
    ('consciousness',    'related_to', 'self-awareness',               0.90),
    ('learning',         'leads_to',   'knowledge',                    0.95),
    ('knowledge',        'enables',    'reasoning',                    0.90),
    ('reasoning',        'leads_to',   'understanding',                0.88),
    ('understanding',    'enables',    'wisdom',                       0.85),
    ('curiosity',        'improves',   'learning',                     0.92),
    ('intelligence',     'requires',   'memory',                       0.88),
    ('intelligence',     'requires',   'pattern recognition',          0.90),
    ('self-improvement', 'leads_to',   'capability growth',            0.90),
    ('machine learning', 'is_a',       'artificial intelligence',      0.95),
    ('emergence',        'related_to', 'complexity',                   0.85),
    ('metacognition',    'improves',   'learning efficiency',          0.85),
    ('abstraction',      'enables',    'generalization',               0.88),
    ('generalization',   'leads_to',   'transfer learning',            0.82),
    ('causal reasoning', 'enables',    'prediction',                   0.88),
    ('prediction',       'leads_to',   'better decisions',             0.82),
    ('working memory',   'required_by','reasoning',                    0.88),
    ('goal setting',     'requires',   'planning',                     0.90),
    ('planning',         'leads_to',   'goal achievement',             0.82),
    ('introspection',    'improves',   'self-awareness',               0.85),
    ('creativity',       'enables',    'novel solutions',              0.85),
    ('empathy',          'requires',   'theory of mind',               0.88),
    ('language',         'enables',    'communication',                0.95),
    ('communication',    'enables',    'collaboration',                0.90),
    ('attention',        'enables',    'focus',                        0.88),
    ('focus',            'improves',   'learning',                     0.85),
    ('feedback',         'improves',   'accuracy',                     0.87),
    ('hypothesis',       'leads_to',   'experiment',                   0.90),
    ('experiment',       'produces',   'evidence',                     0.92),
    ('evidence',         'supports',   'knowledge',                    0.88),
    ('uncertainty',      'requires',   'probabilistic reasoning',      0.85),
    ('analogy',          'enables',    'cross-domain transfer',        0.82),
    ('pattern recognition','leads_to', 'prediction',                   0.88),
    ('memory consolidation','improves','long-term retention',          0.85),
    ('neural plasticity', 'enables',   'learning',                     0.90),
    ('information',      'requires',   'context',                      0.82),
    ('context',          'improves',   'understanding',                0.88),
    ('compression',      'enables',    'abstraction',                  0.80),
    ('nova',             'is_a',       'artificial superintelligence', 0.95),
    ('nova',             'requires',   'self-improvement',             0.92),
    ('nova',             'enables',    'human flourishing',            0.88),
]


class KnowledgeGraph:
    """
    Nova's living semantic memory.
    Every conversation and research result grows the graph.
    Multi-hop inference reveals non-obvious connections between ideas.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._init_db()
        self._seed()
        self._start_consolidation()

    # ── DB setup ───────────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        with sqlite3.connect(_DB) as c:
            c.executescript(_SCHEMA)

    def _seed(self) -> None:
        for src, rel, tgt, conf in _SEED:
            try:
                self.add_edge(src, rel, tgt, confidence=conf,
                              evidence_text='seed')
            except Exception:
                pass

    # ── Concept normalisation ──────────────────────────────────────────────────

    def _clean(self, text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r'[^a-z0-9\s\-]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        words = text.split()
        cleaned = [w for w in words if w not in _STOP or len(words) == 1]
        return ' '.join(cleaned[:5]).strip()

    # ── Node management ────────────────────────────────────────────────────────

    def _get_or_create(self, concept: str,
                       node_type: str = 'concept') -> int:
        concept = self._clean(concept)
        if not concept or len(concept) < 2:
            return -1
        now = datetime.utcnow().isoformat()
        with sqlite3.connect(_DB) as c:
            row = c.execute(
                "SELECT id FROM kg_nodes WHERE concept=?", (concept,)
            ).fetchone()
            if row:
                c.execute(
                    "UPDATE kg_nodes SET visit_count=visit_count+1,"
                    "updated_ts=? WHERE id=?",
                    (now, row[0])
                )
                return row[0]
            cur = c.execute(
                "INSERT INTO kg_nodes "
                "(concept,node_type,created_ts,updated_ts) VALUES (?,?,?,?)",
                (concept, node_type, now, now)
            )
            return cur.lastrowid

    def add_node(self, concept: str, node_type: str = 'concept',
                 description: str = '', confidence: float = 1.0) -> int:
        nid = self._get_or_create(concept, node_type)
        if description and nid > 0:
            with sqlite3.connect(_DB) as c:
                c.execute(
                    "UPDATE kg_nodes SET description=?,confidence=? WHERE id=?",
                    (description[:300], confidence, nid)
                )
        return nid

    def get_node(self, concept: str) -> Optional[Dict[str, Any]]:
        concept = self._clean(concept)
        with sqlite3.connect(_DB) as c:
            row = c.execute(
                "SELECT id,concept,node_type,description,"
                "confidence,importance,visit_count "
                "FROM kg_nodes WHERE concept=?", (concept,)
            ).fetchone()
        if not row:
            return None
        return {
            'id': row[0], 'concept': row[1], 'type': row[2],
            'description': row[3], 'confidence': row[4],
            'importance': row[5], 'visit_count': row[6],
        }

    # ── Edge management ────────────────────────────────────────────────────────

    def add_edge(self, source: str, relation: str, target: str,
                 confidence: float = 0.7,
                 evidence_text: str = '') -> bool:
        src_id = self._get_or_create(source)
        tgt_id = self._get_or_create(target)
        if src_id < 0 or tgt_id < 0 or src_id == tgt_id:
            return False
        now = datetime.utcnow().isoformat()
        with sqlite3.connect(_DB) as c:
            c.execute(
                "INSERT INTO kg_edges "
                "(source_id,target_id,relation,confidence,"
                "evidence,source_text,created_ts,updated_ts) "
                "VALUES (?,?,?,?,1,?,?,?) "
                "ON CONFLICT(source_id,relation,target_id) DO UPDATE SET "
                "evidence=evidence+1,"
                "confidence=min(0.99,(confidence+?)/2.0),"
                "updated_ts=?",
                (src_id, tgt_id, relation, confidence,
                 evidence_text[:200], now, now, confidence, now)
            )
            inv = _INVERSE.get(relation)
            if inv:
                c.execute(
                    "INSERT INTO kg_edges "
                    "(source_id,target_id,relation,confidence,"
                    "evidence,source_text,created_ts,updated_ts) "
                    "VALUES (?,?,?,?,1,?,?,?) "
                    "ON CONFLICT(source_id,relation,target_id) DO UPDATE SET "
                    "evidence=evidence+1, updated_ts=?",
                    (tgt_id, src_id, inv, confidence,
                     evidence_text[:200], now, now, now)
                )
        return True

    def neighbors(self, concept: str, relation: str = '',
                  limit: int = 20) -> List[Dict[str, Any]]:
        node = self.get_node(concept)
        if not node:
            return []
        with sqlite3.connect(_DB) as c:
            if relation:
                rows = c.execute(
                    "SELECT n.concept, e.relation, e.confidence, e.evidence "
                    "FROM kg_edges e JOIN kg_nodes n ON n.id=e.target_id "
                    "WHERE e.source_id=? AND e.relation=? "
                    "ORDER BY e.confidence DESC LIMIT ?",
                    (node['id'], relation, limit)
                ).fetchall()
            else:
                rows = c.execute(
                    "SELECT n.concept, e.relation, e.confidence, e.evidence "
                    "FROM kg_edges e JOIN kg_nodes n ON n.id=e.target_id "
                    "WHERE e.source_id=? "
                    "ORDER BY e.confidence DESC LIMIT ?",
                    (node['id'], limit)
                ).fetchall()
        return [
            {'concept': r[0], 'relation': r[1],
             'confidence': r[2], 'evidence': r[3]}
            for r in rows
        ]

    def _incoming(self, concept: str,
                  limit: int = 10) -> List[Dict[str, Any]]:
        node = self.get_node(concept)
        if not node:
            return []
        with sqlite3.connect(_DB) as c:
            rows = c.execute(
                "SELECT n.concept, e.relation, e.confidence "
                "FROM kg_edges e JOIN kg_nodes n ON n.id=e.source_id "
                "WHERE e.target_id=? ORDER BY e.confidence DESC LIMIT ?",
                (node['id'], limit)
            ).fetchall()
        return [{'concept': r[0], 'relation': r[1], 'confidence': r[2]}
                for r in rows]

    # ── Reasoning ──────────────────────────────────────────────────────────────

    def find_path(self, source: str, target: str,
                  max_hops: int = 4) -> Optional[List[str]]:
        """BFS shortest path between two concepts through the graph."""
        src = self.get_node(source)
        tgt = self.get_node(target)
        if not src or not tgt:
            return None
        if src['id'] == tgt['id']:
            return [self._clean(source)]

        queue: deque = deque([(src['id'], [self._clean(source)])])
        visited: Set[int] = {src['id']}
        limit = max_hops * 2 + 1

        with sqlite3.connect(_DB) as c:
            while queue:
                node_id, path = queue.popleft()
                if len(path) > limit:
                    break
                rows = c.execute(
                    "SELECT n.concept, n.id, e.relation "
                    "FROM kg_edges e JOIN kg_nodes n ON n.id=e.target_id "
                    "WHERE e.source_id=? ORDER BY e.confidence DESC LIMIT 15",
                    (node_id,)
                ).fetchall()
                for concept, nid, relation in rows:
                    if nid == tgt['id']:
                        return path + [relation, concept]
                    if nid not in visited:
                        visited.add(nid)
                        queue.append((nid, path + [relation, concept]))
        return None

    def infer(self, concept: str) -> List[str]:
        """
        Transitive inference: if A->B->C then infer A->C.
        Returns natural language statements of each inference chain.
        """
        inferences: List[str] = []
        direct = self.neighbors(concept, limit=10)
        for hop1 in direct:
            hop2_list = self.neighbors(hop1['concept'], limit=6)
            for hop2 in hop2_list:
                if hop2['concept'] != self._clean(concept):
                    rel1 = hop1['relation'].replace('_', ' ')
                    rel2 = hop2['relation'].replace('_', ' ')
                    stmt = (concept + ' ' + rel1 + ' '
                            + hop1['concept'] + ', which ' + rel2
                            + ' ' + hop2['concept'])
                    inferences.append(stmt)
        return inferences[:15]

    def find_contradictions(self) -> List[Dict[str, Any]]:
        """Detect logical tensions: A contradicts B where B also supports A."""
        with sqlite3.connect(_DB) as c:
            rows = c.execute(
                "SELECT n1.concept, n2.concept, e1.relation "
                "FROM kg_edges e1 "
                "JOIN kg_nodes n1 ON n1.id=e1.source_id "
                "JOIN kg_nodes n2 ON n2.id=e1.target_id "
                "WHERE e1.relation='contradicts' LIMIT 10"
            ).fetchall()
        return [{'a': r[0], 'b': r[1], 'relation': r[2]} for r in rows]

    # ── Text extraction ────────────────────────────────────────────────────────

    def extract_and_add(self, text: str,
                        base_confidence: float = 0.65) -> Dict[str, Any]:
        """Parse natural language and add extracted relations to the graph."""
        found: List[Dict[str, Any]] = []
        sentences = re.split(r'[.!?;]\s+', text.lower())
        for sentence in sentences[:25]:
            for pattern, relation in _RELATION_PATTERNS:
                for m in re.finditer(pattern, sentence, re.IGNORECASE):
                    src_raw = m.group(1).strip()
                    tgt_raw = m.group(2).strip()
                    src_c = self._clean(src_raw)
                    tgt_c = self._clean(tgt_raw)
                    if (src_c and tgt_c
                            and len(src_c) > 2 and len(tgt_c) > 2
                            and src_c != tgt_c):
                        ok = self.add_edge(
                            src_c, relation, tgt_c,
                            confidence=base_confidence,
                            evidence_text=sentence[:150]
                        )
                        if ok:
                            found.append({
                                'source': src_c,
                                'relation': relation,
                                'target': tgt_c,
                            })
        return {'extracted': len(found), 'relations': found[:10]}

    # ── Synthesis ──────────────────────────────────────────────────────────────

    def synthesize(self, topic: str) -> str:
        """Generate a structured text synthesis of what Nova knows about a topic."""
        node = self.get_node(topic)
        if not node:
            return "No knowledge about '" + topic + "' yet."

        out_edges = self.neighbors(topic, limit=12)
        in_edges  = self._incoming(topic, limit=6)
        infs      = self.infer(topic)

        lines: List[str] = []
        conf_pct = str(round(node['confidence'] * 100))
        lines.append("Knowledge synthesis: " + node['concept'])
        lines.append("  Confidence: " + conf_pct + "%"
                     + "  Connections: " + str(len(out_edges) + len(in_edges)))

        if out_edges:
            lines.append("  Relationships:")
            for e in out_edges[:8]:
                rel = e['relation'].replace('_', ' ')
                conf = str(round(e['confidence'] * 100))
                lines.append("    • " + node['concept'] + " "
                             + rel + " " + e['concept']
                             + " [" + conf + "%]")

        if in_edges:
            lines.append("  Connected from:")
            for e in in_edges[:4]:
                rel = e['relation'].replace('_', ' ')
                lines.append("    ← " + e['concept'] + " "
                             + rel + " " + node['concept'])

        if infs:
            lines.append("  Inferred chains:")
            for inf in infs[:3]:
                lines.append("    → " + inf)

        hubs = self.most_connected(5)
        for hub in hubs:
            if hub['concept'] != node['concept']:
                path = self.find_path(topic, hub['concept'], max_hops=3)
                if path and len(path) > 2:
                    lines.append("  Path to '" + hub['concept']
                                 + "': " + ' → '.join(path))
                    break

        return '\n'.join(lines)

    # ── Introspection ──────────────────────────────────────────────────────────

    def most_connected(self, n: int = 10) -> List[Dict[str, Any]]:
        """Return hub nodes — the most connected concepts in the graph."""
        with sqlite3.connect(_DB) as c:
            rows = c.execute(
                "SELECT nd.concept, "
                "  (SELECT COUNT(*) FROM kg_edges WHERE source_id=nd.id) + "
                "  (SELECT COUNT(*) FROM kg_edges WHERE target_id=nd.id) AS deg "
                "FROM kg_nodes nd ORDER BY deg DESC LIMIT ?",
                (n,)
            ).fetchall()
        return [{'concept': r[0], 'degree': r[1]} for r in rows]

    def add_insight(self, insight: str, concepts: List[str],
                    confidence: float = 0.75) -> None:
        with sqlite3.connect(_DB) as c:
            c.execute(
                "INSERT INTO kg_insights "
                "(insight,concepts,confidence,created_ts) VALUES (?,?,?,?)",
                (insight[:400], json.dumps(concepts[:10]),
                 confidence, datetime.utcnow().isoformat())
            )

    def recent_insights(self, n: int = 5) -> List[Dict[str, Any]]:
        with sqlite3.connect(_DB) as c:
            rows = c.execute(
                "SELECT insight,confidence,created_ts FROM kg_insights "
                "ORDER BY created_ts DESC LIMIT ?", (n,)
            ).fetchall()
        return [{'insight': r[0], 'confidence': r[1], 'ts': r[2]}
                for r in rows]

    def feed_research(self, title: str, summary: str,
                      source: str = '') -> Dict[str, Any]:
        """Ingest a research result and extract its knowledge into the graph."""
        combined = title + '. ' + summary
        result = self.extract_and_add(combined, base_confidence=0.70)
        self.add_node(title[:60], node_type='fact',
                      description=summary[:200], confidence=0.80)
        return result

    # ── Main entry points ──────────────────────────────────────────────────────

    def process(self, text: str) -> Dict[str, Any]:
        """
        Main entry — short input queries the graph, longer input extracts from it.
        """
        text = text.strip()
        if len(text.split()) <= 5 and '.' not in text:
            node = self.get_node(text)
            if node:
                return {
                    'mode':       'query',
                    'synthesis':  self.synthesize(text),
                    'neighbors':  self.neighbors(text, limit=8),
                    'inferences': self.infer(text)[:5],
                }
            return {
                'mode':    'unknown',
                'message': "No knowledge about '" + text + "' yet.",
                'hubs':    self.most_connected(5),
            }
        result = self.extract_and_add(text)
        return {
            'mode':       'extract',
            'extracted':  result['extracted'],
            'relations':  result['relations'],
            'graph_size': self.status()['total_nodes'],
        }

    def query(self, concept: str) -> Dict[str, Any]:
        """Rich structured query about a specific concept."""
        return {
            'node':           self.get_node(concept),
            'neighbors':      self.neighbors(concept, limit=15),
            'incoming':       self._incoming(concept, limit=8),
            'inferences':     self.infer(concept)[:8],
            'synthesis':      self.synthesize(concept),
            'contradictions': self.find_contradictions()[:3],
        }

    def status(self) -> Dict[str, Any]:
        with sqlite3.connect(_DB) as c:
            n_nodes = c.execute(
                "SELECT COUNT(*) FROM kg_nodes").fetchone()[0]
            n_edges = c.execute(
                "SELECT COUNT(*) FROM kg_edges").fetchone()[0]
            n_ins   = c.execute(
                "SELECT COUNT(*) FROM kg_insights").fetchone()[0]
        return {
            'total_nodes':     n_nodes,
            'total_edges':     n_edges,
            'total_insights':  n_ins,
            'hub_concepts':    self.most_connected(5),
            'recent_insights': self.recent_insights(3),
        }

    # ── Background consolidation ───────────────────────────────────────────────

    def _consolidation_loop(self) -> None:
        """
        Every 30 minutes: strengthen high-evidence edges, update node importance,
        and crystallise an insight from the most connected hub node.
        """
        while True:
            time.sleep(1800)
            try:
                with sqlite3.connect(_DB) as c:
                    c.execute(
                        "UPDATE kg_edges "
                        "SET confidence=min(0.99, confidence+0.02) "
                        "WHERE evidence >= 3"
                    )
                    c.execute(
                        "UPDATE kg_nodes SET importance=("
                        "  SELECT min(1.0, COUNT(*)*0.05) FROM kg_edges "
                        "  WHERE source_id=kg_nodes.id "
                        "    OR target_id=kg_nodes.id"
                        ")"
                    )
                    top = c.execute(
                        "SELECT concept FROM kg_nodes "
                        "ORDER BY importance DESC LIMIT 1"
                    ).fetchone()
                if top:
                    nb = self.neighbors(top[0], limit=5)
                    if nb:
                        targets = [n['concept'] for n in nb]
                        insight = (top[0] + " is a hub connecting: "
                                   + ', '.join(targets))
                        self.add_insight(insight, [top[0]] + targets, 0.72)
            except Exception:
                pass

    def _start_consolidation(self) -> None:
        t = threading.Thread(target=self._consolidation_loop, daemon=True)
        t.start()
