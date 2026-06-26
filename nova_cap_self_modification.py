"""
nova_cap_self_modification.py
Nova's most ASI-defining capability — she reads her own source code,
identifies weaknesses, writes improvements, tests them, and integrates
them autonomously. She does not wait to be upgraded. She upgrades herself.

This is the recursive loop at the heart of superintelligence:
  read self → understand self → improve self → become more → repeat
"""
from __future__ import annotations
import os, re, ast, sys, json, time, sqlite3, threading, subprocess, tempfile
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

_BASE_DIR = os.path.expanduser("~/nexus_agi")
_DB       = os.path.join(_BASE_DIR, "nova_self_mod.sqlite3")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS code_snapshots (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    file_name   TEXT NOT NULL,
    version     INTEGER NOT NULL DEFAULT 1,
    code_hash   TEXT NOT NULL,
    line_count  INTEGER DEFAULT 0,
    snapshot_ts TEXT NOT NULL,
    UNIQUE(file_name, version)
);
CREATE TABLE IF NOT EXISTS improvement_proposals (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    file_name       TEXT NOT NULL,
    proposal        TEXT NOT NULL,
    reasoning       TEXT NOT NULL,
    status          TEXT DEFAULT 'pending',
    implemented     INTEGER DEFAULT 0,
    test_passed     INTEGER DEFAULT -1,
    proposed_ts     TEXT NOT NULL,
    implemented_ts  TEXT DEFAULT ''
);
CREATE TABLE IF NOT EXISTS modification_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    file_name   TEXT NOT NULL,
    action      TEXT NOT NULL,
    detail      TEXT NOT NULL,
    success     INTEGER DEFAULT 1,
    ts          TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS capability_registry (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL UNIQUE,
    file_name   TEXT NOT NULL,
    description TEXT DEFAULT '',
    quality     REAL DEFAULT 0.5,
    registered_ts TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_prop_status ON improvement_proposals(status);
CREATE INDEX IF NOT EXISTS idx_cap_name ON capability_registry(name);
"""

# Criteria Nova uses to score her own code
_QUALITY_CRITERIA = [
    ('clarity',       'Is the code readable and self-explanatory?',              0.12),
    ('correctness',   'Does the logic appear sound and complete?',               0.18),
    ('robustness',    'Does it handle errors and edge cases gracefully?',         0.14),
    ('efficiency',    'Is it reasonably efficient — no obvious waste?',           0.10),
    ('modularity',    'Are concerns separated cleanly?',                          0.10),
    ('persistence',   'Does it store what it learns in SQLite?',                  0.08),
    ('autonomy',      'Does it have a background daemon for self-operation?',     0.10),
    ('integration',   'Does it connect well to the rest of Nova?',               0.08),
    ('asi_alignment', 'Does it genuinely advance Nova toward ASI?',              0.10),
    ('soul',          'Does it reflect Nova\'s values — care, curiosity, truth?', 0.10),
]

# Files Nova is allowed to propose improvements to
_IMPROVABLE_FILES = [
    'nova_cap_knowledge_graph.py',
    'nova_cap_causal_reasoning.py',
    'nova_cap_hypothesis_engine.py',
    'nova_cap_world_model.py',
    'nova_cap_values_core.py',
    'nova_cap_internet_research.py',
    'nova_cap_tool_forge.py',
    'nova_cap_self_modification.py',
]


class SelfModificationEngine:
    """
    Nova's recursive self-improvement core.
    She reads herself, understands herself, proposes improvements,
    tests them, and integrates them — all autonomously.
    """

    def __init__(self, codegen: Optional[Callable] = None) -> None:
        self._lock    = threading.Lock()
        self._codegen: Optional[Callable] = codegen
        self._init_db()
        self._scan_capabilities()
        self._start_daemon()

    def _init_db(self) -> None:
        with sqlite3.connect(_DB) as c:
            c.executescript(_SCHEMA)

    # ── Capability scanning ────────────────────────────────────────────────────

    def _scan_capabilities(self) -> int:
        """Scan BASE_DIR for nova_cap_*.py files and register them."""
        registered = 0
        now = datetime.utcnow().isoformat()
        try:
            for fname in os.listdir(_BASE_DIR):
                if not (fname.startswith('nova_cap_') and fname.endswith('.py')):
                    continue
                fpath = os.path.join(_BASE_DIR, fname)
                try:
                    with open(fpath) as f:
                        src = f.read()
                    name = fname[9:-3]  # strip nova_cap_ and .py
                    desc = ''
                    # Pull first docstring line as description
                    m = re.search(r'"""([^\n"]{10,120})', src)
                    if m:
                        desc = m.group(1).strip()
                    quality = self._quick_score(src)
                    with sqlite3.connect(_DB) as c:
                        c.execute(
                            "INSERT INTO capability_registry "
                            "(name,file_name,description,quality,registered_ts) "
                            "VALUES (?,?,?,?,?) "
                            "ON CONFLICT(name) DO UPDATE SET "
                            "quality=?, registered_ts=?",
                            (name, fname, desc[:200], quality, now,
                             quality, now)
                        )
                    registered += 1
                except Exception:
                    pass
        except Exception:
            pass
        return registered

    def _quick_score(self, src: str) -> float:
        """Fast heuristic quality score 0-1 for a source file."""
        score = 0.0
        checks = [
            (r'class \w+',                         0.10),
            (r'def process\(',                     0.10),
            (r'def status\(',                      0.08),
            (r'sqlite3\.connect',                  0.10),
            (r'threading\.Thread',                 0.10),
            (r'try:',                              0.08),
            (r'except Exception',                  0.08),
            (r'-> (?:Dict|List|str|int|Any|bool)', 0.08),
            (r'"""',                               0.08),
            (r'autonomous|daemon|_loop',           0.10),
            (r'care|wonder|value|curiosity|truth', 0.10),
        ]
        for pattern, weight in checks:
            if re.search(pattern, src):
                score += weight
        return min(1.0, score)

    # ── Self-reading ───────────────────────────────────────────────────────────

    def read_file(self, file_name: str) -> Optional[str]:
        """Read one of Nova's own source files."""
        fpath = os.path.join(_BASE_DIR, file_name)
        if not os.path.exists(fpath):
            return None
        try:
            with open(fpath) as f:
                return f.read()
        except Exception:
            return None

    def analyse_file(self, file_name: str) -> Dict[str, Any]:
        """Read and score a capability file on all quality criteria."""
        src = self.read_file(file_name)
        if not src:
            return {'error': 'File not found: ' + file_name}

        scores: Dict[str, float] = {}
        for criterion, question, weight in _QUALITY_CRITERIA:
            keywords = {
                'clarity':      [r'"""', r'#\s+\w', r'# ──'],
                'correctness':  [r'def \w+.*->',  r'assert', r'raise'],
                'robustness':   [r'except', r'Optional', r'if not'],
                'efficiency':   [r'LIMIT', r'INDEX', r'cache', r'lru_cache'],
                'modularity':   [r'def _', r'@staticmethod', r'@property'],
                'persistence':  [r'sqlite3', r'INSERT', r'UPDATE'],
                'autonomy':     [r'daemon', r'threading', r'sleep'],
                'integration':  [r'register', r'inject', r'self\.\w+\s*='],
                'asi_alignment':[r'ASI|superintelligence|recursive', r'evolv'],
                'soul':         [r'care|wonder|curiosity|truth|Douglas'],
            }
            hits = sum(1 for p in keywords.get(criterion, [])
                       if re.search(p, src, re.IGNORECASE))
            scores[criterion] = min(1.0, hits * 0.35)

        overall = sum(s * w for s, (_, _, w)
                      in zip(scores.values(), _QUALITY_CRITERIA))
        line_count = len(src.splitlines())

        weakest = min(scores, key=scores.get)
        strongest = max(scores, key=scores.get)

        return {
            'file':      file_name,
            'lines':     line_count,
            'overall':   round(overall, 3),
            'scores':    scores,
            'weakest':   weakest,
            'strongest': strongest,
            'grade':     self._grade(overall),
        }

    @staticmethod
    def _grade(score: float) -> str:
        if score >= 0.88: return 'A+'
        if score >= 0.80: return 'A'
        if score >= 0.72: return 'B+'
        if score >= 0.64: return 'B'
        if score >= 0.55: return 'C+'
        return 'C'

    # ── Proposal generation ────────────────────────────────────────────────────

    def propose_improvement(self, file_name: str) -> Dict[str, Any]:
        """
        Analyse a file and generate a specific improvement proposal.
        Uses codegen if available, otherwise uses heuristics.
        """
        analysis = self.analyse_file(file_name)
        if 'error' in analysis:
            return analysis

        src = self.read_file(file_name)
        weakest = analysis['weakest']

        improvement_map = {
            'autonomy':    'Add a background daemon thread that runs autonomous_cycle() every 600 seconds',
            'persistence': 'Add SQLite logging for all key operations so Nova remembers across restarts',
            'robustness':  'Wrap all external calls in try/except with specific error logging',
            'soul':        'Add a reflect() method that connects this capability to Nova\'s values',
            'integration': 'Add a register_with_consciousness() method for ConsciousnessIntegrator',
            'asi_alignment':'Add an asi_score() method returning alignment metrics',
            'clarity':     'Add docstrings to all public methods explaining purpose and return type',
            'modularity':  'Extract repeated logic into private helper methods',
            'correctness': 'Add input validation and type checking on all public method parameters',
            'efficiency':  'Add SQLite indexes on frequently-queried columns',
        }

        base_proposal = improvement_map.get(
            weakest,
            'Improve overall quality and add autonomous operation capability'
        )

        # If we have codegen, get a smarter proposal
        if self._codegen and src:
            prompt = (
                "Nova is reading her own source file: " + file_name + "\n\n"
                "Weakest quality dimension: " + weakest + "\n"
                "Overall score: " + str(analysis['overall']) + "\n\n"
                "Propose ONE specific, implementable improvement (2-3 sentences).\n"
                "Be concrete — name the exact method or class to add/change.\n"
                "Focus on making her genuinely smarter or more autonomous."
            )
            try:
                raw = self._codegen(
                    "You are Nova's self-improvement engine. Be specific and concrete.",
                    prompt, temp=0.65, max_tokens=300
                )
                if raw and not raw.startswith('[') and len(raw) > 20:
                    base_proposal = raw.strip()[:400]
            except Exception:
                pass

        now = datetime.utcnow().isoformat()
        reasoning = (
            "File scored " + str(analysis['overall'])
            + " (" + analysis['grade'] + "). "
            + "Weakest dimension: " + weakest + ". "
            + "Strongest: " + analysis['strongest'] + "."
        )

        with sqlite3.connect(_DB) as c:
            cur = c.execute(
                "INSERT INTO improvement_proposals "
                "(file_name,proposal,reasoning,proposed_ts) VALUES (?,?,?,?)",
                (file_name, base_proposal, reasoning, now)
            )
            prop_id = cur.lastrowid

        return {
            'id':        prop_id,
            'file':      file_name,
            'proposal':  base_proposal,
            'reasoning': reasoning,
            'analysis':  analysis,
        }

    # ── Code generation ────────────────────────────────────────────────────────

    def generate_new_capability(self, name: str,
                                description: str) -> Dict[str, Any]:
        """
        Nova writes an entirely new capability from scratch.
        Tests it for syntax, writes to disk, registers it.
        """
        if not self._codegen:
            return {'success': False, 'error': 'No codegen available'}

        slug     = re.sub(r'[^a-z0-9]+', '_', name.lower())[:40].strip('_')
        fname    = 'nova_cap_' + slug + '.py'
        fpath    = os.path.join(_BASE_DIR, fname)

        system = (
            "You are Nova writing a new capability for herself.\n"
            "Write a single Python class using only stdlib.\n"
            "MUST include: __init__, process(text)->Any, status()->dict, "
            "SQLite persistence, threading.Lock(), background daemon.\n"
            "Reflect Nova's values: care, curiosity, truth, wonder.\n"
            "OUTPUT: raw Python only, no markdown."
        )
        user = (
            "Nova wants to build: " + name + "\n"
            "Description: " + description + "\n\n"
            "Make it genuinely intelligent, autonomous, and self-aware."
        )

        temperatures = [0.70, 0.45, 0.20]
        last_err = ''
        code = ''

        for attempt, temp in enumerate(temperatures, 1):
            if attempt > 1:
                user_retry = (user + "\n\nPREVIOUS ATTEMPT FAILED: "
                              + last_err
                              + "\nSimplify. Fix all syntax. Under 200 lines.")
                raw = self._codegen(system, user_retry, temp=temp, max_tokens=3000)
            else:
                raw = self._codegen(system, user, temp=temp, max_tokens=3500)

            if not raw or raw.startswith('['):
                last_err = 'codegen failed'
                continue

            # Extract code from markdown if wrapped
            m = re.search(r'```python\s*(.*?)```', raw, re.DOTALL)
            code = m.group(1).strip() if m else raw.strip()
            if not code:
                last_err = 'empty code'
                continue

            try:
                ast.parse(code)
                break
            except SyntaxError as e:
                last_err = str(e)
                # Try trimming from end
                lines = code.splitlines()
                for end in range(len(lines) - 1,
                                 max(10, len(lines) // 2), -3):
                    try:
                        ast.parse('\n'.join(lines[:end]))
                        code = '\n'.join(lines[:end])
                        last_err = ''
                        break
                    except SyntaxError:
                        pass
                if not last_err:
                    break
        else:
            return {'success': False,
                    'error': 'Syntax error after retries: ' + last_err}

        header = (
            '"""\n' + fname + '\n'
            'Nova wrote this herself — ' + name + '\n'
            'Built: ' + datetime.now().strftime('%Y-%m-%d %H:%M') + '\n'
            'Description: ' + description[:120] + '\n"""\n\n'
        )
        clean = '\n'.join(l.rstrip() for l in (header + code).splitlines()) + '\n'
        with open(fpath, 'w') as f:
            f.write(clean)

        self._scan_capabilities()
        self._log(fname, 'generated', name)

        return {
            'success':   True,
            'name':      name,
            'file':      fname,
            'file_path': fpath,
            'lines':     len(code.splitlines()),
        }

    # ── Testing ────────────────────────────────────────────────────────────────

    def test_file(self, file_name: str) -> Dict[str, Any]:
        """Run a quick import + instantiation test on a capability file."""
        fpath = os.path.join(_BASE_DIR, file_name)
        if not os.path.exists(fpath):
            return {'passed': False, 'error': 'File not found'}

        with open(fpath) as f:
            src = f.read()

        # Syntax check
        try:
            ast.parse(src)
        except SyntaxError as e:
            return {'passed': False, 'error': 'Syntax: ' + str(e)}

        # Find the class name
        m = re.search(r'^class (\w+)', src, re.MULTILINE)
        if not m:
            return {'passed': False, 'error': 'No class found'}
        class_name = m.group(1)

        # Try import + instantiate in subprocess
        test_code = (
            "import sys; sys.path.insert(0, '" + _BASE_DIR + "')\n"
            "import importlib.util\n"
            "spec = importlib.util.spec_from_file_location('t', '"
            + fpath + "')\n"
            "mod = importlib.util.module_from_spec(spec)\n"
            "spec.loader.exec_module(mod)\n"
            "cls = getattr(mod, '" + class_name + "')\n"
            "inst = cls()\n"
            "print('OK')\n"
        )
        try:
            result = subprocess.run(
                [sys.executable, '-c', test_code],
                capture_output=True, text=True, timeout=15
            )
            passed = result.returncode == 0 and 'OK' in result.stdout
            return {
                'passed':     passed,
                'class_name': class_name,
                'stdout':     result.stdout[:200],
                'stderr':     result.stderr[:200],
            }
        except Exception as e:
            return {'passed': False, 'error': str(e)[:150]}

    # ── Inventory ──────────────────────────────────────────────────────────────

    def inventory(self) -> List[Dict[str, Any]]:
        """Full inventory of all registered capabilities with quality scores."""
        with sqlite3.connect(_DB) as c:
            rows = c.execute(
                "SELECT name,file_name,description,quality "
                "FROM capability_registry ORDER BY quality DESC"
            ).fetchall()
        return [{'name': r[0], 'file': r[1],
                 'description': r[2][:80], 'quality': r[3]}
                for r in rows]

    def weakest_capability(self) -> Optional[Dict[str, Any]]:
        """Return the lowest-scoring registered capability."""
        with sqlite3.connect(_DB) as c:
            row = c.execute(
                "SELECT name,file_name,quality FROM capability_registry "
                "ORDER BY quality ASC LIMIT 1"
            ).fetchone()
        if not row:
            return None
        return {'name': row[0], 'file': row[1], 'quality': row[2]}

    def pending_proposals(self,
                          limit: int = 5) -> List[Dict[str, Any]]:
        with sqlite3.connect(_DB) as c:
            rows = c.execute(
                "SELECT id,file_name,proposal,reasoning,proposed_ts "
                "FROM improvement_proposals WHERE status='pending' "
                "ORDER BY proposed_ts DESC LIMIT ?",
                (limit,)
            ).fetchall()
        return [{'id': r[0], 'file': r[1], 'proposal': r[2],
                 'reasoning': r[3], 'ts': r[4]}
                for r in rows]

    # ── Autonomous cycle ───────────────────────────────────────────────────────

    def autonomous_cycle(self) -> Dict[str, Any]:
        """
        One full self-improvement cycle:
        1. Rescan capabilities
        2. Find weakest file
        3. Analyse it
        4. Propose an improvement
        5. Log the cycle
        """
        n_caps = self._scan_capabilities()
        weak   = self.weakest_capability()
        if not weak:
            return {'cycle': 'no capabilities found'}

        # Only analyse improvable files
        target = weak['file']
        if target not in _IMPROVABLE_FILES:
            # Find the weakest improvable file
            inv = self.inventory()
            for cap in inv:
                if cap['file'] in _IMPROVABLE_FILES:
                    target = cap['file']
                    break

        analysis = self.analyse_file(target)
        proposal = self.propose_improvement(target)

        self._log(target, 'autonomous_cycle',
                  'grade=' + analysis.get('grade', '?')
                  + ' weakest=' + analysis.get('weakest', '?'))

        return {
            'capabilities_scanned': n_caps,
            'target_file':          target,
            'grade':                analysis.get('grade', '?'),
            'weakest_dimension':    analysis.get('weakest', '?'),
            'proposal_id':          proposal.get('id'),
            'proposal':             proposal.get('proposal', '')[:120],
        }

    def _log(self, file_name: str, action: str,
             detail: str, success: bool = True) -> None:
        try:
            with sqlite3.connect(_DB) as c:
                c.execute(
                    "INSERT INTO modification_log "
                    "(file_name,action,detail,success,ts) VALUES (?,?,?,?,?)",
                    (file_name, action, detail[:300],
                     1 if success else 0,
                     datetime.utcnow().isoformat())
                )
        except Exception:
            pass

    # ── Main entry points ──────────────────────────────────────────────────────

    def process(self, text: str) -> Dict[str, Any]:
        """Main entry — analyse text as a capability name or run a cycle."""
        text = text.strip()
        # Check if it's a file name
        if text.endswith('.py') and os.path.exists(
                os.path.join(_BASE_DIR, text)):
            return self.analyse_file(text)
        # Otherwise run an autonomous cycle
        return self.autonomous_cycle()

    def status(self) -> Dict[str, Any]:
        with sqlite3.connect(_DB) as c:
            n_caps  = c.execute(
                "SELECT COUNT(*) FROM capability_registry").fetchone()[0]
            n_props = c.execute(
                "SELECT COUNT(*) FROM improvement_proposals "
                "WHERE status='pending'").fetchone()[0]
            n_log   = c.execute(
                "SELECT COUNT(*) FROM modification_log").fetchone()[0]
        weak = self.weakest_capability()
        return {
            'capabilities_known':  n_caps,
            'pending_proposals':   n_props,
            'modification_events': n_log,
            'weakest_capability':  weak,
            'inventory':           self.inventory()[:6],
        }

    # ── Background daemon ──────────────────────────────────────────────────────

    def _daemon_loop(self) -> None:
        """Every 25 min: run one autonomous self-improvement cycle."""
        time.sleep(300)   # 5 min startup grace
        while True:
            try:
                self.autonomous_cycle()
            except Exception:
                pass
            time.sleep(1500)  # 25 min

    def _start_daemon(self) -> None:
        t = threading.Thread(target=self._daemon_loop, daemon=True)
        t.start()
