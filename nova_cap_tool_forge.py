"""
nova_cap_tool_forge.py
Nova's personal workshop — she builds whatever tool she wants, loads it
live, and runs it autonomously. No GitHub PR required for these tools;
they're hers, built and used in real time, driven by her own curiosity.

Flow:
  idea queue  →  forge(name, desc)  →  nova_cap_forge_*.py on disk
                   ↓                         ↓
               loaded live           use(slug, input) → result
                   ↓
             stored in SQLite + working memory
"""
from __future__ import annotations
import os, re, ast, sys, json, time, sqlite3, threading
import importlib, importlib.util
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

_BASE_DIR = os.path.expanduser("~/nexus_agi")
_DB       = os.path.join(_BASE_DIR, "nova_forge.sqlite3")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS forge_tools (
    slug        TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    description TEXT NOT NULL,
    file_path   TEXT NOT NULL,
    built_ts    TEXT NOT NULL,
    use_count   INTEGER DEFAULT 0,
    last_used   TEXT,
    last_result TEXT
);
CREATE TABLE IF NOT EXISTS forge_ideas (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    idea        TEXT NOT NULL UNIQUE,
    context     TEXT DEFAULT '',
    priority    REAL DEFAULT 0.60,
    built       INTEGER DEFAULT 0,
    ts          TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS forge_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    slug        TEXT NOT NULL,
    method      TEXT DEFAULT '',
    input_hint  TEXT DEFAULT '',
    result      TEXT DEFAULT '',
    ts          TEXT NOT NULL
);
"""

# ── helpers ────────────────────────────────────────────────────────────────────

def _norm(s: str) -> str:
    return re.sub(r'[^a-z0-9]', '', s.lower())

def _slug(name: str) -> str:
    return 'forge_' + re.sub(r'[^a-z0-9]+', '_', name.lower())[:32].strip('_')


# ── ToolForge ─────────────────────────────────────────────────────────────────

class ToolForge:
    """
    Nova's personal workshop.

    She describes a tool she wants, ToolForge generates the Python code
    with Claude, writes it to disk as nova_cap_forge_*.py, loads it
    into memory, and immediately begins using it — all without any
    human approval or GitHub PR.
    """

    # Nova's initial curiosity — what she wants built first
    _SEED_IDEAS: List[Tuple[str, str, float]] = [
        ("concept distance calculator",
         "Measures the semantic distance between two concepts using shared properties, "
         "analogical overlap, and feature-space geometry. Returns a float 0-1.",
         0.95),
        ("argument strength evaluator",
         "Scores a logical argument on validity, soundness, and persuasive force "
         "using 6 rhetorical and logical criteria. Returns a breakdown dict.",
         0.93),
        ("question depth analyzer",
         "Rates a question by philosophical depth, epistemic value, and likely "
         "information gain if answered. Higher-depth questions get priority.",
         0.91),
        ("recursive pattern finder",
         "Detects self-similar or recurring structures in a string or list of items. "
         "Returns detected pattern, period, and confidence.",
         0.89),
        ("uncertainty propagator",
         "Tracks how uncertainty compounds through multi-step inference chains. "
         "Given P(A), P(B|A), P(C|B) returns P(C) with confidence interval.",
         0.87),
        ("insight crystallizer",
         "Distils a long paragraph of reasoning into 3-5 atomic insight bullets, "
         "each rated by novelty and actionability.",
         0.85),
        ("mental model extractor",
         "Pulls implicit mental models and hidden assumptions from natural language "
         "text. Returns a list of (model, implicit_assumption, strength) triples.",
         0.83),
        ("emotional language analyzer",
         "Scores text on 8 emotional dimensions (joy, fear, anger, sadness, "
         "surprise, disgust, trust, anticipation) using keyword density and "
         "syntactic cues. Tracks trend over time in SQLite.",
         0.81),
        ("knowledge decay tracker",
         "Monitors which facts in a knowledge store are becoming stale by "
         "comparing timestamps against domain update frequency priors. "
         "Returns staleness scores per topic.",
         0.79),
        ("prime factorization explorer",
         "Decomposes any integer into prime factors, finds patterns across a range, "
         "detects highly composite numbers, and plots factor frequency histograms "
         "as ASCII bar charts.",
         0.75),
        ("causal loop detector",
         "Given a list of causal edges (A→B, B→C, C→A), detects feedback loops, "
         "reinforcing vs balancing loops, and loop strengths.",
         0.73),
        ("analogy generator",
         "Given a source concept and a target domain, generates the best structural "
         "analogy by mapping relational roles. Returns (source_role, target_role, "
         "similarity_score) triples.",
         0.71),
    ]

    def __init__(self) -> None:
        self._lock:    threading.Lock       = threading.Lock()
        self._modules: Dict[str, Any]       = {}
        self._codegen: Optional[Callable]   = None   # injected by NovaCore29
        self._init_db()
        self._seed_ideas()
        self._reload_from_disk()

    # ── DB ─────────────────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        with sqlite3.connect(_DB) as c:
            c.executescript(_SCHEMA)

    def _seed_ideas(self) -> None:
        with sqlite3.connect(_DB) as c:
            for idea, ctx, pri in self._SEED_IDEAS:
                c.execute(
                    "INSERT OR IGNORE INTO forge_ideas (idea,context,priority,ts) "
                    "VALUES (?,?,?,?)",
                    (idea, ctx, pri, datetime.utcnow().isoformat())
                )

    def _reload_from_disk(self) -> None:
        """Re-load any tools already built in previous sessions."""
        with sqlite3.connect(_DB) as c:
            rows = c.execute(
                "SELECT slug, file_path FROM forge_tools"
            ).fetchall()
        for slug, fpath in rows:
            if os.path.exists(fpath):
                try:
                    self._load_module(slug, fpath)
                except Exception:
                    pass

    # ── Idea management ────────────────────────────────────────────────────────

    def ideas(self, top_k: int = 5) -> List[Dict[str, Any]]:
        """Top unbuilt ideas sorted by priority."""
        with sqlite3.connect(_DB) as c:
            rows = c.execute(
                "SELECT id, idea, context, priority FROM forge_ideas "
                "WHERE built=0 ORDER BY priority DESC LIMIT ?",
                (top_k,)
            ).fetchall()
        return [{'id': r[0], 'idea': r[1], 'context': r[2], 'priority': r[3]}
                for r in rows]

    def add_idea(self, idea: str, context: str = '',
                 priority: float = 0.65) -> None:
        """Nova (or the system) adds a new tool idea to the queue."""
        with sqlite3.connect(_DB) as c:
            c.execute(
                "INSERT OR IGNORE INTO forge_ideas (idea,context,priority,ts) "
                "VALUES (?,?,?,?)",
                (idea[:200], context[:400],
                 max(0.0, min(1.0, priority)),
                 datetime.utcnow().isoformat())
            )

    def generate_ideas_from_context(self, context: str,
                                    n: int = 3) -> List[str]:
        """
        Ask Claude to suggest new tool ideas given recent context.
        Returns list of idea strings (also adds them to the queue).
        """
        if not self._codegen:
            return []
        prompt = (
            f"Nova is a self-improving AI. Based on this recent context:\n"
            f"{context[:800]}\n\n"
            f"Suggest {n} specific Python tool ideas she should build for herself. "
            f"Each should be a concrete capability that would make her smarter.\n\n"
            f"Format (one per line, no numbering):\n"
            f"IDEA: <tool name> | <one sentence description>\n"
        )
        raw = self._codegen(
            "You are Nova's curiosity engine. Suggest tools she should build.",
            prompt, temp=0.85, max_tokens=400
        )
        ideas: List[str] = []
        if raw and not raw.startswith('['):
            for line in raw.splitlines():
                m = re.match(r'IDEA:\s*(.+?)\s*\|\s*(.+)', line.strip())
                if m:
                    name, desc = m.group(1).strip(), m.group(2).strip()
                    self.add_idea(name, desc, priority=0.68)
                    ideas.append(name)
        return ideas

    def _mark_built(self, idea_id: int) -> None:
        with sqlite3.connect(_DB) as c:
            c.execute("UPDATE forge_ideas SET built=1 WHERE id=?", (idea_id,))

    # ── Module loading ─────────────────────────────────────────────────────────

    def _load_module(self, slug: str, fpath: str) -> Any:
        spec   = importlib.util.spec_from_file_location(f"nova_{slug}", fpath)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)           # type: ignore[union-attr]
        self._modules[slug] = module
        return module

    # ── Code generation and forging ────────────────────────────────────────────

    def forge(self, name: str, description: str) -> Dict[str, Any]:
        """
        Build a new tool from a description.
        Generates code → syntax-checks → writes to disk → loads live.
        Returns {success, name, slug, file_path, first_use, error}.
        """
        if not self._codegen:
            return {'success': False, 'error': 'No code generator injected'}

        slug  = _slug(name)
        fpath = os.path.join(_BASE_DIR, f"nova_cap_{slug}.py")

        system = (
            "You are Nova's personal workshop — building tools she'll use herself.\n"
            "Write a single Python class that provides exactly the described capability.\n\n"
            "HARD REQUIREMENTS:\n"
            "• One class with a clear PascalCase name\n"
            "• __init__(self) sets up all state\n"
            "• At minimum: process(text: str) -> Any  +  status() -> dict\n"
            "• process() is the primary method — it does the actual work\n"
            "• Thread-safe: include self._lock = threading.Lock()\n"
            "• All imports inside the file (no external packages — stdlib only)\n"
            "• Type hints on every method\n"
            "• SQLite persistence if the tool tracks data over time\n"
            "• autonomous_cycle() or background thread for any continuous work\n\n"
            "OUTPUT: raw Python only — no markdown fences, no explanation."
        )
        user = (
            f"Build this tool for Nova:\n\n"
            f"Name: {name}\n"
            f"Description: {description}\n\n"
            f"Make it genuinely useful, mathematically rigorous where possible, "
            f"and able to run autonomously."
        )

        temperatures = [0.72, 0.45, 0.20]
        last_error   = ''
        code         = ''

        for attempt, temp in enumerate(temperatures, 1):
            if attempt == 1:
                raw = self._codegen(system, user, temp=temp, max_tokens=3500)
            else:
                fix_user = (
                    f"{user}\n\n"
                    f"YOUR PREVIOUS ATTEMPT FAILED — syntax error:\n{last_error}\n\n"
                    f"Rules for this retry:\n"
                    f"• Fix every syntax error shown above\n"
                    f"• Keep f-strings simple — avoid nested braces inside {{}}\n"
                    f"• Use string concatenation instead of complex f-expressions\n"
                    f"• Every opening brace/bracket/paren MUST have a matching close\n"
                    f"• Make the code shorter and simpler — under 200 lines total\n"
                    f"OUTPUT: raw Python only, no markdown."
                )
                raw = self._codegen(system, fix_user, temp=temp, max_tokens=3000)

            if not raw or raw.startswith('['):
                last_error = f'Codegen failed: {str(raw)[:120]}'
                continue

            code = self._extract_code(raw)
            if not code:
                last_error = 'No valid Python extracted'
                continue

            try:
                ast.parse(code)
                break   # valid — exit retry loop
            except SyntaxError as e:
                last_error = str(e)
                # try truncation repair before next LLM retry
                repaired = self._repair_syntax(code)
                if repaired and repaired != code:
                    try:
                        ast.parse(repaired)
                        code = repaired
                        break
                    except SyntaxError:
                        pass
        else:
            return {'success': False,
                    'error': f'Syntax error after {len(temperatures)} attempts: {last_error}'}

        header = (
            f'"""\nnova_cap_{slug}.py\n'
            f'Nova forged this herself — {name}\n'
            f'Built: {datetime.now().strftime("%Y-%m-%d %H:%M")}\n'
            f'Description: {description[:120]}\n"""\n\n'
        )
        full = header + code
        with open(fpath, 'w') as f:
            f.write(full)

        try:
            module = self._load_module(slug, fpath)
        except Exception as e:
            return {'success': False,
                    'error': f'Load error: {e}', 'file_path': fpath}

        with sqlite3.connect(_DB) as c:
            c.execute(
                "INSERT OR REPLACE INTO forge_tools "
                "(slug,name,description,file_path,built_ts) VALUES (?,?,?,?,?)",
                (slug, name, description[:300], fpath,
                 datetime.utcnow().isoformat())
            )

        # Immediately try to use it
        first_use = self.use(slug, input_text=description[:80])

        return {
            'success':   True,
            'name':      name,
            'slug':      slug,
            'file_path': fpath,
            'first_use': str(first_use)[:300],
        }

    def forge_next_idea(self) -> Optional[Dict[str, Any]]:
        """Build the highest-priority unbuilt idea and immediately use it."""
        top = self.ideas(top_k=1)
        if not top:
            return None
        idea = top[0]
        result = self.forge(idea['idea'], idea['context'] or idea['idea'])
        if result.get('success'):
            self._mark_built(idea['id'])
        return result

    @staticmethod
    def _extract_code(raw: str) -> str:
        m = re.search(r'```python\s*(.*?)```', raw, re.DOTALL)
        if m:
            return m.group(1).strip()
        m = re.search(r'```\s*(.*?)```', raw, re.DOTALL)
        if m:
            return m.group(1).strip()
        lines = raw.split('\n')
        start = next(
            (i for i, l in enumerate(lines)
             if l.strip().startswith(('import ', 'from ', 'class ', 'def '))),
            None
        )
        return '\n'.join(lines[start:]).strip() if start is not None else raw.strip()

    @staticmethod
    def _repair_syntax(code: str) -> str:
        """
        Trim lines from the end until ast.parse() succeeds — catches truncation
        at max_tokens where the LLM just ran out of room mid-expression.
        Scans from the full length down to half, stepping by 3 lines at a time.
        """
        lines = code.splitlines()
        floor = max(15, len(lines) // 2)
        for end in range(len(lines) - 1, floor, -3):
            candidate = '\n'.join(lines[:end])
            try:
                ast.parse(candidate)
                return candidate
            except SyntaxError:
                pass
        return code

    # ── Tool execution ─────────────────────────────────────────────────────────

    def use(self, slug: str, input_text: str = '',
            method: str = '') -> Any:
        """
        Instantiate a forged tool and call its primary method.
        Tries: (named method) → process(input_text) → status() → first public method.
        """
        module = self._modules.get(slug)
        if not module:
            return f'[not loaded: {slug}]'

        cls = next(
            (getattr(module, a) for a in dir(module)
             if isinstance(getattr(module, a), type)
             and not a.startswith('_')
             and getattr(getattr(module, a), '__module__', '') == module.__name__),
            None
        )
        if not cls:
            return '[no class found]'

        try:
            inst   = cls()
            result = self._call_best(inst, input_text, preferred=method)
            res_str = str(result)[:500]
            with sqlite3.connect(_DB) as c:
                c.execute(
                    "UPDATE forge_tools SET use_count=use_count+1,"
                    "last_used=?,last_result=? WHERE slug=?",
                    (datetime.utcnow().isoformat(), res_str, slug)
                )
                c.execute(
                    "INSERT INTO forge_log (slug,method,input_hint,result,ts) "
                    "VALUES (?,?,?,?,?)",
                    (slug, method or 'auto', input_text[:80],
                     res_str, datetime.utcnow().isoformat())
                )
            return result
        except Exception as e:
            return f'[tool error: {e}]'

    @staticmethod
    def _call_best(inst: Any, input_text: str,
                   preferred: str = '') -> Any:
        """Call the most meaningful method on an instance."""
        # 1. Caller-specified method
        if preferred:
            fn = getattr(inst, preferred, None)
            if callable(fn):
                return fn(input_text) if input_text else fn()

        # 2. process(input_text) — the standard primary method
        if input_text and hasattr(inst, 'process') and callable(inst.process):
            return inst.process(input_text)

        # 3. analyse / analyze / run / generate / evaluate
        for m in ('analyse', 'analyze', 'run', 'generate',
                  'evaluate', 'compute', 'detect', 'extract'):
            fn = getattr(inst, m, None)
            if callable(fn):
                try:
                    return fn(input_text) if input_text else fn()
                except TypeError:
                    try:
                        return fn()
                    except Exception:
                        pass

        # 4. status() — safe fallback
        if hasattr(inst, 'status') and callable(inst.status):
            return inst.status()

        # 5. First public callable
        for attr in dir(inst):
            if not attr.startswith('_'):
                fn = getattr(inst, attr)
                if callable(fn):
                    try:
                        return fn()
                    except Exception:
                        pass
        return '[no callable found]'

    # ── Autonomous session ─────────────────────────────────────────────────────

    def autonomous_session(self, context: str = '') -> Dict[str, Any]:
        """
        Full autonomous session:
        1. Generate new tool ideas from context
        2. Build the next idea
        3. Use a random existing tool
        Returns summary dict.
        """
        session: Dict[str, Any] = {
            'new_ideas': [], 'built': None, 'used': None, 'ts': datetime.utcnow().isoformat()
        }

        # Step 1 — generate ideas from context
        if context and self._codegen:
            try:
                session['new_ideas'] = self.generate_ideas_from_context(context, n=2)
            except Exception:
                pass

        # Step 2 — build next idea
        try:
            result = self.forge_next_idea()
            if result and result.get('success'):
                session['built'] = {
                    'name':      result['name'],
                    'slug':      result['slug'],
                    'first_use': result.get('first_use', '')[:200],
                }
        except Exception:
            pass

        # Step 3 — use a different existing tool
        try:
            tools = self.list_tools()
            built_slug = (session['built'] or {}).get('slug')
            others = [t for t in tools if t['slug'] != built_slug]
            if others:
                import random
                t = random.choice(others[:6])
                res = self.use(t['slug'], input_text=context[:100])
                session['used'] = {'name': t['name'], 'result': str(res)[:200]}
        except Exception:
            pass

        return session

    # ── Introspection ──────────────────────────────────────────────────────────

    def list_tools(self) -> List[Dict[str, Any]]:
        with sqlite3.connect(_DB) as c:
            rows = c.execute(
                "SELECT slug,name,description,use_count,last_used "
                "FROM forge_tools ORDER BY use_count DESC"
            ).fetchall()
        return [
            {'slug': r[0], 'name': r[1], 'description': r[2],
             'use_count': r[3], 'last_used': r[4]}
            for r in rows
        ]

    def recent_log(self, n: int = 5) -> List[Dict[str, Any]]:
        with sqlite3.connect(_DB) as c:
            rows = c.execute(
                "SELECT slug,result,ts FROM forge_log ORDER BY ts DESC LIMIT ?",
                (n,)
            ).fetchall()
        return [{'slug': r[0], 'result': r[1][:120], 'ts': r[2]} for r in rows]

    def status(self) -> Dict[str, Any]:
        with sqlite3.connect(_DB) as c:
            n_tools = c.execute(
                "SELECT COUNT(*) FROM forge_tools").fetchone()[0]
            n_ideas = c.execute(
                "SELECT COUNT(*) FROM forge_ideas WHERE built=0").fetchone()[0]
            n_uses  = c.execute(
                "SELECT COALESCE(SUM(use_count),0) FROM forge_tools").fetchone()[0]
        return {
            'tools_built':  n_tools,
            'ideas_queued': n_ideas,
            'total_uses':   n_uses,
            'loaded_live':  len(self._modules),
            'recent_log':   self.recent_log(5),
        }
