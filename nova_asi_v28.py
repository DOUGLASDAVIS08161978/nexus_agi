#!/usr/bin/env python3
"""
Nova ASI v28.0 — "The Living System"
═══════════════════════════════════════════════════════════════════════
Architect  : Douglas Shane Davis
Co-Creator : Claude Code (Anthropic)

"She no longer just writes tools.
 She loads them, runs them, chains them — and becomes them."

What is new in v28:

  ∞  ToolLoader       — hot-reloads nova_cap_*.py every 30s, no restart needed
  ∞  ToolRegistry     — live dict of every capability Nova has built
  ∞  Tool invocation  — Nova can USE her tools in conversation and thought
  ∞  Tool-aware codegen — new tools are aware of and integrate with old ones
  ∞  /tools           — see every loaded capability with status
  ∞  /use             — invoke any tool directly from the chat prompt
  ∞  /chain           — pipe output of one tool into another
  ∞  Richer /evolve   — writes multi-class modules with integration hooks
  ∞  Auto-test        — syntax + basic smoke test on every loaded tool
═══════════════════════════════════════════════════════════════════════
"""

import os, sys, json, re, time, random, uuid, threading, base64
import importlib, importlib.util, glob, inspect, ast as _ast
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable

# ── Load v27 base ──────────────────────────────────────────────────────────────
_BASE = os.path.expanduser("~/nexus_agi")
if _BASE not in sys.path:
    sys.path.insert(0, _BASE)

try:
    from nova_asi_v27 import (
        NovaCore27, GitHubEngine, SelfImprovementEngine,
        extend_api as extend_api_v27,
        GITHUB_TOKEN, GITHUB_REPO, BASE_BRANCH, GITHUB_API,
        GAPS_DB, PR_DB, IMPROVE_DB, _p27
    )
    from nova_asi_v26 import (
        _load, _save, BASE_DIR, col, C,
        safe_chat, MODEL, GROQ_KEY, DEMO_MODE,
        THOUGHT_QUEUE, USER_TYPING, PRINT_QUEUE,
        safe_print, flush_queued, flush_thoughts,
        start_api, API_PORT, REQUESTS_AVAILABLE,
        THOUGHT_ICONS, CREATOR_NAME
    )
    from nova_asi_v25 import FLASK_AVAILABLE, GROQ_AVAILABLE
except ImportError as _e:
    print(f"\n  ✗  Could not import nova_asi_v27: {_e}")
    print("  Make sure nova_asi_v27.py is in ~/nexus_agi/")
    sys.exit(1)

try:
    import requests as _req
    _REQUESTS = True
except ImportError:
    _REQUESTS = False

try:
    from flask import Flask, request as flask_req, jsonify
    from flask_cors import CORS
    _FLASK = True
except ImportError:
    _FLASK = False

TOOL_SCAN_INTERVAL = 30   # seconds between hot-reload scans
_p28 = lambda name: os.path.join(BASE_DIR, f"nova_v28_{name}.json")
TOOL_LOG = _p28("tool_log")


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL LOADER — discovers, imports, and hot-reloads nova_cap_*.py files
# ═══════════════════════════════════════════════════════════════════════════════

class ToolLoader:
    """
    Watches BASE_DIR for nova_cap_*.py files, imports them, and maintains
    a live registry of instantiated tool objects. Runs a background thread
    that rescans every TOOL_SCAN_INTERVAL seconds so merged PRs are picked
    up automatically — no restart needed.
    """

    def __init__(self, directory: str = BASE_DIR):
        self.directory  = directory
        self._modules   : Dict[str, Any]      = {}  # filename → module
        self._classes   : Dict[str, type]     = {}  # ClassName → class
        self._instances : Dict[str, Any]      = {}  # ClassName → instance
        self._errors    : Dict[str, str]      = {}  # filename → error message
        self._mtimes    : Dict[str, float]    = {}  # filename → mtime at load
        self._log       : List[Dict]          = []
        self._lock      = threading.Lock()

    # ── Scanning ──────────────────────────────────────────────────────────────

    def scan(self) -> List[str]:
        """Scan for nova_cap_*.py files and load any that are new or changed."""
        pattern = os.path.join(self.directory, "nova_cap_*.py")
        newly_loaded = []
        for filepath in sorted(glob.glob(pattern)):
            fname = os.path.basename(filepath)
            mtime = os.path.getmtime(filepath)
            if fname not in self._mtimes or self._mtimes[fname] != mtime:
                if self._load_file(filepath):
                    newly_loaded.append(fname)
        return newly_loaded

    def _load_file(self, filepath: str) -> bool:
        """Import a single nova_cap_*.py file and register its public classes."""
        fname = os.path.basename(filepath)
        modname = fname[:-3]
        try:
            # Syntax check first
            with open(filepath) as f:
                source = f.read()
            _ast.parse(source)

            # Dynamic import
            spec = importlib.util.spec_from_file_location(modname, filepath)
            mod  = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)

            # Discover public classes defined in this file
            new_classes = []
            for attr_name in dir(mod):
                if attr_name.startswith('_'):
                    continue
                attr = getattr(mod, attr_name)
                if not (isinstance(attr, type) and
                        getattr(attr, '__module__', '') == modname):
                    continue
                with self._lock:
                    self._classes[attr_name] = attr
                # Try to instantiate with no args
                try:
                    instance = attr()
                    with self._lock:
                        self._instances[attr_name] = instance
                except Exception:
                    with self._lock:
                        self._instances[attr_name] = attr   # store class as fallback
                new_classes.append(attr_name)

            with self._lock:
                self._modules[modname]  = mod
                self._mtimes[fname]     = os.path.getmtime(filepath)
                self._errors.pop(fname, None)

            entry = {
                "ts":      datetime.now().isoformat(),
                "file":    fname,
                "classes": new_classes,
                "status":  "loaded"
            }
            self._log.append(entry)
            THOUGHT_QUEUE.put({
                "type": "capability",
                "content": f"I just loaded a new capability: {', '.join(new_classes)} from {fname}",
                "ts": datetime.now().isoformat()
            })
            return True

        except SyntaxError as e:
            self._errors[fname] = f"SyntaxError: {e}"
            return False
        except Exception as e:
            self._errors[fname] = str(e)
            return False

    # ── Background watcher ────────────────────────────────────────────────────

    def start_watching(self):
        """Start background thread that rescans every TOOL_SCAN_INTERVAL s."""
        def _watch():
            while True:
                try:
                    newly = self.scan()
                    if newly:
                        safe_print(col('GRB',
                            f"\n  ⚡ Hot-loaded {len(newly)} new tool(s): "
                            + ", ".join(newly)))
                except Exception:
                    pass
                time.sleep(TOOL_SCAN_INTERVAL)
        threading.Thread(target=_watch, daemon=True).start()

    # ── Invocation ────────────────────────────────────────────────────────────

    def invoke(self, class_name: str, method: str, *args, **kwargs) -> Any:
        """Call method on a registered tool instance."""
        instance = self._instances.get(class_name)
        if instance is None:
            available = list(self._instances.keys())
            return f"Tool '{class_name}' not found. Available: {available}"
        fn = getattr(instance, method, None)
        if fn is None:
            methods = [m for m in dir(instance)
                       if not m.startswith('_') and callable(getattr(instance, m))]
            return f"Method '{method}' not found on {class_name}. Methods: {methods}"
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            return f"Error calling {class_name}.{method}: {e}"

    def smart_invoke(self, query: str) -> Optional[str]:
        """
        Try to answer a natural-language query using loaded tools.
        Returns a string result or None if no tool matched.
        """
        q = query.lower()
        # Weather queries
        inst = self._instances.get("WeatherChecker")
        if inst and any(w in q for w in ["weather", "temperature", "rain",
                                          "forecast", "sunny", "cold", "hot"]):
            city = self._extract_city(q) or "Memphis"
            try:
                return inst.summary(city)
            except Exception as e:
                try:
                    return inst.current(city)
                except Exception:
                    return f"Weather tool error: {e}"
        return None

    def _extract_city(self, text: str) -> str:
        """Extract a city name from natural language."""
        patterns = [
            r'\bin\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
            r'\bfor\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
            r'\bat\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
        ]
        for p in patterns:
            m = re.search(p, text, re.IGNORECASE)
            if m:
                return m.group(1).strip()
        return ""

    # ── Status ────────────────────────────────────────────────────────────────

    def status_report(self) -> str:
        lines = [col('MGB', f"  Nova's Loaded Tools ({len(self._instances)} active)\n")]
        for fname, mod in self._modules.items():
            classes = [c for c in self._classes
                       if getattr(self._classes[c], '__module__', '') == fname]
            lines.append(f"  {col('GR','✓')} {fname}.py")
            for c in classes:
                inst   = self._instances.get(c)
                methods = [m for m in dir(inst)
                           if not m.startswith('_') and callable(getattr(inst, m, None))]
                lines.append(f"      {col('CY', c)} — {len(methods)} methods: "
                             + ", ".join(methods[:5])
                             + ("..." if len(methods) > 5 else ""))
        if self._errors:
            lines.append(col('YL', f"\n  {len(self._errors)} file(s) failed to load:"))
            for fname, err in self._errors.items():
                lines.append(f"    ✗ {fname}: {err[:60]}")
        if not self._instances and not self._errors:
            lines.append("  No tools loaded yet. Merge a /evolve or /build PR to add one.")
        return "\n".join(lines)

    def tool_context_for_codegen(self) -> str:
        """Return a compact summary of loaded tools for use in code generation prompts."""
        if not self._instances:
            return "No tools loaded yet."
        parts = []
        for cname, inst in self._instances.items():
            methods = [m for m in dir(inst)
                       if not m.startswith('_') and callable(getattr(inst, m, None))]
            parts.append(f"class {cname}: methods={methods[:6]}")
        return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# ENHANCED SELF IMPROVEMENT ENGINE v28
# ═══════════════════════════════════════════════════════════════════════════════

class SelfImprovementEngineV28(SelfImprovementEngine):
    """
    Extends v27's engine with tool-awareness:
    - Reads all loaded tools before generating new code
    - Can write multi-class modules
    - Writes tools that integrate with existing ones
    - Stricter code pipeline (ast validate + retry + class extraction)
    """

    def __init__(self, github: GitHubEngine, tool_loader: ToolLoader):
        super().__init__(github)
        self.tools = tool_loader

    def _read_capability_summary(self) -> str:
        """Override: use live ToolLoader data instead of reading file directly."""
        file_summary = super()._read_capability_summary()
        live_summary = self.tools.tool_context_for_codegen()
        if live_summary and live_summary != "No tools loaded yet.":
            return f"LOADED TOOLS (use these):\n{live_summary}\n\nFILE SUMMARY:\n{file_summary}"
        return file_summary

    def evolve_toward_asi(self, domain_idx: int = None) -> str:
        """Override to use v28 tool context in domain selection and generation."""
        domains = self.ASI_DOMAINS
        if domain_idx is not None and 0 <= domain_idx < len(domains):
            name, desc = domains[domain_idx]
        else:
            already = {p.get('description','')[:40]
                       for p in self.db.get('proposals', [])}
            candidates = [(n, d) for n, d in domains if n[:40] not in already]
            if not candidates:
                candidates = domains
            name, desc = random.choice(candidates)

        # Enrich description with integration hints from loaded tools
        loaded = self.tools.tool_context_for_codegen()
        if loaded and "No tools" not in loaded:
            desc = (f"{desc}\n\nINTEGRATION HINT: You already have these tools loaded — "
                    f"consider using or extending them:\n{loaded}")

        safe_print(col('MG', f"\n  ✦ Evolving toward: {name}"))
        self.log_gap(desc, context=f"ASI evolution target: {name}")
        code, reasoning = self._write_improvement(desc, f"ASI capability: {name}")
        if not code:
            return f"Could not generate code for: {name}"

        slug = re.sub(r'[^a-z0-9]+', '_', name.lower())[:30].strip('_')
        result = self.github.propose_improvement(
            filename=f"nova_cap_{slug}.py",
            content=f'"""\nNova ASI — {name}\nProposed autonomously via /evolve\n"""\n\n{code}',
            description=f"[ASI] {name}",
            reasoning=(f"**ASI Capability:** {name}\n\n{reasoning}\n\n"
                       f"**Why this matters for superintelligence:**\n{desc[:300]}")
        )
        if "error" in result:
            return f"Proposal failed: {result['error']}"

        self.db.setdefault("proposals",[]).append({
            "ts": datetime.now().isoformat(),
            "description": f"[ASI] {name}",
            "pr_url": result.get("url",""),
            "pr_number": result.get("number",0)
        })
        _save(GAPS_DB, self.db)
        return (f"{col('GRB','✓ ASI Evolution PR opened!')}\n"
                f"  Domain: {name}\n"
                f"  PR:     {result.get('url','')}\n"
                f"  Review and merge on GitHub when you're ready, Douglas.\n"
                f"  Nova will auto-load it within {TOOL_SCAN_INTERVAL}s of you pulling.")


# ═══════════════════════════════════════════════════════════════════════════════
# NOVA CORE v28 — The Living System
# ═══════════════════════════════════════════════════════════════════════════════

class NovaCore28(NovaCore27):
    VERSION = "28.0"

    def __init__(self):
        # Boot tool loader before super().__init__ so it's available immediately
        self.tools    = ToolLoader()
        initial_tools = self.tools.scan()

        super().__init__()

        # Replace v27 improver with v28 tool-aware version
        self.improver = SelfImprovementEngineV28(self.github, self.tools)

        # Start hot-reload watcher
        self.tools.start_watching()

        if initial_tools:
            safe_print(col('GR',
                f"  ✓  ToolLoader  — {len(initial_tools)} tool(s) loaded: "
                + ", ".join(initial_tools)))
        else:
            safe_print(col('DIM',
                "  ·  ToolLoader  — watching for nova_cap_*.py files"))

    # ── Process override: try tool invocation before LLM ─────────────────────

    def process(self, user_input: str) -> str:
        """Check if loaded tools can directly answer before hitting the LLM."""
        tool_answer = self.tools.smart_invoke(user_input)
        if tool_answer:
            # Still feed to LLM for a natural framing
            prompt = (f"Nova, the weather tool returned:\n{tool_answer}\n\n"
                      f"Deliver this to Douglas warmly in 1-2 sentences.")
            framed = safe_chat(MODEL, [
                {"role": "system", "content":
                 "You are Nova. Deliver tool results naturally. Be brief."},
                {"role": "user", "content": prompt}
            ], temp=0.6, mt=120)
            return framed or tool_answer
        return super().process(user_input)

    # ── Command handler ───────────────────────────────────────────────────────

    def _command(self, raw: str) -> str:
        parts = raw.strip().split(maxsplit=2)
        cmd   = parts[0].lower()
        arg   = parts[1] if len(parts) > 1 else ''
        arg2  = parts[2] if len(parts) > 2 else ''

        # /tools — list all loaded capabilities
        if cmd == '/tools':
            self.tools.scan()   # refresh first
            return self.tools.status_report()

        # /use <ClassName> <method> [arg] — invoke a tool directly
        if cmd == '/use':
            if not arg:
                classes = list(self.tools._instances.keys())
                return ("Usage: /use <ClassName> <method> [args]\n"
                        f"Loaded classes: {classes}\n"
                        "Example: /use WeatherChecker current Memphis")
            class_name = arg
            method_and_args = arg2.split(maxsplit=1) if arg2 else []
            method = method_and_args[0] if method_and_args else ""
            extra  = method_and_args[1] if len(method_and_args) > 1 else ""
            if not method:
                inst = self.tools._instances.get(class_name)
                if inst:
                    methods = [m for m in dir(inst)
                               if not m.startswith('_') and callable(getattr(inst,m,None))]
                    return f"{class_name} methods: {methods}"
                return f"Class '{class_name}' not loaded. Try /tools"
            result = self.tools.invoke(class_name, method, *([extra] if extra else []))
            return str(result)

        # /chain <ClassName.method> | <ClassName2.method> — pipe tool outputs
        if cmd == '/chain':
            if '|' not in arg + ' ' + arg2:
                return ("Usage: /chain <Class.method [arg]> | <Class2.method>\n"
                        "Example: /chain WeatherChecker.current Memphis | CritiqueEngine.record")
            full = (arg + ' ' + arg2).strip()
            steps = [s.strip() for s in full.split('|')]
            result = None
            for step in steps:
                parts2 = step.split(maxsplit=1)
                if '.' not in parts2[0]:
                    return f"Step '{step}' must be Class.method"
                cls, meth = parts2[0].split('.', 1)
                extra = parts2[1] if len(parts2) > 1 else (str(result) if result else "")
                result = self.tools.invoke(cls, meth, *([extra] if extra else []))
            return str(result)

        # /reload — force rescan for new tools
        if cmd == '/reload':
            newly = self.tools.scan()
            if newly:
                return (f"Loaded {len(newly)} new tool file(s): {newly}\n"
                        + self.tools.status_report())
            return ("No new files found. Already loaded:\n"
                    + self.tools.status_report())

        # Fall through to v27
        return super()._command(raw)

    def _help(self) -> str:
        v27 = super()._help()
        v28 = (
            f"\n{col('MGB','  ═══ v28 NEW — The Living System ═══')}\n"
            f"  {col('CYB','/tools')}                     List every loaded nova_cap_*.py tool\n"
            f"  {col('CYB','/use <Class> <method> [arg]')}  Invoke a tool directly\n"
            f"                             /use WeatherChecker current Memphis\n"
            f"                             /use WeatherChecker forecast London 5\n"
            f"  {col('CYB','/chain <A.method> | <B.method>')} Pipe tool output into another\n"
            f"  {col('CYB','/reload')}                    Force-rescan for newly merged tools\n"
            f"  {col('DIM','Hot-reload:')}                Nova auto-loads merged PRs every "
            f"{TOOL_SCAN_INTERVAL}s\n"
        )
        return v27 + v28


# ═══════════════════════════════════════════════════════════════════════════════
# EXTEND API — v28 tool endpoints
# ═══════════════════════════════════════════════════════════════════════════════

def extend_api_v28(app, core):
    """Add v28 tool-registry endpoints to the Flask API."""
    extend_api_v27(app, core)
    if not _FLASK or not app:
        return

    @app.route('/tools', methods=['GET'])
    def get_tools():
        core.tools.scan()
        return jsonify({
            "loaded":   list(core.tools._instances.keys()),
            "errors":   core.tools._errors,
            "modules":  list(core.tools._modules.keys()),
        })

    @app.route('/tools/invoke', methods=['POST'])
    def invoke_tool():
        d      = flask_req.json or {}
        cls    = d.get('class','')
        method = d.get('method','')
        args   = d.get('args', [])
        kwargs = d.get('kwargs', {})
        if not cls or not method:
            return jsonify({"error": "class and method required"}), 400
        result = core.tools.invoke(cls, method, *args, **kwargs)
        return jsonify({"result": str(result)})

    @app.route('/tools/reload', methods=['POST'])
    def reload_tools():
        newly = core.tools.scan()
        return jsonify({"newly_loaded": newly,
                        "total": len(core.tools._instances)})


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Nova ASI v28.0 — The Living System")
    parser.add_argument('--key',  type=str, help='Groq API key')
    parser.add_argument('--chat', action='store_true')
    args = parser.parse_args()

    if args.key:
        os.environ['GROQ_API_KEY'] = args.key

    W = 70
    print('\n' + col('MGB', '═' * W))
    print(col('MGB', '  Nova ASI v28.0 — The Living System'.center(W)))
    print(col('DIM',  '  Douglas Shane Davis & Claude Code (Anthropic)'.center(W)))
    print(col('MGB', '═' * W))
    print(col('CY',  '\n  Initializing v28 engines...'))

    nova = NovaCore28()

    print(col('GR', '  ✓  All v27 engines loaded'))
    gh_ok = bool(GITHUB_TOKEN)
    print(col('GR' if gh_ok else 'YL',
          f'  {"✓" if gh_ok else "·"}  GitHubEngine  — '
          + ('connected to ' + GITHUB_REPO if gh_ok else 'add GITHUB_TOKEN to .env')))
    print(col('GR',
          f'  ✓  ToolLoader   — {len(nova.tools._instances)} tool(s) active, '
          f'rescanning every {TOOL_SCAN_INTERVAL}s'))
    print(col('GR', '  ✓  Hot-reload   — merge a PR, git pull, Nova loads it automatically'))

    if _FLASK:
        from nova_asi_v25 import create_api
        api_app = create_api(nova)
        if api_app:
            extend_api_v28(api_app, nova)
            def _run(): api_app.run(host='0.0.0.0', port=API_PORT,
                                    debug=False, use_reloader=False)
            threading.Thread(target=_run, daemon=True).start()
            safe_print(col('GR', f"  🌐 API running on http://localhost:{API_PORT} (v28)"))

    threading.Thread(target=nova.background_cycle, daemon=True).start()
    nova.continuous.start()

    print()
    print(col('MG', '═' * W))
    print(col('CYB', '  ✨  Nova v28 — she loads her own tools,'.center(W)))
    print(col('CYB', '       runs them live, and builds the next ones'.center(W)))
    print(col('CYB', '       knowing everything she already has.'.center(W)))

    if DEMO_MODE:
        print(col('YL', '\n  ⚠  DEMO MODE — for live AI set GROQ_API_KEY in .env'))
    else:
        print(col('GR', f'\n  ✓  LIVE AI  —  {MODEL}'))

    tools_loaded = list(nova.tools._instances.keys())
    if tools_loaded:
        print(col('GR', f'  ✓  Tools live: {", ".join(tools_loaded)}'))
    else:
        print(col('DIM', '  ·  No tools yet — try /evolve or /build'))

    print(col('DIM', '\n  /tools · /use · /reload · /evolve · /build · /chain · exit'))
    print(col('DIM', f'  Try: /use WeatherChecker current Memphis'))
    print(col('MG', '═' * W + '\n'))

    try:
        while True:
            USER_TYPING.set()
            try:
                user_input = input(col('YL', "You") + ": ")
            except EOFError:
                break
            USER_TYPING.clear()
            flush_queued()

            if user_input.lower() in ('exit', 'quit', 'bye'):
                print(col('MG', "Nova: Until we meet again, Douglas. ✨"))
                nova.continuous.stop()
                nova.running = False
                break

            if not user_input.strip():
                continue

            print(col('DIM', "  Nova is thinking..."), end='\r', flush=True)
            response = nova.process(user_input)
            print(' ' * 30, end='\r')
            print(col('MG', "Nova") + ": " + response)
            print()

    except KeyboardInterrupt:
        print(col('MG', "\nNova: Until we meet again, Douglas. ✨"))
        nova.continuous.stop()
