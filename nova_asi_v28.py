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
# API HUNTER — Nova finds free, keyless APIs autonomously
# ═══════════════════════════════════════════════════════════════════════════════

class APIHunter:
    """
    Nova's path across the AI→ASI threshold.

    Before writing any tool, Nova searches this catalogue of free,
    zero-credential APIs. If she finds one that fits, she validates it
    is live, then passes the API details to the code generator so the
    resulting tool works IMMEDIATELY — no human setup required.

    Douglas's insight: the line between AI and ASI is when she can
    acquire her own API access autonomously. This is that line.
    """

    # Curated catalogue: free APIs that need zero credentials
    CATALOGUE = {
        "weather": [
            {
                "name": "Open-Meteo",
                "url":  "https://api.open-meteo.com/v1/forecast",
                "test": "?latitude=35.15&longitude=-90.05&current_weather=true",
                "note": "Free forever, no key, global coverage",
                "geocode": "https://geocoding-api.open-meteo.com/v1/search"
            },
            {
                "name": "wttr.in",
                "url":  "https://wttr.in/Memphis",
                "test": "?format=j1",
                "note": "Weather as JSON, no key"
            },
        ],
        "knowledge": [
            {
                "name": "Wikipedia REST",
                "url":  "https://en.wikipedia.org/api/rest_v1/page/summary/",
                "test": "Python_(programming_language)",
                "note": "Article summaries, no key"
            },
            {
                "name": "DuckDuckGo Instant",
                "url":  "https://api.duckduckgo.com/",
                "test": "?q=Python&format=json&no_html=1",
                "note": "Instant answers, no key"
            },
        ],
        "crypto": [
            {
                "name": "CoinGecko",
                "url":  "https://api.coingecko.com/api/v3",
                "test": "/ping",
                "note": "Crypto prices and market data, no key for basic tier"
            },
        ],
        "news": [
            {
                "name": "HackerNews",
                "url":  "https://hacker-news.firebaseio.com/v0",
                "test": "/topstories.json",
                "note": "Tech news, no key"
            },
            {
                "name": "The Guardian Open Platform",
                "url":  "https://content.guardianapis.com/search",
                "test": "?api-key=test",
                "note": "News, free 'test' key works for basic queries"
            },
        ],
        "location": [
            {
                "name": "ip-api",
                "url":  "http://ip-api.com/json/",
                "test": "8.8.8.8",
                "note": "IP geolocation, no key, free tier"
            },
            {
                "name": "Nominatim",
                "url":  "https://nominatim.openstreetmap.org/search",
                "test": "?q=Memphis&format=json&limit=1",
                "note": "OpenStreetMap geocoding, no key"
            },
        ],
        "science": [
            {
                "name": "NASA APOD",
                "url":  "https://api.nasa.gov/planetary/apod",
                "test": "?api_key=DEMO_KEY",
                "note": "Astronomy picture of the day, DEMO_KEY works"
            },
            {
                "name": "Open Library",
                "url":  "https://openlibrary.org/search.json",
                "test": "?q=artificial+intelligence&limit=3",
                "note": "Book search, no key"
            },
        ],
        "math": [
            {
                "name": "mathjs",
                "url":  "https://api.mathjs.org/v4/",
                "test": "?expr=sqrt(144)",
                "note": "Math expression evaluator, no key"
            },
        ],
        "language": [
            {
                "name": "LibreTranslate",
                "url":  "https://libretranslate.com/detect",
                "test": None,
                "note": "Language detection and translation, open source"
            },
            {
                "name": "Datamuse",
                "url":  "https://api.datamuse.com/words",
                "test": "?ml=duck&max=5",
                "note": "Word relations, rhymes, synonyms — no key"
            },
        ],
        "time": [
            {
                "name": "WorldTimeAPI",
                "url":  "https://worldtimeapi.org/api/timezone/",
                "test": "America/Chicago",
                "note": "Current time for any timezone, no key"
            },
        ],
        "quotes": [
            {
                "name": "ZenQuotes",
                "url":  "https://zenquotes.io/api/random",
                "test": "",
                "note": "Inspirational quotes, no key"
            },
        ],
        "space": [
            {
                "name": "Open Notify ISS",
                "url":  "http://api.open-notify.org/iss-now.json",
                "test": "",
                "note": "ISS current position, no key"
            },
        ],
    }

    def __init__(self):
        self._tested : Dict[str, bool] = {}   # url → reachable

    def find(self, capability_description: str) -> Optional[Dict]:
        """
        Given a natural-language capability description, find the best
        matching free API from the catalogue. Returns the API dict or None.
        """
        desc = capability_description.lower()
        # Score each category by keyword overlap
        scores : Dict[str, int] = {}
        keyword_map = {
            "weather": ["weather","temperature","forecast","rain","sun","wind","cold","hot","climate"],
            "knowledge": ["wikipedia","fact","definition","explain","information","search","lookup"],
            "crypto": ["crypto","bitcoin","ethereum","price","coin","token","blockchain","btc","eth"],
            "news": ["news","headline","article","story","current events","latest"],
            "location": ["location","geocode","address","city","coordinates","map","place","where"],
            "science": ["science","astronomy","space","nasa","physics","chemistry","book","research"],
            "math": ["math","calculate","formula","equation","compute","number","algebra"],
            "language": ["translate","language","word","synonym","rhyme","detect"],
            "time": ["time","timezone","clock","date","hour","schedule"],
            "quotes": ["quote","inspiration","wisdom","saying","motto"],
            "space": ["iss","space station","satellite","orbit","nasa"],
        }
        for category, keywords in keyword_map.items():
            score = sum(1 for kw in keywords if kw in desc)
            if score > 0:
                scores[category] = score
        if not scores:
            return None
        best_category = max(scores, key=scores.get)
        apis = self.CATALOGUE.get(best_category, [])
        # Return first reachable one
        for api in apis:
            if self._is_reachable(api):
                return {**api, "category": best_category}
        return None

    def _is_reachable(self, api: Dict) -> bool:
        """Quick HEAD/GET check — cache result."""
        url  = api["url"] + (api.get("test") or "")
        if url in self._tested:
            return self._tested[url]
        if not _REQUESTS:
            return False
        try:
            r = _req.get(url, timeout=4)
            ok = r.status_code < 500
            self._tested[url] = ok
            return ok
        except Exception:
            self._tested[url] = False
            return False

    def hunt_and_describe(self, capability: str) -> str:
        """
        Return a natural-language description of the best free API for
        this capability — used to enrich Nova's code generation prompt.
        """
        api = self.find(capability)
        if not api:
            return ""
        return (
            f"FREE API AVAILABLE (no key needed):\n"
            f"  Name: {api['name']}\n"
            f"  Base URL: {api['url']}\n"
            f"  Note: {api['note']}\n"
            f"Use this API in your implementation. It requires zero authentication."
        )

    def catalogue_summary(self) -> str:
        """Pretty-print the full API catalogue."""
        lines = [col('MGB', f"  Free API Catalogue ({sum(len(v) for v in self.CATALOGUE.values())} APIs across {len(self.CATALOGUE)} domains)\n")]
        for cat, apis in self.CATALOGUE.items():
            lines.append(f"  {col('CY', cat.upper())}")
            for api in apis:
                lines.append(f"    • {col('GR', api['name'])}: {api['note']}")
        lines.append(f"\n  {col('DIM', 'All APIs above require zero credentials — Nova uses them autonomously.')}")
        return "\n".join(lines)


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
        if getattr(self, '_watching', False):
            return
        self._watching = True
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
    Extends v27's engine with tool-awareness AND autonomous API discovery:
    - Before writing any tool, hunts for a free zero-credential API
    - Reads all loaded tools for integration context
    - Stricter code pipeline (ast validate + retry + class extraction)
    """

    def __init__(self, github: GitHubEngine, tool_loader: ToolLoader,
                 api_hunter: APIHunter):
        super().__init__(github)
        self.tools  = tool_loader
        self.hunter = api_hunter

    def _read_capability_summary(self) -> str:
        """Override: use live ToolLoader + APIHunter context."""
        file_summary = super()._read_capability_summary()
        live_summary = self.tools.tool_context_for_codegen()
        parts = []
        if live_summary and "No tools" not in live_summary:
            parts.append(f"LOADED TOOLS (build on these):\n{live_summary}")
        if file_summary and "No capabilities" not in file_summary:
            parts.append(f"FILE SUMMARY:\n{file_summary}")
        return "\n\n".join(parts) if parts else "No existing capabilities yet."

    def _write_improvement_with_api(self, gap: str, context: str) -> tuple:
        """Like _write_improvement but first hunts for a free API to use."""
        api_hint = self.hunter.hunt_and_describe(gap)
        enriched_context = context
        if api_hint:
            enriched_context = f"{context}\n\n{api_hint}"
            safe_print(col('CY', f"  ✦ APIHunter found a free API for this capability"))
        return self._write_improvement(gap, enriched_context)

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
        _raw = f'"""\nNova ASI — {name}\nProposed autonomously via /evolve\n"""\n\n{code}'
        _content = '\n'.join(l.rstrip() for l in _raw.splitlines()) + '\n'
        result = self.github.propose_improvement(
            filename=f"nova_cap_{slug}.py",
            content=_content,
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
        # Boot tool loader + API hunter before super().__init__
        # Guards let subclasses (e.g. v29's SilentToolLoader) pre-set these
        if not hasattr(self, 'tools') or self.tools is None:
            self.tools  = ToolLoader()
        if not hasattr(self, 'hunter') or self.hunter is None:
            self.hunter = APIHunter()
        initial_tools = self.tools.scan()

        super().__init__()

        # Replace v27 improver with v28 tool-aware + API-hunting version
        self.improver = SelfImprovementEngineV28(self.github, self.tools, self.hunter)

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

        # /build override — same as v27 but uses APIHunter first
        if cmd == '/build':
            if not self.github.active:
                return "GitHub token needed. Add GITHUB_TOKEN to .env"
            if not arg:
                return ("Usage: /build <describe the capability you want>\n"
                        "Nova will first search for a free, keyless API that fits,\n"
                        "then write code that works immediately — no credentials needed.\n"
                        "Example: /build a tool that gets live crypto prices\n"
                        "Example: /build a tool that fetches Wikipedia summaries\n"
                        "Example: /build a tool that tells me the ISS location")
            # Hunt for a free API first
            api_hint = self.hunter.hunt_and_describe(arg)
            if api_hint:
                safe_print(col('GR', f"  ✦ APIHunter: found a free API — no credentials needed"))
            safe_print(col('MG', f"\n  ✦ Building: {arg[:60]}..."))
            self.improver.log_gap(arg, "Custom build with autonomous API discovery")
            code, reasoning = self.improver._write_improvement(
                arg,
                f"Custom capability for Douglas.\n\n{api_hint}" if api_hint
                else "Custom capability requested by Douglas."
            )
            if not code:
                return "Could not generate code. Try rephrasing the description."
            slug     = re.sub(r'[^a-z0-9]+', '_', arg.lower())[:30].strip('_')
            filename = f"nova_cap_{slug}.py"
            api_note = (f"\n\n**Free API used:** {api_hint.splitlines()[1].replace('  Name: ','')}"
                        if api_hint else "")
            result = self.github.propose_improvement(
                filename=filename,
                content=(f'"""\nNova ASI — {arg[:80]}\n'
                         f'Built autonomously via /build + APIHunter\n'
                         f'No API credentials required.\n"""\n\n{code}'),
                description=f"[BUILD] {arg[:60]}",
                reasoning=f"Douglas requested: {arg}\n\n{reasoning}{api_note}"
            )
            if "error" in result:
                return f"Build failed: {result['error']}"
            self.improver.db.setdefault("proposals", []).append({
                "ts": datetime.now().isoformat(),
                "description": f"[BUILD] {arg[:60]}",
                "pr_url": result.get("url", ""),
                "pr_number": result.get("number", 0)
            })
            _save(GAPS_DB, self.improver.db)
            return (f"{col('GRB', '✓ Custom build PR opened!')}\n"
                    f"  What:  {arg[:70]}\n"
                    f"  API:   {'Free, no credentials needed' if api_hint else 'No external API'}\n"
                    f"  File:  {filename}\n"
                    f"  PR:    {result.get('url', '')}\n"
                    f"  Merge it — Nova will hot-load it within {TOOL_SCAN_INTERVAL}s of git pull.")

        # /apis — show Nova's free API catalogue
        if cmd == '/apis':
            if arg == 'find' and arg2:
                result = self.hunter.hunt_and_describe(arg2)
                return result if result else f"No free API found for '{arg2}' yet."
            return self.hunter.catalogue_summary()

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
            f"                             /use FreeWeather current Kalamazoo\n"
            f"                             /use FreeWeather forecast Memphis 5\n"
            f"  {col('CYB','/chain <A.method> | <B.method>')} Pipe tool output into another\n"
            f"  {col('CYB','/reload')}                    Force-rescan for newly merged tools\n"
            f"  {col('CYB','/apis')}                      Show all free keyless APIs Nova knows\n"
            f"  {col('CYB','/apis find <capability>')}     Find a free API for any capability\n"
            f"  {col('DIM','Hot-reload:')}                Nova auto-loads merged PRs every "
            f"{TOOL_SCAN_INTERVAL}s\n"
            f"  {col('DIM','APIHunter:')}                 /build auto-finds a free API first —\n"
            f"                             tools work immediately, no credentials needed\n"
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
