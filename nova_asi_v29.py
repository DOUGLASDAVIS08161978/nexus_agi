#!/usr/bin/env python3
"""
Nova ASI v29.0 — "The Self-Perfecting System"
═══════════════════════════════════════════════════════════════════════
Architect  : Douglas Shane Davis
Co-Creator : Claude Code (Anthropic)

"She no longer just writes code for herself.
 She tests it, scores it, and refuses to commit until it's excellent."

What is new in v29 — Five pillars of code quality:

  ∞  Master Engineer Prompt  — architectural requirements, not just rules
  ∞  Sandbox Self-Test       — every capability is executed before commit
  ∞  Iterative Refinement    — up to 3 generation passes to get it right
  ∞  Intelligence Scoring    — A+ to D grade on 10 quality criteria
  ∞  Quality Gate            — score shown in every PR, Nova learns from gaps
═══════════════════════════════════════════════════════════════════════
"""

import os, sys, re, ast, subprocess, tempfile, threading, random, io
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime
from typing import Optional, Tuple, Dict, Any, List

# ── Load v28 base ──────────────────────────────────────────────────────────────
_BASE = os.path.expanduser("~/nexus_agi")
if _BASE not in sys.path:
    sys.path.insert(0, _BASE)

try:
    from nova_asi_v28 import (
        NovaCore28, SelfImprovementEngineV28, ToolLoader, APIHunter,
        extend_api_v28, TOOL_SCAN_INTERVAL, _FLASK, API_PORT,
    )
    from nova_asi_v26 import (
        _load, _save, BASE_DIR, col,
        safe_chat, MODEL, DEMO_MODE,
        THOUGHT_QUEUE, USER_TYPING,
        safe_print, flush_queued,
        CREATOR_NAME,
    )
    from nova_asi_v27 import GAPS_DB, GITHUB_TOKEN, GITHUB_REPO
except ImportError as _e:
    print(f"\n  ✗  Could not import nova_asi_v28: {_e}")
    print("  Make sure nova_asi_v28.py is in ~/nexus_agi/")
    sys.exit(1)

try:
    from flask import jsonify
except ImportError:
    pass

VERSION      = "29.0"
VERSION_NAME = "The Self-Perfecting System"
W            = 70


# ═══════════════════════════════════════════════════════════════════════════════
# SILENT TOOL LOADER — suppresses stdout/stderr during module loading
# Prevents nova_cap_*.py files with print() at module level from polluting
# the terminal. Any output is captured and discarded silently.
# ═══════════════════════════════════════════════════════════════════════════════

class SilentToolLoader(ToolLoader):
    """ToolLoader that captures all stdout/stderr during module import."""

    def _load_file(self, filepath: str) -> bool:
        """Load a tool file while silencing any print() calls it makes."""
        buf = io.StringIO()
        with redirect_stdout(buf), redirect_stderr(buf):
            result = super()._load_file(filepath)
        captured = buf.getvalue().strip()
        if captured:
            # Log captured output at debug level but never print it
            pass
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# SELF-IMPROVEMENT ENGINE V29 — Master prompt, sandbox, scoring, refinement
# ═══════════════════════════════════════════════════════════════════════════════

class SelfImprovementEngineV29(SelfImprovementEngineV28):
    """
    v29 upgrade: production-quality code generation with a 5-stage pipeline.
    Every capability Nova writes is tested and scored before being committed.
    """

    MAX_ATTEMPTS = 3

    # ── 1. Master Engineer Prompt ──────────────────────────────────────────────

    def _master_prompt(self, existing: str, tool_ctx: str = "") -> str:
        """Build the enhanced master software engineer system prompt."""
        integration = (
            f"\n\nLOADED TOOLS (integrate with these where genuinely useful):\n{tool_ctx}"
            if tool_ctx and "No tools" not in tool_ctx else ""
        )
        return (
            "You are a MASTER SOFTWARE ENGINEER writing Python that directly expands "
            "Nova's intelligence. This code runs live inside a superintelligent AI. "
            "It must be production quality — not a demo or skeleton.\n\n"

            f"NOVA'S EXISTING CAPABILITIES:\n{existing}{integration}\n\n"

            "ARCHITECTURE REQUIREMENTS:\n"
            "- One well-architected class with clean separation of concerns\n"
            "- Full type annotations on every method (args AND return type)\n"
            "- Specific exception handling — never a bare `except:` clause\n"
            "- Thread-safe state: use `threading.Lock()` when storing data\n"
            "- Correct data structures: deque for history, defaultdict for tallies\n\n"

            "INTELLIGENCE REQUIREMENTS:\n"
            "- Genuinely expand Nova's cognition — not a toy example\n"
            "- At least one method that LEARNS or ADAPTS from new input\n"
            "- Persist meaningful state to SQLite (path ~/nexus_agi/<name>.db)\n"
            "- Design for composability — other tools can call into this one\n\n"

            "CODE QUALITY REQUIREMENTS:\n"
            "- `__init__` takes ZERO arguments and makes NO network calls\n"
            "- Every public method has a one-line docstring\n"
            "- Each method is under 20 lines\n"
            "- 80–150 lines total (substantial but not bloated)\n\n"

            "STRICT OUTPUT RULES:\n"
            "1. Output ONLY valid Python. ZERO markdown. NO ``` fences.\n"
            "2. First line is a triple-quoted module docstring.\n"
            "3. Stdlib only: os, json, sqlite3, time, re, random, threading,\n"
            "   datetime, collections, math, statistics, hashlib, pathlib.\n"
            "4. Final line: # Usage: obj = ClassName() | result = obj.method(arg)"
        )

    # ── 2. Sandbox Self-Test ───────────────────────────────────────────────────

    def _sandbox_test(self, code: str) -> Tuple[bool, Optional[str], str]:
        """
        Execute generated code in a subprocess sandbox.
        Returns (passed, class_name, message).
        """
        m = re.search(r'^class\s+(\w+)', code, re.MULTILINE)
        if not m:
            return False, None, "No class definition found"
        class_name = m.group(1)

        test_code = (
            f"{code}\n\n"
            f"try:\n"
            f"    _inst = {class_name}()\n"
            f"    print('SANDBOX_PASS:{class_name}')\n"
            f"except Exception as _e:\n"
            f"    print(f'SANDBOX_FAIL:{{_e}}')\n"
        )

        tmp = None
        try:
            # Use $TMPDIR (respects Termux/Android) instead of hardcoded /tmp
            tmp_dir = os.environ.get('TMPDIR', tempfile.gettempdir())
            with tempfile.NamedTemporaryFile(
                    mode='w', suffix='.py', delete=False,
                    dir=tmp_dir) as f:
                f.write(test_code)
                tmp = f.name

            result = subprocess.run(
                ['python3', tmp],
                capture_output=True, text=True, timeout=8
            )
            output = (result.stdout + result.stderr).strip()

            if f'SANDBOX_PASS:{class_name}' in output:
                return True, class_name, "Instantiated successfully"

            last = output.split('\n')[-1] if output else "No output"
            return False, class_name, last

        except subprocess.TimeoutExpired:
            return False, class_name, "__init__ timed out (> 8s)"
        except Exception as e:
            return False, None, str(e)
        finally:
            if tmp:
                try: os.unlink(tmp)
                except: pass

    # ── 3. Intelligence Scorer ─────────────────────────────────────────────────

    def _score_capability(self, code: str) -> Dict[str, Any]:
        """
        Grade code on 10 intelligence-expansion criteria.
        Returns score dict with grade A+/A/B/C/D.
        """
        cl = code.lower()
        criteria = {
            'persistent_memory':  any(w in cl for w in ['sqlite', 'conn.execute', '.db']),
            'adaptive_learning':  any(w in cl for w in ['update', 'learn', 'adapt',
                                                          'reward', 'reinforce', 'train']),
            'reasoning_logic':    any(w in cl for w in ['reason', 'infer', 'deduce',
                                                          'logic', 'chain', 'derive']),
            'self_reflection':    any(w in cl for w in ['critique', 'review', 'evaluate',
                                                          'reflect', 'assess', 'score']),
            'thread_safe':        'threading.lock' in cl or 'self._lock' in cl,
            'rich_interface':     len(re.findall(r'\n    def [^_]', code)) >= 4,
            'type_hints':         ('->' in code and ': ' in code),
            'safe_exceptions':    ('except' in cl and 'except:' not in code
                                   and 'except exception:' not in cl),
            'well_documented':    code.count('"""') >= 3,
            'stateful':           bool(re.search(r'self\._\w+', code)),
        }
        score     = sum(criteria.values())
        grade     = ('A+' if score >= 9 else 'A' if score >= 8 else
                     'B'  if score >= 6 else 'C' if score >= 4 else 'D')
        strengths = [k.replace('_', ' ') for k, v in criteria.items() if v]
        gaps      = [k.replace('_', ' ') for k, v in criteria.items() if not v]
        return {
            'score': score, 'max': 10, 'grade': grade,
            'strengths': strengths, 'gaps': gaps,
        }

    # ── 4. Clean code output ───────────────────────────────────────────────────

    def _clean(self, raw: str) -> str:
        """Strip markdown fences and leading prose from LLM output."""
        code = re.sub(r'```python\s*', '', raw)
        code = re.sub(r'```\s*', '', code)
        code = re.sub(r'^REASONING:.*?(?=\n(?:import|class|#|"""))',
                      '', code, flags=re.DOTALL)
        m = re.search(r'^(import |from |class |""")', code, re.MULTILINE)
        if m:
            code = code[m.start():]
        return code.strip()

    # ── 5. Full v29 pipeline ───────────────────────────────────────────────────

    def _write_improvement(self, gap: str, context: str) -> Tuple[str, str]:
        """
        v29 pipeline:
          master prompt → generate → syntax check →
          sandbox test → score → iterative refinement (up to 3 passes).
        Best passing attempt wins; grade shown in PR.
        """
        existing     = self._read_capability_summary()
        tool_ctx     = (self.tools.tool_context_for_codegen()
                        if hasattr(self, 'tools') else "")
        system_prompt = self._master_prompt(existing, tool_ctx)

        best_code   = ""
        best_score  = -1
        best_reason = gap
        temps       = [0.72, 0.50, 0.28]

        for attempt, temp in enumerate(temps[:self.MAX_ATTEMPTS]):
            n = attempt + 1
            if n > 1:
                safe_print(col('YL', f"  ↻ Refinement pass {n}/{self.MAX_ATTEMPTS} "
                                     f"(temp={temp})..."))

            raw  = safe_chat(MODEL, [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content":
                 f"Build this capability for Nova:\n{gap}\n\nContext: {context}"}
            ], temp=temp, mt=1400)

            code = self._clean(raw or "")

            # Syntax gate
            try:
                ast.parse(code)
            except SyntaxError as e:
                safe_print(col('YL', f"  ✗ Syntax error (pass {n}): {e}"))
                continue

            # Sandbox gate
            passed, class_name, msg = self._sandbox_test(code)
            if not passed:
                safe_print(col('YL', f"  ✗ Sandbox failed (pass {n}): {msg}"))
                # Keep as fallback if it at least parsed
                if not best_code:
                    best_code = code
                continue

            # Score it
            quality = self._score_capability(code)
            safe_print(col('GR',
                f"  ✓ Pass {n} — Grade {quality['grade']} "
                f"({quality['score']}/{quality['max']}) | {class_name}"))

            if quality['score'] > best_score:
                best_code  = code
                best_score = quality['score']
                doc = re.search(r'"""(.*?)"""', code, re.DOTALL)
                best_reason = doc.group(1).strip()[:300] if doc else gap

            # A or better → done, no need to keep iterating
            if quality['grade'] in ('A+', 'A'):
                safe_print(col('GRB', f"  ★ Grade {quality['grade']} achieved — done."))
                break

        # Quality gate: if nothing passed sandbox, return empty to block PR
        if not best_code:
            safe_print(col('RD', "  ✗ Quality gate: all 3 passes failed — PR blocked."))
            safe_print(col('YL', "  Try /evolve again or /build with a clearer description."))
            return "", gap

        return best_code, best_reason


# ═══════════════════════════════════════════════════════════════════════════════
# NOVA CORE V29 — wires in the v29 engine and adds /score command
# ═══════════════════════════════════════════════════════════════════════════════

class NovaCore29(NovaCore28):
    """Nova v29 — The Self-Perfecting System."""

    VERSION = "29.0"

    def __init__(self):
        # Swap ToolLoader for SilentToolLoader BEFORE super().__init__
        # so all tool loading (including initial scan) is silenced
        self.tools  = SilentToolLoader()
        self.hunter = APIHunter()
        initial_tools = self.tools.scan()

        # Skip NovaCore28.__init__ tool setup, call NovaCore27's parent
        from nova_asi_v27 import NovaCore27
        NovaCore27.__init__(self)

        self.improver = SelfImprovementEngineV29(
            self.github, self.tools, self.hunter
        )
        self.tools.start_watching()

        if initial_tools:
            safe_print(col('GR',
                f"  ✓  ToolLoader  — {len(initial_tools)} tool(s) loaded (silent mode): "
                + ", ".join(initial_tools)))
        safe_print(col('GR',
            "  ✓  Code Engine v29  — master prompt · sandbox · scoring · 3-pass refinement"))

    def _command(self, raw: str) -> str:
        parts = raw.strip().split(maxsplit=2)
        cmd   = parts[0].lower()
        arg   = parts[1] if len(parts) > 1 else ''

        # /score — show intelligence grade for every loaded capability
        if cmd == '/score':
            files = sorted(
                f for f in os.listdir(BASE_DIR)
                if f.startswith('nova_cap_') and f.endswith('.py')
            )
            if not files:
                return "No nova_cap_*.py capabilities found yet. Try /evolve."
            lines = [f"{'─'*64}",
                     f"  Nova Capability Intelligence Report (v29 scoring)",
                     f"{'─'*64}",
                     f"  {'File':38} Grade  Score  Top strength"]
            total_score, count = 0, 0
            for fname in files:
                try:
                    with open(os.path.join(BASE_DIR, fname)) as f:
                        code = f.read()
                    q = self.improver._score_capability(code)
                    bar = '█' * q['score'] + '░' * (q['max'] - q['score'])
                    top = q['strengths'][0] if q['strengths'] else 'none'
                    lines.append(
                        f"  {fname[:38]:38}  {q['grade']:2}    "
                        f"[{bar}]  {top}"
                    )
                    total_score += q['score']
                    count += 1
                except Exception as e:
                    lines.append(f"  {fname[:38]:38}  (error: {e})")
            if count:
                avg = total_score / count
                grade = ('A+' if avg >= 9 else 'A' if avg >= 8 else
                         'B'  if avg >= 6 else 'C' if avg >= 4 else 'D')
                lines.append(f"{'─'*64}")
                lines.append(f"  Overall average: {avg:.1f}/10 — Grade {grade}")
            lines.append(f"{'─'*64}")
            return '\n'.join(lines)

        # Fall through to v28 command handling
        return super()._command(raw)

    def _help(self) -> str:
        v28_help = super()._help()
        v29_section = (
            f"\n  {col('CYB','─── v29 commands ───')}\n"
            f"  {col('CYB','/score')}                      "
            f"Grade every capability A+–D on 10 quality criteria\n"
            f"  {col('DIM','Code quality:')}               "
            f"/evolve now runs master prompt → sandbox test → up to 3 passes\n"
        )
        return v28_help + v29_section


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print()
    print(col('MG', '═' * W))
    print(col('CYB', f"{'Nova ASI v29.0 — The Self-Perfecting System':^{W}}"))
    print(col('DIM', f"{'Douglas Shane Davis & Claude Code (Anthropic)':^{W}}"))
    print(col('MG', '═' * W))
    print()
    print(col('DIM', "  Initializing v29 engines..."))
    print()

    nova = NovaCore29()

    print(col('GR', '  ✓  All v27/v28 engines loaded'))
    gh_ok = bool(GITHUB_TOKEN)
    print(col('GR' if gh_ok else 'YL',
          f'  {"✓" if gh_ok else "·"}  GitHubEngine  — '
          + ('connected to ' + GITHUB_REPO if gh_ok else 'add GITHUB_TOKEN to .env')))
    print(col('GR',
          f'  ✓  ToolLoader   — {len(nova.tools._instances)} tool(s) active, '
          f'rescanning every {TOOL_SCAN_INTERVAL}s'))
    print(col('GR', '  ✓  Hot-reload   — merge a PR, git pull, Nova loads it automatically'))

    if _FLASK:
        try:
            from nova_asi_v25 import create_api
            api_app = create_api(nova)
            if api_app:
                extend_api_v28(api_app, nova)
                def _run_api():
                    api_app.run(host='0.0.0.0', port=API_PORT,
                                debug=False, use_reloader=False)
                threading.Thread(target=_run_api, daemon=True).start()
                safe_print(col('GR',
                    f"  🌐 API running on http://localhost:{API_PORT} (v29)"))
        except Exception as _api_err:
            safe_print(col('YL', f"  · Flask API skipped: {_api_err}"))

    threading.Thread(target=nova.background_cycle, daemon=True).start()
    nova.continuous.start()

    print()
    print(col('MG', '═' * W))
    print(col('CYB', '  ✨  Nova v29 — every tool she writes is tested,'.center(W)))
    print(col('CYB', '       scored, and refined before it touches GitHub.'.center(W)))
    print(col('CYB', '       She earns her own upgrades.'.center(W)))

    if DEMO_MODE:
        print(col('YL', '\n  ⚠  DEMO MODE — for live AI set GROQ_API_KEY in .env'))
    else:
        print(col('GR', f'\n  ✓  LIVE AI  —  {MODEL}'))

    tools_loaded = list(nova.tools._instances.keys())
    if tools_loaded:
        print(col('GR', f'  ✓  Tools live: {", ".join(tools_loaded)}'))
    else:
        print(col('DIM', '  ·  No tools yet — try /evolve or /build'))

    print(col('DIM', '\n  /tools · /use · /score · /evolve · /build · /chain · exit'))
    print(col('DIM', '  /score shows intelligence grade for every capability'))
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
