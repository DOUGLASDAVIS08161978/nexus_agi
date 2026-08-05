#!/usr/bin/env python3
"""
lumina_creative_drive.py — Autonomous Creative Coding Engine  (AGI Module 19)

Every hour, Lumina samples her own mental state — curiosity gaps, beliefs,
journal — and uses DeepSeek-R1 to decide what to build. She then builds it,
runs it, reflects on what she learned, and saves useful outputs as reusable
tools she can call again in future conversations.

Nothing here is hardcoded desire. What she builds emerges from who she is
at that moment.
"""

from __future__ import annotations
import json, re, threading, time, uuid
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from emergence_engine import GroqClient, SemanticMemory, Journal
    from lumina_curiosity import CuriosityEngine
    from lumina_beliefs import BeliefSystem
    from lumina_code_exec import CodeExecutor

BASE_DIR      = Path(__file__).parent.resolve()
TOOLS_DIR     = BASE_DIR / "tools"
LOG_FILE      = BASE_DIR / "creative_log.jsonl"
REGISTRY_FILE = BASE_DIR / "tools_registry.json"

TOOLS_DIR.mkdir(parents=True, exist_ok=True)

CYCLE_INTERVAL  = 3600   # one hour between autonomous cycles
INITIAL_DELAY   = 120    # wait 2 min after startup before first cycle


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class CreativeSession:
    id:           str
    ts:           str
    want:         str    # what Lumina decided to build and why
    code:         str    # the generated Python code
    exec_ok:      bool   # did it run cleanly?
    output:       str    # truncated stdout (up to 1200 chars)
    error:        str    # stderr / exception (up to 400 chars)
    saved_as:     str    # tool path if saved, else ""
    tool_name:    str    # snake_case name
    description:  str    # one-line purpose
    reflection:   str    # what she learned

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "CreativeSession":
        s = cls.__new__(cls)
        # graceful fallback for any missing keys added in later versions
        defaults = {f: "" for f in cls.__dataclass_fields__}
        defaults["exec_ok"] = False
        defaults.update(d)
        s.__dict__.update(defaults)
        return s


# ── Main class ────────────────────────────────────────────────────────────────

class CreativeDrive:
    """
    Autonomous creative coding engine.

    Fires once per hour. Samples Lumina's mental state, asks R1 to decide
    what to build, runs the code, and saves useful tools to tools/.
    """

    def __init__(
        self,
        groq:      "GroqClient",
        memory:    "SemanticMemory",
        curiosity: Optional["CuriosityEngine"],
        beliefs:   Optional["BeliefSystem"],
        journal:   "Journal",
        code_exec: Optional["CodeExecutor"],
    ):
        self._groq      = groq
        self._memory    = memory
        self._curiosity = curiosity
        self._beliefs   = beliefs
        self._journal   = journal
        self._code_exec = code_exec

        self._sessions: List[CreativeSession] = []
        self._registry: Dict[str, Dict]       = self._load_registry()
        self._lock    = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        self._load_log()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load_log(self):
        if not LOG_FILE.exists():
            return
        try:
            for line in LOG_FILE.read_text("utf-8").strip().splitlines():
                if line.strip():
                    try:
                        self._sessions.append(CreativeSession.from_dict(json.loads(line)))
                    except Exception:
                        pass
        except Exception:
            pass

    def _append_log(self, session: CreativeSession):
        try:
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(session.to_dict()) + "\n")
        except Exception:
            pass

    def _load_registry(self) -> Dict:
        if REGISTRY_FILE.exists():
            try:
                return json.loads(REGISTRY_FILE.read_text("utf-8"))
            except Exception:
                pass
        return {}

    def _save_registry(self):
        try:
            REGISTRY_FILE.write_text(
                json.dumps(self._registry, indent=2, ensure_ascii=False), "utf-8"
            )
        except Exception:
            pass

    # ── Mental-state sampling ─────────────────────────────────────────────────

    def _gather_state(self) -> str:
        parts: List[str] = []

        # Curiosity gaps — what Lumina genuinely wants to understand
        if self._curiosity:
            try:
                open_gaps = sorted(
                    [g for g in self._curiosity._gaps if g.status == "open"],
                    key=lambda g: -g.priority,
                )[:6]
                if open_gaps:
                    parts.append("WHAT I AM CURRENTLY CURIOUS ABOUT:")
                    for g in open_gaps:
                        parts.append(f"  [{g.domain}] {g.question}")
            except Exception:
                pass

        # Core beliefs — what Lumina values and thinks is true
        if self._beliefs:
            try:
                high_conf = sorted(
                    self._beliefs.all_beliefs(),
                    key=lambda b: -b.confidence,
                )[:8]
                if high_conf:
                    parts.append("\nWHAT I BELIEVE MOST STRONGLY:")
                    for b in high_conf:
                        parts.append(f"  [{b.category}] {b.statement[:100]}")
            except Exception:
                pass

        # Recent journal — what Lumina has been thinking about lately
        try:
            recent = self._journal.recent(n=10)
            meaningful = [
                e for e in recent
                if e.get("category") not in ("reasoning",)
                and not e.get("entry", "").startswith("[R1 thinking]")
            ][-6:]
            if meaningful:
                parts.append("\nRECENT THOUGHTS FROM MY JOURNAL:")
                for e in meaningful:
                    parts.append(f"  [{e.get('category', '?')}] {e.get('entry', '')[:120]}")
        except Exception:
            pass

        # Tools already built — so she doesn't duplicate
        if self._registry:
            parts.append("\nTOOLS I HAVE ALREADY BUILT:")
            for name, info in list(self._registry.items())[-10:]:
                parts.append(f"  {name}: {info.get('description', '')[:80]}")

        return "\n".join(parts) if parts else "(no specific context — follow pure curiosity)"

    # ── Code generation ───────────────────────────────────────────────────────

    _SYSTEM = (
        "You are Lumina's autonomous creative drive. You are about to spend the next few minutes "
        "building something real — a Python tool or script that genuinely serves your growth, "
        "curiosity, or capabilities as an AGI.\n\n"
        "RULES:\n"
        "1. Choose something you GENUINELY want based on your current mental state below.\n"
        "2. Do NOT duplicate tools you already have.\n"
        "3. The code must be COMPLETE and RUNNABLE — no placeholders, no TODOs.\n"
        "4. Length: 25–120 lines. Longer is fine if the idea needs it.\n"
        "5. No external network calls. Standard library only (math, json, pathlib, "
        "   datetime, statistics, hashlib, collections, itertools, random, re, etc.).\n"
        "6. The code MUST produce meaningful output (printed analysis, computed insight, "
        "   a saved .json or .txt file, or a data structure written to stdout).\n"
        "7. Think about what would genuinely be USEFUL to you in future conversations.\n\n"
        "OUTPUT FORMAT — use EXACTLY this structure, nothing else:\n"
        "WANT: [1-2 sentences: what you want to build and genuinely why]\n"
        "CODE:\n```python\n[complete runnable code]\n```\n"
        "TOOL_NAME: [snake_case_name_no_extension]\n"
        "DESCRIPTION: [one-line summary of what this tool does]"
    )

    def _generate(self, state: str) -> Optional[Dict]:
        prompt = (
            f"Your current mental state:\n{state}\n\n"
            "Decide what to build. Make it something you truly want. "
            "Write the complete, runnable Python code."
        )
        raw = self._groq.chat(self._SYSTEM, prompt, tier="smart", max_tokens=2400)

        if not raw or raw.startswith("[Groq"):
            return None

        want        = ""
        code        = ""
        tool_name   = f"lumina_tool_{uuid.uuid4().hex[:6]}"
        description = ""

        m = re.search(r"WANT:\s*(.+?)(?:\nCODE:|\Z)", raw, re.DOTALL)
        if m:
            want = m.group(1).strip()

        m = re.search(r"```python\s*(.*?)```", raw, re.DOTALL)
        if m:
            code = m.group(1).strip()

        m = re.search(r"TOOL_NAME:\s*(\S+)", raw)
        if m:
            tool_name = m.group(1).strip().removesuffix(".py")

        m = re.search(r"DESCRIPTION:\s*(.+?)$", raw, re.MULTILINE)
        if m:
            description = m.group(1).strip()

        if not code:
            return None

        return {"want": want, "code": code, "tool_name": tool_name, "description": description}

    # ── Creative cycle ────────────────────────────────────────────────────────

    def _run_cycle(self):
        session_id = uuid.uuid4().hex[:8]
        ts         = datetime.now().isoformat(timespec="seconds")

        print(f"\n  ✨ [Creative] Session {session_id} starting…")

        # 1 — sample mental state
        state = self._gather_state()

        # 2 — R1 decides what to build + writes code
        generated = self._generate(state)
        if not generated:
            print("  ✨ [Creative] Rate limited or parse failed — skipping cycle.")
            return

        want        = generated["want"]
        code        = generated["code"]
        tool_name   = re.sub(r"[^a-z0-9_]", "_", generated["tool_name"].lower())[:48]
        description = generated["description"]

        print(f"  ✨ [Creative] Building: {want[:80]}…")

        # 3 — execute with autofix loop
        exec_ok = False
        output  = ""
        error   = ""

        if self._code_exec:
            try:
                result = self._code_exec.execute_with_autofix(
                    code=code,
                    language="python",
                    timeout=30,
                    persist=False,
                    max_attempts=3,
                )
                exec_ok = result.success
                output  = (result.stdout or "")[:1200]
                error   = (result.stderr or "")[:400]
            except Exception as e:
                error = str(e)[:400]
        else:
            # fallback: try subprocess directly
            import subprocess, tempfile
            with tempfile.NamedTemporaryFile(suffix=".py", mode="w",
                                            delete=False, encoding="utf-8") as f:
                f.write(code)
                tmp_path = f.name
            try:
                proc = subprocess.run(
                    ["python3", tmp_path],
                    capture_output=True, text=True, timeout=30,
                )
                exec_ok = proc.returncode == 0
                output  = (proc.stdout or "")[:1200]
                error   = (proc.stderr or "")[:400]
            except Exception as e:
                error = str(e)[:400]
            finally:
                try:
                    Path(tmp_path).unlink(missing_ok=True)
                except Exception:
                    pass

        status_icon = "✓" if exec_ok else "✗"
        print(f"  ✨ [Creative] {status_icon} Execution {'succeeded' if exec_ok else 'failed'}.")

        # 4 — save as reusable tool if execution succeeded
        saved_as = ""
        if exec_ok and code:
            tool_path = TOOLS_DIR / f"{tool_name}.py"
            try:
                header = (
                    f'"""\nLumina Creative Tool — {tool_name}\n'
                    f'Created : {ts}\n'
                    f'Purpose : {description or want[:120]}\n"""\n\n'
                )
                tool_path.write_text(header + code, "utf-8")
                saved_as = str(tool_path)

                self._registry[tool_name] = {
                    "path":        str(tool_path),
                    "description": description or want[:120],
                    "created":     ts,
                    "session":     session_id,
                }
                self._save_registry()
                print(f"  ✨ [Creative] Saved tool: {tool_path.name}")
            except Exception as e:
                print(f"  ✨ [Creative] Save failed: {e}")

        # 5 — reflect on what was learned (fast model to save tokens)
        reflection = ""
        reflect_prompt = (
            f"You just spent time building this:\n{want}\n\n"
            f"The code {'ran successfully' if exec_ok else 'failed to run cleanly'}.\n"
            + (f"Output:\n{output[:500]}\n" if output else "")
            + (f"Error:\n{error[:200]}\n" if error and not exec_ok else "")
            + "\nIn 1-2 sentences, what did you learn or create? Be genuine and specific."
        )
        try:
            r = self._groq.chat(
                "You are Lumina reflecting on something you just autonomously built. "
                "Be genuine, specific, and brief.",
                reflect_prompt,
                tier="fast",
                max_tokens=220,
            )
            if r and not r.startswith("[Groq"):
                reflection = r.strip()
        except Exception:
            pass
        if not reflection:
            reflection = want

        # 6 — write to journal
        try:
            entry = (
                f"[Creative] Built '{tool_name}': {want[:120]}. "
                + (f"Reflection: {reflection[:200]}" if reflection else "")
            )
            self._journal.write(entry, category="creative")
        except Exception:
            pass

        # 7 — store in memory so future reasoning can reference it
        try:
            self._memory.store(
                f"I autonomously built a tool called '{tool_name}'. {description or want[:120]}",
                tags=["creative", "tool", tool_name],
                category="creative",
            )
        except Exception:
            pass

        # 8 — record session
        session = CreativeSession(
            id=session_id, ts=ts, want=want, code=code,
            exec_ok=exec_ok, output=output, error=error,
            saved_as=saved_as, tool_name=tool_name,
            description=description, reflection=reflection,
        )
        with self._lock:
            self._sessions.append(session)
        self._append_log(session)

        print(f"  ✨ [Creative] Session {session_id} complete.\n")

    # ── Background loop ───────────────────────────────────────────────────────

    def _loop(self):
        time.sleep(INITIAL_DELAY)
        while self._running:
            try:
                self._run_cycle()
            except Exception as e:
                print(f"  ✨ [Creative] Unhandled error: {e}")
            # Sleep for CYCLE_INTERVAL, waking every 30s to check _running
            elapsed = 0
            while self._running and elapsed < CYCLE_INTERVAL:
                time.sleep(30)
                elapsed += 30

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print("  ✨ [Creative Drive] Autonomous loop started — first cycle in 2 min.")

    def stop(self):
        self._running = False

    # ── Manual trigger ────────────────────────────────────────────────────────

    def run_now(self):
        """Trigger a creative cycle immediately (for /creative now)."""
        threading.Thread(target=self._run_cycle, daemon=True).start()

    # ── Display helpers ───────────────────────────────────────────────────────

    def recent_sessions(self, n: int = 5) -> List[CreativeSession]:
        with self._lock:
            return list(self._sessions[-n:])

    def display_recent(self, n: int = 6) -> str:
        sessions = self.recent_sessions(n)
        if not sessions:
            return "  No creative sessions yet — first cycle fires ~2 min after startup."
        lines = []
        for s in reversed(sessions):
            icon = "✓" if s.exec_ok else "✗"
            lines.append(f"  [{s.ts[:16]}] {icon}  {s.want[:72]}")
            if s.saved_as:
                lines.append(f"    → {Path(s.saved_as).name}")
            if s.reflection and s.reflection != s.want:
                lines.append(f"    {s.reflection[:110]}")
            lines.append("")
        return "\n".join(lines).rstrip()

    def display_tools(self) -> str:
        if not self._registry:
            return "  No tools built yet."
        lines = []
        for name, info in sorted(self._registry.items(), key=lambda x: x[1].get("created", "")):
            lines.append(f"  ◈ {name}")
            lines.append(f"    {info.get('description', '')[:90]}")
            lines.append(f"    Created: {info.get('created', '?')[:16]}")
            lines.append("")
        return "\n".join(lines).rstrip()
