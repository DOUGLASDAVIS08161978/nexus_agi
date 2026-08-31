#!/usr/bin/env python3
"""
lumina_code_exec.py — Research-grade code execution sandbox for Lumina

Lumina can now RUN code, not just write it. This closes the gap between
reasoning about computation and performing it.

Capabilities:
  Execution     — Python and Bash, subprocess-isolated, timeout-enforced
  Auto-fix loop — DeepSeek-R1 reads tracebacks and self-corrects (3 attempts)
  Persistence   — Optional REPL namespace: variables survive between /run calls
  Safety        — Pre-flight scan blocks destructive operations
  Visualization — Detects matplotlib/PIL output, saves figures automatically
  History       — Full execution log in sandbox/history.jsonl with metrics
  Introspection — After each run, reports what variables were created/changed

Research basis:
  The auto-fix loop mirrors AlphaCode / Reflexion architectures — the model
  observes its own failure signal (traceback) and iterates. Three attempts
  matches empirical diminishing returns in code self-repair literature.
  Persistent namespace enables true REPL-style incremental computation.
"""

from __future__ import annotations
import ast, json, os, pickle, re, subprocess, sys, tempfile, textwrap, time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from emergence_engine import GroqClient

BASE_DIR    = Path(__file__).parent.resolve()
SANDBOX_DIR = BASE_DIR / "sandbox"
HISTORY_FILE = SANDBOX_DIR / "history.jsonl"
STATE_FILE   = SANDBOX_DIR / "repl_state.pkl"
FIG_DIR      = SANDBOX_DIR / "figures"

MAX_OUTPUT_CHARS = 4000
MAX_HISTORY      = 500


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


# ── Safety pre-flight ──────────────────────────────────────────────────────────

# Operations blocked unconditionally — they can cause irreversible damage
_HARD_BLOCKED = [
    r"shutil\.rmtree\s*\(",
    r"os\.system\s*\(",
    r"rmdir\s*\(",
    r"os\.remove\s*\(.*['\"]\/(?:etc|proc|sys|bin|usr)",
    r"open\s*\(['\"]\/(?:etc|proc|sys)",
    r":(){:|:&};:",           # fork bomb
    r"while\s+True.*os\.",
]

# Operations warned about but not blocked (user proceeds knowingly)
_SOFT_WARNED = [
    (r"subprocess\.", "subprocess call — runs shell commands"),
    (r"os\.unlink\s*\(", "file deletion"),
    (r"open\s*\(.*['\"]w['\"]", "file write"),
    (r"requests\.", "network request"),
    (r"socket\.", "raw socket"),
    (r"__import__\s*\(", "dynamic import"),
]


def _safety_check(code: str) -> Tuple[bool, List[str], List[str]]:
    """
    Returns (safe, hard_blocks, soft_warnings).
    safe=False means execution must be aborted.
    """
    hard, soft = [], []
    for pat in _HARD_BLOCKED:
        if re.search(pat, code):
            hard.append(pat)
    for pat, label in _SOFT_WARNED:
        if re.search(pat, code):
            soft.append(label)
    return len(hard) == 0, hard, soft


# ── Execution result ───────────────────────────────────────────────────────────

class ExecutionResult:
    def __init__(self, code: str, output: str, error: str,
                 success: bool, runtime_ms: int, language: str,
                 attempts: int = 1, fixed_code: str = "",
                 figures: List[str] = None, new_vars: Dict = None):
        self.code        = code
        self.output      = output[:MAX_OUTPUT_CHARS]
        self.error       = error[:2000]
        self.success     = success
        self.runtime_ms  = runtime_ms
        self.language    = language
        self.attempts    = attempts
        self.fixed_code  = fixed_code
        self.figures     = figures or []
        self.new_vars    = new_vars or {}
        self.ts          = _now()

    def to_dict(self) -> Dict:
        return {
            "ts": self.ts, "language": self.language,
            "success": self.success, "runtime_ms": self.runtime_ms,
            "attempts": self.attempts, "output": self.output,
            "error": self.error,
            "code_preview": self.code[:200],
            "figures": self.figures,
        }

    def summary(self) -> str:
        """Human-readable one-line summary."""
        status = "✓" if self.success else "✗"
        fix_note = f" (fixed in {self.attempts} attempts)" if self.attempts > 1 else ""
        fig_note = f" | {len(self.figures)} figure(s)" if self.figures else ""
        return f"{status} {self.runtime_ms}ms{fix_note}{fig_note}"

    def display(self) -> str:
        """Full terminal display block."""
        lines = []
        lines.append("  ┌─ EXECUTION RESULT " + "─" * 45)
        lines.append(f"  │  Status   : {'SUCCESS' if self.success else 'FAILED'} "
                     f"({self.runtime_ms}ms, {self.attempts} attempt(s))")
        if self.fixed_code and self.attempts > 1:
            lines.append(f"  │  Auto-fix : applied after {self.attempts-1} error(s)")
        if self.output:
            lines.append("  │  Output ↓")
            for line in self.output.splitlines()[:40]:
                lines.append(f"  │    {line}")
            if self.output.count("\n") > 40:
                lines.append(f"  │    … ({self.output.count(chr(10))} lines total)")
        if self.error and not self.success:
            lines.append("  │  Error ↓")
            for line in self.error.splitlines()[:15]:
                lines.append(f"  │    {line}")
        if self.figures:
            for fig in self.figures:
                lines.append(f"  │  Figure   : {fig}")
        if self.new_vars:
            lines.append(f"  │  New vars : {', '.join(list(self.new_vars.keys())[:8])}")
        lines.append("  └" + "─" * 63)
        return "\n".join(lines)


# ── CodeExecutor ──────────────────────────────────────────────────────────────

class CodeExecutor:

    def __init__(self, groq: Optional["GroqClient"] = None):
        self._groq       = groq
        self._state: Dict[str, Any] = {}   # persistent REPL namespace
        self._history: List[Dict]   = []
        self._session_runs: int     = 0

        SANDBOX_DIR.mkdir(parents=True, exist_ok=True)
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        self._load_state()
        self._load_history()

    # ── State persistence ──────────────────────────────────────────────────

    def _load_state(self):
        if STATE_FILE.exists():
            try:
                with open(STATE_FILE, "rb") as f:
                    self._state = pickle.load(f)
            except Exception:
                self._state = {}

    def _save_state(self, namespace: Dict):
        """Persist only JSON-serialisable values to avoid bloat."""
        clean = {}
        for k, v in namespace.items():
            if k.startswith("_"):
                continue
            try:
                json.dumps(v)
                clean[k] = v
            except (TypeError, ValueError):
                pass
        self._state = clean
        try:
            with open(STATE_FILE, "wb") as f:
                pickle.dump(clean, f)
        except Exception:
            pass

    def clear_state(self):
        self._state = {}
        STATE_FILE.unlink(missing_ok=True)

    # ── History ────────────────────────────────────────────────────────────

    def _load_history(self):
        if HISTORY_FILE.exists():
            try:
                lines = HISTORY_FILE.read_text("utf-8").strip().splitlines()
                self._history = [json.loads(l) for l in lines[-MAX_HISTORY:]]
            except Exception:
                self._history = []

    def _log_result(self, result: ExecutionResult):
        entry = result.to_dict()
        self._history.append(entry)
        self._history = self._history[-MAX_HISTORY:]
        try:
            with open(HISTORY_FILE, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            pass

    def stats(self) -> str:
        if not self._history:
            return "  No executions yet this session."
        total   = len(self._history)
        success = sum(1 for h in self._history if h["success"])
        avg_ms  = sum(h["runtime_ms"] for h in self._history) // max(total, 1)
        multi   = sum(1 for h in self._history if h.get("attempts", 1) > 1)
        return (f"  Executions: {total} | Success: {success}/{total} "
                f"| Avg: {avg_ms}ms | Auto-fixed: {multi}")

    # ── Safety + language detection ────────────────────────────────────────

    @staticmethod
    def detect_language(code: str) -> str:
        code = code.strip()
        if code.startswith("#!") and "python" in code.splitlines()[0]:
            return "python"
        if code.startswith("#!") and ("bash" in code.splitlines()[0] or "sh" in code.splitlines()[0]):
            return "bash"
        # Heuristic: bash if starts with common shell keywords
        bash_starters = ("echo ", "cd ", "ls ", "cat ", "grep ", "export ",
                         "git ", "pip ", "apt ", "curl ", "wget ")
        if any(code.lstrip().startswith(s) for s in bash_starters):
            return "bash"
        return "python"

    # ── Core execution ─────────────────────────────────────────────────────

    def _run_python(self, code: str, timeout: int,
                    persist: bool = False) -> Tuple[str, str, bool, int, Dict]:
        """
        Run Python code in a subprocess.
        Returns (stdout, stderr, success, runtime_ms, new_vars).
        """
        # Inject persistent state as variable assignments
        preamble = ""
        if persist and self._state:
            lines = []
            for k, v in self._state.items():
                try:
                    lines.append(f"{k} = {repr(v)}")
                except Exception:
                    pass
            preamble = "\n".join(lines) + "\n\n"

        # Append variable dump at end for state capture
        epilogue = ""
        if persist:
            epilogue = textwrap.dedent("""
            import json as __json, builtins as __builtins_mod
            __safe_types = (int, float, str, bool, list, dict, tuple, type(None))
            __state_dump = {
                k: v for k, v in locals().items()
                if not k.startswith('_') and isinstance(v, __safe_types)
            }
            print("__STATE_DUMP__:" + __json.dumps(__state_dump))
            """)

        full_code = preamble + code + epilogue

        with tempfile.NamedTemporaryFile(suffix=".py", mode="w",
                                         dir=str(SANDBOX_DIR), delete=False) as f:
            f.write(full_code)
            fname = f.name

        t0 = time.monotonic()
        try:
            result = subprocess.run(
                [sys.executable, fname],
                capture_output=True, text=True,
                timeout=timeout, cwd=str(SANDBOX_DIR),
            )
            elapsed = int((time.monotonic() - t0) * 1000)
            stdout = result.stdout
            stderr = result.stderr
            success = result.returncode == 0

            # Extract and save state dump
            new_vars = {}
            if persist and "__STATE_DUMP__:" in stdout:
                parts = stdout.split("__STATE_DUMP__:")
                stdout = parts[0]
                try:
                    new_vars = json.loads(parts[1].strip())
                    old_keys = set(self._state.keys())
                    self._save_state(new_vars)
                    new_vars = {k: v for k, v in new_vars.items() if k not in old_keys}
                except Exception:
                    pass

            # Detect matplotlib figures
            figures = self._collect_figures(SANDBOX_DIR)

            return stdout.strip(), stderr.strip(), success, elapsed, new_vars
        except subprocess.TimeoutExpired:
            return "", f"TimeoutError: exceeded {timeout}s limit", False, timeout*1000, {}
        except Exception as e:
            return "", str(e), False, 0, {}
        finally:
            try:
                os.unlink(fname)
            except Exception:
                pass

    def _run_bash(self, code: str, timeout: int) -> Tuple[str, str, bool, int]:
        t0 = time.monotonic()
        try:
            result = subprocess.run(
                ["bash", "-c", code],
                capture_output=True, text=True,
                timeout=timeout, cwd=str(SANDBOX_DIR),
            )
            elapsed = int((time.monotonic() - t0) * 1000)
            return (result.stdout.strip(), result.stderr.strip(),
                    result.returncode == 0, elapsed)
        except subprocess.TimeoutExpired:
            return "", f"Timeout: exceeded {timeout}s", False, timeout*1000
        except Exception as e:
            return "", str(e), False, 0

    def _collect_figures(self, directory: Path) -> List[str]:
        """Move any new .png/.jpg files to FIG_DIR and return their paths."""
        found = []
        for ext in ("*.png", "*.jpg", "*.svg"):
            for p in directory.glob(ext):
                if p.parent != FIG_DIR:
                    dest = FIG_DIR / f"{int(time.time())}_{p.name}"
                    try:
                        p.rename(dest)
                        found.append(str(dest))
                    except Exception:
                        found.append(str(p))
        return found

    # ── Public API ─────────────────────────────────────────────────────────

    def execute(self, code: str, language: str = "auto",
                timeout: int = 15, persist: bool = False) -> ExecutionResult:
        """Single execution — no auto-fix."""
        if language == "auto":
            language = self.detect_language(code)

        safe, hard, soft = _safety_check(code)
        if not safe:
            return ExecutionResult(
                code=code, output="",
                error=f"BLOCKED: dangerous operations detected: {hard}",
                success=False, runtime_ms=0, language=language,
            )

        if language == "bash":
            out, err, ok, ms = self._run_bash(code, timeout)
            result = ExecutionResult(code=code, output=out, error=err,
                                     success=ok, runtime_ms=ms, language=language)
        else:
            out, err, ok, ms, new_vars = self._run_python(code, timeout, persist)
            result = ExecutionResult(code=code, output=out, error=err,
                                     success=ok, runtime_ms=ms, language=language,
                                     new_vars=new_vars)

        self._session_runs += 1
        self._log_result(result)
        return result

    def execute_with_autofix(self, code: str, language: str = "auto",
                              timeout: int = 15, persist: bool = False,
                              max_attempts: int = 3) -> ExecutionResult:
        """
        Research-grade self-repair loop.
        On failure, DeepSeek-R1 reads the traceback and generates a fix.
        Mirrors the Reflexion / self-debugging paradigm.
        """
        if language == "auto":
            language = self.detect_language(code)

        safe, hard, soft = _safety_check(code)
        if not safe:
            return ExecutionResult(
                code=code, output="",
                error=f"BLOCKED: {hard}",
                success=False, runtime_ms=0, language=language,
            )

        current_code = code
        last_result  = None
        total_ms     = 0

        for attempt in range(1, max_attempts + 1):
            if language == "bash":
                out, err, ok, ms = self._run_bash(current_code, timeout)
                new_vars = {}
            else:
                out, err, ok, ms, new_vars = self._run_python(
                    current_code, timeout, persist)
            total_ms += ms

            if ok:
                result = ExecutionResult(
                    code=code, output=out, error=err,
                    success=True, runtime_ms=total_ms,
                    language=language, attempts=attempt,
                    fixed_code=current_code if attempt > 1 else "",
                    new_vars=new_vars,
                )
                self._log_result(result)
                return result

            last_result = (out, err, ok)

            # Don't attempt fix on last iteration
            if attempt == max_attempts or not self._groq:
                break

            # Ask DeepSeek-R1 to fix the code
            fixed = self._ai_fix(current_code, err or out, language, attempt)
            if not fixed or fixed == current_code:
                break
            current_code = fixed

        # All attempts failed
        out, err, ok = last_result or ("", "Unknown error", False)
        result = ExecutionResult(
            code=code, output=out, error=err,
            success=False, runtime_ms=total_ms,
            language=language, attempts=max_attempts,
            fixed_code=current_code if current_code != code else "",
        )
        self._log_result(result)
        return result

    def _ai_fix(self, code: str, error: str, language: str, attempt: int) -> str:
        """Ask the LLM to fix a failing code snippet given the error message."""
        if not self._groq:
            return ""
        prompt = (
            f"Fix this {language} code that produced an error.\n\n"
            f"Code:\n```{language}\n{code}\n```\n\n"
            f"Error (attempt {attempt}):\n{error[:800]}\n\n"
            "Return ONLY the corrected code, no explanation, no markdown fences."
        )
        fix = self._groq.chat(
            system=(
                "You are an expert debugging assistant. "
                "Return only the fixed code, nothing else. "
                "No markdown, no explanation."
            ),
            user=prompt,
            tier="code",
            max_tokens=1000,
        )
        # Strip markdown fences if model added them
        fix = re.sub(r"^```\w*\n?", "", fix.strip())
        fix = re.sub(r"\n?```$", "", fix)
        return fix.strip()

    # ── Convenience ────────────────────────────────────────────────────────

    def extract_code_blocks(self, text: str) -> List[Tuple[str, str]]:
        """
        Extract ```language ... ``` blocks from a markdown string.
        Returns list of (language, code) tuples.
        """
        pattern = r"```(\w*)\n(.*?)```"
        blocks  = re.findall(pattern, text, re.DOTALL)
        return [(lang or "python", code.strip()) for lang, code in blocks]

    def capability_description(self) -> str:
        return (
            "\n\nCode Execution: You can run Python and Bash code directly. "
            "Use /run to execute code. If code fails, I will attempt to fix and re-run "
            f"automatically (up to 3 attempts). {self.stats()}"
        )
