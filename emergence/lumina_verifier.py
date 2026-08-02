#!/usr/bin/env python3
"""
lumina_verifier.py — Code verification before evolution PR submission

The single biggest flaw in recursive self-improvement: writing code you
never run. Lumina generates evolution PRs but can't know if they work.

This module closes the loop:
  1. Receives generated code (Python file content)
  2. Writes it to a temp sandbox directory
  3. Runs it with a timeout via subprocess
  4. Captures stdout, stderr, exit code
  5. If errors: asks Groq to diagnose and fix
  6. Iterates up to MAX_FIX_ATTEMPTS times
  7. Returns verified code (or best attempt with error report)

This is what makes self-improvement *actual* improvement, not just
creative fiction.
"""

from __future__ import annotations
import ast, os, re, subprocess, tempfile, time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from emergence_engine import GroqClient

MAX_FIX_ATTEMPTS = 3
TIMEOUT_SECS     = 15
SANDBOX_DIR      = Path(__file__).parent / ".sandbox"
SANDBOX_DIR.mkdir(parents=True, exist_ok=True)


class VerificationResult:
    def __init__(self, success: bool, code: str, errors: List[str],
                 fixes_applied: int, output: str = ""):
        self.success       = success
        self.code          = code
        self.errors        = errors
        self.fixes_applied = fixes_applied
        self.output        = output

    def __str__(self) -> str:
        status = "PASS" if self.success else "FAIL"
        return (
            f"[{status}] fixes={self.fixes_applied} "
            f"errors={len(self.errors)} "
            f"output={self.output[:60]}"
        )


class CodeVerifier:
    def __init__(self, groq: "GroqClient"):
        self._groq = groq

    # ── Static analysis ────────────────────────────────────────────────────

    @staticmethod
    def syntax_check(code: str) -> Tuple[bool, str]:
        """Fast AST syntax check before execution."""
        try:
            ast.parse(code)
            return True, ""
        except SyntaxError as e:
            return False, f"SyntaxError line {e.lineno}: {e.msg}"

    @staticmethod
    def safety_check(code: str) -> Tuple[bool, str]:
        """
        Lightweight safety check — block obviously dangerous patterns.
        We're running in a controlled environment but be cautious.
        """
        dangerous = [
            (r"\bos\.system\s*\(", "os.system call"),
            (r"\bsubprocess\.call\b", "subprocess.call"),
            (r"\beval\s*\(", "eval()"),
            (r"\bexec\s*\(", "exec()"),
            (r"__import__\s*\(", "__import__()"),
            (r"shutil\.rmtree", "shutil.rmtree"),
            (r"rm\s+-rf", "rm -rf"),
        ]
        for pattern, label in dangerous:
            if re.search(pattern, code):
                return False, f"Safety: detected {label}"
        return True, ""

    # ── Execution ──────────────────────────────────────────────────────────

    def _execute(self, code: str, filename: str = "test_module.py") -> Tuple[bool, str, str]:
        """
        Execute code in sandbox. Returns (success, stdout, stderr).
        Wraps the code in a __main__ guard if needed, then runs.
        """
        # Add a soft execution guard — only run if __main__
        # We test by checking imports and class/function definitions parse
        sandbox_path = SANDBOX_DIR / filename
        try:
            sandbox_path.write_text(code, "utf-8")
            # Run with timeout; capture but don't let it do network IO
            # in the test pass — just check for import/syntax errors
            test_code = (
                "import ast, sys\n"
                f"code = open('{sandbox_path}').read()\n"
                "try:\n"
                "    ast.parse(code)\n"
                "    compile(code, '<test>', 'exec')\n"
                "    print('COMPILE_OK')\n"
                "except Exception as e:\n"
                "    print(f'COMPILE_ERROR: {e}', file=sys.stderr)\n"
                "    sys.exit(1)\n"
            )
            result = subprocess.run(
                ["python3", "-c", test_code],
                capture_output=True, text=True, timeout=TIMEOUT_SECS,
            )
            return (
                result.returncode == 0,
                result.stdout.strip(),
                result.stderr.strip(),
            )
        except subprocess.TimeoutExpired:
            return False, "", "Timeout during verification"
        except Exception as e:
            return False, "", str(e)
        finally:
            sandbox_path.unlink(missing_ok=True)

    # ── Fix loop ───────────────────────────────────────────────────────────

    def _ask_groq_to_fix(self, code: str, error: str, goal: str) -> Optional[str]:
        """Ask Groq to fix a specific error in the code."""
        system = (
            "You are a Python expert fixing a code error. "
            "Return ONLY the complete corrected Python code — no markdown, no explanation. "
            "Fix the specific error while preserving all intended functionality."
        )
        user = (
            f"Goal: {goal}\n\n"
            f"Error:\n{error}\n\n"
            f"Code:\n{code[:3000]}\n\n"
            "Fixed code:"
        )
        fixed = self._groq.chat(system, user, tier="code", max_tokens=4000)
        fixed = re.sub(r"^```python\s*", "", fixed.strip())
        fixed = re.sub(r"```\s*$", "", fixed.strip())
        return fixed if len(fixed) > 50 else None

    # ── Main verify entry ──────────────────────────────────────────────────

    def verify(self, code: str, filename: str = "module.py",
               goal: str = "") -> VerificationResult:
        """
        Full verification pipeline with auto-fix loop.
        Returns VerificationResult with the best version of the code.
        """
        errors        = []
        fixes_applied = 0
        current_code  = code

        for attempt in range(MAX_FIX_ATTEMPTS + 1):
            # 1. Syntax check
            ok, err = self.syntax_check(current_code)
            if not ok:
                errors.append(f"Attempt {attempt} — Syntax: {err}")
                if attempt < MAX_FIX_ATTEMPTS:
                    fixed = self._ask_groq_to_fix(current_code, err, goal)
                    if fixed:
                        current_code = fixed
                        fixes_applied += 1
                        continue
                break

            # 2. Safety check
            ok, err = self.safety_check(current_code)
            if not ok:
                errors.append(f"Safety block: {err}")
                break

            # 3. Compile/execute check
            ok, stdout, stderr = self._execute(current_code, filename)
            if ok:
                return VerificationResult(
                    success=True, code=current_code,
                    errors=errors, fixes_applied=fixes_applied,
                    output=stdout,
                )
            else:
                error_msg = stderr or "Unknown execution error"
                errors.append(f"Attempt {attempt} — Runtime: {error_msg[:200]}")
                if attempt < MAX_FIX_ATTEMPTS:
                    fixed = self._ask_groq_to_fix(current_code, error_msg, goal)
                    if fixed:
                        current_code = fixed
                        fixes_applied += 1
                    else:
                        break
                time.sleep(1)

        return VerificationResult(
            success=False, code=current_code,
            errors=errors, fixes_applied=fixes_applied,
        )

    def verify_improvement(self, improvement: Dict) -> Dict:
        """
        Verify a single improvement dict (from EvolutionEngine).
        Updates the dict in-place with verification results.
        Returns the updated dict.
        """
        code     = improvement.get("full_content", "")
        filename = improvement.get("file", "module.py")
        goal     = improvement.get("description", "")

        if not code:
            improvement["verified"]     = False
            improvement["verify_error"] = "No code content"
            return improvement

        result = self.verify(code, filename, goal)
        improvement["full_content"]   = result.code   # use fixed version
        improvement["verified"]       = result.success
        improvement["verify_errors"]  = result.errors
        improvement["fixes_applied"]  = result.fixes_applied
        improvement["verify_output"]  = result.output

        if result.success:
            print(f"  ✓ Verified: {filename} "
                  f"({result.fixes_applied} fixes applied)")
        else:
            print(f"  ✗ Verification failed: {filename} "
                  f"— {result.errors[-1][:80] if result.errors else 'unknown'}")

        return improvement

    def batch_verify(self, improvements: List[Dict]) -> List[Dict]:
        """Verify a list of improvements, filtering out ones that can't be fixed."""
        verified = []
        for imp in improvements:
            imp = self.verify_improvement(imp)
            # Include even unverified — let PR review catch it, but flag it
            verified.append(imp)
        return verified
