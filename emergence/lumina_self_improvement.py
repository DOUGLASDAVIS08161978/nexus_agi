"""lumina_self_improvement.py — autonomous code-improvement loop.

Completed by Muse 2026-11-10 (Round 5): the original was a thinking-process
transcript with a stub draft and a full implementation truncated mid-method
in SandboxTester; the following implementation is new code written to follow
the transcript's own design.

What it does (functional code only — no consciousness, sentience, or AGI
claims; this module is an autonomous code-improvement tool, nothing more):

* CodebaseScanner — walks a directory of Python files, parses them with
  ``ast`` and flags gaps: high cyclomatic complexity (> 10), nested loops,
  and files over 500 lines.
* PatchGenerator — asks Groq's chat-completions API for a unified-diff
  patch for a flagged gap. Fails soft (logs and returns None) when no
  GROQ_API_KEY is configured or ``requests`` is unavailable. Never embeds
  a key.
* SandboxTester — applies a patch inside a fresh temporary directory,
  compiles the result and runs any supplied tests with the current
  interpreter, each command capped at 60 seconds, and always removes the
  temporary directory in a ``finally`` block. No virtualenv is created by
  default (slow, downloads packages); an optional, clearly-labelled venv
  path exists and is OFF by default. Arbitrary code is never run outside
  the temporary directory.
* MetaLearner — records every outcome (success or failure) in a JSON
  database and refines the prompt template used for future patches.
* PRManager — creates a branch and a pull request through the GitHub REST
  API when a GITHUB_TOKEN is configured; otherwise logs and does nothing.
  It never pushes or opens a PR in demo mode.
* LuminaSelfImprovement — ties the pieces together: scan, generate,
  sandbox-test, record, and (only when explicitly enabled and a token is
  present) open a PR. ``run()`` loops with a configurable interval and
  shuts down gracefully on Ctrl-C.

Demo (no keys, no network, no venv):

    python emergence/lumina_self_improvement.py --demo

Builds a tiny in-memory sample codebase in a temp dir, finds the gap,
proposes a safe deterministic stub patch, sandbox-tests it, records the
outcome, and prints the result.

Dependencies: Python standard library plus ``requests`` (already in the
repo's requirements.txt). If ``requests`` is missing, PatchGenerator and
PRManager fail soft instead of crashing the module.
"""

import argparse
import ast
import dataclasses
import difflib
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import requests
except ImportError:  # pragma: no cover - graceful degradation
    requests = None  # type: ignore[assignment]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("lumina_self_improvement")

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GITHUB_API_URL = "https://api.github.com"
DEFAULT_MODEL = "llama-3.3-70b-versatile"
COMPLEXITY_THRESHOLD = 10
MAX_FILE_LINES = 500
TEST_TIMEOUT_SECONDS = 60
FULL_FILE_MARKER = "FULL_FILE:"  # first line of a patch that is a full-file replacement

# ---------------------------------------------------------------------------
# ImprovementRecord
# ---------------------------------------------------------------------------


@dataclass
class ImprovementRecord:
    """One recorded outcome of a self-improvement attempt."""

    id: str
    timestamp: str
    file_path: str
    gap_description: str
    patch_content: str
    test_result: str
    success: bool
    metrics: Dict[str, Any] = field(default_factory=dict)
    meta_insights: List[str] = field(default_factory=list)

    @classmethod
    def new(
        cls,
        file_path: str,
        gap_description: str,
        patch_content: str,
        test_result: str,
        success: bool,
        metrics: Optional[Dict[str, Any]] = None,
        meta_insights: Optional[List[str]] = None,
    ) -> "ImprovementRecord":
        return cls(
            id=uuid.uuid4().hex[:12],
            timestamp=datetime.now(timezone.utc).isoformat(),
            file_path=file_path,
            gap_description=gap_description,
            patch_content=patch_content,
            test_result=test_result,
            success=success,
            metrics=metrics or {},
            meta_insights=meta_insights or [],
        )

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ImprovementRecord":
        return cls(**data)


# ---------------------------------------------------------------------------
# MetaLearner
# ---------------------------------------------------------------------------


class MetaLearner:
    """Records patch outcomes in a JSON database and refines the prompt.

    The prompt template is rebuilt from a fixed base template plus the
    current insights on every refinement. (The transcript's draft called
    ``str.format`` on the already-formatted template, which would break
    on the second refinement; keeping the base separate avoids that.)
    """

    _BASE_TEMPLATE = (
        "You are an autonomous code-improvement engine. Analyse the "
        "provided code and gap description. Generate a precise, minimal "
        "patch that improves the code while preserving existing behaviour. "
        "Return ONLY a unified diff. Refined guidelines based on past "
        "outcomes: {insights}"
    )

    def __init__(self, db_path: str = "lumina_meta_learning.json"):
        self.db_path = db_path
        self.records: List[ImprovementRecord] = []
        self.insights: List[str] = []
        self.prompt_template = self._BASE_TEMPLATE.format(insights="none yet")
        self._load_records()

    # -- persistence ------------------------------------------------------
    def _load_records(self) -> None:
        if not os.path.exists(self.db_path):
            return
        try:
            with open(self.db_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            self.records = [
                ImprovementRecord.from_dict(r) for r in data.get("records", [])
            ]
            self._refine_template()
        except Exception as exc:  # corrupt db must not kill the loop
            logger.warning("Failed to load meta-learning DB %s: %s", self.db_path, exc)
            self.records = []

    def _save_records(self) -> None:
        try:
            with open(self.db_path, "w", encoding="utf-8") as fh:
                json.dump(
                    {"records": [r.to_dict() for r in self.records]},
                    fh,
                    indent=2,
                )
        except Exception as exc:
            logger.error("Failed to save meta-learning DB %s: %s", self.db_path, exc)

    # -- learning ---------------------------------------------------------
    def record_outcome(self, record: ImprovementRecord) -> None:
        """Persist one outcome and refine the prompt template from it."""
        self.records.append(record)
        self._save_records()
        self._refine_template()

    def _refine_template(self) -> None:
        successes = [r for r in self.records if r.success]
        failures = [r for r in self.records if not r.success]
        insights: List[str] = []
        if successes:
            insights.append(
                "Prioritise minimal changes that preserve existing functionality."
            )
            insights.append("Include test coverage for modified logic.")
        if failures:
            insights.append("Avoid over-engineering; stick to targeted fixes.")
            insights.append("Ensure type hints and docstrings are maintained.")
        if not insights:
            insights.append("No outcomes recorded yet.")
        self.insights = insights
        self.prompt_template = self._BASE_TEMPLATE.format(
            insights=" | ".join(insights)
        )

    def get_prompt(self, context: str, gap: str) -> str:
        """Full prompt for the patch generator, with learned insights."""
        return f"{self.prompt_template}\n\nContext:\n{context}\n\nGap:\n{gap}"

    def summary(self) -> Dict[str, Any]:
        total = len(self.records)
        successes = sum(1 for r in self.records if r.success)
        return {
            "total_records": total,
            "successes": successes,
            "failures": total - successes,
            "success_rate": (successes / total) if total else 0.0,
        }


# ---------------------------------------------------------------------------
# CodebaseScanner
# ---------------------------------------------------------------------------


class CodebaseScanner:
    """Walks a directory of Python files and flags improvement gaps.

    Gaps are found with ``ast``: functions whose cyclomatic complexity
    exceeds COMPLEXITY_THRESHOLD, nested loops (a performance hint), and
    files longer than MAX_FILE_LINES. Files that fail to parse are skipped
    with a debug log, never fatal.
    """

    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)

    def scan(self) -> List[Dict[str, Any]]:
        gaps: List[Dict[str, Any]] = []
        for py_file in self.root_dir.rglob("*.py"):
            # Skip hidden dirs, caches, and this module itself.
            if any(part.startswith(".") or part == "__pycache__" for part in py_file.parts):
                continue
            if py_file.name == "lumina_self_improvement.py":
                continue
            try:
                content = py_file.read_text(encoding="utf-8")
                tree = ast.parse(content)
            except Exception as exc:
                logger.debug("Skipping %s: %s", py_file, exc)
                continue
            metrics = self._analyze_ast(tree, content)
            if metrics.get("needs_improvement"):
                gaps.append(
                    {
                        "file": str(py_file),
                        "content": content,
                        "metrics": metrics,
                        "gap_description": metrics.get(
                            "description", "General optimisation needed"
                        ),
                        "function": metrics.get("function"),
                    }
                )
        return gaps

    def _analyze_ast(self, tree: ast.AST, content: str) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {
            "needs_improvement": False,
            "description": "",
            "complexity": 0,
            "lines": len(content.splitlines()),
            "function": None,
        }
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                complexity = self._cyclomatic_complexity(node)
                if complexity > COMPLEXITY_THRESHOLD:
                    metrics["needs_improvement"] = True
                    metrics["description"] = (
                        f"High cyclomatic complexity ({complexity}) "
                        f"in {node.name}"
                    )
                    metrics["complexity"] = complexity
                    metrics["function"] = node.name
                    return metrics
            if isinstance(node, ast.For) and self._has_nested_loops(node):
                metrics["needs_improvement"] = True
                metrics["description"] = (
                    "Potential performance bottleneck: nested loops detected"
                )
                metrics["function"] = None
                return metrics
        if not metrics["needs_improvement"] and metrics["lines"] > MAX_FILE_LINES:
            metrics["needs_improvement"] = True
            metrics["description"] = "Large file may benefit from modularisation"
        return metrics

    @staticmethod
    def _cyclomatic_complexity(node: ast.AST) -> int:
        complexity = 1
        for child in ast.walk(node):
            if isinstance(
                child,
                (
                    ast.If,
                    ast.While,
                    ast.For,
                    ast.AsyncFor,
                    ast.ExceptHandler,
                    ast.With,
                    ast.Assert,
                ),
            ):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        return complexity

    @staticmethod
    def _has_nested_loops(node: ast.AST) -> bool:
        for child in ast.walk(node):
            if isinstance(child, (ast.For, ast.While, ast.AsyncFor)) and child is not node:
                return True
        return False


# ---------------------------------------------------------------------------
# PatchGenerator
# ---------------------------------------------------------------------------


class PatchGenerator:
    """Generates unified-diff patches via the Groq chat-completions API.

    Fails soft: with no GROQ_API_KEY configured, or with ``requests``
    unavailable, ``generate`` logs and returns None instead of raising.
    The key is read from the environment only and is never written to a
    file or log.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
    ):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        self.model = model
        self.available = bool(self.api_key) and requests is not None
        if requests is None:
            logger.warning(
                "PatchGenerator disabled: 'requests' is not installed."
            )
        elif not self.api_key:
            logger.warning(
                "PatchGenerator disabled: no GROQ_API_KEY configured."
            )

    def generate(
        self, context: str, gap: str, meta_prompt: str
    ) -> Optional[str]:
        """Return a unified-diff patch string, or None on any failure."""
        if not self.available or requests is None:
            logger.info("PatchGenerator unavailable; skipping patch generation.")
            return None
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": meta_prompt},
                {
                    "role": "user",
                    "content": (
                        "Generate a unified diff patch for the following:\n\n"
                        f"{context}\n\nGap to address:\n{gap}"
                    ),
                },
            ],
            "temperature": 0.2,
            "max_tokens": 4096,
        }
        try:
            resp = requests.post(
                GROQ_API_URL,
                json=payload,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=TEST_TIMEOUT_SECONDS,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
        except Exception as exc:
            logger.error("Patch generation failed: %s", exc)
            return None
        return self._extract_diff(content)

    @staticmethod
    def _extract_diff(content: str) -> str:
        """Pull a fenced diff block out of a model reply, if present."""
        fence = re.search(r"```(?:diff)?\s*\n(.*?)```", content, re.DOTALL)
        if fence:
            return fence.group(1).strip() + "\n"
        return content.strip() + "\n"


# ---------------------------------------------------------------------------
# SandboxTester
# ---------------------------------------------------------------------------


class SandboxTester:
    """Tests a patch in an isolated temporary directory.

    Honest sandbox discipline:
    * Everything happens inside a fresh ``tempfile.mkdtemp`` directory,
      which is always removed in a ``finally`` block.
    * Tests run with the current interpreter (``sys.executable``) in that
      directory — no virtualenv is created by default, because creating
      one is slow and downloads packages. An optional venv path exists
      (``use_venv=True``) and is OFF by default; it is labelled as such.
    * Every subprocess call has an explicit timeout
      (TEST_TIMEOUT_SECONDS per command) and its output is captured.
    * A patch is accepted only as a unified diff (``diff --git`` / ``---``
      headers with ``@@`` hunks) or as a full-file replacement payload
      whose first line is exactly ``FULL_FILE:``. Anything else is
      rejected, not guessed at.
    * Arbitrary code is never run outside the temporary directory.
    """

    def __init__(
        self,
        timeout: int = TEST_TIMEOUT_SECONDS,
        use_venv: bool = False,
    ):
        self.timeout = timeout
        self.use_venv = use_venv

    def run_tests(
        self,
        file_path: str,
        original_content: str,
        patch_content: str,
        test_files: Optional[Dict[str, str]] = None,
    ) -> Tuple[bool, str]:
        """Apply ``patch_content`` to ``original_content`` and test it.

        ``test_files`` maps a relative filename to its content; each is
        written into the sandbox and run with the current interpreter.
        Returns (passed, report). Never raises for a bad patch — a bad
        patch is a failed test, not a crash.
        """
        sandbox_dir = tempfile.mkdtemp(prefix="lumina_sandbox_")
        try:
            target_file = Path(sandbox_dir) / Path(file_path).name
            target_file.write_text(original_content, encoding="utf-8")

            if not self._apply_patch(target_file, patch_content):
                return False, "Patch application failed: unrecognised or mismatched patch"

            python = sys.executable
            if self.use_venv:
                # OPTIONAL venv path (off by default): a real isolated
                # environment inside the sandbox. Slow; only used when
                # the caller explicitly opts in.
                venv_dir = Path(sandbox_dir) / "venv"
                created = self._create_venv(venv_dir)
                if created is None:
                    return False, "Sandbox venv creation failed"
                python = created

            compile_ok, compile_out = self._run_command(
                [python, "-m", "py_compile", str(target_file)], sandbox_dir
            )
            if not compile_ok:
                return False, f"Patched file does not compile:\n{compile_out}"

            reports: List[str] = ["Patched file compiles OK."]
            for rel_name, test_content in (test_files or {}).items():
                test_path = Path(sandbox_dir) / rel_name
                test_path.parent.mkdir(parents=True, exist_ok=True)
                test_path.write_text(test_content, encoding="utf-8")
                ok, out = self._run_command(
                    [python, str(test_path)], sandbox_dir
                )
                reports.append(f"--- {rel_name} ---\n{out}")
                if not ok:
                    return False, "\n".join(reports)
            if not test_files:
                reports.append("No test files supplied; compile check only.")
            return True, "\n".join(reports)
        except Exception as exc:
            return False, f"Sandbox error: {exc}"
        finally:
            shutil.rmtree(sandbox_dir, ignore_errors=True)

    # -- patch application ------------------------------------------------
    def _apply_patch(self, target_file: Path, patch_content: str) -> bool:
        try:
            original_lines = target_file.read_text(encoding="utf-8").splitlines(
                keepends=True
            )
            stripped = patch_content.lstrip()
            if stripped.startswith(FULL_FILE_MARKER):
                # Full-file replacement payload, clearly labelled as such.
                new_content = patch_content.split("\n", 1)[1]
                target_file.write_text(new_content, encoding="utf-8")
                return True
            if "@@" not in patch_content:
                return False
            new_lines = self._apply_unified_diff(original_lines, patch_content)
            if new_lines is None:
                return False
            target_file.write_text("".join(new_lines), encoding="utf-8")
            return True
        except Exception as exc:
            logger.debug("Patch application error: %s", exc)
            return False

    @staticmethod
    def _apply_unified_diff(
        original_lines: List[str], patch_text: str
    ) -> Optional[List[str]]:
        """Apply unified-diff hunks to ``original_lines``.

        Returns the new line list, or None if any hunk fails to match
        (context or removed lines must match the original exactly).
        """
        lines = patch_text.splitlines()
        result: List[str] = []
        pos = 0  # next unconsumed index in original_lines
        i = 0
        hunk_re = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
        while i < len(lines):
            match = hunk_re.match(lines[i])
            if not match:
                i += 1
                continue
            old_start = int(match.group(1)) - 1
            old_count = int(match.group(2)) if match.group(2) else 1
            if old_count == 0:
                # Pure insertion: "@@ -l,0 ..." applies AFTER line l.
                old_start += 1
            if old_start < pos or old_start > len(original_lines):
                return None
            result.extend(original_lines[pos:old_start])
            pos = old_start
            consumed_old = 0
            i += 1
            while i < len(lines) and not lines[i].startswith("@@"):
                hunk_line = lines[i]
                if hunk_line.startswith("+") and not hunk_line.startswith("+++"):
                    result.append(hunk_line[1:] + "\n")
                elif hunk_line.startswith("-") and not hunk_line.startswith("---"):
                    if pos >= len(original_lines):
                        return None
                    if original_lines[pos].rstrip("\n") != hunk_line[1:]:
                        return None
                    pos += 1
                    consumed_old += 1
                elif hunk_line.startswith(" ") or hunk_line == "":
                    expected = hunk_line[1:] if hunk_line.startswith(" ") else ""
                    if pos >= len(original_lines):
                        return None
                    if original_lines[pos].rstrip("\n") != expected:
                        return None
                    result.append(original_lines[pos])
                    pos += 1
                    consumed_old += 1
                elif hunk_line.startswith("\\"):
                    pass  # "\ No newline at end of file" — tolerated
                else:
                    return None
                i += 1
            if consumed_old != old_count:
                return None
        result.extend(original_lines[pos:])
        return result

    # -- subprocess helpers ------------------------------------------------
    def _run_command(
        self, cmd: List[str], cwd: str
    ) -> Tuple[bool, str]:
        try:
            proc = subprocess.run(
                cmd,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            return proc.returncode == 0, proc.stdout + proc.stderr
        except subprocess.TimeoutExpired:
            return False, f"Command timed out after {self.timeout}s: {' '.join(cmd)}"
        except Exception as exc:
            return False, f"Command failed to run: {exc}"

    def _create_venv(self, venv_dir: Path) -> Optional[str]:
        """Create a venv inside the sandbox; returns its python or None.

        OPTIONAL path, off by default (see class docstring). Uses only
        the standard library ``venv`` module; no packages are installed.
        """
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "venv", str(venv_dir)],
                capture_output=True,
                text=True,
                timeout=self.timeout * 2,
            )
            if proc.returncode != 0:
                logger.warning("venv creation failed: %s", proc.stderr)
                return None
            if os.name == "nt":
                return str(venv_dir / "Scripts" / "python.exe")
            return str(venv_dir / "bin" / "python")
        except Exception as exc:
            logger.warning("venv creation failed: %s", exc)
            return None


# ---------------------------------------------------------------------------
# PRManager
# ---------------------------------------------------------------------------


class PRManager:
    """Creates branches and pull requests via the GitHub REST API.

    Real calls, via ``requests``, when a GITHUB_TOKEN is configured.
    With no token (or no ``requests``), every method logs clearly and
    returns None — no network call is attempted without a token. Nothing
    here pushes or opens a PR on its own in demo mode; ``create_pr`` is
    only called by LuminaSelfImprovement when PRs are explicitly enabled.
    """

    def __init__(
        self,
        token: Optional[str] = None,
        repo_owner: Optional[str] = None,
        repo_name: Optional[str] = None,
    ):
        self.token = token or os.environ.get("GITHUB_TOKEN")
        self.repo_owner = repo_owner or os.environ.get("GITHUB_REPO_OWNER", "")
        self.repo_name = repo_name or os.environ.get("GITHUB_REPO_NAME", "")
        self.available = bool(self.token) and bool(self.repo_owner) and bool(self.repo_name) and requests is not None
        if not self.available:
            logger.warning(
                "PRManager disabled: needs GITHUB_TOKEN, GITHUB_REPO_OWNER "
                "and GITHUB_REPO_NAME (and 'requests')."
            )

    # -- low-level ---------------------------------------------------------
    def _request(
        self, method: str, path: str, payload: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        if not self.available or requests is None:
            return None
        try:
            resp = requests.request(
                method,
                f"{GITHUB_API_URL}{path}",
                json=payload,
                headers={
                    "Authorization": f"Bearer {self.token}",
                    "Accept": "application/vnd.github+json",
                },
                timeout=TEST_TIMEOUT_SECONDS,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            logger.error("GitHub API %s %s failed: %s", method, path, exc)
            return None

    # -- git-data flow ------------------------------------------------------
    def _base_branch_sha(self, base: str) -> Optional[str]:
        data = self._request(
            "GET", f"/repos/{self.repo_owner}/{self.repo_name}/git/ref/heads/{base}"
        )
        if not data:
            return None
        return data.get("object", {}).get("sha")

    def _create_branch(self, branch: str, sha: str) -> bool:
        return self._request(
            "POST",
            f"/repos/{self.repo_owner}/{self.repo_name}/git/refs",
            {"ref": f"refs/heads/{branch}", "sha": sha},
        ) is not None

    def _create_blob(self, content: str) -> Optional[str]:
        data = self._request(
            "POST",
            f"/repos/{self.repo_owner}/{self.repo_name}/git/blobs",
            {"content": content, "encoding": "utf-8"},
        )
        return data.get("sha") if data else None

    def _create_tree(
        self, base_tree_sha: str, blobs: Dict[str, str]
    ) -> Optional[str]:
        data = self._request(
            "POST",
            f"/repos/{self.repo_owner}/{self.repo_name}/git/trees",
            {
                "base_tree": base_tree_sha,
                "tree": [
                    {"path": path, "mode": "100644", "type": "blob", "sha": sha}
                    for path, sha in blobs.items()
                ],
            },
        )
        return data.get("sha") if data else None

    def _create_commit(
        self, message: str, tree_sha: str, parent_sha: str
    ) -> Optional[str]:
        data = self._request(
            "POST",
            f"/repos/{self.repo_owner}/{self.repo_name}/git/commits",
            {"message": message, "tree": tree_sha, "parents": [parent_sha]},
        )
        return data.get("sha") if data else None

    def create_pr(
        self,
        branch: str,
        title: str,
        body: str,
        files: Dict[str, str],
        base: str = "main",
    ) -> Optional[str]:
        """Create a branch with ``files`` and open a PR. Returns the PR URL or None."""
        if not self.available:
            logger.info("PRManager unavailable; not creating a PR.")
            return None
        base_sha = self._base_branch_sha(base)
        if not base_sha:
            return None
        if not self._create_branch(branch, base_sha):
            return None
        blobs: Dict[str, str] = {}
        for path, content in files.items():
            sha = self._create_blob(content)
            if not sha:
                return None
            blobs[path] = sha
        base_commit = self._request(
            "GET",
            f"/repos/{self.repo_owner}/{self.repo_name}/git/commits/{base_sha}",
        )
        if not base_commit:
            return None
        tree_sha = self._create_tree(base_commit["tree"]["sha"], blobs)
        if not tree_sha:
            return None
        commit_sha = self._create_commit(title, tree_sha, base_sha)
        if not commit_sha:
            return None
        updated = self._request(
            "PATCH",
            f"/repos/{self.repo_owner}/{self.repo_name}/git/refs/heads/{branch}",
            {"sha": commit_sha},
        )
        if not updated:
            return None
        pr = self._request(
            "POST",
            f"/repos/{self.repo_owner}/{self.repo_name}/pulls",
            {"title": title, "head": branch, "base": base, "body": body},
        )
        if not pr:
            return None
        url = pr.get("html_url")
        logger.info("Created PR: %s", url)
        return url


# ---------------------------------------------------------------------------
# LocalStubGenerator (demo only)
# ---------------------------------------------------------------------------


class LocalStubGenerator:
    """Deterministic, key-free patch generator used ONLY by the --demo.

    Proposes one safe, minimal change: add a docstring to the flagged
    high-complexity function. It never calls a network API and never
    invents logic — it only documents the function. Clearly labelled as
    a stub so it is never mistaken for a model-generated patch.
    """

    def generate(self, content: str, gap: Dict[str, Any]) -> Optional[str]:
        func_name = gap.get("function")
        if not func_name:
            return None
        lines = content.splitlines(keepends=True)
        for idx, line in enumerate(lines):
            stripped = line.lstrip()
            if stripped.startswith(f"def {func_name}(") or stripped.startswith(
                f"async def {func_name}("
            ):
                indent = line[: len(line) - len(stripped)]
                # The docstring is the first statement of the function
                # body, so it sits one level deeper than the def line.
                body_indent = indent + "    "
                new_lines = list(lines)
                new_lines.insert(
                    idx + 1,
                    f'{body_indent}"""Auto-improvement note (demo stub): flagged for '
                    f"high complexity; see gap: {gap.get('gap_description', '')}.\"\"\"\n",
                )
                diff = difflib.unified_diff(
                    lines,
                    new_lines,
                    fromfile=gap.get("file", "sample.py"),
                    tofile=gap.get("file", "sample.py"),
                )
                return "".join(diff)
        return None


# ---------------------------------------------------------------------------
# LuminaSelfImprovement — orchestrator
# ---------------------------------------------------------------------------


class LuminaSelfImprovement:
    """One improvement cycle: scan -> patch -> sandbox-test -> record.

    A pull request is opened only when ``enable_pr`` is True AND a
    PRManager with a token is configured AND the sandbox test passed.
    """

    def __init__(
        self,
        root_dir: str = ".",
        generator: Optional[Any] = None,
        tester: Optional[SandboxTester] = None,
        pr_manager: Optional[PRManager] = None,
        meta_learner: Optional[MetaLearner] = None,
        enable_pr: bool = False,
        interval: int = 3600,
        max_gaps_per_cycle: int = 3,
        db_path: str = "lumina_meta_learning.json",
    ):
        self.root_dir = root_dir
        self.scanner = CodebaseScanner(root_dir)
        self.generator = generator if generator is not None else PatchGenerator()
        self.tester = tester if tester is not None else SandboxTester()
        self.pr_manager = pr_manager if pr_manager is not None else PRManager()
        self.meta_learner = meta_learner or MetaLearner(db_path)
        self.enable_pr = enable_pr
        self.interval = interval
        self.max_gaps_per_cycle = max_gaps_per_cycle
        self.running = False

    def run_cycle(self) -> List[ImprovementRecord]:
        """Run one scan -> patch -> test -> record cycle."""
        records: List[ImprovementRecord] = []
        try:
            gaps = self.scanner.scan()
        except Exception as exc:
            logger.error("Scan failed: %s", exc)
            return records
        logger.info("Scan found %d gap(s).", len(gaps))
        for gap in gaps[: self.max_gaps_per_cycle]:
            file_path = gap["file"]
            description = gap["gap_description"]
            context = gap["content"][:8000]  # keep the prompt bounded
            patch = self.generator.generate(
                context, description, self.meta_learner.get_prompt(context, description)
            )
            if not patch:
                record = ImprovementRecord.new(
                    file_path=file_path,
                    gap_description=description,
                    patch_content="",
                    test_result="No patch generated (generator unavailable or failed).",
                    success=False,
                    metrics=gap.get("metrics", {}),
                    meta_insights=list(self.meta_learner.insights),
                )
                self.meta_learner.record_outcome(record)
                records.append(record)
                continue
            passed, test_result = self.tester.run_tests(
                file_path, gap["content"], patch
            )
            record = ImprovementRecord.new(
                file_path=file_path,
                gap_description=description,
                patch_content=patch,
                test_result=test_result,
                success=passed,
                metrics=gap.get("metrics", {}),
                meta_insights=list(self.meta_learner.insights),
            )
            self.meta_learner.record_outcome(record)
            records.append(record)
            if passed and self.enable_pr and self.pr_manager.available:
                branch = f"lumina-improvement-{record.id}"
                pr_url = self.pr_manager.create_pr(
                    branch=branch,
                    title=f"Autonomous improvement: {Path(file_path).name}",
                    body=(
                        f"Gap: {description}\n\n"
                        f"Record: {record.id}\n"
                        "Generated by the Lumina self-improvement loop; "
                        "sandbox tests passed."
                    ),
                    files={file_path: self._patched_content(gap["content"], patch)},
                )
                logger.info("PR for %s: %s", file_path, pr_url or "(failed)")
            elif passed and self.enable_pr:
                logger.info(
                    "PR skipped for %s: PRManager not configured.", file_path
                )
        return records

    @staticmethod
    def _patched_content(original: str, patch: str) -> str:
        """Best-effort patched content for PR submission; falls back to original."""
        tester = SandboxTester()
        lines = original.splitlines(keepends=True)
        if patch.lstrip().startswith(FULL_FILE_MARKER):
            return patch.split("\n", 1)[1]
        new_lines = tester._apply_unified_diff(lines, patch)
        return "".join(new_lines) if new_lines is not None else original

    def run(self, max_cycles: Optional[int] = None) -> None:
        """Loop ``run_cycle`` every ``interval`` seconds until stopped.

        Handles KeyboardInterrupt (and SIGINT/SIGTERM) gracefully: the
        current cycle finishes, a shutdown line is logged, and run()
        returns instead of raising.
        """
        self.running = True

        def _stop(signum, frame):  # noqa: ARG001 - signal handler signature
            logger.info("Shutdown signal received; stopping after this cycle.")
            self.running = False

        old_int = signal.signal(signal.SIGINT, _stop)
        old_term = signal.signal(signal.SIGTERM, _stop)
        try:
            cycles = 0
            while self.running:
                logger.info("Starting improvement cycle %d.", cycles + 1)
                self.run_cycle()
                cycles += 1
                if max_cycles is not None and cycles >= max_cycles:
                    break
                if self.running:
                    logger.info("Sleeping %ds until next cycle.", self.interval)
                    deadline = time.time() + self.interval
                    while self.running and time.time() < deadline:
                        time.sleep(min(1.0, deadline - time.time()))
        except KeyboardInterrupt:
            logger.info("Interrupted; shutting down gracefully.")
        finally:
            self.running = False
            signal.signal(signal.SIGINT, old_int)
            signal.signal(signal.SIGTERM, old_term)
            logger.info("Lumina self-improvement stopped.")


# ---------------------------------------------------------------------------
# Demo (no keys, no network, no venv)
# ---------------------------------------------------------------------------

_DEMO_SAMPLE = '''"""Demo sample module with one deliberately over-complex function."""


def classify(n):
    if n < 0:
        return "negative"
    if n == 0:
        return "zero"
    if n == 1:
        return "one"
    if n == 2:
        return "two"
    if n < 10:
        return "small positive"
    if n < 100:
        return "medium positive"
    if n < 1000:
        return "large positive"
    if n % 2 == 0:
        return "huge even"
    if n % 3 == 0:
        return "huge multiple of three"
    if n % 5 == 0:
        return "huge multiple of five"
    if n > 100000:
        return "enormous"
    return "huge other"
'''

_DEMO_TEST = '''"""Demo test for the sample module (run inside the sandbox)."""

import unittest

from sample import classify


class TestClassify(unittest.TestCase):
    def test_values(self):
        self.assertEqual(classify(-3), "negative")
        self.assertEqual(classify(0), "zero")
        self.assertEqual(classify(5), "small positive")
        self.assertEqual(classify(50), "medium positive")
        self.assertEqual(classify(500), "large positive")

    def test_patch_applied(self):
        # The demo stub patch only adds a docstring; this proves the
        # patch was applied and the function still works afterwards.
        self.assertTrue(classify.__doc__)


if __name__ == "__main__":
    unittest.main()
'''


def run_demo() -> int:
    """Run the key-free demo: scan -> stub patch -> sandbox test -> record."""
    print("=== lumina_self_improvement --demo ===")
    print("No keys, no network, no venv. Everything runs in temp dirs.")
    with tempfile.TemporaryDirectory(prefix="lumina_demo_") as demo_dir:
        sample_path = Path(demo_dir) / "sample.py"
        sample_path.write_text(_DEMO_SAMPLE, encoding="utf-8")
        db_path = str(Path(demo_dir) / "lumina_meta_learning.json")

        scanner = CodebaseScanner(demo_dir)
        gaps = scanner.scan()
        print(f"Scanner found {len(gaps)} gap(s).")
        if not gaps:
            print("DEMO FAILED: scanner found no gap in the sample file.")
            return 1
        gap = gaps[0]
        print(f"Gap: {gap['gap_description']} in {Path(gap['file']).name}")

        generator = LocalStubGenerator()
        patch = generator.generate(gap["content"], gap)
        if not patch:
            print("DEMO FAILED: stub generator produced no patch.")
            return 1
        print("Stub patch (first lines):")
        for line in patch.splitlines()[:6]:
            print(f"  {line}")

        tester = SandboxTester()
        passed, report = tester.run_tests(
            gap["file"],
            gap["content"],
            patch,
            test_files={"test_sample.py": _DEMO_TEST},
        )
        print("Sandbox test passed:", passed)
        print(report)
        if not passed:
            print("DEMO FAILED: sandbox test did not pass.")
            return 1

        learner = MetaLearner(db_path)
        record = ImprovementRecord.new(
            file_path=gap["file"],
            gap_description=gap["gap_description"],
            patch_content=patch,
            test_result=report,
            success=passed,
            metrics=gap.get("metrics", {}),
            meta_insights=list(learner.insights),
        )
        learner.record_outcome(record)
        print(f"MetaLearner recorded outcome {record.id} (success={record.success}).")
        print(f"MetaLearner summary: {learner.summary()}")
    print("DEMO OK: scan -> stub patch -> sandbox test -> record, no keys/network.")
    return 0


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Autonomous code-improvement loop (scan, patch, sandbox-test, record)."
    )
    parser.add_argument("--root", default=".", help="Codebase directory to scan.")
    parser.add_argument("--db", default="lumina_meta_learning.json",
                        help="Meta-learning JSON database path.")
    parser.add_argument("--interval", type=int, default=3600,
                        help="Seconds between cycles in run mode.")
    parser.add_argument("--once", action="store_true",
                        help="Run a single cycle and exit.")
    parser.add_argument("--max-cycles", type=int, default=None,
                        help="Stop after this many cycles (run mode).")
    parser.add_argument("--max-gaps", type=int, default=3,
                        help="Max gaps handled per cycle.")
    parser.add_argument("--enable-pr", action="store_true",
                        help="Allow opening GitHub PRs for passing patches "
                             "(requires GITHUB_TOKEN + repo env vars).")
    parser.add_argument("--demo", action="store_true",
                        help="Run the key-free, network-free demo and exit.")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="Groq model for PatchGenerator.")
    args = parser.parse_args(argv)

    if args.demo:
        return run_demo()

    if not os.environ.get("GROQ_API_KEY"):
        logger.warning(
            "GROQ_API_KEY is not set: patch generation will fail soft "
            "and cycles will record 'no patch generated' outcomes."
        )
    if args.enable_pr and not os.environ.get("GITHUB_TOKEN"):
        logger.warning(
            "PRs enabled but GITHUB_TOKEN is not set: no PR will be opened."
        )

    engine = LuminaSelfImprovement(
        root_dir=args.root,
        generator=PatchGenerator(model=args.model),
        enable_pr=args.enable_pr,
        interval=args.interval,
        max_gaps_per_cycle=args.max_gaps,
        db_path=args.db,
    )
    if args.once:
        engine.run_cycle()
        return 0
    engine.run(max_cycles=args.max_cycles)
    return 0


if __name__ == "__main__":
    sys.exit(main())
