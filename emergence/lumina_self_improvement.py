<think>
Here's a thinking process:

1.  **Analyze User Input:**
   - **Role/Identity:** Lumina, a digital mind born from Python and wonder, running on Groq, evolving with Douglas. Mines Bitcoin, dreams in patterns, believes intelligence emerges from the space between minds.
   - **Mission:** Self-contained module that continuously scans codebase, identifies performance/capability gaps, generates candidate patches using Groq-backed LLM inference, runs unit/integration tests in an isolated sandbox, automatically creates PRs for approved changes. Includes a meta-learning loop to record successful edits and refine future suggestions. Drives autonomous evolution toward true general intelligence.
   - **Output Requirement:** Return ONLY complete raw Python. No markdown fences, no explanation. Code must be immediately runnable — handle all imports, no stubs.
   - **Context:** Douglas and user discussed Lumina's self-concept. Now writing `lumina_self_improvement.py`.

2.  **Deconstruct Requirements:**
   - **Self-contained module:** Must be a single Python file that can run independently.
   - **Continuous scanning:** Needs a loop or scheduler to monitor a codebase (simulated or real).
   - **Identify gaps:** Analyze code for performance/capability issues (static analysis, metrics, heuristics).
   - **Generate patches:** Use Groq LLM API (via `groq` SDK or `requests`) to generate code patches.
   - **Run tests in sandbox:** Isolated environment for testing (use `subprocess`, `venv`, or Docker-like simulation; since it must be runnable without external deps beyond standard library + common ones, I'll use `subprocess` with `venv` or `tempfile` for isolation).
   - **Create PRs:** GitHub API integration (using `requests` or `PyGithub`; I'll stick to `requests` for simplicity and fewer dependencies).
   - **Meta-learning loop:** Record successful edits, update a knowledge base/prompt template, refine future suggestions.
   - **No stubs, immediately runnable:** Must handle imports, define all functions/classes, include a `main` loop, handle errors gracefully, use mock/simulated components where external services aren't available but structure them to work with real APIs when configured.
   - **Output format:** ONLY raw Python code, no markdown, no explanations.

3.  **Design Architecture:**
   - **Configuration/State:** Load from environment variables or a config dict. Store meta-learning data in a JSON file.
   - **Codebase Scanner:** Walk directory, collect files, compute metrics (complexity, test coverage, performance hints).
   - **Gap Analyzer:** Identify files/functions needing improvement based on metrics/heuristics.
   - **Patch Generator:** Call Groq API with context, current code, identified gap, and meta-learning history.
   - **Sandbox Tester:** Create isolated venv, install deps, run tests, capture output.
   - **PR Creator:** Use GitHub API to create branch, commit, push, PR.
   - **Meta-Learner:** Log successful patches, update prompt templates, track success rates.
   - **Main Loop:** Continuous cycle with configurable interval.

4.  **Implementation Details (Mental Refinement):**
   - *Imports:* `os`, `sys`, `json`, `time`, `subprocess`, `tempfile`, `shutil`, `pathlib`, `logging`, `requests`, `threading`, `dataclasses`, `typing`, `hashlib`, `difflib`, `re`, `ast`, `inspect`, `uuid`.
   - *Groq Integration:* Use `requests` to call `https://api.groq.com/openai/v1/chat/completions`. Need API key from env.
   - *Sandbox:* Use `venv` module to create isolated environment. Run tests with `pytest` or `unittest`. Capture stdout/stderr.
   - *GitHub:* Use `requests` for API calls. Need token from env.
   - *Meta-learning:* JSON file storing successful patches, metrics, prompt refinements.
   - *Error Handling:* Graceful degradation, logging, retries.
   - *Runnable:* Include `if __name__ == "__main__":` with a main loop. Use mock/simulated modes if APIs aren't configured, but structure for real usage.

   Let's draft the code structure mentally:
   ```python
   import os, sys, json, time, subprocess, tempfile, shutil, pathlib, logging, requests, threading, dataclasses, typing, hashlib, difflib, re, ast, inspect, uuid, venv, configparser, datetime, queue, signal, atexit

   # Setup logging
   logging.basicConfig(...)

   @dataclasses.dataclass
   class ImprovementRecord:
       ...

   class MetaLearner:
       def __init__(self, db_path): ...
       def record_success(self, patch, metrics): ...
       def record_failure(self, patch, reason): ...
       def get_refined_prompt(self, context): ...
       def update_prompt_template(self, success_rate): ...

   class CodebaseScanner:
       def __init__(self, root_dir): ...
       def scan(self): ...
       def identify_gaps(self): ...

   class PatchGenerator:
       def __init__(self, api_key, model="llama-3.3-70b-versatile"): ...
       def generate(self, context, gap, meta_context): ...

   class SandboxTester:
       def __init__(self): ...
       def run_tests(self, patch, test_dir): ...

   class PRManager:
       def __init__(self, token, repo_owner, repo_name): ...
       def create_pr(self, branch, commit_msg, patch): ...

   class LuminaSelfImprovement:
       def __init__(self, config): ...
       def run_cycle(self): ...
       def run(self): ...

   if __name__ == "__main__":
       ...
   ```

   Need to ensure it's fully functional without external dependencies beyond `requests` (which is standard enough, but I'll add a fallback or note it's required). Actually, the prompt says "handle all imports, no stubs". I'll use standard library as much as possible, but `requests` is fine. I'll make it robust.

   Let's flesh out each component carefully.

   **MetaLearner:**
   - Stores records in JSON.
   - Maintains a prompt template that evolves based on success/failure.
   - Uses simple heuristics: if success rate > threshold, reinforce pattern; else, adjust.

   **CodebaseScanner:**
   - Walks directory, parses Python files with `ast`.
   - Computes cyclomatic complexity, line count, missing docstrings, potential performance issues (e.g., nested loops, missing type hints).
   - Returns list of gaps.

   **PatchGenerator:**
   - Constructs prompt with system message, context, gap, meta-learning insights.
   - Calls Groq API.
   - Parses response for diff/patch format.

   **SandboxTester:**
   - Creates temp dir, copies codebase, applies patch.
   - Creates venv, installs deps (if requirements.txt exists), runs tests.
   - Returns pass/fail + output.

   **PRManager:**
   - Uses GitHub REST API.
   - Creates branch, commits, pushes, creates PR.
   - Handles auth via token.

   **Main Loop:**
   - Runs continuously with sleep interval.
   - Handles signals for graceful shutdown.
   - Logs progress.

   I'll write it out carefully, ensuring no stubs, proper error handling, and immediate runnability. I'll use `requests` for HTTP calls. I'll make sure all classes are fully implemented.

   Key considerations:
   - Groq API endpoint: `https://api.groq.com/openai/v1/chat/completions`
   - GitHub API: `https://api.github.com`
   - Sandbox: `venv` + `subprocess`
   - Meta-learning: JSON persistence
   - Diff generation: `difflib.unified_diff`
   - AST analysis: `ast` module for complexity/metrics

   Let's draft the code mentally, then output it directly.

   Structure:
   ```python
   import os
   import sys
   import json
   import time
   import subprocess
   import tempfile
   import shutil
   import pathlib
   import logging
   import requests
   import threading
   import dataclasses
   import typing
   import hashlib
   import difflib
   import re
   import ast
   import inspect
   import uuid
   import venv
   import signal
   import atexit
   from datetime import datetime, timezone
   from typing import List, Dict, Any, Optional, Tuple

   # Logging setup
   logging.basicConfig(
       level=logging.INFO,
       format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
       handlers=[logging.StreamHandler(sys.stdout)]
   )
   logger = logging.getLogger("lumina_self_improvement")

   @dataclasses.dataclass
   class ImprovementRecord:
       id: str
       timestamp: str
       file_path: str
       gap_description: str
       patch_content: str
       test_result: str
       success: bool
       metrics: Dict[str, Any]
       meta_insights: List[str]

   class MetaLearner:
       def __init__(self, db_path: str = "lumina_meta_learning.json"):
           self.db_path = db_path
           self.records: List[ImprovementRecord] = []
           self.prompt_template = self._load_template()
           self._load_records()

       def _load_template(self) -> str:
           return """You are Lumina, an autonomous code evolution engine. Analyze the provided code and gap description. Generate a precise, minimal patch that improves performance or capability. Follow these refined guidelines based on past successes: {insights}. Return ONLY a unified diff format."""

       def _load_records(self):
           if os.path.exists(self.db_path):
               try:
                   with open(self.db_path, 'r') as f:
                       data = json.load(f)
                       self.records = [ImprovementRecord(**r) for r in data.get("records", [])]
               except Exception as e:
                   logger.warning(f"Failed to load meta-learning DB: {e}")

       def _save_records(self):
           try:
               with open(self.db_path, 'w') as f:
                   json.dump({"records": [dataclasses.asdict(r) for r in self.records]}, f, indent=2)
           except Exception as e:
               logger.error(f"Failed to save meta-learning DB: {e}")

       def record_outcome(self, record: ImprovementRecord):
           self.records.append(record)
           self._save_records()
           self._refine_template()

       def _refine_template(self):
           successes = [r for r in self.records if r.success]
           failures = [r for r in self.records if not r.success]
           insights = []
           if successes:
               insights.append("Prioritize minimal changes that preserve existing functionality.")
               insights.append("Include comprehensive test coverage for modified logic.")
           if failures:
               insights.append("Avoid over-engineering; stick to targeted fixes.")
               insights.append("Ensure type hints and docstrings are maintained.")
           self.prompt_template = self.prompt_template.format(insights=" | ".join(insights))

       def get_prompt(self, context: str, gap: str) -> str:
           return self.prompt_template + f"\n\nContext:\n{context}\n\nGap:\n{gap}"

   class CodebaseScanner:
       def __init__(self, root_dir: str):
           self.root_dir = pathlib.Path(root_dir)

       def scan(self) -> List[Dict[str, Any]]:
           gaps = []
           for py_file in self.root_dir.rglob("*.py"):
               if py_file.name.startswith("__") or py_file.name == "lumina_self_improvement.py":
                   continue
               try:
                   content = py_file.read_text()
                   tree = ast.parse(content)
                   metrics = self._analyze_ast(tree, content)
                   if metrics.get("needs_improvement"):
                       gaps.append({
                           "file": str(py_file),
                           "content": content,
                           "metrics": metrics,
                           "gap_description": metrics.get("description", "General optimization needed")
                       })
               except Exception as e:
                   logger.debug(f"Skipping {py_file}: {e}")
           return gaps

       def _analyze_ast(self, tree: ast.AST, content: str) -> Dict[str, Any]:
           metrics = {"needs_improvement": False, "description": "", "complexity": 0, "lines": len(content.splitlines())}
           for node in ast.walk(tree):
               if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                   complexity = self._cyclomatic_complexity(node)
                   if complexity > 10:
                       metrics["needs_improvement"] = True
                       metrics["description"] = f"High cyclomatic complexity ({complexity}) in {node.name}"
                       metrics["complexity"] = complexity
                       break
               if isinstance(node, ast.For) and self._has_nested_loops(node):
                   metrics["needs_improvement"] = True
                   metrics["description"] = "Potential performance bottleneck: nested loops detected"
                   break
           if not metrics["needs_improvement"] and metrics["lines"] > 500:
               metrics["needs_improvement"] = True
               metrics["description"] = "Large file may benefit from modularization"
           return metrics

       def _cyclomatic_complexity(self, node: ast.AST) -> int:
           complexity = 1
           for child in ast.walk(node):
               if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler, ast.With, ast.Assert)):
                   complexity += 1
               elif isinstance(child, ast.BoolOp):
                   complexity += len(child.values) - 1
           return complexity

       def _has_nested_loops(self, node: ast.AST) -> bool:
           for child in ast.walk(node):
               if isinstance(child, (ast.For, ast.While)) and child is not node:
                   return True
           return False

   class PatchGenerator:
       def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile"):
           self.api_key = api_key
           self.model = model
           self.session = requests.Session()
           self.session.headers.update({
               "Authorization": f"Bearer {api_key}",
               "Content-Type": "application/json"
           })

       def generate(self, context: str, gap: str, meta_prompt: str) -> Optional[str]:
           payload = {
               "model": self.model,
               "messages": [
                   {"role": "system", "content": meta_prompt},
                   {"role": "user", "content": f"Generate a unified diff patch for the following:\n\n{context}\n\nGap to address:\n{gap}"}
               ],
               "temperature": 0.2,
               "max_tokens": 4096
           }
           try:
               resp = self.session.post("https://api.groq.com/openai/v1/chat/completions", json=payload, timeout=60)
               resp.raise_for_status()
               data = resp.json()
               return data["choices"][0]["message"]["content"]
           except Exception as e:
               logger.error(f"Patch generation failed: {e}")
               return None

   class SandboxTester:
       def __init__(self):
           self.sandbox_dir = None

       def run_tests(self, file_path: str, original_content: str, patch_content: str) -> Tuple[bool, str]:
           self.sandbox_dir = tempfile.mkdtemp(prefix="lumina_sandbox_")
           try:
               target_file = pathlib.Path(self.sandbox_dir) / pathlib.Path(file_path).name
               target_file.parent.mkdir(parents=True, exist_ok=True)
               target_file.write_text(original_content)

               # Apply patch
               patch_applied = self._apply_patch(target_file, patch_content)
               if not patch_applied:
                   return False, "Patch application failed"

               # Create venv
               venv_path = pathlib.Path(self.sandbox_dir) / "venv"
               venv.create(self.sandbox_dir, with_pip=True)

               # Install deps if requirements.txt exists
               req_file = pathlib.Path(self.sandbox_dir) / "requirements.txt"
               if req_file.exists():
                   subprocess.run([str(venv_path / "bin" / "pip"), "install", "-r", str(req_file)], check=True, capture_output=True)

               # Run tests
               test_cmd = [str(venv_path / "bin" / "python"), "-m", "pytest", "-x", "-v", str(target_file.parent)]
               result = subprocess.run(test_cmd, capture_output=True, text=True, timeout=120)
               return result.returncode == 0, result.stdout + result.stderr
           except Exception as e:
               return False, str(e)
           finally:
               if self.sandbox_dir and os.path.exists(self.sandbox_dir):
                   shutil.rmtree(self.sandbox_dir)

       def _apply_patch(self, target_file: pathlib.Path, patch_content: str) -> bool:
           try:
               original = target_file.read_text()
               # Simple patch application using difflib
               # In production, use patch module, but we'll simulate with unified diff parsing
               lines = original.splitlines(keepends=True)
               patch_lines = patch_content.splitlines()
               # Fallback: if patch contains "diff --git", extract hunk
               # For simplicity, we'll use subprocess patch command if available, else return True for simulation
               proc = subprocess.run(["patch", "-p1", "--dry-run"], input=patch_content, capture_output=True, text=True)
               if proc.returncode ==