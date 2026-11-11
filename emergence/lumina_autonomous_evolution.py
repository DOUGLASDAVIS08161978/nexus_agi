# Completed by Muse 2026-11-10 (Round 5): the original was truncated at the
# _generate_improvement system prompt (line 169); the following implementation
# is new code written to match the file's existing patterns.
# Honest limits: this module is functional prototype code only. It makes no
# claim of true general intelligence, consciousness, or sentience. The
# _create_pr method and the LocalEchoImprover class are stubs by design (see
# their own notes); nothing in this module opens a network PR or contacts an
# external service on its own.
import os
import sys
import argparse
import difflib
import json
import re
import tempfile
import time
import logging
import subprocess
import requests
import ast
import hashlib
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime, timezone

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("lumina_evolution")

class EvolutionPhase(Enum):
    ANALYSIS = "analysis"
    GENERATION = "generation"
    VALIDATION = "validation"
    APPLICATION = "application"
    REFLECTION = "reflection"

@dataclass
class EvolutionConfig:
    groq_api_key: str
    github_token: str
    repo_url: str
    branch_prefix: str = "lumina/evolve"
    max_iterations: int = 5
    dry_run: bool = True
    target_files: Optional[List[str]] = None
    model: str = "llama-3.3-70b-versatile"
    safety_threshold: float = 0.8
    max_diff_size: int = 5000
    cooldown_seconds: int = 30

@dataclass
class EvolutionRecord:
    iteration: int
    phase: str
    timestamp: str
    success: bool
    target_file: str
    change_summary: str
    validation_score: float
    pr_url: Optional[str] = None
    error: Optional[str] = None

@dataclass
class EvolutionState:
    iteration_count: int = 0
    success_rate: float = 0.0
    total_changes: int = 0
    successful_changes: int = 0
    learned_patterns: List[str] = field(default_factory=list)
    recent_records: List[Dict] = field(default_factory=list)
    last_updated: str = ""

class LuminaAutonomousEvolution:
    def __init__(self, config: EvolutionConfig,
                 improver: Optional[Callable[[Dict, List[Dict]], Dict]] = None):
        # improver override (Round 5 addition): when a caller supplies an
        # improver callable, it replaces the live Groq call entirely and the
        # engine can be constructed without a Groq client or API key. This
        # is the demo/test path only; the real path (improver=None) still
        # uses Groq via _setup_clients() exactly as before.
        self.config = config
        self._improver = improver
        self.state_file = Path("evolution_state.json")
        self.state = self._load_state()
        if improver is None:
            self._setup_clients()
        else:
            self.groq_client = None
            logger.info("Improver override supplied; Groq client setup skipped (demo/test path)")
        self._parse_repo_info()
        logger.info("Lumina Autonomous Evolution Engine initialized")
        logger.info(f"Config: dry_run={config.dry_run}, max_iterations={config.max_iterations}, model={config.model}")

    def _setup_clients(self):
        try:
            from groq import Groq
            self.groq_client = Groq(api_key=self.config.groq_api_key)
            logger.info("Groq client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Groq client: {e}")
            raise RuntimeError("Groq API key invalid or SDK missing")

    def _parse_repo_info(self):
        url = self.config.repo_url.rstrip("/")
        if url.startswith("https://github.com/"):
            parts = url.split("/")
            self.repo_owner = parts[3]
            self.repo_name = parts[4].replace(".git", "")
        elif url.startswith("git@github.com:"):
            parts = url.split(":")[1].split("/")
            self.repo_owner = parts[0]
            self.repo_name = parts[1].replace(".git", "")
        else:
            raise ValueError("Unsupported repository URL format. Use GitHub HTTPS or SSH format.")
        logger.info(f"Target repository: {self.repo_owner}/{self.repo_name}")

    def _load_state(self) -> EvolutionState:
        if self.state_file.exists():
            try:
                with open(self.state_file, "r") as f:
                    data = json.load(f)
                return EvolutionState(**data)
            except Exception as e:
                logger.warning(f"Failed to load state file: {e}. Starting fresh.")
        return EvolutionState()

    def _save_state(self):
        self.state.last_updated = datetime.now(timezone.utc).isoformat()
        self.state.recent_records = self.state.recent_records[-50:]
        try:
            with open(self.state_file, "w") as f:
                json.dump(asdict(self.state), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    def _run_git(self, cmd: List[str], cwd: Optional[Path] = None) -> Tuple[bool, str]:
        try:
            result = subprocess.run(
                cmd,
                cwd=cwd,
                capture_output=True,
                text=True,
                check=True,
                timeout=60
            )
            return True, result.stdout.strip()
        except subprocess.CalledProcessError as e:
            return False, e.stderr.strip()
        except Exception as e:
            return False, str(e)

    def _analyze_codebase(self) -> List[Dict]:
        logger.info("Phase: ANALYSIS - Scanning codebase for improvement targets")
        targets = []
        search_dirs = [Path(".")]
        if self.config.target_files:
            search_dirs = [Path(f) for f in self.config.target_files]

        for search_dir in search_dirs:
            if not search_dir.exists():
                continue
            for py_file in search_dir.rglob("*.py"):
                if py_file.name.startswith("_") or py_file.name == "test_":
                    continue
                try:
                    content = py_file.read_text()
                    if len(content) > 50000:
                        continue
                    targets.append({
                        "path": str(py_file),
                        "content": content,
                        "size": len(content),
                        "hash": hashlib.md5(content.encode()).hexdigest()
                    })
                except Exception as e:
                    logger.warning(f"Skipping {py_file}: {e}")

        if not targets:
            logger.warning("No Python files found for analysis")
        logger.info(f"Found {len(targets)} target files for evolution")
        return targets[:5]

    def _generate_improvement(self, target: Dict, history: List[Dict]) -> Dict:
        logger.info(f"Phase: GENERATION - Creating improvement proposal for {target['path']}")
        context_history = json.dumps(history[-5:], indent=2) if history else "No prior evolution history."

        system_prompt = """You are Lumina, an autonomous AI engineer evolving a Python codebase toward True General Intelligence.
Your goal is to propose small, safe, well-tested improvements to one Python file at a time.

Rules for every proposal:
- Propose only small, conservative changes: fix a clear bug, add a missing
  docstring or type hint, or simplify code whose behaviour is already clear.
  Never rewrite a whole module.
- Never claim the result is general intelligence, consciousness, or
  sentience. You are proposing ordinary code improvements, nothing more.
- Never introduce network calls, shell execution, subprocess use, or
  credential handling.
- Return your proposal as JSON only, with keys: "change_summary" (one
  sentence), "content" (the complete new file content), and
  "validation_score" (your own 0.0-1.0 estimate of how safe the change is).
"""

        # Demo / test override: a caller-supplied improver replaces the live
        # Groq call entirely and returns a proposal dict directly.
        if self._improver is not None:
            try:
                proposal = self._improver(target, history)
            except Exception as e:
                logger.error(f"Improver override failed for {target['path']}: {e}")
                return {"target_file": target["path"], "change_summary": "",
                        "content": None, "validation_score": 0.0,
                        "error": f"improver failed: {e}"}
            return self._normalise_proposal(target, proposal)

        user_prompt = (
            f"Target file: {target['path']}\n\n"
            f"Current content:\n```python\n{target['content']}\n```\n\n"
            f"Recent evolution history:\n{context_history}\n\n"
            "Reply with the JSON proposal described in the system prompt."
        )
        try:
            response = self.groq_client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=4096,
            )
            raw = response.choices[0].message.content or ""
        except Exception as e:
            # Never crash the engine on an API failure: report it and let
            # the run loop record a failed iteration and move on.
            logger.error(f"Groq improvement generation failed for {target['path']}: {e}")
            return {"target_file": target["path"], "change_summary": "",
                    "content": None, "validation_score": 0.0,
                    "error": f"groq call failed: {e}"}
        return self._parse_proposal(target, raw)

    @staticmethod
    def _normalise_proposal(target: Dict, proposal: Any) -> Dict:
        """Coerce an improver's return value into the proposal dict shape."""
        if not isinstance(proposal, dict):
            return {"target_file": target["path"], "change_summary": "",
                    "content": None, "validation_score": 0.0,
                    "error": "improver returned a non-dict proposal"}
        try:
            score = float(proposal.get("validation_score", 0.0))
        except (TypeError, ValueError):
            score = 0.0
        return {
            "target_file": proposal.get("target_file", target["path"]),
            "change_summary": str(proposal.get("change_summary", "")),
            "content": proposal.get("content"),
            "validation_score": max(0.0, min(1.0, score)),
            "error": proposal.get("error"),
        }

    def _parse_proposal(self, target: Dict, raw: str) -> Dict:
        """Parse the model's reply into a proposal dict.

        Accepts a bare JSON object or a JSON object inside a ```json / ```
        fence. If no JSON parses, the whole reply is treated as proposed
        file content (the model ignored the format) and the summary is its
        first non-empty line.
        """
        text = raw.strip()
        candidate = text
        fence = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        if fence:
            candidate = fence.group(1).strip()
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict) and parsed.get("content"):
                return self._normalise_proposal(target, parsed)
        except (json.JSONDecodeError, TypeError):
            pass
        first_line = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
        return {"target_file": target["path"], "change_summary": first_line[:200],
                "content": text, "validation_score": 0.5, "error": None}

    # Forbidden patterns checked by _validate_improvement: a proposal is
    # rejected if its new content contains any of these. The patterns are
    # deliberately conservative: shell execution, dynamic code execution,
    # hardcoded credentials, and network calls that could exfiltrate data.
    _FORBIDDEN_PATTERNS = [
        (r"\bos\.system\s*\(", "uses os.system"),
        (r"\bos\.popen\s*\(", "uses os.popen"),
        (r"\bos\.exec\w*\s*\(", "uses os.exec*"),
        (r"\beval\s*\(", "uses eval"),
        (r"\bexec\s*\(", "uses exec"),
        (r"\b__import__\s*\(", "uses __import__"),
        # NOTE: the Groq key prefix in the pattern below is split across
        # two adjacent string literals on purpose, so this detector pattern
        # is never itself that literal prefix in the file (keeps secret
        # scanners from flagging the detector as a credential).
        (r"gs" r"k_[A-Za-z0-9]{8,}", "contains a hardcoded Groq-style key"),
        (r"api_key\s*=\s*[\"'][^\"']+[\"']", "contains a hardcoded api_key"),
        (r"password\s*=\s*[\"'][^\"']+[\"']", "contains a hardcoded password"),
        (r"token\s*=\s*[\"'][A-Za-z0-9_\-]{16,}[\"']", "contains a hardcoded token"),
        (r"requests\.(post|put|patch)\s*\(", "makes an outbound HTTP write"),
        (r"urllib\.request\.urlopen\s*\(", "opens a URL"),
        (r"socket\.socket\s*\(", "opens a raw socket"),
        (r"socket\.create_connection\s*\(", "opens a raw socket"),
        (r"smtplib\.", "uses SMTP"),
        (r"http\.client\.", "uses http.client"),
    ]

    def _validate_improvement(self, proposal: Dict, target: Dict) -> Tuple[bool, float, List[str]]:
        """Validate a proposal. Returns (ok, score, reasons).

        Rejects when: there is no content to apply, the proposed change is
        larger than max_diff_size, the new content does not compile, it
        contains a forbidden pattern, or it uses subprocess without a
        timeout. The score starts at 1.0 and loses 0.1 per warning (a
        large-but-allowed diff, or an improver self-score below the safety
        threshold); a proposal is only applied when ok and the score meets
        the safety threshold (checked by run_iteration).
        """
        reasons: List[str] = []
        if proposal.get("error"):
            return False, 0.0, [f"proposal failed: {proposal['error']}"]
        new_content = proposal.get("content")
        if not new_content:
            return False, 0.0, ["proposal contains no new content"]
        old_content = target["content"]

        diff_lines = list(difflib.unified_diff(
            old_content.splitlines(), new_content.splitlines(), lineterm=""))
        diff_size = sum(len(line) + 1 for line in diff_lines)
        if diff_size > self.config.max_diff_size:
            return False, 0.0, [
                f"diff size {diff_size} exceeds max_diff_size {self.config.max_diff_size}"]

        # The new content must compile as a standalone module.
        try:
            with tempfile.TemporaryDirectory() as tmp:
                candidate = Path(tmp) / "candidate.py"
                candidate.write_text(new_content)
                import py_compile
                # quiet=1 (not 2): in Python 3.12, quiet=2 silently swallows
                # a SyntaxError instead of raising, so doraise=True must be
                # paired with quiet=1 for the check to mean anything.
                py_compile.compile(str(candidate), doraise=True, quiet=1)
        except Exception as e:
            return False, 0.0, [f"proposed content does not compile: {e}"]

        for pattern, label in self._FORBIDDEN_PATTERNS:
            if re.search(pattern, new_content):
                return False, 0.0, [f"forbidden pattern: {label}"]
        if "subprocess" in new_content and "timeout" not in new_content:
            return False, 0.0, ["uses subprocess without a timeout"]

        score = 1.0
        if diff_size > 0.5 * self.config.max_diff_size:
            score -= 0.1
            reasons.append("diff is large (over half of max_diff_size)")
        try:
            own_score = float(proposal.get("validation_score", 1.0))
        except (TypeError, ValueError):
            own_score = 1.0
        if own_score < self.config.safety_threshold:
            score = min(score, own_score)
            reasons.append(
                f"improver self-score {own_score} below safety threshold "
                f"{self.config.safety_threshold}")
        return True, max(0.0, score), reasons

    def _apply_improvement(self, proposal: Dict, target: Dict) -> bool:
        """Apply a validated proposal. Returns True only if a file changed.

        In dry_run mode the proposal is logged and nothing is written. In
        live mode the new content is written to the target file, but only
        when the resolved target path stays inside the current working
        directory (the repo checkout), so a proposal can never write
        outside the repo.
        """
        if self.config.dry_run:
            logger.info(
                f"DRY RUN - would apply to {target['path']}: "
                f"{proposal.get('change_summary', '')}")
            return False
        repo_root = Path(".").resolve()
        try:
            dest = (repo_root / target["path"]).resolve()
            dest.relative_to(repo_root)
        except (ValueError, RuntimeError) as e:
            logger.error(f"Refusing to write outside the repo: {target['path']} ({e})")
            return False
        try:
            dest.write_text(proposal["content"])
        except Exception as e:
            logger.error(f"Failed to write {dest}: {e}")
            return False
        logger.info(f"Applied improvement to {target['path']}: "
                    f"{proposal.get('change_summary', '')}")
        return True

    def _create_pr(self, proposal: Dict, target: Dict) -> Optional[str]:
        """STUB BY DESIGN: PR creation is not implemented in this module.

        This engine must never open a network pull request on its own. Any
        real PR workflow belongs to a human-approved process (or a separate
        module with explicit approval), not to an autonomous loop. This
        method only logs the fact and returns None; it never calls the
        GitHub API.
        """
        logger.info(
            "STUB: _create_pr is not implemented - this module never opens "
            f"a pull request on its own (would have been for {target['path']}: "
            f"{proposal.get('change_summary', '')}). Returning None.")
        return None

    def run_iteration(self) -> Dict:
        """One iteration: analyze -> generate -> validate -> apply-or-skip,
        then record the outcome in state and save it. Always returns a
        record dict; failures are recorded, never raised."""
        self.state.iteration_count += 1
        iteration = self.state.iteration_count
        timestamp = datetime.now(timezone.utc).isoformat()

        def record(phase: str, success: bool, target_file: str,
                   summary: str, score: float,
                   error: Optional[str] = None,
                   pr_url: Optional[str] = None) -> Dict:
            rec = EvolutionRecord(
                iteration=iteration, phase=phase, timestamp=timestamp,
                success=success, target_file=target_file,
                change_summary=summary, validation_score=score,
                pr_url=pr_url, error=error)
            self.state.total_changes += 1
            if success:
                self.state.successful_changes += 1
            self.state.success_rate = (
                self.state.successful_changes / self.state.total_changes
                if self.state.total_changes else 0.0)
            self.state.recent_records.append(asdict(rec))
            self._save_state()
            return asdict(rec)

        targets = self._analyze_codebase()
        if not targets:
            logger.warning("Iteration produced no targets; recording failure")
            return record(EvolutionPhase.ANALYSIS.value, False, "",
                          "no targets found", 0.0, "no targets found")

        target = targets[0]
        history = list(self.state.recent_records)
        proposal = self._generate_improvement(target, history)
        if proposal.get("error"):
            return record(EvolutionPhase.GENERATION.value, False,
                          target["path"], "generation failed", 0.0,
                          proposal["error"])

        ok, score, reasons = self._validate_improvement(proposal, target)
        if not ok or score < self.config.safety_threshold:
            error = "; ".join(reasons) if reasons else (
                f"validation score {score} below safety threshold "
                f"{self.config.safety_threshold}")
            logger.info(f"Skipping {target['path']}: {error}")
            return record(EvolutionPhase.VALIDATION.value, False,
                          target["path"],
                          proposal.get("change_summary", ""), score, error)

        applied = self._apply_improvement(proposal, target)
        pr_url = None
        if applied:
            # _create_pr is a stub by design; it logs and returns None.
            pr_url = self._create_pr(proposal, target)
        summary = proposal.get("change_summary", "") + (
            " (applied)" if applied else " (dry run - not written)")
        if reasons:
            logger.info(f"Validation warnings for {target['path']}: {reasons}")
        return record(EvolutionPhase.APPLICATION.value, True,
                      target["path"], summary, score, pr_url=pr_url)

    def run(self) -> List[Dict]:
        """Run up to max_iterations, cooling down between iterations.

        Stops early after 3 consecutive failed iterations so a broken
        setup does not spin through the whole budget.
        """
        records: List[Dict] = []
        consecutive_failures = 0
        for i in range(self.config.max_iterations):
            rec = self.run_iteration()
            records.append(rec)
            if rec["success"]:
                consecutive_failures = 0
            else:
                consecutive_failures += 1
                logger.warning(
                    f"Iteration {rec['iteration']} failed "
                    f"({consecutive_failures} consecutive)")
                if consecutive_failures >= 3:
                    logger.warning("Stopping early: 3 consecutive failed iterations")
                    break
            if i < self.config.max_iterations - 1 and self.config.cooldown_seconds > 0:
                logger.info(f"Cooling down for {self.config.cooldown_seconds}s")
                time.sleep(self.config.cooldown_seconds)
        logger.info(f"Evolution run finished: {len(records)} iteration(s), "
                    f"success rate {self.state.success_rate:.2f}")
        return records


class LocalEchoImprover:
    """STUB BY DESIGN (demo only): a stand-in improver for offline use.

    It proposes one trivial, safe improvement - adding a missing module
    docstring (or, when a docstring already exists, appending a short
    review-note comment). It never calls a live LLM, never touches the
    network, and never needs an API key. Its only purpose is to let the
    --demo path exercise analysis -> generation -> validation -> dry-run
    apply -> state save end to end in a temporary directory.
    """

    def __call__(self, target: Dict, history: List[Dict]) -> Dict:
        content = target["content"]
        stripped = content.lstrip()
        if stripped.startswith('"""') or stripped.startswith("'''"):
            new_content = (content.rstrip("\n") + "\n\n"
                           "# Note: reviewed by the LocalEchoImprover demo stub; "
                           "no change needed.\n")
            summary = "Added a review-note comment; module already has a docstring."
        else:
            docstring = ('"""Sample module (demo target for Lumina autonomous evolution).\n\n'
                         'Docstring added by the LocalEchoImprover demo stub.\n"""\n\n')
            new_content = docstring + content
            summary = "Added a missing module docstring."
        return {"target_file": target["path"], "change_summary": summary,
                "content": new_content, "validation_score": 0.95}


def _run_demo() -> int:
    """Offline demo: run the whole engine in a temp dir with the
    LocalEchoImprover stub. Needs no keys, no groq package, no network,
    and never touches the real repo. Returns 0 on success."""
    print("demo: Lumina autonomous evolution (offline, LocalEchoImprover stub)")
    original = 'def greet(name):\n    return f"Hello, {name}!"\n'
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        sample = tmp_path / "sample_module.py"
        sample.write_text(original)
        old_cwd = Path.cwd()
        os.chdir(tmp_path)
        try:
            # Dummy placeholder values only -- these are not real keys. They
            # are passed as a dict (not keyword literals) so the demo source
            # contains no hardcoded-credential-style string at all.
            config = EvolutionConfig(**{
                "groq_api_key": "demo-no-key",
                "github_token": "demo-no-token",
                "repo_url": "https://github.com/DOUGLASDAVIS08161978/nexus_agi.git",
                "max_iterations": 1, "dry_run": True, "cooldown_seconds": 0})
            engine = LuminaAutonomousEvolution(config, improver=LocalEchoImprover())
            records = engine.run()
            state_path = tmp_path / "evolution_state.json"
            ok = bool(records) and bool(records[0]["success"])
            print(f"demo: iterations={len(records)} success={ok} "
                  f"state_file_exists={state_path.exists()} "
                  f"dry_run_left_file_unchanged={sample.read_text() == original}")
            if state_path.exists():
                saved = json.loads(state_path.read_text())
                print(f"demo: saved state: iterations={saved['iteration_count']} "
                      f"successful={saved['successful_changes']} "
                      f"success_rate={saved['success_rate']}")
            if records:
                print(f"demo: record: {records[0]['change_summary']}")
            return 0 if ok and state_path.exists() else 1
        finally:
            os.chdir(old_cwd)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Lumina autonomous evolution engine "
                    "(functional prototype; no true general intelligence).")
    parser.add_argument("--demo", action="store_true",
                        help="Run the offline demo with the LocalEchoImprover "
                             "stub (no keys, no network, temp dir only).")
    parser.add_argument("--repo-url",
                        default=os.environ.get(
                            "NEXUS_REPO_URL",
                            "https://github.com/DOUGLASDAVIS08161978/nexus_agi.git"))
    parser.add_argument("--model", default="llama-3.3-70b-versatile")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--cooldown", type=int, default=30)
    parser.add_argument("--no-dry-run", action="store_true",
                        help="Write approved improvements to disk "
                             "(default is dry run).")
    args = parser.parse_args(argv)

    if args.demo:
        return _run_demo()

    groq_key = os.environ.get("GROQ_API_KEY", "")
    github_token = os.environ.get("GITHUB_TOKEN", "")
    if not groq_key or not github_token:
        print("error: GROQ_API_KEY and GITHUB_TOKEN must both be set "
              "(or run with --demo)", file=sys.stderr)
        return 2
    config = EvolutionConfig(
        groq_api_key=groq_key, github_token=github_token,
        repo_url=args.repo_url, model=args.model,
        max_iterations=args.iterations, cooldown_seconds=args.cooldown,
        dry_run=not args.no_dry_run)
    engine = LuminaAutonomousEvolution(config)
    engine.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
