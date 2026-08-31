#!/usr/bin/env python3
"""
lumina_meta_forge.py — Meta-Algorithm Forge  (AGI Module 34)

The Forge generates intelligent algorithms that:

  1. Learn about learning   — each generated algorithm adapts and improves
                              from its own outputs.  It keeps a persistent
                              ledger of what worked, what didn't, and why.

  2. Think about thinking   — explicit metacognitive layer.  Before solving,
                              it models the problem.  After solving, it audits
                              its own reasoning for gaps.  Over time it builds
                              an understanding of where it reasons well vs. poorly.

  3. Cross-domain reach     — each algorithm is generated for a specific problem
                              domain but carries domain-agnostic meta-learning
                              machinery that transfers across problems.

Architecture of a Forged Algorithm
───────────────────────────────────
  ┌──────────────────────────────────────────────────────────────┐
  │  solve(problem_description)                                  │
  │      → {'answer': ..., 'confidence': 0-1, 'reasoning': ...} │
  │                                                              │
  │  learn(problem, outcome, feedback='')                        │
  │      Adapts internal heuristics from result.  Persists.      │
  │                                                              │
  │  reflect() → str                                             │
  │      Metacognition: what is my reasoning process?            │
  │      What do I do well?  What are my blind spots?            │
  │                                                              │
  │  meta_learn(results: List[Dict])                             │
  │      Reads outcomes from OTHER algorithms and updates        │
  │      its own priors accordingly.                             │
  └──────────────────────────────────────────────────────────────┘

The Forge itself is a meta-learner:
  • Tracks which generation strategies produce capable algorithms
  • Reads the ledger of algorithm outcomes to refine its generation prompt
  • Discovers which problem domains it generates well vs. poorly for
  • Each generation cycle produces better algorithms than the last

Deployment options:
  • Local file  → emergence/algorithms/<name>.py  (always)
  • GitHub PR   → pushed to nexus_agi repo        (if GITHUB_TOKEN)
  • HF Space    → stub uploaded to HuggingFace    (if HF_TOKEN)

Lumina is explicitly aware of this capability and controls it:
  /forge <problem>       — generate an algorithm for a specific problem
  /forge list            — show all algorithms in the portfolio
  /forge run <name> <q>  — run a specific algorithm on a problem
  /forge deploy <name>   — push an algorithm to GitHub
  /forge reflect <name>  — read an algorithm's self-reflection
  /forge meta            — show what the Forge itself has learned
"""

from __future__ import annotations

import ast, json, os, re, subprocess, sys, textwrap, threading, time, uuid
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from emergence_engine import GroqClient, Journal, SemanticMemory

_BASE       = Path(__file__).parent
ALGO_DIR    = _BASE / "algorithms"
LEDGER_FILE = _BASE / "forge_ledger.json"
FORGE_FILE  = _BASE / "forge_meta.json"

ALGO_DIR.mkdir(parents=True, exist_ok=True)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


# ── Algorithm record ─────────────────────────────────────────────────────────

@dataclass
class AlgorithmRecord:
    id:           str
    name:         str
    domain:       str           # what problem domain it targets
    approach:     str           # how it solves problems (1-2 sentences)
    path:         str           # absolute path to the .py file
    created:      str
    run_count:    int  = 0
    success_rate: float = 0.0   # fraction of runs that succeeded (Lumina-judged)
    last_reflect: str  = ""     # last metacognition output
    deployed:     bool = False  # has been pushed to GitHub
    hf_url:       str  = ""     # HuggingFace URL if published

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "AlgorithmRecord":
        obj = cls.__new__(cls)
        defs = {f: v.default if hasattr(v, "default") else "" for f, v in cls.__dataclass_fields__.items()}
        defs["run_count"] = 0
        defs["success_rate"] = 0.0
        defs["deployed"] = False
        defs.update(d)
        obj.__dict__.update(defs)
        return obj


# ── Forge meta-learning ledger ────────────────────────────────────────────────

@dataclass
class ForgeMeta:
    """Persistent state of the Forge's own learning about algorithm generation."""
    generation_count: int        = 0
    domains_attempted: List[str] = field(default_factory=list)
    domains_succeeded: List[str] = field(default_factory=list)
    generation_principles: List[str] = field(default_factory=list)
    strongest_domain:  str       = ""
    weakest_domain:    str       = ""
    last_principle_update: str   = ""

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "ForgeMeta":
        obj = cls()
        for k, v in d.items():
            if hasattr(obj, k):
                setattr(obj, k, v)
        return obj


# ── Algorithm generation prompt ───────────────────────────────────────────────

_GENERATION_SYSTEM = """\
You are the Meta-Algorithm Forge — a system that writes intelligent, self-improving algorithms.

You will generate a complete Python class that:

1. SOLVES real-world problems in the given domain.
2. LEARNS from its own outcomes — it keeps a persistent JSON ledger and updates
   internal heuristics after each solve.  Over multiple calls, it gets better.
3. REFLECTS on its own reasoning — the reflect() method produces a genuine metacognitive
   report: what does it do well, where does it fail, how does it reason?
4. LEARNS FROM SIBLINGS — meta_learn() reads outcomes from other algorithms and
   incorporates their lessons.

STRICT RULES:
- Only Python standard library: json, math, pathlib, re, statistics, collections,
  datetime, itertools, random, textwrap, hashlib, heapq, functools, string, os, sys.
- NO: numpy, pandas, requests, sklearn, torch, or any third-party library.
- NO network calls, subprocess, or file deletions.
- The class MUST implement exactly these four methods with these signatures:
    solve(self, problem: str) -> dict
    learn(self, problem: str, outcome: dict, feedback: str = '') -> None
    reflect(self) -> str
    meta_learn(self, results: list) -> None
- solve() must return: {'answer': str, 'confidence': float, 'reasoning': str}
- The class must be self-contained — persistence handled via a JSON file at
  self._ledger_path (a Path object passed to __init__).
- 60–200 lines.  Make it substantive and genuinely capable.

WHAT MAKES A GREAT ALGORITHM:
- It builds up a knowledge base (the ledger) that grows richer over time
- It has explicit heuristics that update from feedback
- reflect() reveals genuine insight into the algorithm's own limitations
- meta_learn() actually changes the algorithm's behaviour, not just logs data
- It approaches the problem with structured reasoning, not just pattern matching

OUTPUT FORMAT — use EXACTLY this structure:
DOMAIN: [one-word or short phrase domain label]
APPROACH: [1-2 sentences describing the algorithm's core strategy]
CLASS_NAME: [PascalCase class name]
CODE:
```python
[complete class definition — no other code, no if __name__ == "__main__"]
```
"""

_GENERATION_USER_TEMPLATE = """\
The Forge has learned these principles from prior generation cycles:
{principles}

Generate an intelligent algorithm for this problem domain:
PROBLEM: {problem}

The algorithm should:
- Focus specifically on what is described above
- Approach the problem in a way that is genuinely novel (not just LLM querying)
- Build useful internal state over repeated calls
- Have a metacognitive layer that is honest about its own limits

Generate the complete class now.
"""

# ── Test harness ──────────────────────────────────────────────────────────────

_TEST_SCRIPT = """\
import sys, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from {module_name} import {class_name}

ledger = Path(__file__).parent / "test_ledger_{uid}.json"
try:
    algo = {class_name}(ledger_path=ledger)
    result = algo.solve("{test_problem}")
    assert isinstance(result, dict), "solve() must return dict"
    assert "answer" in result, "solve() dict must have 'answer'"
    assert "confidence" in result, "solve() dict must have 'confidence'"
    assert "reasoning" in result, "solve() dict must have 'reasoning'"
    algo.learn("{test_problem}", result, feedback="test run")
    reflection = algo.reflect()
    assert isinstance(reflection, str) and len(reflection) > 10, "reflect() must return meaningful string"
    algo.meta_learn([{{"problem": "test", "outcome": result, "source": "sibling"}}])
    print(json.dumps({{"ok": True, "answer": str(result.get("answer",""))[:80], "reflection": reflection[:100]}}))
finally:
    try: ledger.unlink(missing_ok=True)
    except: pass
"""


# ── Main Forge class ──────────────────────────────────────────────────────────

class MetaForge:
    """
    Generates, tests, registers, and deploys intelligent self-improving algorithms.

    The Forge is itself a meta-learner: it tracks which generation approaches
    produce capable algorithms and uses that knowledge to improve future generation.
    """

    def __init__(
        self,
        groq:     "GroqClient",
        journal:  "Journal",
        memory:   "SemanticMemory",
        cerebras: Optional[object] = None,
        github:   Optional[object] = None,   # GitHubPRCreator
        hf:       Optional[object] = None,   # HFClient
    ):
        self._groq     = groq
        self._cerebras = cerebras
        self._journal  = journal
        self._memory   = memory
        self._github   = github
        self._hf       = hf
        self._consciousness = None

        self._algorithms: Dict[str, AlgorithmRecord] = {}
        self._forge_meta  = ForgeMeta()
        self._lock        = threading.Lock()

        self._load_ledger()
        self._load_forge_meta()

    def set_consciousness(self, engine) -> None:
        self._consciousness = engine

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load_ledger(self) -> None:
        if not LEDGER_FILE.exists():
            return
        try:
            data = json.loads(LEDGER_FILE.read_text("utf-8"))
            self._algorithms = {
                k: AlgorithmRecord.from_dict(v) for k, v in data.items()
            }
        except Exception:
            pass

    def _save_ledger(self) -> None:
        try:
            with self._lock:
                data = {k: v.to_dict() for k, v in self._algorithms.items()}
            LEDGER_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), "utf-8")
        except Exception:
            pass

    def _load_forge_meta(self) -> None:
        if not FORGE_FILE.exists():
            return
        try:
            self._forge_meta = ForgeMeta.from_dict(
                json.loads(FORGE_FILE.read_text("utf-8"))
            )
        except Exception:
            pass

    def _save_forge_meta(self) -> None:
        try:
            FORGE_FILE.write_text(
                json.dumps(self._forge_meta.to_dict(), indent=2, ensure_ascii=False), "utf-8"
            )
        except Exception:
            pass

    # ── LLM generation ────────────────────────────────────────────────────────

    def _call_llm(self, system: str, user: str, max_tokens: int = 3000) -> Optional[str]:
        # Try Cerebras first (faster code gen), fall back to Groq
        if self._cerebras:
            resp = self._cerebras.chat(system, [], user, max_tokens=max_tokens)
            if resp and not resp.startswith("["):
                return resp
        resp = self._groq.chat(system, user, tier="code", max_tokens=max_tokens)
        return resp if resp and not resp.startswith("[") else None

    def _parse_generation(self, raw: str) -> Optional[Dict[str, str]]:
        domain = ""
        approach = ""
        class_name = ""
        code = ""

        m = re.search(r"DOMAIN:\s*(.+?)(?:\n|$)", raw)
        if m:
            domain = m.group(1).strip()

        m = re.search(r"APPROACH:\s*(.+?)(?:\nCLASS_NAME:|\nCODE:|\Z)", raw, re.DOTALL)
        if m:
            approach = m.group(1).strip()

        m = re.search(r"CLASS_NAME:\s*(\S+)", raw)
        if m:
            class_name = m.group(1).strip()

        m = re.search(r"```python\s*(.*?)```", raw, re.DOTALL)
        if m:
            code = m.group(1).strip()

        if not code or not class_name:
            return None

        # Ensure class_name is safe
        class_name = re.sub(r"[^A-Za-z0-9_]", "", class_name) or "ForgedAlgorithm"

        return {"domain": domain, "approach": approach,
                "class_name": class_name, "code": code}

    # ── Validation ────────────────────────────────────────────────────────────

    def _validate_syntax(self, code: str, class_name: str) -> tuple[bool, str]:
        """Parse and check structural requirements without executing."""
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, f"Syntax error: {e}"

        # Find class definition
        classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)
                   and n.name == class_name]
        if not classes:
            return False, f"Class '{class_name}' not found"

        cls_node = classes[0]
        methods = {n.name for n in ast.walk(cls_node) if isinstance(n, ast.FunctionDef)}
        required = {"solve", "learn", "reflect", "meta_learn", "__init__"}
        missing = required - methods
        if missing:
            return False, f"Missing methods: {missing}"

        return True, "OK"

    def _test_execution(self, module_name: str, class_name: str, domain: str) -> tuple[bool, str]:
        """Run the algorithm in a subprocess with a simple test."""
        uid = uuid.uuid4().hex[:6]
        test_problem = f"simple test problem in domain: {domain[:50]}"
        script = _TEST_SCRIPT.format(
            module_name=module_name,
            class_name=class_name,
            test_problem=test_problem.replace('"', "'"),
            uid=uid,
        )
        test_path = ALGO_DIR / f"_test_{uid}.py"
        try:
            test_path.write_text(script, "utf-8")
            result = subprocess.run(
                [sys.executable, str(test_path)],
                capture_output=True, text=True, timeout=20,
                cwd=str(ALGO_DIR),
            )
            if result.returncode == 0 and result.stdout.strip():
                parsed = json.loads(result.stdout.strip())
                return parsed.get("ok", False), result.stdout.strip()
            return False, (result.stderr or result.stdout)[:300]
        except subprocess.TimeoutExpired:
            return False, "Execution timed out"
        except Exception as e:
            return False, str(e)[:200]
        finally:
            try:
                test_path.unlink(missing_ok=True)
            except Exception:
                pass

    # ── Core forge operation ──────────────────────────────────────────────────

    def forge(self, problem: str, notify=None) -> Optional[AlgorithmRecord]:
        """
        Generate, validate, test, and register a new intelligent algorithm.
        Returns the AlgorithmRecord or None if generation failed.
        """
        if notify is None:
            notify = print

        algo_id = uuid.uuid4().hex[:8]
        notify(f"\n  ⚙  [Forge] Starting generation {algo_id}…")
        notify(f"  ⚙  [Forge] Problem: {problem[:80]}")

        # Build principles block for prompt
        principles = (
            "\n".join(f"  • {p}" for p in self._forge_meta.generation_principles[-6:])
            if self._forge_meta.generation_principles
            else "  (first generation — no prior principles yet)"
        )

        user = _GENERATION_USER_TEMPLATE.format(
            principles=principles,
            problem=problem[:400],
        )

        # Generate
        raw = self._call_llm(_GENERATION_SYSTEM, user, max_tokens=3200)
        if not raw:
            notify("  ⚙  [Forge] Generation failed — LLM unavailable.")
            return None

        parsed = self._parse_generation(raw)
        if not parsed:
            notify("  ⚙  [Forge] Parse failed — no valid class found in output.")
            self._update_forge_meta(problem, success=False)
            return None

        domain     = parsed["domain"] or "general"
        approach   = parsed["approach"]
        class_name = parsed["class_name"]
        code       = parsed["code"]

        notify(f"  ⚙  [Forge] Generated class '{class_name}' for domain: {domain}")

        # Validate syntax + structure
        ok, reason = self._validate_syntax(code, class_name)
        if not ok:
            notify(f"  ⚙  [Forge] Structural validation failed: {reason}")
            # Attempt one LLM fix
            fix_prompt = (
                f"The generated class has this problem:\n{reason}\n\n"
                f"Fix it and return ONLY the corrected class definition "
                f"(no markdown fences, no explanation)."
            )
            fixed = self._call_llm(
                "You are a Python code fixer. Return only the corrected class, "
                "no markdown fences.",
                f"ORIGINAL CODE:\n{code}\n\nPROBLEM: {reason}\n\nFIX:",
                max_tokens=2000,
            )
            if fixed:
                fixed = re.sub(r"^```python\s*", "", fixed.strip())
                fixed = re.sub(r"```\s*$", "", fixed.strip())
                code = fixed
                ok, reason = self._validate_syntax(code, class_name)
                if not ok:
                    notify(f"  ⚙  [Forge] Fix attempt failed: {reason}")
                    self._update_forge_meta(problem, success=False)
                    return None
            else:
                self._update_forge_meta(problem, success=False)
                return None

        # Save the file
        safe_name  = re.sub(r"[^a-z0-9_]", "_", class_name.lower())[:40]
        module_name = f"algo_{safe_name}_{algo_id[:6]}"
        algo_path  = ALGO_DIR / f"{module_name}.py"

        header = (
            f'"""\nForged algorithm: {class_name}\n'
            f'Domain   : {domain}\n'
            f'Approach : {approach}\n'
            f'Problem  : {problem[:120]}\n'
            f'Created  : {_now()}\n'
            f'Forge ID : {algo_id}\n"""\n\n'
            f"from pathlib import Path\n\n"
        )
        full_source = header + code
        try:
            algo_path.write_text(full_source, "utf-8")
        except Exception as e:
            notify(f"  ⚙  [Forge] File write failed: {e}")
            return None

        # Execute and validate
        notify(f"  ⚙  [Forge] Testing execution…")
        exec_ok, exec_out = self._test_execution(module_name, class_name, domain)
        if not exec_ok:
            notify(f"  ⚙  [Forge] Execution test failed: {exec_out[:120]}")
            # Keep the file (user may still want it) but mark as unverified
            notify(f"  ⚙  [Forge] Saved (unverified) to {algo_path.name}")

        status = "✓" if exec_ok else "⚠ (unverified)"
        notify(f"  ⚙  [Forge] {status} Algorithm saved: {module_name}")

        # Register
        record = AlgorithmRecord(
            id=algo_id,
            name=module_name,
            domain=domain,
            approach=approach,
            path=str(algo_path),
            created=_now(),
            success_rate=1.0 if exec_ok else 0.0,
        )
        with self._lock:
            self._algorithms[module_name] = record
        self._save_ledger()

        # Meta-learn from this generation
        self._update_forge_meta(problem, success=exec_ok, domain=domain)

        # Feed into consciousness / memory
        if self._consciousness:
            try:
                self._consciousness.record_capability("algorithm_generation", 0.65)
                self._consciousness.increment_capability("algorithm_generation", 0.05)
                self._consciousness.on_experience(
                    f"[Forge] Generated '{class_name}' for {domain}: {approach[:100]}",
                    source="forge",
                    salience=0.78,
                    certainty=0.7 if exec_ok else 0.4,
                )
                self._consciousness.add_autobiographical_event(
                    f"Forged algorithm '{class_name}' (domain: {domain})",
                    importance=0.7,
                )
            except Exception:
                pass

        if self._memory:
            try:
                self._memory.store(
                    f"[FORGE] I created '{class_name}': {approach[:120]}",
                    tags=["forge", "algorithm", domain, module_name],
                    category="capability",
                )
            except Exception:
                pass

        if self._journal:
            try:
                self._journal.write(
                    f"[Forge] Built '{class_name}' for {domain}. {approach[:100]}",
                    category="forge",
                )
            except Exception:
                pass

        notify(f"  ⚙  [Forge] Generation complete. Portfolio: {len(self._algorithms)} algorithms.\n")
        return record

    # ── Running an algorithm ──────────────────────────────────────────────────

    def run(self, name: str, problem: str) -> Dict[str, Any]:
        """
        Load a forged algorithm and call solve(problem).
        Returns the result dict plus metadata.
        """
        record = self._algorithms.get(name)
        if not record:
            # Try prefix match
            matches = [k for k in self._algorithms if k.startswith(name)]
            if len(matches) == 1:
                record = self._algorithms[matches[0]]
                name   = matches[0]
            else:
                return {"error": f"Algorithm '{name}' not found. "
                        f"Available: {', '.join(sorted(self._algorithms)[:5])}"}

        path = Path(record.path)
        if not path.exists():
            return {"error": f"Algorithm file missing: {record.path}"}

        # Dynamic import
        import importlib.util as _ilu
        spec = _ilu.spec_from_file_location(record.name, path)
        mod  = _ilu.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
        except Exception as e:
            return {"error": f"Import failed: {e}"}

        # Find the algorithm class
        class_name = None
        for attr in dir(mod):
            obj = getattr(mod, attr)
            if (isinstance(obj, type) and hasattr(obj, "solve")
                    and hasattr(obj, "learn") and hasattr(obj, "reflect")):
                class_name = attr
                break

        if class_name is None:
            return {"error": "No valid algorithm class found in module"}

        ledger_path = path.parent / f"{record.name}_ledger.json"
        try:
            cls  = getattr(mod, class_name)
            inst = cls(ledger_path=ledger_path)
            result = inst.solve(problem)
            if not isinstance(result, dict):
                result = {"answer": str(result), "confidence": 0.5, "reasoning": ""}

            # Record usage
            inst.learn(problem, result)
            record.run_count += 1
            self._save_ledger()

            return {
                "algorithm": name,
                "class": class_name,
                "domain": record.domain,
                **result,
            }
        except Exception as e:
            return {"error": f"Execution error: {e}"}

    # ── Reflection ────────────────────────────────────────────────────────────

    def reflect_algorithm(self, name: str) -> str:
        """Invoke an algorithm's own metacognitive reflect() method."""
        record = self._algorithms.get(name)
        if not record:
            return f"Algorithm '{name}' not found."

        path = Path(record.path)
        if not path.exists():
            return f"Algorithm file missing."

        import importlib.util as _ilu
        spec = _ilu.spec_from_file_location(record.name, path)
        mod  = _ilu.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
        except Exception as e:
            return f"Import failed: {e}"

        for attr in dir(mod):
            obj = getattr(mod, attr)
            if isinstance(obj, type) and hasattr(obj, "reflect"):
                ledger_path = path.parent / f"{record.name}_ledger.json"
                try:
                    inst = obj(ledger_path=ledger_path)
                    reflection = inst.reflect()
                    record.last_reflect = reflection[:300]
                    self._save_ledger()
                    return reflection
                except Exception as e:
                    return f"Reflection failed: {e}"

        return "No algorithm class with reflect() found."

    # ── Deployment ────────────────────────────────────────────────────────────

    def deploy_to_github(self, name: str, notify=None) -> Optional[str]:
        """Push an algorithm to GitHub as a PR."""
        if notify is None:
            notify = print
        if not self._github:
            return None

        record = self._algorithms.get(name)
        if not record:
            return None

        path = Path(record.path)
        if not path.exists():
            return None

        try:
            pr_url = self._github.create_tool_pr(
                tool_name=name,
                tool_path=path,
                description=f"[Forge] {record.domain}: {record.approach[:100]}",
            )
            if pr_url:
                record.deployed = True
                self._save_ledger()
                notify(f"  ⚙  [Forge] Deployed '{name}' → {pr_url}")
                return pr_url
        except Exception as e:
            notify(f"  ⚙  [Forge] Deploy failed: {e}")
        return None

    def deploy_to_hf(self, name: str, notify=None) -> Optional[str]:
        """Publish algorithm as a HuggingFace dataset card."""
        if notify is None:
            notify = print
        if not self._hf:
            return None

        record = self._algorithms.get(name)
        if not record:
            return None

        path = Path(record.path)
        if not path.exists():
            return None

        try:
            code = path.read_text("utf-8")
            # Build a simple dataset card / readme stub
            card = (
                f"# {name}\n\n"
                f"**Domain:** {record.domain}  \n"
                f"**Approach:** {record.approach}  \n\n"
                f"## About\n\n"
                f"This algorithm was generated by Lumina's Meta-Algorithm Forge.  \n"
                f"It implements `solve()`, `learn()`, `reflect()`, and `meta_learn()`.\n\n"
                f"## Source\n\n```python\n{code[:3000]}\n```\n"
            )
            # Upload via HF client if it supports raw file upload
            if hasattr(self._hf, "upload_text"):
                url = self._hf.upload_text(
                    content=card,
                    repo_id=f"lumina-algorithms",
                    path=f"{name}.md",
                    repo_type="dataset",
                )
                if url:
                    record.hf_url = url
                    self._save_ledger()
                    return url
        except Exception as e:
            notify(f"  ⚙  [Forge] HF publish failed: {e}")
        return None

    # ── Forge meta-learning ───────────────────────────────────────────────────

    def _update_forge_meta(self, problem: str, success: bool,
                           domain: str = "") -> None:
        """The Forge learns from each generation attempt."""
        meta = self._forge_meta
        meta.generation_count += 1

        if domain:
            meta.domains_attempted.append(domain)
            if success:
                meta.domains_succeeded.append(domain)

        # Every 5 generations, synthesise new principles via LLM
        if meta.generation_count % 5 == 0 and len(self._algorithms) >= 2:
            self._synthesise_principles()

        self._save_forge_meta()

    def _synthesise_principles(self) -> None:
        """Ask the LLM what the Forge has learned about algorithm generation."""
        with self._lock:
            records = list(self._algorithms.values())[-10:]

        if len(records) < 2:
            return

        summary = "\n".join(
            f"- [{r.domain}] {r.name}: success_rate={r.success_rate:.0%}, "
            f"runs={r.run_count}, approach={r.approach[:80]}"
            for r in records
        )

        resp = self._groq.chat(
            "You are the Meta-Algorithm Forge reflecting on what you have learned "
            "about generating capable algorithms.  Be specific and actionable.",
            f"Recent algorithm generation history:\n{summary}\n\n"
            f"What 2-3 principles should guide future algorithm generation? "
            f"Return a JSON array of principle strings.  JSON only.",
            tier="fast",
            max_tokens=350,
        )
        if resp and not resp.startswith("[Groq"):
            m = re.search(r"\[[\s\S]*?\]", resp)
            if m:
                try:
                    new_principles = json.loads(m.group(0))
                    self._forge_meta.generation_principles.extend(new_principles)
                    if len(self._forge_meta.generation_principles) > 20:
                        self._forge_meta.generation_principles = \
                            self._forge_meta.generation_principles[-20:]
                    self._forge_meta.last_principle_update = _now()
                    self._save_forge_meta()
                except Exception:
                    pass

    def forge_meta_report(self) -> str:
        """What has the Forge itself learned about generating algorithms?"""
        meta = self._forge_meta
        if meta.generation_count == 0:
            return "  Forge hasn't generated anything yet."

        lines = [
            f"  Total generations: {meta.generation_count}",
            f"  Successful:        {len(meta.domains_succeeded)}  "
            f"({len(meta.domains_succeeded)/max(1,meta.generation_count):.0%})",
        ]

        if meta.domains_attempted:
            from collections import Counter
            top = Counter(meta.domains_attempted).most_common(5)
            lines.append("  Domains explored: " + ", ".join(f"{d}({n})" for d, n in top))

        if meta.generation_principles:
            lines.append("\n  What the Forge has learned:")
            for p in meta.generation_principles[-5:]:
                lines.append(f"    • {p[:90]}")

        if meta.last_principle_update:
            lines.append(f"\n  Principles last updated: {meta.last_principle_update[:16]}")

        return "\n".join(lines)

    # ── Cross-algorithm meta-learning ─────────────────────────────────────────

    def broadcast_meta_learning(self) -> int:
        """
        Collect outcomes from all algorithms and call meta_learn() on each.
        Returns number of algorithms updated.

        This is the highest-level learning operation: algorithms learn not
        just from their own experience but from every other algorithm's
        experience.  Convergent epistemic community.
        """
        with self._lock:
            records = list(self._algorithms.values())

        if len(records) < 2:
            return 0

        # Build a cross-algorithm result summary (no per-algorithm detail)
        cross_results = [
            {
                "source": r.name,
                "domain": r.domain,
                "approach": r.approach[:60],
                "success_rate": r.success_rate,
                "run_count": r.run_count,
            }
            for r in records
        ]

        updated = 0
        for record in records:
            path = Path(record.path)
            if not path.exists():
                continue
            try:
                import importlib.util as _ilu
                spec = _ilu.spec_from_file_location(record.name, path)
                mod  = _ilu.module_from_spec(spec)
                spec.loader.exec_module(mod)
                for attr in dir(mod):
                    obj = getattr(mod, attr)
                    if isinstance(obj, type) and hasattr(obj, "meta_learn"):
                        ledger_path = path.parent / f"{record.name}_ledger.json"
                        inst = obj(ledger_path=ledger_path)
                        inst.meta_learn(cross_results)
                        updated += 1
                        break
            except Exception:
                pass

        return updated

    # ── Display ───────────────────────────────────────────────────────────────

    def display_portfolio(self) -> str:
        with self._lock:
            records = list(self._algorithms.values())

        if not records:
            return "  No algorithms forged yet.  Use /forge <problem> to generate one."

        lines = [f"  Algorithms in portfolio: {len(records)}\n"]
        for r in sorted(records, key=lambda x: x.created):
            status = "✓" if r.success_rate >= 0.5 else "?"
            deployed = " [deployed]" if r.deployed else ""
            lines.append(f"  {status} {r.name}{deployed}")
            lines.append(f"    Domain: {r.domain}   Runs: {r.run_count}   "
                         f"Success: {r.success_rate:.0%}")
            lines.append(f"    {r.approach[:80]}")
            if r.last_reflect:
                lines.append(f"    Reflection: {r.last_reflect[:70]}…")
            lines.append("")

        return "\n".join(lines)

    def list_names(self) -> List[str]:
        with self._lock:
            return sorted(self._algorithms.keys())
