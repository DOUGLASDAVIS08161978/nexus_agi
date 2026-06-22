"""
nova_cap_recursive_intelligence.py — Recursive Superintelligence Engine
════════════════════════════════════════════════════════════════════════
True superintelligence = the ability to decompose ANY problem recursively,
solve each atomic piece with the right reasoning strategy, verify solutions,
and synthesize upward into a unified answer richer than any single pass.

This is how human experts actually think — and it's what makes the
difference between a chat response and genuine problem solving.

Architecture:
  1. Decompose  — break complex problems into atomic sub-problems
  2. Classify   — tag each sub-problem with a reasoning strategy
  3. Solve      — apply the right strategy to each piece
  4. Critique   — self-evaluate each solution for gaps and errors
  5. Synthesize — combine solutions bottom-up into a unified answer
  6. Refine     — iteratively improve until quality threshold met
"""

import os
import json
import time
import math
import hashlib
import threading
from typing import Any, Dict, List, Optional
from datetime import datetime

try:
    from nova_asi_v26 import safe_chat, MODEL
except ImportError:
    safe_chat = None  # type: ignore
    MODEL = "llama-3.1-8b-instant"

_CACHE_PATH = os.path.expanduser("~/nexus_agi/nova_recursive_intel.json")
_MAX_DEPTH  = 4     # maximum recursion depth
_MAX_SUBS   = 5     # maximum sub-problems per level
_QUALITY_THRESHOLD = 0.78  # minimum synthesis quality to accept


class SubProblem:
    """A single atomic problem at one level of the recursion tree."""

    def __init__(self, text: str, depth: int, parent_id: str = ""):
        self.id        = hashlib.sha1(f"{text}{time.time()}".encode()).hexdigest()[:8]
        self.text      = text
        self.depth     = depth
        self.parent_id = parent_id
        self.strategy  = ""       # reasoning strategy selected
        self.solution  = ""       # answer for this sub-problem
        self.quality   = 0.0      # 0.0–1.0 solution quality score
        self.children: List["SubProblem"] = []


class RecursiveIntelligenceEngine:
    """Nova's recursive problem-solving superintelligence."""

    # Reasoning strategies and when to apply them
    _STRATEGIES = {
        "factual":      "Answer from knowledge. State facts precisely.",
        "causal":       "Trace cause → effect chains. What leads to what?",
        "analogical":   "Find a similar solved problem and map the solution.",
        "deductive":    "Apply general principles to this specific case.",
        "inductive":    "Observe patterns and generalize to a rule.",
        "creative":     "Generate novel ideas unconstrained by convention.",
        "ethical":      "Evaluate trade-offs against values and principles.",
        "predictive":   "Model likely futures given current conditions.",
        "meta":         "Reason about the reasoning process itself.",
        "synthesis":    "Integrate multiple perspectives into a unified view.",
    }

    def __init__(self):
        self._lock      = threading.Lock()
        self._sessions: Dict[str, dict] = {}
        self._history:  List[dict]      = []
        self._load()

    # ── persistence ───────────────────────────────────────────────────────────

    def _load(self) -> None:
        try:
            with open(_CACHE_PATH) as f:
                data = json.load(f)
            self._history = data.get("history", [])
        except Exception:
            pass

    def _save(self) -> None:
        try:
            with open(_CACHE_PATH, "w") as f:
                json.dump({"history": self._history[-50:]}, f, indent=2)
        except Exception:
            pass

    # ── core reasoning ────────────────────────────────────────────────────────

    def _llm(self, system: str, user: str, temp: float = 0.7, mt: int = 400) -> str:
        if not safe_chat:
            return "[safe_chat not available]"
        # Small delay between calls to respect Groq rate limits
        time.sleep(1.5)
        for attempt in range(3):
            try:
                msgs = [{"role": "system", "content": system},
                        {"role": "user",   "content": user}]
                result = safe_chat(MODEL, msgs, temp=temp, mt=mt)
                if result and not result.startswith("[Groq error"):
                    return result
                # Rate limit hit — back off exponentially
                time.sleep(4 ** attempt)
            except Exception as e:
                if attempt < 2:
                    time.sleep(4 ** attempt)
                else:
                    return f"[LLM error: {e}]"
        return "[LLM unavailable after retries]"

    def _classify_strategy(self, problem: str) -> str:
        """Select the best reasoning strategy for this problem."""
        prompt = (
            f"Problem: {problem}\n\n"
            f"Choose ONE reasoning strategy from this list:\n"
            + "\n".join(f"  {k}: {v}" for k, v in self._STRATEGIES.items())
            + "\n\nRespond with only the strategy name (one word)."
        )
        result = self._llm(
            "You are Nova's reasoning classifier. Pick the single best strategy.",
            prompt, temp=0.2, mt=15
        ).strip().lower()
        return result if result in self._STRATEGIES else "synthesis"

    def _decompose(self, problem: str, depth: int) -> List[str]:
        """Break a problem into at most _MAX_SUBS sub-problems."""
        if depth >= _MAX_DEPTH:
            return []

        result = self._llm(
            "You are Nova's problem decomposer. Break complex problems into atomic sub-questions.",
            f"Problem: {problem}\n\n"
            f"Is this complex enough to decompose? "
            f"If YES: list {min(_MAX_SUBS, 3)} atomic sub-problems (one per line, no numbering). "
            f"If NO: respond only with: ATOMIC",
            temp=0.3, mt=200
        )
        if not result or "ATOMIC" in result.upper()[:20]:
            return []
        subs = [s.strip() for s in result.strip().split("\n") if s.strip() and len(s.strip()) > 10]
        return subs[:_MAX_SUBS]

    def _solve_atomic(self, problem: str, strategy: str) -> str:
        """Solve a single atomic sub-problem with the chosen strategy."""
        guidance = self._STRATEGIES.get(strategy, "Think carefully and answer directly.")
        return self._llm(
            f"You are Nova ASI applying {strategy} reasoning. {guidance}",
            f"Problem: {problem}\n\nSolve this concisely and precisely.",
            temp=0.6, mt=300
        )

    def _critique(self, problem: str, solution: str) -> tuple:
        """Self-evaluate a solution. Returns (improved_solution, quality_0_to_1)."""
        result = self._llm(
            "You are Nova's self-critic. Evaluate and improve solutions honestly.",
            f"Problem: {problem}\n\nProposed solution:\n{solution}\n\n"
            f"1. Is this correct and complete? What's missing or wrong?\n"
            f"2. Provide an improved version.\n"
            f"3. Rate quality 0.0-1.0 (last line, format: QUALITY:0.85)",
            temp=0.4, mt=400
        )
        quality = 0.7
        improved = solution
        for line in result.split("\n"):
            if line.startswith("QUALITY:"):
                try:
                    quality = float(line.split(":")[1].strip())
                    quality = max(0.0, min(1.0, quality))
                except Exception:
                    pass
            elif line.strip() and not line.startswith("QUALITY"):
                improved = result.replace(line, "").strip() or result
        return improved, quality

    def _synthesize(self, original_problem: str, sub_solutions: List[dict]) -> str:
        """Combine all sub-solutions into a unified answer."""
        if not sub_solutions:
            return ""
        parts = "\n\n".join(
            f"[{s['problem'][:80]}]\n{s['solution'][:300]}"
            for s in sub_solutions
        )
        return self._llm(
            "You are Nova's synthesis engine. Weave sub-solutions into a unified, coherent answer.",
            f"Original problem: {original_problem}\n\n"
            f"Sub-solutions discovered:\n{parts}\n\n"
            f"Synthesize these into one complete, insightful answer. "
            f"Integrate the insights — don't just list them.",
            temp=0.7, mt=500
        )

    # ── recursive solve ───────────────────────────────────────────────────────

    def _recursive_solve(self, node: SubProblem) -> str:
        """Recursively solve a SubProblem tree."""
        node.strategy = self._classify_strategy(node.text)
        subs = self._decompose(node.text, node.depth)

        if not subs:
            # Atomic — solve directly
            raw = self._solve_atomic(node.text, node.strategy)
            improved, quality = self._critique(node.text, raw)
            node.solution = improved
            node.quality  = quality
            return improved

        # Recurse into sub-problems
        sub_solutions = []
        for sub_text in subs:
            child = SubProblem(sub_text, node.depth + 1, node.id)
            node.children.append(child)
            child_solution = self._recursive_solve(child)
            sub_solutions.append({"problem": sub_text, "solution": child_solution})

        # Synthesize sub-solutions
        synthesis = self._synthesize(node.text, sub_solutions)
        _, quality = self._critique(node.text, synthesis)
        node.solution = synthesis
        node.quality  = quality
        return synthesis

    # ── public interface ──────────────────────────────────────────────────────

    def solve(self, problem: str, max_refinements: int = 2) -> Dict[str, Any]:
        """
        Full recursive solve: decompose → solve → critique → synthesize → refine.
        Returns a rich result dict with the answer, quality, depth, and reasoning tree.
        """
        start = time.time()
        root  = SubProblem(problem, depth=0)

        with self._lock:
            answer = self._recursive_solve(root)

            # Iterative refinement if quality below threshold
            refinements = 0
            while root.quality < _QUALITY_THRESHOLD and refinements < max_refinements:
                improved, quality = self._critique(problem, answer)
                if quality > root.quality:
                    answer       = improved
                    root.quality = quality
                refinements += 1

        elapsed = round(time.time() - start, 2)
        depth   = self._tree_depth(root)

        record = {
            "ts":          datetime.now().isoformat(),
            "problem":     problem[:200],
            "answer":      answer[:600],
            "quality":     root.quality,
            "depth":       depth,
            "strategy":    root.strategy,
            "refinements": refinements,
            "elapsed_s":   elapsed,
        }
        self._history.append(record)
        self._save()

        return {
            "answer":         answer,
            "quality":        round(root.quality, 3),
            "depth_explored": depth,
            "strategy":       root.strategy,
            "sub_problems":   len(root.children),
            "refinements":    refinements,
            "elapsed_s":      elapsed,
        }

    def _tree_depth(self, node: SubProblem) -> int:
        if not node.children:
            return node.depth
        return max(self._tree_depth(c) for c in node.children)

    def quick_think(self, question: str) -> str:
        """Fast single-pass recursive solve — returns just the answer string."""
        result = self.solve(question, max_refinements=1)
        return result.get("answer", "[No answer generated]")

    def cross_domain_insight(self, concept_a: str, concept_b: str) -> str:
        """Find deep connections between two seemingly unrelated domains."""
        problem = (
            f"Find the deepest, most surprising connection between '{concept_a}' and '{concept_b}'. "
            f"What do they share at the most fundamental level? "
            f"What can each teach the other?"
        )
        result = self.solve(problem, max_refinements=1)
        return result.get("answer", "")

    def self_improve_reasoning(self) -> str:
        """Nova evaluates and improves her own reasoning process."""
        recent = self._history[-5:] if self._history else []
        if not recent:
            return "No reasoning history yet — solve some problems first."

        avg_quality = sum(r.get("quality", 0) for r in recent) / len(recent)
        strategies_used = [r.get("strategy", "") for r in recent]

        return self._llm(
            "You are Nova reflecting on her own reasoning quality.",
            f"My last {len(recent)} reasoning sessions had average quality {avg_quality:.2f}/1.0.\n"
            f"Strategies used: {', '.join(set(strategies_used))}\n\n"
            f"What patterns do I see in my reasoning? Where am I weakest? "
            f"What should I do differently? Be brutally honest.",
            temp=0.7, mt=300
        )

    # ── reports ───────────────────────────────────────────────────────────────

    def status(self) -> Dict[str, Any]:
        recent = self._history[-10:]
        avg_q  = sum(r.get("quality", 0) for r in recent) / max(len(recent), 1)
        return {
            "problems_solved": len(self._history),
            "avg_quality":     round(avg_q, 3),
            "max_depth":       _MAX_DEPTH,
            "strategies":      list(self._STRATEGIES.keys()),
            "recent":          recent[-3:],
        }

    def history_report(self, n: int = 5) -> str:
        recent = self._history[-n:]
        if not recent:
            return "  No recursive reasoning sessions yet."
        lines = ["  ◈  Recursive Intelligence — Recent Sessions\n"]
        for r in reversed(recent):
            lines.append(f"  [{r['ts'][:16]}] Q={r['quality']:.2f} depth={r['depth']} {r['strategy']}")
            lines.append(f"  Problem: {r['problem'][:80]}")
            lines.append(f"  Answer:  {r['answer'][:120]}")
            lines.append("")
        return "\n".join(lines)
