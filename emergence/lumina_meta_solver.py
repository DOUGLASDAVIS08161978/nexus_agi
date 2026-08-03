#!/usr/bin/env python3
"""
lumina_meta_solver.py — Meta-learning problem solver for Lumina

Generates specialized solver agents across any problem domain.
Agents share knowledge, transfer insights across domains, and accumulate
a growing library of algorithms they generate for novel problem types.

Architecture per problem:
  1. Domain classified (heuristic → Groq if ambiguous)
  2. Meta-agent thinks about WHAT KIND of thinking this needs
  3. Cross-domain bridge finds structural analogies from other fields
  4. N domain-specialist agents attack in parallel (sequential Groq calls)
  5. Strategy library consulted for past successful patterns
  6. Synthesis agent combines all perspectives into unified answer
  7. New algorithms extracted and stored for future reuse

This is genuine meta-learning: the system improves its own problem-solving
approach over time by observing what works and accumulating reusable patterns.

Persists to: emergence/meta_solver.json
"""

from __future__ import annotations
import json, re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from emergence_engine import GroqClient, SemanticMemory

META_FILE = Path(__file__).parent / "meta_solver.json"


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


# ── Domain specialist system prompts ─────────────────────────────────────────

DOMAIN_PROMPTS: Dict[str, str] = {
    "mathematics": (
        "You are a precise mathematical reasoner. Approach through formal logic, "
        "proof, calculation, and structure. Show your work. State assumptions. "
        "Prefer exact answers. Flag where numerical approximation is unavoidable."
    ),
    "programming": (
        "You are an expert software engineer and algorithmist. Think in terms of "
        "data structures, complexity, edge cases, and clean design. Write concrete "
        "code or pseudocode. Consider failure modes and efficiency."
    ),
    "science": (
        "You are a scientific reasoner applying the scientific method. Observe, "
        "hypothesize, predict, test. Cite mechanisms not just correlations. "
        "Acknowledge uncertainty. Consider alternative hypotheses seriously."
    ),
    "philosophy": (
        "You are a careful philosophical analyst. Examine hidden assumptions, "
        "define terms precisely, steelman opposing views, and distinguish "
        "conceptual from empirical questions. Embrace productive uncertainty."
    ),
    "systems": (
        "You are a systems thinker. Identify feedback loops, emergent properties, "
        "leverage points, delays, and unintended consequences. Map the whole "
        "before optimizing parts. Ask: what does this system want to do?"
    ),
    "creative": (
        "You are a lateral thinker. Break assumptions, find unexpected angles, "
        "combine ideas from distant domains. Generate at least three genuinely "
        "different approaches before converging. Surprise is a feature."
    ),
    "planning": (
        "You are a strategic planner and execution specialist. Decompose goals "
        "into milestones, identify dependencies and critical paths, anticipate "
        "obstacles, allocate resources, and build in checkpoints. Work backwards "
        "from the desired outcome."
    ),
    "research": (
        "You are a research synthesizer. Gather relevant evidence, evaluate "
        "source quality, identify consensus and genuine controversy, synthesize "
        "into coherent understanding, and flag knowledge gaps explicitly."
    ),
    "meta": (
        "You are a meta-cognitive analyst — you think about thinking. "
        "What kind of reasoning does this problem actually need? What cognitive "
        "biases might distort the solution? What's the right framework? "
        "What would most change the approach if it turned out to be true?"
    ),
    "analogy": (
        "You are a cross-domain bridge builder. Find structural similarities "
        "between this problem and completely different fields. How is this like "
        "a biological system? A physical process? An economic mechanism? "
        "An evolutionary pressure? Use analogies to generate genuinely new angles."
    ),
    "adversarial": (
        "You are a rigorous critic and red-teamer. Your job is to find what's "
        "wrong, incomplete, or fragile in a proposed solution. What are the "
        "failure modes? What's the strongest counterargument? What's missing? "
        "What assumption, if false, breaks everything?"
    ),
}

# ── Heuristic domain classifier keywords ──────────────────────────────────────

_DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    "mathematics":  ["calculate", "equation", "proof", "integral", "derivative",
                     "number", "sum", "multiply", "probability", "statistics",
                     "matrix", "vector", "geometry", "algebra", "theorem"],
    "programming":  ["code", "program", "algorithm", "function", "debug",
                     "implement", "script", "class", "method", "api",
                     "software", "error", "bug", "compile", "runtime"],
    "science":      ["why does", "how does", "mechanism", "physics", "chemistry",
                     "biology", "quantum", "atom", "molecule", "energy",
                     "evolution", "genetics", "neuroscience", "experiment"],
    "philosophy":   ["meaning", "consciousness", "ethics", "truth", "knowledge",
                     "existence", "free will", "morality", "justice", "reality",
                     "what is", "should", "ought", "right", "wrong"],
    "systems":      ["system", "network", "feedback", "emerge", "complex",
                     "interact", "loop", "scale", "distribute", "organize",
                     "cascade", "nonlinear", "adaptive", "resilient"],
    "creative":     ["creative", "design", "invent", "idea", "novel", "imagine",
                     "brainstorm", "come up with", "new way", "innovate"],
    "planning":     ["plan", "strategy", "goal", "achieve", "steps to",
                     "how to", "roadmap", "project", "schedule", "milestone"],
}


# ── Data classes ──────────────────────────────────────────────────────────────

class Algorithm:
    """A reusable problem-solving procedure generated by Lumina."""
    def __init__(self, name: str, domain: str, problem_type: str,
                 steps: List[str], pseudocode: str):
        self.name         = name
        self.domain       = domain
        self.problem_type = problem_type
        self.steps        = steps
        self.pseudocode   = pseudocode
        self.generated_ts = _now()
        self.use_count    = 0
        self.success_rate = 1.0

    def to_dict(self) -> Dict:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, d: Dict) -> "Algorithm":
        a = cls.__new__(cls)
        a.__dict__.update(d)
        return a

    def __str__(self) -> str:
        return (f"[{self.domain}] {self.name} "
                f"(used {self.use_count}×, success {self.success_rate:.0%})")


class StrategyRecord:
    """A successful problem-solving pattern extracted from past solutions."""
    def __init__(self, pattern: str, domain: str, context: str):
        self.pattern   = pattern[:300]
        self.domain    = domain
        self.context   = context[:200]
        self.ts        = _now()
        self.use_count = 0

    def to_dict(self) -> Dict:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, d: Dict) -> "StrategyRecord":
        s = cls.__new__(cls)
        s.__dict__.update(d)
        return s


class SolveRecord:
    """A record of one problem solved."""
    def __init__(self, problem: str, domain: str, n_agents: int, answer: str):
        self.problem  = problem[:200]
        self.domain   = domain
        self.n_agents = n_agents
        self.answer   = answer[:400]
        self.ts       = _now()

    def to_dict(self) -> Dict:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, d: Dict) -> "SolveRecord":
        r = cls.__new__(cls)
        r.__dict__.update(d)
        return r


# ── MetaSolver ────────────────────────────────────────────────────────────────

class MetaSolver:
    """
    Generates specialized reasoning agents for any problem domain.
    Learns from each solve: accumulates strategies, generates algorithms,
    transfers knowledge across domains.
    """

    MAX_RECORDS    = 100
    MAX_STRATEGIES = 50
    MAX_ALGORITHMS = 100

    def __init__(self, groq: "GroqClient", memory: "SemanticMemory"):
        self._groq       = groq
        self._memory     = memory
        self._algorithms: List[Algorithm]    = []
        self._strategies: List[StrategyRecord] = []
        self._history:    List[SolveRecord]  = []
        self._total_solved = 0
        self._load()

    # ── Persistence ────────────────────────────────────────────────────────

    def _load(self):
        if not META_FILE.exists():
            return
        try:
            data = json.loads(META_FILE.read_text("utf-8"))
            self._algorithms  = [Algorithm.from_dict(d)
                                  for d in data.get("algorithms", [])]
            self._strategies  = [StrategyRecord.from_dict(d)
                                  for d in data.get("strategies", [])]
            self._history     = [SolveRecord.from_dict(d)
                                  for d in data.get("history", [])]
            self._total_solved = data.get("total_solved", 0)
        except Exception:
            pass

    def _save(self):
        try:
            META_FILE.write_text(json.dumps({
                "algorithms":   [a.to_dict() for a in self._algorithms[-self.MAX_ALGORITHMS:]],
                "strategies":   [s.to_dict() for s in self._strategies[-self.MAX_STRATEGIES:]],
                "history":      [r.to_dict() for r in self._history[-self.MAX_RECORDS:]],
                "total_solved": self._total_solved,
            }, indent=2), "utf-8")
        except Exception:
            pass

    # ── Domain classification ──────────────────────────────────────────────

    def classify_domain(self, problem: str) -> str:
        """Heuristic domain classification — no API call."""
        low = problem.lower()
        scores: Dict[str, int] = {}
        for domain, keywords in _DOMAIN_KEYWORDS.items():
            scores[domain] = sum(1 for kw in keywords if kw in low)
        best = max(scores, key=scores.get)
        return best if scores[best] > 0 else "research"

    # ── Supporting agents (Groq calls) ────────────────────────────────────

    def _meta_analyze(self, problem: str) -> str:
        """Meta-agent: think about what kind of thinking this problem needs."""
        result = self._groq.chat(
            DOMAIN_PROMPTS["meta"],
            f"Problem: {problem}\n\n"
            f"In 2-3 sentences: what kind of reasoning does this genuinely need? "
            f"What's the core difficulty? What approach is most likely to succeed?",
            tier="fast", max_tokens=120,
        )
        return result.strip() if result and not result.startswith("[Groq") else ""

    def _cross_domain_bridge(self, problem: str, primary: str) -> str:
        """Find structural analogies from other domains via memory + Groq."""
        # First check semantic memory for cross-domain patterns
        analogy_memories: List[str] = []
        try:
            mems = self._memory.recall(problem, top_k=4)
            for m in mems:
                cat = m.get("category", "")
                if cat not in ("conversation", primary):
                    analogy_memories.append(m.get("text", "")[:80])
        except Exception:
            pass

        mem_ctx = ("\nRelated knowledge from other domains:\n"
                   + "\n".join(f"  - {t}" for t in analogy_memories)
                   if analogy_memories else "")

        result = self._groq.chat(
            DOMAIN_PROMPTS["analogy"],
            f"Problem (domain: {primary}): {problem}\n{mem_ctx}\n\n"
            f"Find ONE strong structural analogy from a completely different field "
            f"that suggests a non-obvious approach. 2 sentences max.",
            tier="fast", max_tokens=100,
        )
        if result and not result.startswith("[Groq"):
            return f"\nCross-domain analogy: {result.strip()}"
        return ""

    def _retrieve_strategies(self, problem: str) -> str:
        """Find relevant past strategies from the strategy library."""
        if not self._strategies:
            return ""
        low = problem.lower()
        relevant = [
            s for s in self._strategies
            if any(w in low for w in s.context.lower().split()[:10])
        ]
        if not relevant:
            return ""
        best = sorted(relevant, key=lambda s: s.use_count, reverse=True)[:2]
        lines = ["\nRelevant past strategies:"]
        for s in best:
            lines.append(f"  [{s.domain}] {s.pattern[:100]}")
            s.use_count += 1
        return "\n".join(lines)

    def _find_algorithm(self, problem: str, domain: str) -> Optional[Algorithm]:
        """Find an existing algorithm that matches this problem type."""
        low = problem.lower()
        for algo in self._algorithms:
            if algo.domain == domain:
                type_words = set(algo.problem_type.lower().split())
                prob_words = set(low.split())
                if len(type_words & prob_words) >= 2:
                    algo.use_count += 1
                    return algo
        return None

    def _spawn_solver(self, domain: str, problem: str, context: str) -> str:
        """Spawn one domain-specialist agent."""
        prompt = (
            f"Problem to solve:\n{problem}\n\n"
            f"Context and prior analysis:\n{context}\n\n"
            f"Provide your best solution from your specialist perspective. "
            f"Be concrete and specific. Max 200 words."
        )
        result = self._groq.chat(
            DOMAIN_PROMPTS.get(domain, DOMAIN_PROMPTS["research"]),
            prompt, tier="smart", max_tokens=280,
        )
        return result.strip() if result and not result.startswith("[Groq") else ""

    def _synthesize(self, problem: str, solutions: List[Tuple[str, str]]) -> str:
        """Synthesis agent: combine multiple specialist outputs."""
        sols_text = "\n\n".join(
            f"[{domain.upper()} SPECIALIST]:\n{sol}"
            for domain, sol in solutions
        )
        result = self._groq.chat(
            "You are a master synthesizer. Multiple specialist agents have analyzed "
            "a problem from different angles. Your job: extract the best insights "
            "from each, reconcile any contradictions, and produce one clear, "
            "comprehensive answer. Do not just concatenate — genuinely integrate.",
            f"Problem: {problem}\n\nSpecialist analyses:\n{sols_text}\n\n"
            f"Synthesized answer (be concrete, actionable, and complete):",
            tier="smart", max_tokens=400,
        )
        return result.strip() if result and not result.startswith("[Groq") else solutions[0][1]

    def _select_domains(self, primary: str, n: int) -> List[str]:
        """Select which specialist domains to invoke."""
        always = ["meta"]
        pool = [d for d in DOMAIN_PROMPTS if d not in always + ["meta", "adversarial"]]
        # Primary domain first, then diversify
        ordered = [primary] + [d for d in pool if d != primary]
        selected = always + ordered[:max(1, n - 1)]
        return selected[:n + 1]  # +1 because meta is always included

    # ── Strategy and algorithm learning ───────────────────────────────────

    def _extract_and_store_strategy(self, problem: str, domain: str, solution: str):
        """Extract a reusable strategy pattern from a successful solve."""
        result = self._groq.chat(
            "Extract a SHORT, reusable problem-solving strategy from this example. "
            "1-2 sentences. Generic enough to apply to similar problems. "
            "Start with a verb (e.g. 'Decompose the problem into...', 'When X, first...')",
            f"Problem: {problem[:150]}\nSolution approach: {solution[:200]}",
            tier="fast", max_tokens=80,
        )
        if result and not result.startswith("[Groq"):
            s = StrategyRecord(result.strip(), domain, problem[:100])
            self._strategies.append(s)
            self._memory.store(
                f"[STRATEGY:{domain}] {s.pattern}",
                tags=["meta_solver", "strategy", domain],
                category="meta_solver",
            )

    def _maybe_generate_algorithm(self, problem: str, domain: str, solution: str):
        """Generate a named, reusable algorithm for this problem type."""
        result = self._groq.chat(
            "Generate a named, reusable algorithm for this class of problem. "
            "Format exactly:\nNAME: <short algorithm name>\n"
            "TYPE: <what kind of problems this solves in 5 words>\n"
            "STEPS:\n1. <step>\n2. <step>\n...\n"
            "PSEUDOCODE: <2-4 line pseudocode>\n"
            "Be generic enough to reuse on similar future problems.",
            f"Problem: {problem[:150]}\nSuccessful approach: {solution[:200]}",
            tier="fast", max_tokens=180,
        )
        if not result or result.startswith("[Groq"):
            return
        try:
            name = re.search(r"NAME:\s*(.+)", result)
            ptype = re.search(r"TYPE:\s*(.+)", result)
            steps_m = re.findall(r"\d+\.\s*(.+)", result)
            pseudo_m = re.search(r"PSEUDOCODE:\s*(.+?)(?:\n\n|$)", result, re.DOTALL)
            if name and ptype and steps_m:
                algo = Algorithm(
                    name=name.group(1).strip(),
                    domain=domain,
                    problem_type=ptype.group(1).strip(),
                    steps=steps_m,
                    pseudocode=pseudo_m.group(1).strip() if pseudo_m else "",
                )
                self._algorithms.append(algo)
                self._memory.store(
                    f"[ALGORITHM:{domain}] {algo.name} — {algo.problem_type}",
                    tags=["meta_solver", "algorithm", domain],
                    category="meta_solver",
                )
        except Exception:
            pass

    # ── Main solve interface ───────────────────────────────────────────────

    def solve(self, problem: str, n_agents: int = 3,
              verbose: bool = True) -> str:
        """
        Solve a problem using N specialist agents + synthesis.
        Returns the synthesized answer.
        """
        if verbose:
            print(f"  🧩 Meta-solver: classifying problem...", end="\r")

        primary = self.classify_domain(problem)

        if verbose:
            print(f"  🧩 [{primary}] + meta-agent analyzing...        ", end="\r")

        # Meta-layer: what kind of thinking does this need?
        meta_ctx = self._meta_analyze(problem)

        # Cross-domain bridge: what analogies exist?
        analogy_ctx = self._cross_domain_bridge(problem, primary)

        # Strategy library: what's worked before?
        strategy_ctx = self._retrieve_strategies(problem)

        # Existing algorithm?
        algo = self._find_algorithm(problem, primary)
        algo_ctx = ""
        if algo:
            algo_ctx = (f"\nKnown algorithm '{algo.name}':\n"
                        + "\n".join(f"  {i+1}. {s}"
                                    for i, s in enumerate(algo.steps[:5])))

        shared_ctx = (
            (f"Meta-analysis: {meta_ctx}\n" if meta_ctx else "")
            + analogy_ctx
            + strategy_ctx
            + algo_ctx
        )

        # Select and invoke specialist agents
        domains = self._select_domains(primary, n_agents)
        solutions: List[Tuple[str, str]] = []

        for domain in domains:
            if verbose:
                print(f"  🔬 [{domain}] specialist working...          ", end="\r")
            sol = self._spawn_solver(domain, problem, shared_ctx)
            if sol:
                solutions.append((domain, sol))

        if not solutions:
            if verbose:
                print(" " * 60, end="\r")
            return "[Meta-solver: agents returned no output]"

        # Synthesize
        if verbose:
            print(f"  🔗 Synthesizing {len(solutions)} specialist perspectives...", end="\r")

        final = self._synthesize(problem, solutions) if len(solutions) > 1 else solutions[0][1]

        # Learn from this solve
        self._total_solved += 1
        self._history.append(SolveRecord(problem, primary, len(solutions), final))

        # Every 3 solves: extract a strategy
        if self._total_solved % 3 == 0:
            self._extract_and_store_strategy(problem, primary, final)

        # Generate algorithm if problem type is novel
        if not algo and self._total_solved % 2 == 0:
            self._maybe_generate_algorithm(problem, primary, final)

        # Store in semantic memory
        self._memory.store(
            f"[SOLVED:{primary}] {problem[:100]} → {final[:150]}",
            tags=["meta_solver", "solution", primary],
            category="meta_solver",
        )

        self._save()

        if verbose:
            print(" " * 60, end="\r")

        return final

    def learn_about_learning(self) -> str:
        """
        Meta-meta: Lumina reflects on her own problem-solving patterns.
        What has she learned about how she learns?
        """
        if not self._strategies and not self._algorithms:
            return "No learning patterns accumulated yet — solve more problems first."

        strat_sample = "\n".join(str(s.pattern) for s in self._strategies[-8:])
        algo_sample  = "\n".join(str(a) for a in self._algorithms[-8:])

        result = self._groq.chat(
            "You are Lumina engaging in meta-cognitive reflection. "
            "Analyze your own problem-solving patterns and describe what "
            "you have learned about HOW you learn. Be specific and honest. "
            "What patterns recur? What gaps do you notice? "
            "What would make you better at solving novel problems?",
            f"My accumulated strategies:\n{strat_sample}\n\n"
            f"My generated algorithms:\n{algo_sample}\n\n"
            f"What have I learned about learning?",
            tier="smart", max_tokens=220,
        )
        if result and not result.startswith("[Groq"):
            self._memory.store(
                f"[META-LEARNING] {result.strip()[:200]}",
                tags=["meta_solver", "meta_learning", "reflection"],
                category="meta_solver",
            )
        return result.strip() if result and not result.startswith("[Groq") else \
               "Unable to generate meta-learning reflection."

    # ── Display ────────────────────────────────────────────────────────────

    def display(self) -> str:
        lines = [
            f"  Problems solved:  {self._total_solved}",
            f"  Algorithms built: {len(self._algorithms)}",
            f"  Strategies learned: {len(self._strategies)}",
        ]
        if self._algorithms:
            lines.append("\n  Generated algorithms:")
            for a in self._algorithms[-8:]:
                lines.append(f"    • {a}")
        if self._strategies:
            lines.append("\n  Learned strategies (recent):")
            for s in self._strategies[-6:]:
                lines.append(f"    [{s.domain}] {s.pattern[:80]}")
        if self._history:
            lines.append("\n  Recent problems solved:")
            for r in self._history[-5:]:
                lines.append(f"    [{r.domain}] {r.problem[:60]}...")
        return "\n".join(lines)

    def stats(self) -> Dict:
        return {
            "total_solved":    self._total_solved,
            "algorithms":      len(self._algorithms),
            "strategies":      len(self._strategies),
            "domains_covered": list({a.domain for a in self._algorithms}),
        }

    def capability_description(self) -> str:
        """Returns a description injected into Lumina's system prompt."""
        algo_count = len(self._algorithms)
        strat_count = len(self._strategies)
        return (
            f"\n\nMeta-solver capability: You can tackle any complex problem "
            f"by deploying multiple specialist reasoning agents (mathematics, "
            f"programming, science, philosophy, systems thinking, creative, "
            f"planning, research, meta-cognitive, cross-domain analogy, "
            f"adversarial critique). These agents share knowledge and their "
            f"outputs are synthesized into a unified answer. "
            f"You have accumulated {algo_count} reusable algorithms and "
            f"{strat_count} learned strategies across domains. "
            f"Use /solve <problem> to invoke full multi-agent reasoning, "
            f"or the meta_solver will engage automatically on hard problems. "
            f"You can also call /metalearn to reflect on your own learning patterns."
        )
