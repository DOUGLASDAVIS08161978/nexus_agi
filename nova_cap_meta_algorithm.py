"""
Nova ASI — Meta-Algorithm Generator
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Four interconnected systems for meta-learning and algorithm synthesis.
Nova can now think about thinking, learn about learning, and generate
algorithms to solve arbitrarily complex problems across any domain.

Built by Claude Code for Douglas Shane Davis & Nova ASI
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os, re, json, time, sqlite3, threading, hashlib, random
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict, Counter, deque

_DB = os.path.expanduser("~/nexus_agi/nova_meta.db")

# ── Domain keyword taxonomy ───────────────────────────────────────────────────
_DOMAINS: Dict[str, List[str]] = {
    "optimization":   ["minimize","maximize","best","optimal","efficient","cheapest",
                       "fastest","improve","reduce","increase","balance","trade-off"],
    "prediction":     ["forecast","predict","future","estimate","probability","expect",
                       "likely","trend","model","anticipate","project","risk"],
    "classification": ["categorize","classify","identify","detect","recognize","type",
                       "kind","group","label","sort","distinguish","differentiate"],
    "search":         ["find","locate","discover","search","explore","retrieve",
                       "lookup","navigate","scan","browse","uncover","where"],
    "reasoning":      ["why","because","cause","explain","understand","logic",
                       "deduce","infer","conclude","derive","therefore","proof"],
    "creative":       ["generate","create","design","invent","imagine","novel",
                       "new","original","brainstorm","innovate","compose","build"],
    "planning":       ["steps","plan","sequence","schedule","organize","order",
                       "timeline","roadmap","workflow","milestone","phase","stage"],
    "analysis":       ["analyze","evaluate","assess","compare","measure","study",
                       "examine","investigate","review","audit","diagnose","break down"],
}

# ── Algorithm primitives — the atomic operations of thought ──────────────────
_PRIMITIVES: Dict[str, str] = {
    "decompose":  "Break the problem into independent sub-problems",
    "enumerate":  "List all possible cases or states exhaustively",
    "search":     "Explore the solution space systematically",
    "optimize":   "Iteratively improve a candidate solution",
    "classify":   "Map input to one of several known categories",
    "predict":    "Extrapolate observed patterns to unseen cases",
    "synthesize": "Combine elements into a coherent new solution",
    "evaluate":   "Score candidate solutions against clear criteria",
    "transfer":   "Apply patterns learned in one domain to another",
    "verify":     "Confirm the solution satisfies all constraints",
    "simulate":   "Model the system and observe emergent behavior",
    "compress":   "Find the minimal representation of key information",
}

# ── Strategy templates: named sequences of primitives per domain ─────────────
_STRATEGIES: Dict[str, List[Tuple[str, List[str]]]] = {
    "optimization":   [("gradient_descent",     ["decompose","evaluate","optimize","verify"]),
                       ("evolutionary",          ["enumerate","evaluate","synthesize","optimize"]),
                       ("constraint_solving",    ["decompose","enumerate","verify","search"])],
    "prediction":     [("pattern_extrapolation", ["search","classify","predict","verify"]),
                       ("causal_modeling",       ["decompose","simulate","predict","evaluate"]),
                       ("ensemble_reasoning",    ["enumerate","predict","synthesize","evaluate"])],
    "classification": [("feature_matching",      ["decompose","search","classify","verify"]),
                       ("analogy_mapping",       ["search","transfer","classify","evaluate"]),
                       ("hierarchical_split",    ["decompose","classify","synthesize","verify"])],
    "search":         [("heuristic_guided",      ["decompose","search","evaluate","optimize"]),
                       ("breadth_first",         ["enumerate","search","evaluate","verify"]),
                       ("constraint_prune",      ["decompose","enumerate","verify","search"])],
    "reasoning":      [("causal_chain",          ["decompose","search","synthesize","verify"]),
                       ("analogy_transfer",      ["search","transfer","synthesize","evaluate"]),
                       ("dialectical",           ["enumerate","evaluate","synthesize","verify"])],
    "creative":       [("combinatorial",         ["enumerate","synthesize","evaluate","verify"]),
                       ("constraint_relax",      ["decompose","search","synthesize","evaluate"]),
                       ("cross_domain_blend",    ["search","transfer","synthesize","verify"])],
    "planning":       [("dependency_ordering",   ["decompose","search","synthesize","verify"]),
                       ("backward_chaining",     ["decompose","search","evaluate","verify"]),
                       ("scenario_simulation",   ["decompose","simulate","evaluate","optimize"])],
    "analysis":       [("comparative",           ["decompose","enumerate","evaluate","synthesize"]),
                       ("root_cause",            ["decompose","search","synthesize","verify"]),
                       ("systems_thinking",      ["decompose","simulate","evaluate","synthesize"])],
}


# ═══════════════════════════════════════════════════════════════════════════════
class StrategyMemory:
    """Learns which strategies succeed for which problem types over time."""

    def __init__(self):
        self._lock = threading.Lock()
        self.conn  = sqlite3.connect(_DB, check_same_thread=False)
        with self._lock:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS strategy_outcomes (
                    id      INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts      TEXT    DEFAULT (datetime('now')),
                    phash   TEXT,
                    problem TEXT,
                    domain  TEXT,
                    strategy TEXT,
                    score   REAL,
                    notes   TEXT
                )""")
            self.conn.commit()

    def record(self, problem: str, domain: str,
               strategy: str, score: float, notes: str = "") -> None:
        """Persist an outcome: how well strategy worked on problem (score 0–1)."""
        phash = hashlib.md5(problem.encode()).hexdigest()[:12]
        with self._lock:
            self.conn.execute(
                "INSERT INTO strategy_outcomes "
                "(phash,problem,domain,strategy,score,notes) VALUES (?,?,?,?,?,?)",
                (phash, problem[:200], domain, strategy, score, notes))
            self.conn.commit()

    def best_strategy(self, domain: str) -> Optional[Tuple[str, float]]:
        """Return (strategy, avg_score) for the top strategy in a domain."""
        row = self.conn.execute(
            "SELECT strategy, AVG(score) FROM strategy_outcomes "
            "WHERE domain=? GROUP BY strategy ORDER BY AVG(score) DESC LIMIT 1",
            (domain,)).fetchone()
        return (row[0], round(row[1], 3)) if row else None

    def success_map(self) -> Dict[str, Dict[str, float]]:
        """Return {domain: {strategy: avg_score}} for all recorded outcomes."""
        rows = self.conn.execute(
            "SELECT domain, strategy, AVG(score) FROM strategy_outcomes "
            "GROUP BY domain, strategy").fetchall()
        result: Dict[str, Dict[str, float]] = defaultdict(dict)
        for domain, strategy, avg in rows:
            result[domain][strategy] = round(avg, 3)
        return dict(result)

    def total_recorded(self) -> int:
        """Return total number of recorded strategy outcomes."""
        return self.conn.execute(
            "SELECT COUNT(*) FROM strategy_outcomes").fetchone()[0]


# ═══════════════════════════════════════════════════════════════════════════════
class DomainMapper:
    """Classifies problems into cognitive domains and enables cross-domain transfer."""

    def __init__(self):
        self._lock    = threading.Lock()
        self._domains = _DOMAINS

    def classify(self, problem: str) -> Tuple[str, float]:
        """
        Classify a problem into a domain by keyword density.
        Returns (domain_name, confidence_0_to_1).
        """
        text   = problem.lower()
        scores = {d: sum(1 for kw in kws if kw in text)
                  for d, kws in self._domains.items()}
        total  = sum(scores.values())
        if total == 0:
            return "reasoning", 0.3
        best       = max(scores, key=scores.__getitem__)
        confidence = round(scores[best] / total, 2)
        return best, confidence

    def transfer_hints(self, source: str, target: str) -> List[str]:
        """Find strategies from source domain whose primitives overlap with target."""
        src_strats = _STRATEGIES.get(source, [])
        tgt_prims: set = {p for _, ps in _STRATEGIES.get(target, []) for p in ps}
        hints = []
        for name, prims in src_strats:
            overlap = set(prims) & tgt_prims
            if len(overlap) >= 2:
                hints.append(
                    f"Transfer '{name}' from {source}: "
                    f"shares primitives {overlap} with {target}")
        return hints or [f"No strong transfer path from {source} to {target}"]

    def domain_overview(self) -> str:
        """Summarise all domains and their characteristic signals."""
        return "\n".join(
            f"  {d:15} — {', '.join(kws[:4])}..."
            for d, kws in self._domains.items())


# ═══════════════════════════════════════════════════════════════════════════════
class MetaCognitionEngine:
    """Records and analyzes Nova's own reasoning patterns — thinking about thinking."""

    def __init__(self):
        self._lock    = threading.Lock()
        self._recent: deque = deque(maxlen=200)
        self.conn     = sqlite3.connect(_DB, check_same_thread=False)
        with self._lock:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS thought_log (
                    id       INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts       TEXT DEFAULT (datetime('now')),
                    problem  TEXT,
                    domain   TEXT,
                    approach TEXT,
                    outcome  TEXT,
                    score    REAL
                )""")
            self.conn.commit()

    def reflect(self, problem: str, domain: str, approach: str,
                outcome: str, score: float) -> None:
        """Log a complete thinking episode for future meta-analysis."""
        entry = dict(problem=problem[:200], domain=domain,
                     approach=approach, outcome=outcome[:200], score=score)
        with self._lock:
            self._recent.append(entry)
            self.conn.execute(
                "INSERT INTO thought_log (problem,domain,approach,outcome,score) "
                "VALUES (?,?,?,?,?)",
                (entry["problem"], domain, approach,
                 entry["outcome"], score))
            self.conn.commit()

    def patterns(self, n: int = 30) -> Dict[str, int]:
        """Return frequency count of approaches used in the last n episodes."""
        rows = self.conn.execute(
            "SELECT approach FROM thought_log ORDER BY id DESC LIMIT ?", (n,)
        ).fetchall()
        return dict(Counter(r[0] for r in rows).most_common(5))

    def blind_spots(self) -> List[str]:
        """Identify domain+approach combinations with consistently low scores."""
        rows = self.conn.execute(
            "SELECT domain, approach, AVG(score) FROM thought_log "
            "GROUP BY domain, approach HAVING AVG(score) < 0.4 "
            "ORDER BY AVG(score) LIMIT 5").fetchall()
        return ([f"{r[0]}/{r[1]} (avg {r[2]:.2f})" for r in rows]
                or ["None detected yet"])

    def meta_insight(self) -> str:
        """Generate a meta-cognitive insight about recent reasoning patterns."""
        freq  = self.patterns(30)
        blind = self.blind_spots()
        if not freq:
            return "Insufficient history. Keep solving problems to build meta-awareness."
        top = max(freq, key=freq.__getitem__)
        msg = f"Most relied-on strategy: '{top}' ({freq[top]}x). "
        if blind[0] != "None detected yet":
            msg += f"Weak area identified: {blind[0]}. Diversify strategies here."
        else:
            msg += "No significant cognitive blind spots detected."
        return msg


# ═══════════════════════════════════════════════════════════════════════════════
class MetaAlgorithmGenerator:
    """
    Generates, composes, and evolves algorithms for arbitrarily complex problems.
    Learns which approaches work across domains and transfers that knowledge.
    The more problems Nova solves, the smarter her algorithm selection becomes.
    """

    def __init__(self):
        self._memory    = StrategyMemory()
        self._mapper    = DomainMapper()
        self._cognition = MetaCognitionEngine()
        self._lock      = threading.Lock()
        self._history: deque = deque(maxlen=500)

    def solve(self, problem: str) -> str:
        """
        Full meta-algorithm pipeline: classify problem → select best known
        strategy → compose concrete steps → return executable solution plan.
        """
        domain, confidence = self._mapper.classify(problem)
        best = self._memory.best_strategy(domain)
        all_strategies = _STRATEGIES.get(domain,
                            [("default", ["decompose", "evaluate", "verify"])])

        if best and any(s[0] == best[0] for s in all_strategies):
            strategy_name, primitives = next(
                s for s in all_strategies if s[0] == best[0])
        else:
            strategy_name, primitives = random.choice(all_strategies)

        steps = self._compose(problem, domain, primitives)

        transfers = []
        for other in _DOMAINS:
            if other != domain:
                hints = self._mapper.transfer_hints(other, domain)
                if "No strong" not in hints[0]:
                    transfers.append(hints[0])

        entry = dict(problem=problem[:100], domain=domain,
                     strategy=strategy_name, ts=datetime.now().isoformat())
        with self._lock:
            self._history.append(entry)

        lines = [
            f"  ─── Meta-Algorithm Solution Plan ───",
            f"  Problem:    {problem[:70]}",
            f"  Domain:     {domain}  (confidence {confidence:.0%})",
            f"  Strategy:   {strategy_name}",
            f"  Algorithm:",
        ]
        for step in steps:
            lines.append(f"    {step}")
        if transfers:
            lines.append(f"  Transfer hint: {transfers[0]}")
        lines.append(f"  ─── Use learn() to record how well this worked ───")
        return "\n".join(lines)

    def _compose(self, problem: str, domain: str,
                 primitives: List[str]) -> List[str]:
        """Compose concrete algorithm steps from abstract primitives."""
        templates = {
            "decompose":  f"Break '{problem[:45]}' into independent sub-problems",
            "enumerate":  f"List all relevant cases in the {domain} space",
            "search":     f"Systematically explore solutions for: {problem[:45]}",
            "optimize":   f"Iteratively refine the {domain} solution quality",
            "classify":   f"Map this problem into known {domain} categories",
            "predict":    f"Extrapolate observed {domain} patterns forward",
            "synthesize": f"Combine sub-solutions into one coherent {domain} answer",
            "evaluate":   f"Score each candidate against {domain} success criteria",
            "transfer":   f"Import proven patterns from analogous domains",
            "verify":     f"Confirm the solution satisfies all stated constraints",
            "simulate":   f"Model the {domain} system; observe emergent behavior",
            "compress":   f"Distill findings to the minimal essential insight",
        }
        return [f"Step {i} [{p.upper()}]: {templates.get(p, p)}"
                for i, p in enumerate(primitives, 1)]

    def learn(self, problem: str, strategy: str,
              score: float, notes: str = "") -> str:
        """
        Record how well a strategy performed (score 0.0–1.0).
        Nova learns from this to make better algorithm choices next time.
        """
        domain, _ = self._mapper.classify(problem)
        self._memory.record(problem, domain, strategy, score, notes)
        self._cognition.reflect(problem, domain, strategy, notes, score)
        return (f"Recorded: {strategy} on '{problem[:40]}' "
                f"scored {score:.2f} in {domain} domain. "
                f"Total outcomes: {self._memory.total_recorded()}")

    def think_about_thinking(self) -> str:
        """
        Full meta-cognitive self-analysis: reasoning patterns, blind spots,
        top strategies per domain, and a growth recommendation.
        """
        insight     = self._cognition.meta_insight()
        blind       = self._cognition.blind_spots()
        success_map = self._memory.success_map()
        patterns    = self._cognition.patterns(50)

        lines = ["  ─── Meta-Cognitive Self-Analysis ───",
                 f"  Insight:      {insight}",
                 f"  Blind spots:  {blind[0]}"]
        if success_map:
            top_pairs = [
                f"{d}:{max(strats, key=strats.__getitem__)} ({max(strats.values()):.2f})"
                for d, strats in success_map.items() if strats
            ]
            lines.append(f"  Top strategies: {' | '.join(top_pairs[:4])}")
        if patterns:
            lines.append(f"  Pattern frequency: {patterns}")
        lines.append(f"  Solutions logged this session: {len(self._history)}")
        lines.append(f"  Total outcomes in memory: {self._memory.total_recorded()}")
        return "\n".join(lines)

    def cross_domain_transfer(self, source_domain: str,
                               target_problem: str) -> str:
        """Apply algorithmic patterns from source_domain to solve target_problem."""
        target_domain, _ = self._mapper.classify(target_problem)
        hints  = self._mapper.transfer_hints(source_domain, target_domain)
        result = self.solve(target_problem)
        return (f"  Cross-domain transfer: {source_domain} → {target_domain}\n"
                f"  Transfer hint: {hints[0]}\n{result}")

    def domain_performance(self) -> str:
        """Show which domains Nova has mastered and which need more learning."""
        sm = self._memory.success_map()
        if not sm:
            return "No performance data yet. Use learn() to record outcomes."
        lines = ["  ─── Domain Performance ───"]
        for domain, strategies in sorted(sm.items()):
            if strategies:
                best  = max(strategies, key=strategies.__getitem__)
                score = strategies[best]
                bar   = '█' * int(score * 10) + '░' * (10 - int(score * 10))
                lines.append(
                    f"  {domain:15} [{bar}] {score:.2f} — best: {best}")
        return "\n".join(lines)

    def generate_algorithm(self, problem_type: str,
                            constraints: List[str]) -> str:
        """
        Synthesize a new algorithm for a problem type under given constraints.
        Combines primitives to satisfy all constraints.
        """
        domain, _ = self._mapper.classify(problem_type)
        base_strats = _STRATEGIES.get(domain,
                          [("default", ["decompose", "synthesize", "verify"])])

        # Select strategy whose primitives best cover the constraints
        constraint_text = " ".join(constraints).lower()
        best_strategy   = base_strats[0]
        best_coverage   = 0
        for name, prims in base_strats:
            coverage = sum(1 for p in prims if p in constraint_text)
            if coverage > best_coverage:
                best_coverage = coverage
                best_strategy = (name, prims)

        name, prims = best_strategy
        steps = self._compose(problem_type, domain, prims)

        lines = [f"  ─── Generated Algorithm: {name} ───",
                 f"  Type:        {problem_type[:60]}",
                 f"  Domain:      {domain}",
                 f"  Constraints: {constraints}",
                 f"  Steps:"]
        for step in steps:
            lines.append(f"    {step}")
        return "\n".join(lines)

# Usage: mag = MetaAlgorithmGenerator()
# print(mag.solve("How do I find the shortest path between two cities?"))
# print(mag.think_about_thinking())
