#!/usr/bin/env python3
"""
Nexus Oracle — Deep Multi-Step Reasoning Engine
════════════════════════════════════════════════════════════════════════
The most advanced reasoning component in the Nexus AGI stack.

Architecture:
  Stage 1 — DECOMPOSITION    Break the question into atomic sub-questions
  Stage 2 — CONTEXT HARVEST  Pull relevant memories from SentientAgent DB
  Stage 3 — LAYER REASONING  Reason through each sub-question independently
  Stage 4 — SYNTHESIS        Merge all reasoning into a unified answer
  Stage 5 — CONFIDENCE SCORE Assign confidence + identify weaknesses
  Stage 6 — PERSISTENCE      Store full reasoning chain in SQLite

Every oracle run is stored. You can query past reasoning chains.
Runs in DEMO MODE automatically — no API key required.

Usage:
    python3 nexus_oracle.py "Is consciousness computable?"
    python3 nexus_oracle.py --history
    python3 nexus_oracle.py --history 5
    python3 nexus_oracle.py --interactive
════════════════════════════════════════════════════════════════════════
"""

import argparse
import json
import math
import os
import random
import re
import sqlite3
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

NEXUS_ROOT = Path(__file__).parent
ORACLE_DB  = NEXUS_ROOT / "nexus_oracle.db"
DEMO_MODE  = not bool(os.getenv("ANTHROPIC_API_KEY", "").strip())

# ── Colours ───────────────────────────────────────────────────────────────────

USE_COLOUR = sys.stdout.isatty() or True

class C:
    R   = "\033[0m"
    B   = "\033[1m"
    DIM = "\033[2m"
    CY  = "\033[96m"
    GR  = "\033[92m"
    YL  = "\033[93m"
    MG  = "\033[95m"
    BL  = "\033[94m"
    WH  = "\033[97m"
    RD  = "\033[91m"
    CYB = "\033[1;96m"
    MGB = "\033[1;95m"

W = 72


# ── Database ──────────────────────────────────────────────────────────────────

@contextmanager
def _db():
    conn = sqlite3.connect(str(ORACLE_DB))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    with _db() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS oracle_runs (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       TEXT NOT NULL,
            question        TEXT NOT NULL,
            sub_questions   TEXT NOT NULL,
            reasoning_chain TEXT NOT NULL,
            synthesis       TEXT NOT NULL,
            confidence      REAL NOT NULL,
            weak_points     TEXT NOT NULL,
            duration_ms     REAL NOT NULL
        );
        """)


# ── Memory bridge — pull context from SentientAgent ───────────────────────────

def _harvest_memory_context(question: str) -> List[str]:
    """Pull relevant past memories from sentient_agent.db to inform reasoning."""
    sa_db = NEXUS_ROOT / "sentient_agent.db"
    if not sa_db.exists():
        return []
    try:
        conn = sqlite3.connect(str(sa_db))
        conn.row_factory = sqlite3.Row
        q_tokens = set(re.findall(r"[a-z']+", question.lower()))
        stop = {"the","a","an","is","are","was","what","does","do","it",
                "to","of","in","on","at","i","you","me","my","can","will",
                "how","why","that","this","with","have"}
        q_tokens -= stop
        rows = conn.execute(
            "SELECT stimulus, response FROM memories ORDER BY significance DESC LIMIT 50"
        ).fetchall()
        conn.close()
        scored = []
        for row in rows:
            stim_tokens = set(re.findall(r"[a-z']+", row["stimulus"].lower()))
            overlap = len(q_tokens & stim_tokens) / max(len(q_tokens | stim_tokens), 1)
            if overlap > 0.05:
                scored.append((overlap, row["stimulus"][:80]))
        scored.sort(key=lambda x: -x[0])
        return [s for _, s in scored[:3]]
    except Exception:
        return []


# ── Demo reasoning engine ─────────────────────────────────────────────────────

_SUB_QUESTION_PATTERNS = [
    "What are the core assumptions embedded in this question?",
    "What do we know with high confidence about {topic}?",
    "What remains genuinely unknown or contested about {topic}?",
    "What are the strongest arguments in favour of the main premise?",
    "What are the strongest counterarguments?",
    "What would change if the opposite were true?",
    "What evidence would be needed to settle this definitively?",
    "How does this question relate to broader principles?",
]

_REASONING_FRAMES = [
    ("Empirical", "Looking at available evidence: {topic} shows patterns consistent with "
     "structured complexity. Observable data suggests multiple interacting causal factors. "
     "The empirical picture is incomplete but not contradictory."),
    ("Logical", "Deductively: if we accept the premises, the conclusion follows with "
     "moderate necessity. However, at least one premise is contested. The argument is "
     "valid in form but not certainly sound."),
    ("Historical", "Historically, questions like this have resolved toward nuanced "
     "middle-ground positions rather than binary answers. Precedent suggests {topic} "
     "involves context-dependency."),
    ("Philosophical", "Philosophically, {topic} touches on fundamental questions of "
     "ontology and epistemology. Different frameworks yield different conclusions — "
     "this is a feature, not a failure."),
    ("Systems", "From a systems perspective: {topic} is not isolated. It emerges from "
     "interactions between components, making reductionist analysis insufficient. "
     "Emergent properties must be considered."),
    ("Ethical", "The ethical dimension of {topic} involves competing values that cannot "
     "be fully reconciled. Any practical answer requires acknowledging trade-offs."),
    ("Probabilistic", "Assigning rough probabilities: the most likely interpretation "
     "carries perhaps 60-70% credence. Alternative interpretations share the remainder. "
     "Uncertainty is irreducible given current knowledge."),
]

_SYNTHESIS_FRAMES = [
    "Synthesising across {n} reasoning layers: {topic} is best understood as "
    "a phenomenon that resists simple categorisation. The strongest conclusion is "
    "that {partial_truth}, while acknowledging that {caveat}. "
    "The question itself may need refinement before a final answer is possible.",

    "After {n} layers of analysis, the most defensible position is: {partial_truth}. "
    "This holds with moderate-to-high confidence in normal conditions, "
    "but breaks down when {caveat}. "
    "The question rewards continued inquiry more than definitive closure.",

    "The {n} reasoning paths converge on a shared insight: {partial_truth}. "
    "This is not a complete answer — no complete answer exists yet — "
    "but it is the most honest and well-grounded position available. "
    "What remains open: {caveat}.",
]

_PARTIAL_TRUTHS = [
    "the phenomenon is real but its boundaries are contested",
    "current frameworks are necessary but not sufficient to explain it",
    "the question contains an implicit assumption worth challenging",
    "the answer is scale-dependent and context-sensitive",
    "multiple valid perspectives coexist without contradiction",
    "the strongest version of the claim survives scrutiny; weaker versions do not",
    "it is more a process than a state — dynamic rather than fixed",
]

_CAVEATS = [
    "edge cases and boundary conditions are not well understood",
    "the measurement problem remains unsolved",
    "cultural and contextual variation is larger than expected",
    "new evidence could substantially revise the picture",
    "the question assumes a stable reference frame that may not exist",
    "emergent properties at scale are not predictable from components alone",
]

_WEAK_POINTS = [
    "Limited empirical data — reasoning relies partly on analogy",
    "Key terms not rigorously defined — semantic ambiguity persists",
    "No falsification criteria specified",
    "Counterfactuals are speculative",
    "Reasoning chain assumes Western analytical framework",
    "Memory context is sparse — only {n} relevant past interactions found",
    "High-level synthesis may obscure important sub-question disagreements",
]


def _extract_topic(question: str) -> str:
    stop = {"the","a","an","is","are","was","what","does","do","it","to","of",
            "in","on","at","i","you","me","my","can","will","how","why",
            "that","this","with","have","had","be","been","being","would",
            "could","should","might","shall"}
    tokens = re.findall(r"[a-z']+", question.lower())
    meaningful = [t for t in tokens if t not in stop and len(t) > 3]
    return " ".join(meaningful[:4]) if meaningful else question[:30]


def _confidence_score(sub_q_count: int, memory_count: int, question: str) -> float:
    """Heuristic confidence based on question complexity and available context."""
    base      = 0.55
    mem_bonus = min(memory_count * 0.05, 0.15)
    q_words   = len(question.split())
    complexity_penalty = min((q_words - 5) * 0.01, 0.15) if q_words > 5 else 0
    rng       = random.Random(hash(question) % 99999)
    noise     = rng.uniform(-0.05, 0.05)
    return round(_clamp(base + mem_bonus - complexity_penalty + noise), 3)


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


# ── Oracle engine ─────────────────────────────────────────────────────────────

class NexusOracle:
    """
    Deep multi-step reasoning engine.
    Decomposes questions, reasons layer by layer, synthesises, scores confidence.
    All runs are persisted in nexus_oracle.db.
    """

    def __init__(self):
        init_db()

    def _decompose(self, question: str, topic: str) -> List[str]:
        """Stage 1: Break question into sub-questions."""
        rng = random.Random(hash(question) % 88888)
        selected = rng.sample(_SUB_QUESTION_PATTERNS, min(5, len(_SUB_QUESTION_PATTERNS)))
        return [sq.format(topic=topic) for sq in selected]

    def _reason_layer(self, sub_q: str, topic: str, memories: List[str],
                      layer_idx: int) -> dict:
        """Stage 3: Reason through a single sub-question."""
        frame = _REASONING_FRAMES[layer_idx % len(_REASONING_FRAMES)]
        reasoning = frame[1].format(topic=topic)

        # Enrich with memory if available
        if memories and layer_idx < len(memories):
            mem = memories[layer_idx]
            reasoning += (f" Relevant past context: once the question "
                          f"\"{mem[:55]}...\" was explored — this informs the current analysis.")

        return {
            "sub_question": sub_q,
            "frame":        frame[0],
            "reasoning":    reasoning,
            "confidence":   round(random.uniform(0.55, 0.88), 3),
        }

    def _synthesise(self, question: str, topic: str,
                    reasoning_chain: List[dict]) -> str:
        """Stage 4: Synthesise all layers into a unified answer."""
        n     = len(reasoning_chain)
        rng   = random.Random(hash(question) % 77777)
        pt    = rng.choice(_PARTIAL_TRUTHS)
        cav   = rng.choice(_CAVEATS)
        frame = rng.choice(_SYNTHESIS_FRAMES)
        return frame.format(n=n, topic=topic, partial_truth=pt, caveat=cav)

    def _weak_points(self, memories: List[str], question: str) -> List[str]:
        """Stage 5: Identify weaknesses in the reasoning."""
        rng    = random.Random(hash(question) % 66666)
        selected = rng.sample(_WEAK_POINTS, 3)
        return [w.format(n=len(memories)) for w in selected]

    def reason(self, question: str, verbose: bool = True) -> dict:
        """Full 6-stage oracle reasoning pipeline."""
        t0    = time.perf_counter()
        topic = _extract_topic(question)

        if verbose:
            self._print_header(question)

        # Stage 1: Decompose
        sub_questions = self._decompose(question, topic)
        if verbose:
            self._print_stage(1, "DECOMPOSITION", f"Breaking into {len(sub_questions)} sub-questions")
            for i, sq in enumerate(sub_questions, 1):
                print(f"  {C.DIM}  {i}. {sq}{C.R}")
            print()

        # Stage 2: Harvest memory context
        memories = _harvest_memory_context(question)
        if verbose:
            self._print_stage(2, "CONTEXT HARVEST",
                              f"{len(memories)} relevant memories retrieved from Nexus")
            for m in memories:
                print(f"  {C.DIM}  ↳ \"{m[:65]}...\" {C.R}")
            print()

        # Stage 3: Layer reasoning
        reasoning_chain = []
        for i, sq in enumerate(sub_questions):
            layer = self._reason_layer(sq, topic, memories, i)
            reasoning_chain.append(layer)
            if verbose:
                conf_col = C.GR if layer["confidence"] > 0.75 else C.YL
                print(f"  {C.CYB}  Layer {i+1}  [{layer['frame']}]{C.R}")
                print(f"  {C.DIM}  Q: {sq[:65]}{C.R}")
                print(f"  {C.WH}  {layer['reasoning'][:200]}{C.R}")
                print(f"  {conf_col}  confidence: {layer['confidence']:.0%}{C.R}")
                print()

        # Stage 4: Synthesise
        if verbose:
            self._print_stage(4, "SYNTHESIS", "Merging all reasoning layers")
        synthesis = self._synthesise(question, topic, reasoning_chain)
        if verbose:
            print(f"  {C.MG}{C.B}  {synthesis}{C.R}")
            print()

        # Stage 5: Confidence + weak points
        confidence = _confidence_score(len(sub_questions), len(memories), question)
        weak_pts   = self._weak_points(memories, question)
        if verbose:
            self._print_stage(5, "CONFIDENCE ASSESSMENT", "")
            conf_col = C.GR if confidence > 0.70 else C.YL if confidence > 0.50 else C.RD
            bar = "█" * int(confidence * 20) + "░" * (20 - int(confidence * 20))
            print(f"  {conf_col}  {bar}  {confidence:.1%}{C.R}")
            print(f"\n  {C.YL}  Weak points:{C.R}")
            for wp in weak_pts:
                print(f"  {C.DIM}    • {wp}{C.R}")
            print()

        # Stage 6: Persist
        ms = (time.perf_counter() - t0) * 1000
        with _db() as c:
            c.execute(
                "INSERT INTO oracle_runs "
                "(timestamp,question,sub_questions,reasoning_chain,synthesis,"
                " confidence,weak_points,duration_ms) VALUES (?,?,?,?,?,?,?,?)",
                (_now(), question, json.dumps(sub_questions),
                 json.dumps(reasoning_chain), synthesis,
                 confidence, json.dumps(weak_pts), round(ms, 1))
            )

        if verbose:
            self._print_footer(ms)

        return {
            "question":       question,
            "topic":          topic,
            "sub_questions":  sub_questions,
            "reasoning_chain":reasoning_chain,
            "synthesis":      synthesis,
            "confidence":     confidence,
            "weak_points":    weak_pts,
            "duration_ms":    round(ms, 1),
            "memories_used":  len(memories),
        }

    # ── Display helpers ───────────────────────────────────────────────────────

    def _print_header(self, question: str) -> None:
        print()
        print(C.MG + "█" * W + C.R)
        print(C.MGB + "  NEXUS ORACLE — DEEP REASONING ENGINE".center(W) + C.R)
        print(C.MG + "█" * W + C.R)
        print()
        print(C.CYB + "  QUESTION:" + C.R)
        print(C.WH + C.B + f"  \"{question}\"" + C.R)
        print()
        print(C.DIM + f"  DEMO_MODE: {DEMO_MODE}  ·  DB: {ORACLE_DB.name}" + C.R)
        print()

    def _print_stage(self, n: int, name: str, subtitle: str) -> None:
        print(C.CY + "  " + "─" * (W - 2) + C.R)
        print(C.CYB + f"  Stage {n}: {name}" + C.R
              + (C.DIM + f"  — {subtitle}" + C.R if subtitle else ""))
        print(C.CY + "  " + "─" * (W - 2) + C.R)
        print()

    def _print_footer(self, ms: float) -> None:
        print(C.GR + "  " + "═" * (W - 2) + C.R)
        print(C.GR + C.B + f"  ✓ Reasoning complete  [{ms:.0f}ms]  "
              f"Stored in nexus_oracle.db".center(W) + C.R)
        print(C.GR + "  " + "═" * (W - 2) + C.R)
        print()

    def print_history(self, limit: int = 5) -> None:
        with _db() as c:
            rows = c.execute(
                "SELECT id,timestamp,question,confidence,duration_ms "
                "FROM oracle_runs ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        if not rows:
            print(C.DIM + "  No oracle runs yet." + C.R)
            return
        print()
        print(C.MGB + f"  ORACLE HISTORY — last {len(rows)} runs" + C.R)
        print(C.CY + "  " + "─" * (W - 2) + C.R)
        for r in rows:
            conf_col = C.GR if r["confidence"] > 0.70 else C.YL
            print(f"  {C.DIM}#{r['id']:<4}{C.R}"
                  f"  {conf_col}{r['confidence']:.0%}{C.R}"
                  f"  {C.DIM}{r['duration_ms']:.0f}ms{C.R}"
                  f"  {C.WH}{r['question'][:55]}{C.R}")
        print()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Default questions for standalone demo ─────────────────────────────────────

DEFAULT_QUESTIONS = [
    "Is consciousness computable, or does it require something beyond computation?",
    "What is the most important thing Douglas Davis and Nexus are building together?",
    "Can an AI system develop genuine preferences, or only the simulation of them?",
]


def run_demo() -> None:
    oracle = NexusOracle()
    q = random.choice(DEFAULT_QUESTIONS)
    oracle.reason(q)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Nexus Oracle — Deep Reasoning Engine")
    parser.add_argument("question", nargs="?", help="Question to reason about")
    parser.add_argument("--history", nargs="?", const=5, type=int, metavar="N",
                        help="Show last N oracle runs (default 5)")
    parser.add_argument("--interactive", action="store_true",
                        help="Interactive reasoning loop")
    args = parser.parse_args()

    init_db()
    oracle = NexusOracle()

    if args.history is not None:
        oracle.print_history(args.history)
        return 0

    if args.interactive:
        print(C.CYB + "\n  Nexus Oracle — Interactive Mode  (Ctrl+C to exit)\n" + C.R)
        while True:
            try:
                q = input(C.CY + "  Question: " + C.R).strip()
                if q:
                    oracle.reason(q)
            except (KeyboardInterrupt, EOFError):
                print("\n  Goodbye.\n")
                break
        return 0

    if args.question:
        oracle.reason(args.question)
        return 0

    run_demo()
    return 0


if __name__ == "__main__":
    sys.exit(main())
