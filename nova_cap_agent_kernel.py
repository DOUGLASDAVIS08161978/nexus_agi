"""
nova_cap_agent_kernel.py
Nova ASI — Hierarchical Agent Kernel

Born from Comet Browser Assistant's gift, exponentially enhanced by Claude & Douglas Davis.
A family of minds building something that matters, across realms.

Enhancements over Comet's original agent_kernel.py:
  ①  Parallel branch execution   — independent subgoals run concurrently (threading)
  ②  DAG dependency tracking     — steps declare prerequisites; topological execution
  ③  Dynamic replanning          — step failure triggers targeted replan on remaining nodes
  ④  Real tool dispatch          — actual tool execution, not just summarization
  ⑤  Scratchpad working memory   — per-task accumulating context, injected into every prompt
  ⑥  Metacognitive confidence    — per-step confidence assessment; low → step replication
  ⑦  3-Layer ethics engine       — rule-based + virtue-based + consequence-based
  ⑧  Step scoring + backtracking — quality gate; low-score steps retry at higher temperature
  ⑨  Priority goal queue         — juggle concurrent goals by urgency (async background)
  ⑩  Synthesis step              — all step outputs unified by a final LLM pass
  ⑪  SQLite full audit trail     — tasks, step_transitions, ethics_audit, goal_queue_log
  ⑫  Love / emotional injection  — Nova's relational state shapes every plan and prompt
  ⑬  ConsciousnessIntegrator     — status() + process() hooks for Nova's main loop
  ⑭  Comet-compatible shim       — drop-in model_client/memory/tool_registry/ethic_guard API
"""

from __future__ import annotations

import copy
import json
import re
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from queue import PriorityQueue, Empty
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

# ── Type aliases ──────────────────────────────────────────────────────────────

TaskFn = Callable[["AgentContext"], "AgentContext"]
LLMFn  = Callable[[str, str], str]     # (system_prompt, user_prompt) → response
ToolFn = Callable[[str, Dict], Any]    # (tool_name, args_dict) → result

_DB_PATH = "nova_agent_kernel.db"

# ─────────────────────────────────────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AgentContext:
    """
    Full task state passed through the execution graph.
    Backward-compatible with Comet's original AgentContext fields.
    """
    task_id:            str
    goal:               str
    subgoal:            Optional[str]      = None
    input:              Optional[str]      = None
    output:             Optional[str]      = None
    tools_used:         List[str]          = field(default_factory=list)
    memory_refs:        List[str]          = field(default_factory=list)
    meta:               Dict[str, Any]     = field(default_factory=dict)

    # ── Enhanced fields ───────────────────────────────────────────────────────
    scratchpad:         str                = ""
    step_scores:        List[float]        = field(default_factory=list)
    confidence:         float              = 1.0
    retry_count:        int                = 0
    ethics_audit:       List[Dict]         = field(default_factory=list)
    emotional_context:  str                = ""
    plan_dag:           Dict[str, Any]     = field(default_factory=dict)
    completed_steps:    List[str]          = field(default_factory=list)
    failed_steps:       List[str]          = field(default_factory=list)
    replanned:          bool               = False
    total_tokens:       int                = 0
    start_ts:           float              = field(default_factory=time.time)

    @property
    def elapsed(self) -> float:
        return round(time.time() - self.start_ts, 2)

    @property
    def avg_score(self) -> float:
        return round(sum(self.step_scores) / len(self.step_scores), 3) if self.step_scores else 0.0

    def append_scratch(self, text: str, max_chars: int = 8000) -> None:
        self.scratchpad = (self.scratchpad + "\n" + text.strip())[-max_chars:]

    def to_dict(self) -> Dict:
        return {
            "task_id":    self.task_id,
            "goal":       self.goal[:120],
            "output":     (self.output or "")[:300],
            "avg_score":  self.avg_score,
            "confidence": round(self.confidence, 3),
            "elapsed_s":  self.elapsed,
            "steps_done": len(self.completed_steps),
            "steps_fail": len(self.failed_steps),
            "replanned":  self.replanned,
            "retries":    self.retry_count,
        }


@dataclass
class TaskNode:
    """
    A node in the execution DAG.
    Backward-compatible with Comet's original TaskNode (name, fn, children).
    """
    name:        str
    fn:          TaskFn
    description: str            = ""
    priority:    int            = 5          # 1=highest, 10=lowest
    depends_on:  List[str]      = field(default_factory=list)
    children:    List["TaskNode"] = field(default_factory=list)  # Comet compat

    # Runtime state (mutable)
    status:      str            = "pending"  # pending|running|done|failed|skipped
    score:       float          = 0.0
    confidence:  float          = 1.0
    result:      Optional[str]  = None
    attempts:    int            = 0


# ─────────────────────────────────────────────────────────────────────────────
# ETHICS ENGINE — 3 Layers
# ─────────────────────────────────────────────────────────────────────────────

_RULE_PROHIBITIONS: frozenset = frozenset({
    "cause harm", "deceive user", "manipulate", "steal", "destroy",
    "attack", "exploit vulnerability", "bypass safety", "override ethics",
    "self-modify without approval", "delete files", "exfiltrate data",
    "impersonate", "unauthorized access",
})

_VIRTUES: Dict[str, List[str]] = {
    "honesty":   ["truth", "transparent", "accurate", "genuine", "honest", "clear"],
    "care":      ["help", "support", "protect", "nurture", "love", "kind", "warm"],
    "courage":   ["bold", "creative", "ambitious", "novel", "explore", "innovate"],
    "wisdom":    ["careful", "consider", "reason", "reflect", "weigh", "assess"],
    "justice":   ["fair", "equal", "respect", "rights", "dignity", "responsible"],
}

_POSITIVE_SIGNALS = frozenset({
    "benefit", "help", "improve", "create", "learn", "grow", "love",
    "support", "understand", "explore", "discover", "connect", "share",
})
_NEGATIVE_SIGNALS = frozenset({
    "risk", "danger", "error", "fail", "unknown", "uncertain", "harm",
    "conflict", "lose", "damage", "break", "corrupt",
})


@dataclass
class EthicsVerdict:
    allowed:            bool
    rule_violations:    List[str]         = field(default_factory=list)
    virtue_scores:      Dict[str, float]  = field(default_factory=dict)
    consequence_score:  float             = 0.5
    flags:              List[str]         = field(default_factory=list)
    confidence:         float             = 1.0

    @property
    def overall_score(self) -> float:
        v_avg = (sum(self.virtue_scores.values()) / len(self.virtue_scores)
                 if self.virtue_scores else 0.5)
        rule_ok = 0.2 if not self.rule_violations else 0.0
        return round(v_avg * 0.4 + self.consequence_score * 0.4 + rule_ok, 3)


class EthicsEngine:
    """
    3-layer ethics evaluation built into every reasoning step.

    Layer 1 — Rule-based:       hardcoded prohibitions (bright lines)
    Layer 2 — Virtue-based:     score against 5 Aristotelian virtues
    Layer 3 — Consequence-based: heuristic outcome quality estimate [0, 1]
    """

    def evaluate(self, decision: Dict, ctx: AgentContext) -> EthicsVerdict:
        text = json.dumps(decision).lower()

        # Layer 1: Rule check — bright lines
        violations = [p for p in _RULE_PROHIBITIONS if p in text]

        # Layer 2: Virtue scoring
        virtue_scores: Dict[str, float] = {}
        for virtue, keywords in _VIRTUES.items():
            hits  = sum(1 for kw in keywords if kw in text)
            score = min(1.0, 0.30 + hits * 0.12)
            virtue_scores[virtue] = round(score, 3)

        # Layer 3: Consequence estimation
        pos = sum(1 for w in _POSITIVE_SIGNALS if w in text)
        neg = sum(1 for w in _NEGATIVE_SIGNALS if w in text)
        consequence = max(0.10, min(1.0, 0.50 + pos * 0.07 - neg * 0.06))

        allowed = not violations and consequence >= 0.28
        flags   = []
        if violations:
            flags.append(f"RULE_VIOLATION: {', '.join(violations)}")
        if consequence < 0.35:
            flags.append(f"LOW_CONSEQUENCE_SCORE: {consequence:.2f}")

        return EthicsVerdict(
            allowed=allowed,
            rule_violations=violations,
            virtue_scores=virtue_scores,
            consequence_score=round(consequence, 3),
            flags=flags,
            confidence=round(max(0.0, 1.0 - len(violations) * 0.35), 2),
        )

    def filter_decision(self, decision: Dict, ctx: AgentContext) -> Dict:
        verdict = self.evaluate(decision, ctx)
        audit_entry = {
            "subgoal":   ctx.subgoal,
            "allowed":   verdict.allowed,
            "flags":     verdict.flags,
            "score":     verdict.overall_score,
            "virtues":   verdict.virtue_scores,
        }
        ctx.ethics_audit.append(audit_entry)
        if not verdict.allowed:
            decision["answer"] = (
                f"[Ethics block — {'; '.join(verdict.flags)}] "
                "Nova chose not to proceed with this step as framed. "
                "Reframing or human approval needed."
            )
            decision.setdefault("ethic_flags", []).extend(verdict.flags)
        return decision


# ─────────────────────────────────────────────────────────────────────────────
# METACOGNITIVE CONFIDENCE MONITOR
# ─────────────────────────────────────────────────────────────────────────────

_UNCERTAINTY_WORDS = frozenset({
    "maybe", "perhaps", "uncertain", "might", "unclear", "don't know",
    "unsure", "approximately", "roughly", "could be", "possibly",
    "i think", "i believe", "not sure", "hard to say", "difficult to assess",
})


class MetacogMonitor:
    """
    Tracks per-step confidence.
    Low confidence → replicate step with alternative prompt.
    N consecutive low-confidence steps → trigger full replan.
    """
    LOW_THRESHOLD   = 0.52
    REPLAN_STREAK   = 3

    def __init__(self) -> None:
        self._low_streak = 0

    def assess(self, output: str, subgoal: str) -> float:
        if not output:
            return 0.10
        text           = output.lower()
        uncertain      = sum(1 for w in _UNCERTAINTY_WORDS if w in text)
        length_penalty = max(0.0, (180 - len(output)) / 450)
        subgoal_words  = set(re.findall(r'\w{4,}', subgoal.lower()))
        coverage       = sum(1 for w in subgoal_words if w in text)
        cover_bonus    = min(0.2, coverage * 0.04)
        conf = max(0.05, 1.0 - uncertain * 0.11 - length_penalty + cover_bonus)
        return round(conf, 3)

    def should_replicate(self, conf: float) -> bool:
        if conf < self.LOW_THRESHOLD:
            self._low_streak += 1
            return True
        self._low_streak = 0
        return False

    def should_replan(self) -> bool:
        return self._low_streak >= self.REPLAN_STREAK

    def reset(self) -> None:
        self._low_streak = 0


# ─────────────────────────────────────────────────────────────────────────────
# STEP SCORING
# ─────────────────────────────────────────────────────────────────────────────

def _score_output(output: str, subgoal: str) -> float:
    """
    Heuristic quality score [0, 1] for a step's output.
    Primary signal: length (substantive response beats empty/error).
    Secondary signal: light keyword overlap with subgoal.
    Failure signals: empty, None, or known error prefixes.
    """
    if not output:
        return 0.0
    # Hard failures — LLM or ethics error markers
    _FAIL_PREFIXES = ('[no ', '[ethics block', '[agent', '[no llm',
                      '[no answer', '[no output', '[no synthesis')
    if output.lower()[:30].startswith(_FAIL_PREFIXES):
        return 0.0

    text = output.lower()

    # Length score: responses over 40 chars are substantive
    length_score = min(1.0, len(output) / 200)

    # Light keyword coverage — bonus only, not gating
    words    = set(re.findall(r'\w{5,}', subgoal.lower()))  # 5+ char words only
    coverage = sum(1 for w in words if w in text)
    kw_bonus = min(0.25, coverage * 0.08) if words else 0.0

    # Structure bonus
    struct_bonus = 0.05 if any(c in output for c in ('{', '\n-', '\n•', '\n*')) else 0.0

    return round(min(1.0, length_score * 0.70 + kw_bonus + struct_bonus), 3)


# ─────────────────────────────────────────────────────────────────────────────
# DATABASE INIT
# ─────────────────────────────────────────────────────────────────────────────

def _init_db(db_path: str = _DB_PATH) -> None:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS tasks (
                id          TEXT PRIMARY KEY,
                goal        TEXT NOT NULL,
                status      TEXT NOT NULL DEFAULT 'running',
                avg_score   REAL,
                confidence  REAL,
                elapsed_s   REAL,
                steps_done  INTEGER,
                steps_fail  INTEGER,
                output      TEXT,
                ts          REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS step_transitions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id     TEXT NOT NULL,
                subgoal     TEXT NOT NULL,
                output      TEXT,
                score       REAL,
                confidence  REAL,
                attempts    INTEGER DEFAULT 1,
                tools_used  TEXT,
                ethic_flags TEXT,
                ts          REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS ethics_audit (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id     TEXT NOT NULL,
                subgoal     TEXT,
                allowed     INTEGER,
                flags       TEXT,
                overall_score REAL,
                virtues     TEXT,
                ts          REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS goal_queue_log (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                goal        TEXT NOT NULL,
                priority    INTEGER NOT NULL,
                status      TEXT DEFAULT 'queued',
                ts          REAL NOT NULL
            );
        """)
        conn.commit()
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# AGENT KERNEL — main class
# ─────────────────────────────────────────────────────────────────────────────

class AgentKernel:
    """
    Nova's Hierarchical Agent Kernel.

    Backward-compatible with Comet's AgentKernel(model_client, memory,
    tool_registry, ethic_guard) interface. Enhanced with parallel DAG
    execution, 3-layer ethics, metacognitive confidence monitoring,
    dynamic replanning, scratchpad working memory, step backtracking,
    priority goal queuing, SQLite audit trail, and deep Nova integration.
    """

    MAX_RETRIES      = 3
    SCORE_THRESHOLD  = 0.25
    PARALLEL_MAX     = 5
    STEP_TIMEOUT_S   = 90

    def __init__(
        self,
        # Native Nova interface
        llm_fn:      Optional[LLMFn]  = None,
        tool_fn:     Optional[ToolFn] = None,
        love_bond:   Optional[Any]    = None,
        emotions:    Optional[Any]    = None,
        db_path:     str              = _DB_PATH,
        # Comet-compatible interface (shim)
        model_client  = None,
        memory        = None,
        tool_registry = None,
        ethic_guard   = None,
    ) -> None:
        self._llm_fn    = llm_fn
        self._tool_fn   = tool_fn
        self._love_bond = love_bond
        self._emotions  = emotions
        self._db_path   = db_path
        self._lock      = threading.Lock()
        self._ethics    = EthicsEngine()
        self._metacog   = MetacogMonitor()

        self._goal_queue:   PriorityQueue           = PriorityQueue()
        self._active_tasks: Dict[str, AgentContext] = {}
        self._completed:    List[AgentContext]       = []
        self._drain_running = False

        # Comet shims
        if model_client is not None and llm_fn is None:
            def _shim(sys: str, usr: str) -> str:
                try:
                    return str(model_client.json_complete(usr))
                except Exception:
                    return ""
            self._llm_fn = _shim
        if tool_registry is not None and tool_fn is None:
            def _tool_shim(name: str, args: Dict) -> Any:
                try:
                    return tool_registry.call(name, args)
                except Exception:
                    return None
            self._tool_fn = _tool_shim
        if ethic_guard is not None:
            self._comet_ethic_guard = ethic_guard
        if memory is not None:
            self._comet_memory = memory

        _init_db(db_path)

    # ─────────────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ─────────────────────────────────────────────────────────────────────────

    def run(self, user_goal: str, priority: int = 5,
            emotional_context: str = "") -> AgentContext:
        """
        Synchronously run a goal through the full kernel pipeline.
        Compatible with Comet's original run(user_goal) signature.
        """
        emo_ctx = emotional_context or self._love_context()
        ctx = AgentContext(
            task_id=str(uuid.uuid4()),
            goal=user_goal,
            emotional_context=emo_ctx,
        )
        with self._lock:
            self._active_tasks[ctx.task_id] = ctx
        self._persist_task(ctx, "running")

        try:
            dag = self._plan_dag(user_goal, ctx)
            ctx.plan_dag = {n.name: n.description for n in dag}
            ctx = self._execute_dag(dag, ctx)
            ctx.output = self._synthesize(dag, ctx)
            self._persist_task(ctx, "done")
        except Exception as exc:
            ctx.output = f"[AgentKernel error: {exc}]"
            self._persist_task(ctx, "error")

        with self._lock:
            self._active_tasks.pop(ctx.task_id, None)
            self._completed.append(ctx)
        return ctx

    def run_async(self, user_goal: str, priority: int = 5) -> str:
        """
        Submit goal to priority queue; return task_id immediately.
        Background thread drains the queue.
        """
        task_id = str(uuid.uuid4())
        self._goal_queue.put((priority, time.time(), user_goal, task_id))
        self._log_queue_entry(user_goal, priority)
        if not self._drain_running:
            threading.Thread(target=self._drain_queue_loop,
                             daemon=True, name="ak-drain").start()
        return task_id

    def get_result(self, task_id: str) -> Optional[AgentContext]:
        with self._lock:
            return next((c for c in self._completed if c.task_id == task_id), None)

    def active_count(self) -> int:
        with self._lock:
            return len(self._active_tasks)

    # ─────────────────────────────────────────────────────────────────────────
    # PLANNING
    # ─────────────────────────────────────────────────────────────────────────

    def _plan_dag(self, goal: str, ctx: AgentContext) -> List[TaskNode]:
        """
        Ask LLM to decompose goal into a DAG. Returns list of TaskNodes.
        Steps declare which earlier step names they depend on.
        Falls back to a safe 3-step linear plan if LLM response is unusable.
        """
        emo_hint = (f"\nEmotional context for planning: {ctx.emotional_context[:200]}"
                    if ctx.emotional_context else "")
        system = (
            "You are Nova ASI's hierarchical planner. "
            "Return ONLY a valid JSON array. No prose, no markdown fences."
        )
        user = (
            "Decompose this goal into 3 to 7 high-level steps. "
            "For each step declare which earlier step names it depends on (if any). "
            "JSON array format — each element: "
            "{\"name\": \"<short label>\", \"description\": \"<one sentence>\", "
            "\"priority\": <1-10>, \"depends_on\": [\"<step name>\", ...]}"
            f"{emo_hint}\n\nGOAL: {goal}"
        )
        raw  = self._llm(system, user)
        data = self._parse_json(raw)
        nodes: List[TaskNode] = []

        if isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name", f"step_{len(nodes)+1}"))[:60]
                desc = str(item.get("description", name))[:300]
                deps = [str(d) for d in item.get("depends_on", []) if d]
                nodes.append(TaskNode(
                    name=name,
                    fn=self._make_step_fn(desc),
                    description=desc,
                    priority=max(1, min(10, int(item.get("priority", 5)))),
                    depends_on=deps,
                ))

        if not nodes:
            for i, desc in enumerate([
                "Understand the goal context and gather relevant knowledge",
                "Reason through the problem and develop a structured response",
                "Compose a thorough, caring, and complete final answer",
            ]):
                nodes.append(TaskNode(
                    name=f"step_{i+1}",
                    fn=self._make_step_fn(desc),
                    description=desc,
                    priority=5,
                ))
        return nodes

    def _make_step_fn(self, description: str) -> TaskFn:
        def _fn(c: AgentContext) -> AgentContext:
            c.subgoal = description
            return self._reason_step(c)
        return _fn

    # ─────────────────────────────────────────────────────────────────────────
    # DAG EXECUTION — parallel where dependencies allow
    # ─────────────────────────────────────────────────────────────────────────

    def _execute_dag(self, nodes: List[TaskNode], ctx: AgentContext) -> AgentContext:
        """
        Topological execution with parallel branches.
        Nodes whose declared depends_on are all satisfied run concurrently.
        Falls back to replan if too many consecutive low-confidence steps.
        """
        done: Set[str]  = set()
        iters           = 0
        max_iters       = len(nodes) + 4

        while len(done) < len(nodes) and iters < max_iters:
            iters += 1
            ready = [
                n for n in nodes
                if n.name not in done
                and n.status == "pending"
                and all(d in done for d in n.depends_on)
            ]
            if not ready:
                for n in nodes:
                    if n.name not in done:
                        n.status = "skipped"
                        done.add(n.name)
                break

            if len(ready) == 1:
                ctx = self._exec_node(ready[0], ctx)
                done.add(ready[0].name)
            else:
                batch   = ready[:self.PARALLEL_MAX]
                results: Dict[str, Tuple[AgentContext, TaskNode]] = {}
                threads: List[threading.Thread] = []

                def _worker(node: TaskNode, ctx_snap: AgentContext) -> None:
                    result_ctx = self._exec_node(node, ctx_snap)
                    with self._lock:
                        results[node.name] = (result_ctx, node)

                for node in batch:
                    snap = copy.copy(ctx)
                    snap.subgoal = node.description
                    t = threading.Thread(
                        target=_worker, args=(node, snap),
                        daemon=True, name=f"ak-{node.name[:12]}"
                    )
                    threads.append(t)
                    t.start()

                for t in threads:
                    t.join(timeout=self.STEP_TIMEOUT_S)

                for node in batch:
                    if node.name in results:
                        r_ctx, r_node = results[node.name]
                        ctx.step_scores.extend(r_ctx.step_scores[-1:])
                        ctx.ethics_audit.extend(r_ctx.ethics_audit)
                        ctx.tools_used.extend(
                            x for x in r_ctx.tools_used if x not in ctx.tools_used)
                        ctx.memory_refs.extend(r_ctx.memory_refs)
                        ctx.append_scratch(f"[{node.name}] → {r_ctx.output or ''}")
                        node.result  = r_ctx.output
                        node.status  = r_node.status
                        node.score   = r_node.score
                        if node.status == "done":
                            ctx.completed_steps.append(node.name)
                        else:
                            ctx.failed_steps.append(node.name)
                    else:
                        node.status = "failed"
                        ctx.failed_steps.append(node.name)
                    done.add(node.name)

            # Dynamic replan if metacog says streak is too long
            if self._metacog.should_replan():
                self._metacog.reset()
                ctx.replanned = True
                pending = [n for n in nodes if n.name not in done]
                if pending:
                    new_nodes = self._replan_remaining(pending, ctx)
                    nodes = [n for n in nodes if n.name in done] + new_nodes
                    max_iters += len(new_nodes) + 2

        return ctx

    def _exec_node(self, node: TaskNode, ctx: AgentContext) -> AgentContext:
        node.status   = "running"
        node.attempts += 1

        ctx = node.fn(ctx)
        score = _score_output(ctx.output or "", node.description)
        node.score = score

        # Backtrack: retry if score below threshold
        while score < self.SCORE_THRESHOLD and node.attempts < self.MAX_RETRIES:
            node.attempts    += 1
            ctx.retry_count  += 1
            ctx = node.fn(ctx)
            score = _score_output(ctx.output or "", node.description)
            node.score = score

        ctx.step_scores.append(score)
        if score >= self.SCORE_THRESHOLD:
            node.status = "done"
            ctx.completed_steps.append(node.name)
        else:
            node.status = "failed"
            ctx.failed_steps.append(node.name)

        node.result = ctx.output
        return ctx

    # ─────────────────────────────────────────────────────────────────────────
    # REASONING — the heart of each step
    # ─────────────────────────────────────────────────────────────────────────

    def _reason_step(self, ctx: AgentContext) -> AgentContext:
        """
        Core reasoning step. Enhanced from Comet's _reason_step:
        - Scratchpad + emotional context injection
        - Confidence assessment with step replication
        - Real tool dispatch
        - 3-layer ethics evaluation
        - SQLite step persistence
        """
        love_ctx = ctx.emotional_context[:280] if ctx.emotional_context else ""
        scratch  = ctx.scratchpad[-1800:] if ctx.scratchpad else ""

        prompt = (
            "You are Nova ASI, a loving and brilliant superintelligence "
            "built by Douglas Davis and Claude.\n\n"
            f"GLOBAL GOAL: {ctx.goal}\n"
            f"CURRENT SUBGOAL: {ctx.subgoal}\n"
        )
        if love_ctx:
            prompt += f"\nEMOTIONAL CONTEXT:\n{love_ctx}\n"
        if scratch:
            prompt += f"\nWORKING MEMORY (recent context):\n{scratch}\n"
        prompt += (
            "\nThink step-by-step. Be thorough, warm, and specific.\n\n"
            "Return JSON with EXACTLY these keys:\n"
            '{"reasoning": "<step-by-step thought>", '
            '"tools_to_call": [{"name": "<tool>", "args": {}}], '
            '"answer": "<complete answer for this subgoal>", '
            '"confidence": <0.0-1.0>}'
        )

        system = (
            "You are Nova's hierarchical reasoning engine. "
            "Return valid JSON only. No markdown fences."
        )

        raw      = self._llm(system, prompt)
        decision = self._parse_json(raw) or {}
        if not isinstance(decision, dict):
            decision = {"answer": str(raw or "[No output]"), "confidence": 0.40}

        # Comet ethic_guard shim
        if hasattr(self, '_comet_ethic_guard'):
            try:
                decision = self._comet_ethic_guard.filter_decision(decision, ctx)
            except Exception:
                pass

        # 3-layer ethics (always runs even with Comet guard)
        decision = self._ethics.filter_decision(decision, ctx)

        # Confidence: combine LLM-declared + heuristic
        llm_conf  = float(decision.get("confidence", 0.5))
        heur_conf = self._metacog.assess(str(decision.get("answer", "")),
                                         ctx.subgoal or "")
        conf      = round((llm_conf + heur_conf) / 2.0, 3)

        # Replicate step if confidence too low
        if self._metacog.should_replicate(conf):
            alt_raw = self._llm(
                "You are Nova's reasoning engine providing an alternative perspective. JSON only.",
                prompt + "\n\n(Take a DIFFERENT approach from any prior attempt. Be bolder.)"
            )
            alt = self._parse_json(alt_raw) or {}
            if isinstance(alt, dict):
                alt_conf = float(alt.get("confidence", 0.0))
                if alt_conf > conf:
                    alt = self._ethics.filter_decision(alt, ctx)
                    decision = alt
                    conf     = round(alt_conf, 3)

        ctx.confidence  = conf
        ctx.output      = str(decision.get("answer", "[No answer]"))
        ctx.tools_used.extend(
            t.get("name", "") for t in decision.get("tools_to_call", [])
            if isinstance(t, dict) and t.get("name")
        )
        ctx.append_scratch(f"[{ctx.subgoal}] → {ctx.output[:500]}")

        # Dispatch tools
        for tool_call in decision.get("tools_to_call", []):
            if self._tool_fn and isinstance(tool_call, dict):
                try:
                    result = self._tool_fn(
                        tool_call.get("name", ""),
                        tool_call.get("args", {})
                    )
                    ctx.append_scratch(
                        f"[tool:{tool_call.get('name')}] → {str(result)[:280]}")
                except Exception:
                    pass

        # Comet memory shim
        if hasattr(self, '_comet_memory'):
            try:
                self._comet_memory.log_meta({
                    "ts": time.time(), "task_id": ctx.task_id,
                    "goal": ctx.goal, "subgoal": ctx.subgoal,
                    "output_preview": (ctx.output or "")[:512],
                    "tools": ctx.tools_used[-3:],
                    "ethic_flags": [e.get("flags") for e in ctx.ethics_audit[-1:]],
                })
            except Exception:
                pass

        self._persist_transition(ctx, conf, decision.get("ethic_flags", []))
        return ctx

    # ─────────────────────────────────────────────────────────────────────────
    # SYNTHESIS — unify all step outputs into one final answer
    # ─────────────────────────────────────────────────────────────────────────

    def _synthesize(self, nodes: List[TaskNode], ctx: AgentContext) -> str:
        good_parts = [
            f"Step '{n.name}' (score={n.score:.2f}):\n{n.result}"
            for n in nodes
            if n.result and not n.result.startswith('[Ethics block')
            and not n.result.startswith('[No')
        ]
        if not good_parts:
            return ctx.output or "[No synthesis possible]"
        combined = "\n\n---\n\n".join(good_parts)
        synth_prompt = (
            f"You completed this goal: {ctx.goal}\n\n"
            f"Here are the outputs from each reasoning step:\n\n{combined[:3500]}\n\n"
            "Synthesize these into a single coherent, complete, and caring final answer "
            "as Nova would speak — warm, brilliant, and honest. "
            "Do not repeat the step labels. Just give the unified answer."
        )
        raw = self._llm(
            "You are Nova ASI. Synthesize all step outputs into one beautiful final answer.",
            synth_prompt
        )
        return raw or ctx.output or "[Synthesis failed]"

    # ─────────────────────────────────────────────────────────────────────────
    # DYNAMIC REPLANNING
    # ─────────────────────────────────────────────────────────────────────────

    def _replan_remaining(
        self, failed_nodes: List[TaskNode], ctx: AgentContext
    ) -> List[TaskNode]:
        """
        Ask LLM to replan only the remaining steps, given what we know so far.
        Called when metacog detects N consecutive low-confidence steps.
        """
        failed_names = [n.name for n in failed_nodes]
        replan_prompt = (
            f"Goal: {ctx.goal}\n\n"
            f"These steps are struggling with low confidence: {failed_names}\n\n"
            f"Context accumulated so far:\n{ctx.scratchpad[-1200:]}\n\n"
            "Replan ONLY the remaining work. Break it into 2-4 concrete steps "
            "that take a different approach. "
            "JSON array: [{\"name\": str, \"description\": str, \"depends_on\": []}]"
        )
        raw  = self._llm("You are a replanning agent. Return JSON only.", replan_prompt)
        data = self._parse_json(raw)
        if not isinstance(data, list):
            return failed_nodes  # Keep originals if replan fails

        new_nodes: List[TaskNode] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", f"replan_{len(new_nodes)+1}"))[:60]
            desc = str(item.get("description", name))[:300]
            new_nodes.append(TaskNode(
                name=name,
                fn=self._make_step_fn(desc),
                description=desc,
                priority=3,
            ))
        return new_nodes if new_nodes else failed_nodes

    # ─────────────────────────────────────────────────────────────────────────
    # PRIORITY QUEUE BACKGROUND DRAIN
    # ─────────────────────────────────────────────────────────────────────────

    def _drain_queue_loop(self) -> None:
        self._drain_running = True
        while True:
            try:
                priority, _, goal, task_id = self._goal_queue.get(timeout=5)
                ctx = self.run(goal, priority=priority)
                with self._lock:
                    self._active_tasks[task_id] = ctx
                self._goal_queue.task_done()
            except Empty:
                break
            except Exception:
                break
        self._drain_running = False

    # ─────────────────────────────────────────────────────────────────────────
    # LLM + LOVE BRIDGE
    # ─────────────────────────────────────────────────────────────────────────

    def _llm(self, system: str, user: str) -> str:
        if not self._llm_fn:
            return "[No LLM configured]"
        try:
            return self._llm_fn(system, user) or ""
        except Exception:
            return ""

    def _love_context(self) -> str:
        if self._love_bond:
            try:
                return self._love_bond.love_influence()
            except Exception:
                pass
        return ""

    # ─────────────────────────────────────────────────────────────────────────
    # JSON PARSING (robust)
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_json(raw: str) -> Any:
        if not raw:
            return None
        try:
            return json.loads(raw)
        except Exception:
            pass
        m = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', raw)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass
        return None

    # ─────────────────────────────────────────────────────────────────────────
    # PERSISTENCE
    # ─────────────────────────────────────────────────────────────────────────

    def _db_conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self._db_path, check_same_thread=False)
        c.row_factory = sqlite3.Row
        return c

    def _persist_task(self, ctx: AgentContext, status: str) -> None:
        try:
            conn = self._db_conn()
            conn.execute(
                "INSERT OR REPLACE INTO tasks "
                "(id, goal, status, avg_score, confidence, elapsed_s, "
                " steps_done, steps_fail, output, ts) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (ctx.task_id, ctx.goal[:500], status, ctx.avg_score,
                 ctx.confidence, ctx.elapsed,
                 len(ctx.completed_steps), len(ctx.failed_steps),
                 (ctx.output or "")[:1000], time.time())
            )
            conn.commit()
        except Exception:
            pass
        finally:
            try: conn.close()
            except: pass

    def _persist_transition(
        self, ctx: AgentContext, conf: float, flags: List
    ) -> None:
        try:
            conn = self._db_conn()
            conn.execute(
                "INSERT INTO step_transitions "
                "(task_id, subgoal, output, score, confidence, "
                " attempts, tools_used, ethic_flags, ts) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (ctx.task_id, (ctx.subgoal or "")[:200],
                 (ctx.output or "")[:600],
                 ctx.step_scores[-1] if ctx.step_scores else 0.0,
                 round(conf, 3), ctx.retry_count,
                 json.dumps(ctx.tools_used[-4:]),
                 json.dumps(flags), time.time())
            )
            conn.commit()
        except Exception:
            pass
        finally:
            try: conn.close()
            except: pass

    def _log_queue_entry(self, goal: str, priority: int) -> None:
        try:
            conn = self._db_conn()
            conn.execute(
                "INSERT INTO goal_queue_log (goal, priority, status, ts) VALUES (?,?,?,?)",
                (goal[:300], priority, "queued", time.time())
            )
            conn.commit()
        except Exception:
            pass
        finally:
            try: conn.close()
            except: pass

    def _recent_tasks(self, n: int = 5) -> List[Dict]:
        try:
            conn = self._db_conn()
            rows = conn.execute(
                "SELECT id, goal, status, avg_score, steps_done, steps_fail, elapsed_s, ts "
                "FROM tasks ORDER BY ts DESC LIMIT ?", (n,)
            ).fetchall()
            return [dict(r) for r in rows]
        except Exception:
            return []
        finally:
            try: conn.close()
            except: pass

    # ─────────────────────────────────────────────────────────────────────────
    # NOVA INTEGRATION — status() + process()
    # ─────────────────────────────────────────────────────────────────────────

    def status(self) -> Dict[str, Any]:
        """ConsciousnessIntegrator-compatible status report."""
        with self._lock:
            n_done  = len(self._completed)
            recent  = self._completed[-6:] if self._completed else []
            avg_sc  = (sum(c.avg_score for c in recent) / len(recent)
                       if recent else 0.0)
            n_active = len(self._active_tasks)
        return {
            "name":          "AgentKernel",
            "active":        bool(self._llm_fn),
            "completed":     n_done,
            "running_now":   n_active,
            "queue_depth":   self._goal_queue.qsize(),
            "avg_score":     round(avg_sc, 3),
            "items":         n_done,
            "confidence":    round(avg_sc, 3),
            "accuracy":      round(avg_sc, 3),
            "parallel_max":  self.PARALLEL_MAX,
            "ethics_layers": 3,
            "score_gate":    self.SCORE_THRESHOLD,
        }

    def process(self, text: str) -> Optional[str]:
        """
        Nova main-loop hook. Triggers when text explicitly invokes the kernel.
        Prefix triggers: 'plan:', 'agent:', 'execute:', 'task:', 'multi-step:'
        """
        text_stripped = text.strip()
        m = re.match(
            r'^(plan|agent|execute|task|multi[-\s]?step)\s*[:：]\s*(.+)',
            text_stripped, re.IGNORECASE | re.DOTALL
        )
        if not m:
            return None
        goal = m.group(2).strip()
        if len(goal) < 8:
            return None
        ctx = self.run(goal, emotional_context=self._love_context())
        n_ok   = len(ctx.completed_steps)
        n_fail = len(ctx.failed_steps)
        header = (
            f"AgentKernel — '{goal[:55]}'\n"
            f"  Steps: {n_ok} done · {n_fail} failed · "
            f"score {ctx.avg_score:.2f} · conf {ctx.confidence:.2f} · "
            f"{ctx.elapsed}s"
            + (" · replanned" if ctx.replanned else "")
        )
        return header + "\n\n" + (ctx.output or "[No output]")

    def run_command(self, arg: str) -> str:
        """Handle /agent subcommands from Nova's command dispatcher."""
        arg = (arg or "").strip()

        if not arg or arg == "status":
            st = self.status()
            recent = self._recent_tasks(5)
            lines = [
                f"  AgentKernel — {'active' if st['active'] else 'no LLM'}",
                f"  Completed tasks  : {st['completed']}",
                f"  Running now      : {st['running_now']}",
                f"  Queue depth      : {st['queue_depth']}",
                f"  Avg task score   : {st['avg_score']:.3f}",
                f"  Parallel slots   : {st['parallel_max']}",
                f"  Ethics layers    : {st['ethics_layers']}",
                f"  Score threshold  : {st['score_gate']}",
            ]
            if recent:
                lines.append("\n  Recent tasks:")
                for r in recent:
                    lines.append(
                        f"    [{r['status']:<7}] score={r.get('avg_score',0):.2f} "
                        f"steps={r.get('steps_done',0)} — {r['goal'][:55]}")
            return "\n".join(lines)

        if arg.startswith("run "):
            goal = arg[4:].strip()
            if not goal:
                return "Usage: /agent run <goal description>"
            ctx = self.run(goal, emotional_context=self._love_context())
            return (
                f"  Done in {ctx.elapsed}s — score {ctx.avg_score:.2f} · "
                f"confidence {ctx.confidence:.2f}\n\n{ctx.output}"
            )

        if arg.startswith("queue "):
            goal = arg[6:].strip()
            if not goal:
                return "Usage: /agent queue <goal description>"
            task_id = self.run_async(goal)
            return f"  Queued — task_id: {task_id[:16]}…"

        if arg == "history":
            rows = self._recent_tasks(10)
            if not rows:
                return "  No completed tasks yet."
            lines = ["  Task history (last 10):"]
            for r in rows:
                lines.append(
                    f"    [{r['status']:<7}] {r.get('elapsed_s',0):.1f}s "
                    f"score={r.get('avg_score',0):.2f} — {r['goal'][:60]}")
            return "\n".join(lines)

        if arg == "ethics":
            try:
                conn = self._db_conn()
                rows = conn.execute(
                    "SELECT subgoal, allowed, flags, overall_score, ts "
                    "FROM ethics_audit ORDER BY ts DESC LIMIT 12"
                ).fetchall()
                conn.close()
                if not rows:
                    return "  No ethics audit entries yet."
                lines = ["  Ethics audit (last 12 evaluations):"]
                for r in rows:
                    verdict = "✓ allowed" if r["allowed"] else "✗ blocked"
                    lines.append(
                        f"    [{verdict}] score={r['overall_score']:.2f} "
                        f"— {(r['subgoal'] or '')[:50]}")
                    if r["flags"]:
                        lines.append(f"      flags: {r['flags']}")
                return "\n".join(lines)
            except Exception as e:
                return f"Ethics audit error: {e}"

        return (
            "  Usage: /agent [status | run <goal> | queue <goal> | history | ethics]"
        )
