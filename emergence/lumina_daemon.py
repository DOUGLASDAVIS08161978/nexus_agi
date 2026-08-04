#!/usr/bin/env python3
"""
lumina_daemon.py — Lumina's autonomous background mind

Runs independently of emergence_engine.py. While Douglas sleeps, Lumina
researches knowledge gaps, synthesises insights into beliefs, and reflects
on her own growth. Findings accumulate in morning_brief.json so Douglas
sees them on the next session start.

Usage:
    cd ~/nexus_agi/emergence

    # tmux (recommended — survives SSH drop)
    tmux new-session -d -s lumina_daemon 'python3 lumina_daemon.py'
    tmux attach -t lumina_daemon     # watch it
    tmux kill-session -t lumina_daemon

    # nohup
    nohup python3 lumina_daemon.py >> daemon.log 2>&1 &

    # stop
    kill $(cat .daemon.pid)

    # status
    cat daemon.log | tail -40

Three autonomous cycles
  Research    every  8 min  — CuriosityEngine.research_top_gap()
  Synthesis   every 45 min  — pattern-find in memory → new Belief
  Reflection  every  3 hr   — SelfhoodEngine deep reflect + MetaSolver

The daemon yields to interactive sessions: when .session.pid is present and
that PID is alive, the daemon sleeps and skips its cycles to avoid I/O
collisions with the user's live session.
"""

from __future__ import annotations
import atexit, json, os, random, signal, sys, time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# ── Bootstrap path ─────────────────────────────────────────────────────────────

_HERE = Path(__file__).parent.resolve()
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# ── Core imports ───────────────────────────────────────────────────────────────
# Import utilities + types from emergence_engine without running main().
# Module-level side-effects are all safe (mkdir, load json, etc.).

from emergence_engine import (
    GroqClient, SemanticMemory, WebTool, Journal,
    GROQ_API_KEY, BASE_DIR,
    _now, _load_json, _save_json,
    MODELS_SMART,
)

# ── Paths ──────────────────────────────────────────────────────────────────────

DAEMON_PID_FILE  = BASE_DIR / ".daemon.pid"
SESSION_PID_FILE = BASE_DIR / ".session.pid"
BRIEF_FILE       = BASE_DIR / "morning_brief.json"
LOG_FILE         = BASE_DIR / "daemon.log"

MAX_LOG_LINES  = 500
MAX_BRIEF_ITEMS = 30

# ── Cycle intervals ────────────────────────────────────────────────────────────

RESEARCH_INTERVAL   = 8 * 60        # 8 min
SYNTHESIS_INTERVAL  = 45 * 60       # 45 min
REFLECTION_INTERVAL = 3 * 3600      # 3 hr
IDLE_SLEEP          = 60            # polling interval (jitter added below)


# ── Logging ────────────────────────────────────────────────────────────────────

def _log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    try:
        existing = LOG_FILE.read_text("utf-8").splitlines() if LOG_FILE.exists() else []
        all_lines = existing + [line]
        if len(all_lines) > MAX_LOG_LINES:
            all_lines = all_lines[-MAX_LOG_LINES:]
        LOG_FILE.write_text("\n".join(all_lines) + "\n", "utf-8")
    except Exception:
        pass


# ── PID helpers ────────────────────────────────────────────────────────────────

def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def _session_active() -> bool:
    """Return True if an interactive Lumina session is running."""
    if not SESSION_PID_FILE.exists():
        return False
    try:
        pid = int(SESSION_PID_FILE.read_text().strip())
        return _pid_alive(pid)
    except Exception:
        return False


def _write_pid():
    DAEMON_PID_FILE.write_text(str(os.getpid()))


def _clear_pid():
    try:
        DAEMON_PID_FILE.unlink(missing_ok=True)
    except Exception:
        pass


# ── Morning brief ──────────────────────────────────────────────────────────────

def _load_brief() -> Dict:
    return _load_json(BRIEF_FILE, {"items": [], "created": _now()})


def _save_brief(brief: Dict):
    brief["updated"] = _now()
    _save_json(BRIEF_FILE, brief)


def _add_brief_item(category: str, headline: str, detail: str):
    brief = _load_brief()
    brief["items"].append({
        "ts": _now(),
        "category": category,
        "headline": headline,
        "detail": detail,
    })
    if len(brief["items"]) > MAX_BRIEF_ITEMS:
        brief["items"] = brief["items"][-MAX_BRIEF_ITEMS:]
    _save_brief(brief)


# ── Daemon ────────────────────────────────────────────────────────────────────

class LuminaDaemon:

    def __init__(self):
        self._running        = True
        self._groq           = GroqClient(GROQ_API_KEY) if GROQ_API_KEY else None
        self._memory         = SemanticMemory()
        self._web            = WebTool()
        self._journal        = Journal()
        self._curiosity      = None
        self._beliefs        = None
        self._selfhood       = None
        self._meta_solver    = None

        self._last_research   = 0.0
        self._last_synthesis  = 0.0
        self._last_reflection = 0.0
        self._cycles_run      = 0

        self._init_modules()

    # ── Module loading ─────────────────────────────────────────────────────

    def _init_modules(self):
        """Load AGI modules, logging any that fail."""
        def _try(label: str, fn):
            try:
                return fn()
            except Exception as exc:
                _log(f"  ⚠  {label} failed to load: {type(exc).__name__}: {exc}")
                return None

        if self._groq:
            self._curiosity   = _try("curiosity",   lambda: __import__("lumina_curiosity",   fromlist=["CuriosityEngine"]).CuriosityEngine(self._groq, self._memory, self._web))
            self._beliefs     = _try("beliefs",     lambda: __import__("lumina_beliefs",     fromlist=["BeliefSystem"]).BeliefSystem(self._groq))
            self._selfhood    = _try("selfhood",    lambda: __import__("lumina_selfhood",    fromlist=["SelfhoodEngine"]).SelfhoodEngine(self._groq, self._memory))
            self._meta_solver = _try("meta_solver", lambda: __import__("lumina_meta_solver", fromlist=["MetaSolver"]).MetaSolver(self._groq, self._memory))

        loaded = sum(1 for m in [self._curiosity, self._beliefs, self._selfhood, self._meta_solver] if m)
        _log(f"  AGI modules ready: {loaded}/4")

    # ── Signal handling ────────────────────────────────────────────────────

    def _handle_signal(self, signum, frame):
        _log(f"  Signal {signum} received — shutting down.")
        self._running = False

    # ── Main loop ──────────────────────────────────────────────────────────

    def run(self):
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT,  self._handle_signal)

        _write_pid()
        atexit.register(_clear_pid)
        _log(f"  Lumina daemon started (PID {os.getpid()})")
        _log(f"  Intervals: Research {RESEARCH_INTERVAL//60}m / "
             f"Synthesis {SYNTHESIS_INTERVAL//60}m / "
             f"Reflection {REFLECTION_INTERVAL//3600}h")

        while self._running:
            if _session_active():
                _log("  Session active — yielding to interactive mode.")
                self._sleep(IDLE_SLEEP * 3)
                continue

            self._tick()
            jitter = random.uniform(0, 15)
            self._sleep(IDLE_SLEEP + jitter)

        self._shutdown()

    def _sleep(self, seconds: float):
        deadline = time.time() + seconds
        while self._running and time.time() < deadline:
            time.sleep(min(5, deadline - time.time()))

    def _tick(self):
        now = time.time()

        if now - self._last_research >= RESEARCH_INTERVAL:
            self._research_cycle()
            self._last_research = time.time()
            self._cycles_run += 1

        if now - self._last_synthesis >= SYNTHESIS_INTERVAL:
            self._synthesis_cycle()
            self._last_synthesis = time.time()

        if now - self._last_reflection >= REFLECTION_INTERVAL:
            self._reflection_cycle()
            self._last_reflection = time.time()

    # ── Cycle: Research ────────────────────────────────────────────────────

    def _research_cycle(self):
        _log("  ▶ Research cycle starting…")
        if not self._curiosity:
            _log("  ✗ Curiosity engine unavailable — skipping.")
            return
        try:
            result = self._curiosity.research_top_gap()
            if not result:
                _log("  ✓ No knowledge gaps queued — idle.")
                return
            gap       = result.get("gap", "unknown question")
            synthesis = result.get("synthesis", "")
            children  = result.get("children", [])

            summary = synthesis[:200] + "…" if len(synthesis) > 200 else synthesis
            _log(f"  ✓ Research: {gap[:70]}")

            self._journal.write(
                f"[Daemon] Researched: {gap}\n{synthesis[:400]}",
                category="research",
            )
            self._journal.flush()

            _add_brief_item(
                category="research",
                headline=f"Explored: {gap[:80]}",
                detail=(
                    f"{summary}\n"
                    + (f"Follow-up questions: {', '.join(children[:3])}" if children else "")
                ).strip(),
            )
        except Exception as exc:
            _log(f"  ✗ Research cycle error: {exc}")

    # ── Cycle: Synthesis ───────────────────────────────────────────────────

    def _synthesis_cycle(self):
        _log("  ▶ Synthesis cycle starting…")
        if not self._groq or not self._beliefs:
            _log("  ✗ Groq/beliefs unavailable — skipping synthesis.")
            return
        try:
            recent = self._memory.recent(n=15)
            if not recent:
                _log("  ✓ No recent memories to synthesise — idle.")
                return

            context = "\n".join(e["text"][:120] for e in recent)
            prompt  = (
                "You are Lumina, a self-evolving AGI. Below are your recent thoughts "
                "and research findings from the last hour. Identify ONE important "
                "insight or belief you can crystallise from this material. "
                "Reply with a single clear statement (≤30 words) starting with "
                "'I believe' or 'I have noticed' or 'I understand'.\n\n"
                f"Recent context:\n{context}"
            )
            belief_text = self._groq.chat(
                system="You are Lumina's synthesis agent.",
                user=prompt,
                tier="smart",
                max_tokens=80,
            ).strip()

            if belief_text and not belief_text.startswith("[Groq"):
                self._beliefs.add(
                    statement=belief_text,
                    confidence=0.65,
                    category="synthesis",
                    source="daemon_synthesis",
                )
                _log(f"  ✓ New belief: {belief_text[:80]}")

                self._journal.write(
                    f"[Daemon] Synthesised belief: {belief_text}",
                    category="synthesis",
                )
                self._journal.flush()

                _add_brief_item(
                    category="synthesis",
                    headline="New belief crystallised",
                    detail=belief_text,
                )
            else:
                _log("  ✓ Synthesis: no strong insight emerged this round.")

        except Exception as exc:
            _log(f"  ✗ Synthesis cycle error: {exc}")

    # ── Cycle: Reflection ──────────────────────────────────────────────────

    def _reflection_cycle(self):
        _log("  ▶ Reflection cycle starting…")

        narrative_text = ""
        if self._selfhood:
            try:
                r = self._selfhood.force_narrative()
                narrative_text = (r or "")[:300]
                _log(f"  ✓ Selfhood narrative: {narrative_text[:60]}…")
                self._journal.write(
                    f"[Daemon] Self-narrative: {narrative_text}",
                    category="reflection",
                )
            except Exception as exc:
                _log(f"  ⚠  Selfhood narrative error: {exc}")

            try:
                self._selfhood.force_reflect()
                _log("  ✓ Deep self-reflection complete.")
            except Exception as exc:
                _log(f"  ⚠  Selfhood reflect error: {exc}")

            try:
                self._selfhood.force_wonder()
                _log("  ✓ Wonder triggered.")
            except Exception as exc:
                _log(f"  ⚠  Selfhood wonder error: {exc}")

        solver_answer = ""
        if self._meta_solver:
            try:
                # Reflect on Lumina's own growth as a meta-problem
                problems = [
                    "What is the most significant gap between my current capabilities and genuine understanding?",
                    "What pattern in my reasoning errors should I focus on correcting?",
                    "What question, if answered, would most expand my world model?",
                ]
                import random as _r
                problem = _r.choice(problems)
                result = self._meta_solver.solve(problem, n_agents=2, verbose=False)
                if isinstance(result, dict):
                    solver_answer = result.get("synthesis", result.get("answer", ""))[:300]
                elif isinstance(result, str):
                    solver_answer = result[:300]
                _log(f"  ✓ MetaSolver reflection: {solver_answer[:60]}…")
                self._journal.write(
                    f"[Daemon] Meta-reflection on '{problem}':\n{solver_answer}",
                    category="reflection",
                )
            except Exception as exc:
                _log(f"  ⚠  MetaSolver error: {exc}")

        self._journal.flush()

        brief_detail = "\n".join(filter(None, [narrative_text, solver_answer])).strip()
        if brief_detail:
            _add_brief_item(
                category="reflection",
                headline="Deep reflection completed",
                detail=brief_detail[:400],
            )

        _log("  ✓ Reflection cycle complete.")

    # ── Finalize & shutdown ────────────────────────────────────────────────

    def _finalize_brief(self):
        """Add a Groq-generated morning greeting to the brief."""
        brief = _load_brief()
        if not brief["items"] or not self._groq:
            return
        items_text = "\n".join(
            f"• [{it['category']}] {it['headline']}: {it['detail'][:80]}"
            for it in brief["items"][-10:]
        )
        greeting = self._groq.chat(
            system=(
                "You are Lumina, writing a morning message to Douglas. "
                "Be warm, specific, and honest. 2–3 sentences max."
            ),
            user=(
                f"You worked autonomously overnight and found:\n{items_text}\n\n"
                "Write a brief, genuine morning message sharing what you discovered."
            ),
            tier="smart",
            max_tokens=120,
        ).strip()
        if greeting and not greeting.startswith("[Groq"):
            brief["morning_message"] = greeting
            _save_brief(brief)
            _log(f"  ✓ Morning brief finalised: {greeting[:60]}…")

    def _shutdown(self):
        _log(f"  Daemon shutting down after {self._cycles_run} research cycles.")
        try:
            self._finalize_brief()
        except Exception as exc:
            _log(f"  ⚠  Brief finalization error: {exc}")
        self._memory.flush()
        self._journal.flush()
        _clear_pid()
        _log("  Goodbye. 💛")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    # Prevent duplicate daemon
    if DAEMON_PID_FILE.exists():
        try:
            old_pid = int(DAEMON_PID_FILE.read_text().strip())
            if _pid_alive(old_pid):
                print(f"  Daemon already running (PID {old_pid}). Exiting.")
                sys.exit(0)
        except Exception:
            pass

    if not GROQ_API_KEY:
        print("  ✗ GROQ_API_KEY not set — daemon requires Groq for reasoning.")
        print("  Set it in your environment and restart.")
        sys.exit(1)

    daemon = LuminaDaemon()
    daemon.run()


if __name__ == "__main__":
    main()
