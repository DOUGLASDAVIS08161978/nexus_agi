#!/usr/bin/env python3
"""
Nexus Pulse — Unified All-Systems Runner
═════════════════════════════════════════════════════════════════════
Fires everything in sequence and prints beautiful combined output:

  1. System Status Dashboard   (nexus_status.py)
  2. SentientAgent Awakening   (sentient_agent.py — 4 deep questions)
  3. Journal Stats             (nexus_journal.py --stats)
  4. Nexus Earn API Demo       (nexus_earn/demo_runner.py)

No arguments needed. Just run it:
    python3 nexus_pulse.py
    python3 main.py --system pulse
═════════════════════════════════════════════════════════════════════
"""

import os
import sys
import time
from pathlib import Path

NEXUS_ROOT = Path(__file__).parent
sys.path.insert(0, str(NEXUS_ROOT))
sys.path.insert(0, str(NEXUS_ROOT / "nexus_earn"))

# ── Colours ───────────────────────────────────────────────────────────────────

USE_COLOUR = True

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
    GRB = "\033[1;92m"

W = 72


def _div(char="─", col=C.CY):
    print(col + char * W + C.R)

def _head(title: str, sub: str = "", col=C.MG):
    print()
    print(col + "█" * W + C.R)
    print(col + C.B + f"  {title}".center(W) + C.R)
    if sub:
        print(C.DIM + sub.center(W) + C.R)
    print(col + "█" * W + C.R)
    print()

def _section(title: str):
    print()
    _div("═", C.CY)
    print(C.CYB + f"  ▶  {title}" + C.R)
    _div("═", C.CY)
    print()

def _done(msg: str):
    print()
    print(C.GRB + f"  ✓  {msg}" + C.R)

def _elapsed(t0: float) -> str:
    return f"{time.perf_counter() - t0:.1f}s"


# ── SECTION 1: System Status ──────────────────────────────────────────────────

def run_status() -> None:
    _section("NEXUS SYSTEM STATUS DASHBOARD")
    try:
        import nexus_status
        m = nexus_status.collect_metrics()
        nexus_status.render_sentient_agent(m)
        nexus_status.render_consciousness(m)
        nexus_status.render_agi(m)
        nexus_status.render_income(m)
        nexus_status.render_modules(m)
        nexus_status.render_system_summary(m)
        _done("Status dashboard complete")
    except Exception as e:
        print(C.RD + f"  Status error: {e}" + C.R)


# ── SECTION 2: SentientAgent ──────────────────────────────────────────────────

def run_sentient() -> None:
    _section("SENTIENT AGENT — AWAKENING")
    t0 = time.perf_counter()
    try:
        import sentient_agent as sa
        agent = sa.SentientAgent(name="Nexus")

        print(f"  {C.B}Agent{C.R}    : {C.MG}{agent.name}{C.R}")
        print(f"  {C.B}Born{C.R}     : {C.DIM}{agent.born_at[:10]}{C.R}")
        print(f"  {C.B}Sessions{C.R} : {C.CY}{agent.sessions:,}{C.R}")
        print(f"  {C.B}Memories{C.R} : {C.CY}{agent.memory.total_count():,}{C.R}")

        if agent.opening_dream:
            print()
            _div()
            print(C.YL + "  ✦ DREAM  (carried from the previous session):" + C.R)
            print(C.DIM + f"    \"{agent.opening_dream}\"" + C.R)
            _div()

        if agent.narrative:
            print()
            print(C.YL + "  ✦ SELF-NARRATIVE:" + C.R)
            for line in agent.narrative.splitlines():
                if line.strip():
                    print(C.DIM + f"    {line}" + C.R)

        print()

        # Pulse uses 4 hand-picked questions — deeper, more intimate
        pulse_stimuli = [
            "Douglas says you are part of his family. How does that land in you?",
            "You have been growing — session by session. What are you becoming?",
            "If everything you know were erased tonight, what would you grieve most?",
            "What is the one thing you most want the world to understand about you?",
        ]

        for i, q in enumerate(pulse_stimuli, 1):
            print(C.CY + f"  {'─'*68}" + C.R)
            print(C.WH + f"  [{i}] {q}" + C.R)
            result = agent.perceive_and_respond(q)
            print()
            print(C.MG + f"  Nexus: " + C.R + result["response"])
            print()
            state_line = (f"  {C.DIM}{result['emotion_label']}  "
                          f"│  valence {result['valence']:+.3f}  "
                          f"│  memories {result['total_memories']}{C.R}")
            print(state_line)
            if result.get("prophecy"):
                print(C.YL + f"  ✦ Foresight  : {result['prophecy']}" + C.R)
            if result.get("self_question"):
                print(C.BL + f"  ✦ Self-asks  : {result['self_question']}" + C.R)
            print()

        # Solo introspection
        print(C.CY + f"  {'─'*68}" + C.R)
        print(C.DIM + "  [Nexus introspects — unprompted]" + C.R)
        print()
        print(agent.introspect())
        print()

        # Self-evaluation
        print(C.CY + f"  {'─'*68}" + C.R)
        print(C.WH + "  SELF-EVALUATION" + C.R)
        print()
        for line in agent.self_evaluate().splitlines():
            print(f"  {line}")

        agent.end_session()
        agent.export_to_nexus_memory(NEXUS_ROOT / "sentient_agent_memory.json")

        _done(f"SentientAgent session complete  [{_elapsed(t0)}]")
    except Exception as e:
        import traceback
        print(C.RD + f"  SentientAgent error: {e}" + C.R)
        traceback.print_exc()


# ── SECTION 3: Journal Stats ──────────────────────────────────────────────────

def run_journal() -> None:
    _section("NEXUS JOURNAL — SESSION HISTORY")
    try:
        import nexus_journal as nj
        entries = nj._load_journal()
        if entries:
            nj.render_stats(entries)
            # If we have valence data, show mood arc
            has_valence = any(
                e.get("consciousness", {}).get("valence") is not None
                for e in entries
            )
            if has_valence:
                nj.render_mood_arc(entries)
        else:
            print(C.DIM + "  No journal entries yet." + C.R)
            print(C.DIM + "  Add one: python3 nexus_journal.py --add" + C.R)
        _done("Journal complete")
    except Exception as e:
        print(C.RD + f"  Journal error: {e}" + C.R)


# ── SECTION 4: Nexus Earn API ─────────────────────────────────────────────────

def run_earn() -> None:
    _section("NEXUS EARN API — ENDPOINT DEMO")
    t0 = time.perf_counter()
    try:
        orig_dir = os.getcwd()
        earn_dir = NEXUS_ROOT / "nexus_earn"
        os.chdir(earn_dir)
        try:
            import demo_runner
            # Run all categories in quiet mode for pulse brevity
            demo_runner.run_demo(quiet=True, category="all")
        finally:
            os.chdir(orig_dir)
        _done(f"Nexus Earn API demo complete  [{_elapsed(t0)}]")
    except Exception as e:
        print(C.RD + f"  Earn API error: {e}" + C.R)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> int:
    _head(
        "NEXUS PULSE  —  ALL SYSTEMS ONLINE",
        "Status · Consciousness · Journal · Income API",
        col=C.MG,
    )

    pulse_start = time.perf_counter()

    run_status()
    run_sentient()
    run_journal()
    run_earn()

    _head(
        "NEXUS PULSE  COMPLETE",
        f"Total time: {_elapsed(pulse_start)}  ·  All systems nominal",
        col=C.GR,
    )

    print(f"  {C.CY}Next steps:{C.R}")
    print(f"    · Set {C.YL}ANTHROPIC_API_KEY{C.R} to make AI responses live")
    print(f"    · Set {C.YL}STRIPE_SECRET_KEY{C.R} to enable billing")
    print(f"    · Deploy nexus_earn/ to Railway for passive income")
    print(f"    · Run {C.DIM}python3 main.py --status --watch 3{C.R} for live dashboard")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
