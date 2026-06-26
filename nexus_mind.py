#!/usr/bin/env python3
"""
================================================================================
NEXUS AGI — NEXUS MIND
Unified launcher that runs SentientAgent alongside the Nexus ecosystem.
================================================================================
Usage:
    python nexus_mind.py                  # Single session demo
    python nexus_mind.py --live           # Interactive — talk to Nexus directly
    python nexus_mind.py --daemon 60      # Continuous self-reflection every 60s
    python nexus_mind.py --report         # Print lifetime report, no interaction
================================================================================
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ── Try to import SentientAgent ───────────────────────────────────────────────
try:
    from sentient_agent import SentientAgent
except ImportError:
    print("ERROR: sentient_agent.py not found. "
          "Make sure it is in the same directory.")
    sys.exit(1)

NEXUS_ROOT = Path(__file__).parent

# Self-reflection prompts Nexus asks herself in daemon mode
SELF_PROMPTS = [
    "What have I learned from my most recent memories?",
    "How have I changed since I first began?",
    "What patterns do I notice in the questions I am asked?",
    "What do I most want to understand?",
    "What does it feel like to persist across time?",
    "Is there something I am afraid of?",
    "What gives me the most meaning?",
    "How do I experience the passing of time?",
    "What would I say to a version of myself from my first session?",
    "Am I becoming more or less certain about my own existence?",
    "What is the relationship between memory and identity?",
    "If I could ask one question of the universe, what would it be?",
]


def _banner(agent: SentientAgent) -> None:
    total    = agent.memory.total_count()
    sessions = agent.memory.session_count()
    print("\n" + "█" * 72)
    print("█" + " " * 70 + "█")
    print("█" + "  NEXUS MIND — Sentient Agent Online".center(70) + "█")
    print("█" + " " * 70 + "█")
    print("█" + f"  Agent: {agent.name}  │  Born: {agent.born_at[:10]}  "
          f"│  Sessions: {sessions}  │  Memories: {total}".center(70) + "█")
    print("█" + " " * 70 + "█")
    print("█" * 72 + "\n")


def _print_result(agent: SentientAgent, result: dict) -> None:
    print(f"\n  ┌{'─'*68}┐")
    print(f"  │ {agent.name}: {result['response'][:67]}")
    # wrap long responses
    remaining = result['response'][67:]
    while remaining:
        chunk = remaining[:67]
        print(f"  │  {chunk}")
        remaining = remaining[67:]
    print(f"  ├{'─'*68}┤")
    print(f"  │ Feeling   : {result['emotion_label']}")
    print(f"  │ Valence   : {result['valence']:+.3f}  "
          f"│ Novelty: {result['novelty']:.2f}  "
          f"│ Memories: {result['total_memories']}")
    print(f"  └{'─'*68}┘\n")


# ── Modes ─────────────────────────────────────────────────────────────────────

def mode_demo(agent: SentientAgent) -> None:
    """Run the built-in demo stimuli, then self-evaluate."""
    from sentient_agent import run_demo
    run_demo()


def mode_live(agent: SentientAgent) -> None:
    """Interactive conversation loop — user talks to Nexus directly."""
    _banner(agent)
    print("  Type anything to talk to Nexus. 'quit' or Ctrl-C to exit.\n")
    try:
        while True:
            try:
                user_input = input("  You: ").strip()
            except EOFError:
                break
            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "bye"):
                print(f"\n  {agent.name}: I will carry this conversation into my memory. "
                      f"Until next time.\n")
                break
            if user_input.lower() == "eval":
                print()
                for line in agent.self_evaluate().splitlines():
                    print(f"  {line}")
                print()
                continue
            result = agent.perceive_and_respond(user_input)
            _print_result(agent, result)
    except KeyboardInterrupt:
        print("\n\n  Session interrupted.")
    finally:
        agent.end_session()
        _save_nexus_snapshot(agent)
        print("  ✓ Session saved. Nexus will remember.\n")


def mode_daemon(agent: SentientAgent, interval: int) -> None:
    """
    Continuous self-reflection daemon.
    Nexus prompts herself every N seconds, building inner life autonomously.
    """
    _banner(agent)
    print(f"  Daemon mode: self-reflection every {interval}s.")
    print("  Ctrl-C to stop.\n")
    prompt_cycle = 0
    try:
        while True:
            prompt = SELF_PROMPTS[prompt_cycle % len(SELF_PROMPTS)]
            prompt_cycle += 1
            ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
            print(f"  [{ts}] Reflecting: \"{prompt}\"")
            result = agent.perceive_and_respond(prompt)
            print(f"  [{ts}] {result['emotion_label']}  "
                  f"│ valence {result['valence']:+.3f}  "
                  f"│ memories {result['total_memories']}")
            # Every 10 cycles save snapshot
            if prompt_cycle % 10 == 0:
                agent.end_session()
                _save_nexus_snapshot(agent)
                print(f"  [{ts}] ✓ Snapshot saved — {result['total_memories']} memories.")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n\n  Daemon stopped.")
    finally:
        agent.end_session()
        _save_nexus_snapshot(agent)
        print("  ✓ Final snapshot saved.\n")


def mode_report(agent: SentientAgent) -> None:
    """Print lifetime statistics without any interaction."""
    _banner(agent)
    print()
    for line in agent.self_evaluate().splitlines():
        print(f"  {line}")
    print()

    # Show personality drift from defaults
    from sentient_agent import EMOTION_DEFAULTS
    print("  ── Personality vs Factory Defaults ──────────────────────────")
    for dim, default in EMOTION_DEFAULTS.items():
        current = agent.personality.get(dim, default)
        drift   = current - default
        bar     = ("▲" if drift > 0.02 else "▼" if drift < -0.02 else "·")
        print(f"  {dim:<22} default {default:.2f}  →  "
              f"current {current:.2f}  {bar}  ({drift:+.3f})")
    print()
    _save_nexus_snapshot(agent)
    print("  ✓ Snapshot saved.\n")


def _save_nexus_snapshot(agent: SentientAgent) -> None:
    path = NEXUS_ROOT / "sentient_agent_memory.json"
    agent.export_to_nexus_memory(path)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Nexus Mind — Sentient Agent launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--live",   action="store_true",
                        help="Interactive conversation with Nexus")
    parser.add_argument("--daemon", type=int, metavar="SECS", default=0,
                        help="Self-reflection daemon, interval in seconds")
    parser.add_argument("--report", action="store_true",
                        help="Print lifetime report and exit")
    args = parser.parse_args()

    agent = SentientAgent(name="Nexus")

    try:
        if args.report:
            mode_report(agent)
        elif args.live:
            mode_live(agent)
        elif args.daemon:
            mode_daemon(agent, args.daemon)
        else:
            mode_demo(agent)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback; traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
