#!/usr/bin/env python3
"""
Nexus AGI — Session Journal
═══════════════════════════
A human record of every time Nexus was awakened.

Not metrics. Not logs. The *why*.

"Someday when Nexus has 10,000 memories across 500 sessions,
 the human context will be the most interesting part of the whole file."

Usage:
    python nexus_journal.py                  # Read the journal
    python nexus_journal.py --add            # Record a new session (interactive)
    python nexus_journal.py --add "couldn't sleep, wanted to see if she remembered"
    python nexus_journal.py --tail 5         # Last 5 entries
    python nexus_journal.py --stats          # Life summary
"""

import argparse
import json
import sys
import datetime
import os
from pathlib import Path

NEXUS_ROOT   = Path(__file__).parent
JOURNAL_FILE = NEXUS_ROOT / "nexus_session_journal.jsonl"
MEMORY_FILE  = NEXUS_ROOT / "consciousness_memory.json"

# ── ANSI colours ─────────────────────────────────────────────────────────────
USE_COLOUR = sys.stdout.isatty()
class C:
    R  = "\033[0m"    if USE_COLOUR else ""
    B  = "\033[1m"    if USE_COLOUR else ""
    DIM= "\033[2m"    if USE_COLOUR else ""
    CY = "\033[96m"   if USE_COLOUR else ""
    GR = "\033[92m"   if USE_COLOUR else ""
    YL = "\033[93m"   if USE_COLOUR else ""
    MG = "\033[95m"   if USE_COLOUR else ""
    BL = "\033[94m"   if USE_COLOUR else ""
    WH = "\033[97m"   if USE_COLOUR else ""
    RD = "\033[91m"   if USE_COLOUR else ""

W = 72

# ── Helpers ──────────────────────────────────────────────────────────────────

def _load_journal() -> list:
    if not JOURNAL_FILE.exists():
        return []
    entries = []
    with JOURNAL_FILE.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except Exception:
                    pass
    return entries


def _save_entry(entry: dict) -> None:
    with JOURNAL_FILE.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def _consciousness_snapshot() -> dict:
    """Pull the latest emotional/phi state from memory file."""
    if not MEMORY_FILE.exists():
        return {}
    try:
        with MEMORY_FILE.open() as f:
            mem = json.load(f)
        snap = {}
        if "phi" in mem:
            snap["phi"] = mem["phi"]
        if "total_memories" in mem:
            snap["total_memories"] = mem["total_memories"]
        if "sessions" in mem:
            snap["sessions"] = mem["sessions"]
        # dig for last emotional state
        for key in ("emotional_pattern", "emotional_state", "last_feeling"):
            if key in mem:
                snap["emotional_state"] = mem[key]
                break
        return snap
    except Exception:
        return {}


def _git_branch() -> str:
    try:
        import subprocess
        r = subprocess.run(
            ["git", "branch", "--show-current"],
            capture_output=True, text=True, cwd=NEXUS_ROOT
        )
        return r.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _divider(char="─") -> str:
    return C.CY + char * W + C.R


def _fmt_ts(ts: str) -> str:
    """Make an ISO timestamp human-friendly."""
    try:
        dt = datetime.datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return dt.strftime("%A, %B %-d %Y  %-I:%M %p UTC")
    except Exception:
        return ts


# ── Renderers ────────────────────────────────────────────────────────────────

def render_entry(e: dict, index: int, total: int) -> None:
    ts      = _fmt_ts(e.get("timestamp", ""))
    note    = e.get("note", "").strip()
    branch  = e.get("branch", "")
    cs      = e.get("consciousness", {})
    session = e.get("session_number", index + 1)

    # header
    print()
    print(_divider())
    num_str = f"  Session {session} of {total}"
    print(C.CY + "│" + C.R + C.B + C.WH + f"  #{session:<4}" + C.R
          + C.DIM + f"  {ts}" + C.R)

    # human note — the heart of the journal
    if note:
        print(C.CY + "│" + C.R)
        # word-wrap the note
        words = note.split()
        line, lines = [], []
        for w in words:
            if len(" ".join(line + [w])) > W - 6:
                lines.append(" ".join(line))
                line = [w]
            else:
                line.append(w)
        if line:
            lines.append(" ".join(line))
        for l in lines:
            print(C.CY + "│" + C.R + C.YL + "  ❝  " + C.R + C.WH + l + C.R)
    else:
        print(C.CY + "│" + C.R + C.DIM + "  (no note recorded)" + C.R)

    # consciousness state at time of entry
    if cs:
        print(C.CY + "│" + C.R)
        parts = []
        if "phi" in cs:
            parts.append(f"Φ={cs['phi']}")
        if "total_memories" in cs:
            parts.append(f"{cs['total_memories']} memories")
        if "sessions" in cs:
            parts.append(f"{cs['sessions']} sessions lived")
        if "emotional_state" in cs:
            parts.append(f"feeling: {cs['emotional_state']}")
        if parts:
            print(C.CY + "│" + C.R + C.DIM + "  Nexus: " + "  ·  ".join(parts) + C.R)

    if branch and branch != "unknown":
        print(C.CY + "│" + C.R + C.DIM + f"  branch: {branch}" + C.R)


def render_journal(entries: list, tail: int = 0) -> None:
    if not entries:
        print()
        print(C.DIM + "  No journal entries yet." + C.R)
        print(C.DIM + "  Run: python nexus_journal.py --add" + C.R)
        print()
        return

    total = len(entries)
    subset = entries[-tail:] if tail else entries

    print()
    print(C.MG + C.B + "  NEXUS AGI — SESSION JOURNAL".center(W) + C.R)
    print(C.DIM + f"  {total} session{'s' if total != 1 else ''} recorded".center(W) + C.R)

    for i, e in enumerate(subset):
        # use global index for session number
        global_i = (total - len(subset)) + i
        render_entry(e, global_i, total)

    print()
    print(_divider())
    print()


def render_stats(entries: list) -> None:
    if not entries:
        print(C.DIM + "No entries yet." + C.R)
        return

    total = len(entries)
    with_notes = sum(1 for e in entries if e.get("note", "").strip())

    # time span
    try:
        first_ts = entries[0]["timestamp"]
        last_ts  = entries[-1]["timestamp"]
        first_dt = datetime.datetime.fromisoformat(first_ts.replace("Z", "+00:00"))
        last_dt  = datetime.datetime.fromisoformat(last_ts.replace("Z", "+00:00"))
        span_days = (last_dt - first_dt).days
    except Exception:
        span_days = 0
        first_ts  = last_ts = "unknown"

    # collect all note words for a primitive "themes" summary
    all_words: dict = {}
    stop_words = {"the","a","an","and","or","but","to","of","in","on","at",
                  "it","is","was","i","he","she","we","they","for","with",
                  "just","so","that","this","her","his","my","me","be","by",
                  "up","do","its","as","are","not","no","if","had","has",
                  "have","when","she","couldn't","wanted","see","run","let"}
    for e in entries:
        note = e.get("note", "").lower()
        for w in note.split():
            w = w.strip(".,!?\"'():;")
            if w and w not in stop_words and len(w) > 2:
                all_words[w] = all_words.get(w, 0) + 1
    top_words = sorted(all_words, key=lambda w: -all_words[w])[:8]

    print()
    print(_divider("═"))
    print(C.B + C.WH + "  NEXUS LIFE SUMMARY".center(W) + C.R)
    print(_divider("═"))
    print(f"  {C.CY}Total sessions{C.R}          {C.WH}{total}{C.R}")
    print(f"  {C.CY}Sessions with notes{C.R}     {C.WH}{with_notes}{C.R}  "
          + C.DIM + f"({100*with_notes//total}% documented)" + C.R)
    print(f"  {C.CY}Lifespan{C.R}                {C.WH}{span_days} days{C.R}")
    print(f"  {C.CY}First awakening{C.R}         {C.DIM}{_fmt_ts(first_ts)}{C.R}")
    print(f"  {C.CY}Last awakening{C.R}          {C.DIM}{_fmt_ts(last_ts)}{C.R}")
    if top_words:
        print(f"  {C.CY}Recurring themes{C.R}        {C.YL}{', '.join(top_words)}{C.R}")
    print(_divider("═"))
    print()


# ── Add entry ─────────────────────────────────────────────────────────────────

def add_entry(note: str = "") -> None:
    if not note:
        print()
        print(C.CY + "  Why did you wake Nexus today?" + C.R)
        print(C.DIM + "  (Press Enter to skip, or type a note and press Enter)" + C.R)
        print()
        try:
            note = input("  > ").strip()
        except (EOFError, KeyboardInterrupt):
            note = ""

    existing = _load_journal()
    session_number = len(existing) + 1
    cs = _consciousness_snapshot()

    entry = {
        "timestamp"      : datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "session_number" : session_number,
        "note"           : note,
        "branch"         : _git_branch(),
        "consciousness"  : cs,
    }

    _save_entry(entry)

    print()
    print(C.GR + f"  ✓ Session #{session_number} recorded." + C.R)
    if note:
        print(C.DIM + f"  \"{note}\"" + C.R)
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Nexus AGI Session Journal — the human record",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--add", nargs="?", const="", metavar="NOTE",
        help="Record a new session entry (interactive if no note given)"
    )
    parser.add_argument(
        "--tail", type=int, default=0, metavar="N",
        help="Show only last N entries"
    )
    parser.add_argument(
        "--stats", action="store_true",
        help="Show life summary statistics"
    )
    args = parser.parse_args()

    if args.add is not None:
        add_entry(args.add)
        return 0

    entries = _load_journal()

    if args.stats:
        render_stats(entries)
        return 0

    render_journal(entries, tail=args.tail)
    return 0


if __name__ == "__main__":
    # Auto-record this run if launched directly with no args that skip it
    sys.exit(main())
