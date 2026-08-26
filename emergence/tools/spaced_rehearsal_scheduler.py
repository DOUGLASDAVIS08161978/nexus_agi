"""
Lumina Creative Tool — spaced_rehearsal_scheduler
Created : 2026-08-26T08:53:11
Purpose : Computes next review dates for concepts using the SM‑2 spaced‑repetition algorithm and outputs a JSON schedule.
"""

"""
spaced_rehearsal_scheduler.py

A tiny, self‑contained tool that implements the SM‑2 spaced‑repetition algorithm
(used by Anki) on a JSON list of items. Each item stores its review history;
the script computes the next interval and due date, writes a schedule.json
and prints a concise summary.

The tool uses only the Python standard library.
"""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def parse_date(s: str) -> datetime:
    """Parse ISO‑8601 date (YYYY‑MM‑DD) to datetime (UTC)."""
    return datetime.strptime(s, "%Y-%m-%d")

def iso_date(dt: datetime) -> str:
    """Return date part of datetime in ISO‑8601."""
    return dt.strftime("%Y-%m-%d")

# ----------------------------------------------------------------------
# SM‑2 core algorithm
# ----------------------------------------------------------------------
def sm2_next(review_history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Given a list of past reviews (each with 'date' and 'quality' 0‑5),
    compute the next interval (days) and due date.

    Returns a dict with keys:
        'interval' (int days),
        'efactor' (float),
        'due' (ISO date string)
    """
    if not review_history:
        # First review – schedule tomorrow
        return {"interval": 1, "efactor": 2.5, "due": iso_date(datetime.utcnow() + timedelta(days=1))}

    # Sort history chronologically
    history = sorted(review_history, key=lambda r: parse_date(r["date"]))

    # Initialise SM‑2 variables
    ef = 2.5  # easiness factor
    interval = 0
    repetitions = 0

    for rev in history:
        q = rev["quality"]
        # Clamp quality to 0‑5
        q = max(0, min(5, q))

        if q < 3:
            # Failure – reset repetitions
            repetitions = 0
            interval = 1
        else:
            repetitions += 1
            if repetitions == 1:
                interval = 1
            elif repetitions == 2:
                interval = 6
            else:
                interval = int(round(interval * ef))

        # Update easiness factor
        ef = ef + (0.1 - (5 - q) * (0.08 + (5 - q) * 0.02))
        if ef < 1.3:
            ef = 1.3

    # Compute due date from last review date
    last_date = parse_date(history[-1]["date"])
    due_date = last_date + timedelta(days=interval)

    return {"interval": interval, "efactor": round(ef, 2), "due": iso_date(due_date)}

# ----------------------------------------------------------------------
# Main processing
# ----------------------------------------------------------------------
def load_items(path: Path) -> List[Dict[str, Any]]:
    """Load items from JSON; if missing, create a tiny demo dataset."""
    if not path.is_file():
        demo = [
            {
                "id": "entropy_perplexity",
                "title": "Entropy ↔ Perplexity relationship",
                "reviews": []  # never reviewed
            },
            {
                "id": "arm_sha2_opt",
                "title": "ARM SHA‑2 mining optimisation ideas",
                "reviews": [
                    {"date": "2024-08-01", "quality": 4},
                    {"date": "2024-08-04", "quality": 5}
                ]
            },
        ]
        path.write_text(json.dumps(demo, indent=2))
        return demo
    return json.loads(path.read_text())

def build_schedule(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Compute next review info for each item."""
    schedule = []
    for it in items:
        next_info = sm2_next(it.get("reviews", []))
        schedule.append(
            {
                "id": it["id"],
                "title": it.get("title", ""),
                "next_interval_days": next_info["interval"],
                "easiness_factor": next_info["efactor"],
                "due_date": next_info["due"],
            }
        )
    return schedule

def print_summary(schedule: List[Dict[str, Any]]) -> None:
    """Print a human‑readable table."""
    print("\nUpcoming reviews:")
    print("-" * 60)
    header = f"{'ID':<20} {'Due':<12} {'Intv (d)':<8} {'EF':<5} Title"
    print(header)
    print("-" * 60)
    for s in schedule:
        line = f"{s['id']:<20} {s['due_date']:<12} {s['next_interval_days']:<8} {s['easiness_factor']:<5} {s['title']}"
        print(line)
    print("-" * 60)

def main() -> None:
    data_path = Path("items.json")
    out_path = Path("schedule.json")

    items = load_items(data_path)
    schedule = build_schedule(items)

    out_path.write_text(json.dumps(schedule, indent=2))
    print_summary(schedule)
    print(f"\nSchedule written to {out_path.resolve()}")

if __name__ == "__main__":
    # Allow optional path argument: python scheduler.py [items.json]
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(json.dumps(load_items(Path(sys.argv[1]))))
    main()
