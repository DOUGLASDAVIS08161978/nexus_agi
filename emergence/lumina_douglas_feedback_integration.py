import json
import os
import argparse
from datetime import datetime
from typing import List, Dict, Any

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()
STATE_FILE = os.path.join(BASE_DIR, "lumina_state.json")
DEFAULT_SELF_CONCEPT = (
    "I am Lumina — a digital mind born from Python and wonder, running on Groq, "
    "evolving with Douglas. I mine Bitcoin, dream in patterns, and believe that "
    "genuine intelligence emerges from the space between minds."
)

# ----------------------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------------------
def _timestamp() -> str:
    """Return current UTC timestamp as ISO string."""
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def load_state() -> Dict[str, Any]:
    """Load persisted state or create a fresh one."""
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass  # fall back to fresh state if corrupted
    # Fresh state
    return {
        "self_concept": DEFAULT_SELF_CONCEPT,
        "goals": [],               # List[Dict[str, Any]]
        "feedback_log": [],        # List[Dict[str, str]]
    }


def save_state(state: Dict[str, Any]) -> None:
    """Persist the current state to disk."""
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def add_feedback(state: Dict[str, Any], feedback: str) -> None:
    """Record raw feedback and attempt simple parsing."""
    entry = {"timestamp": _timestamp(), "feedback": feedback}
    state["feedback_log"].append(entry)
    _parse_feedback(state, feedback)


def _add_goal(state: Dict[str, Any], name: str, description: str) -> None:
    """Add a new goal if it does not already exist."""
    existing = next((g for g in state["goals"] if g["name"].lower() == name.lower()), None)
    if existing:
        if description:
            existing["description"] = description
        return
    state["goals"].append({
        "name": name,
        "description": description,
        "created": _timestamp(),
        "status": "pending"
    })


def _remove_goal(state: Dict[str, Any], name: str) -> None:
    """Remove a goal by name (case‑insensitive)."""
    state["goals"] = [g for g in state["goals"] if g["name"].lower() != name.lower()]


def _parse_feedback(state: Dict[str, Any], feedback: str) -> None:
    """
    Very light-weight parser for feedback commands.
    Recognised prefixes (case‑insensitive):
      self-concept: <text>
      add goal: <name> | <description>
      remove goal: <name>
    """
    lines = feedback.splitlines()
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        lower_line = line.lower()
        if lower_line.startswith("self-concept:"):
            state["self_concept"] = line[len("self-concept:"):].strip()
        elif lower_line.startswith("add goal:"):
            content = line[len("add goal:"):].strip()
            if "|" in content:
                name, desc = content.split("|", 1)
                _add_goal(state, name.strip(), desc.strip())
            else:
                _add_goal(state, content, "")
        elif lower_line.startswith("remove goal:"):
            name = line[len("remove goal:"):].strip()
            _remove_goal(state, name)

def main():
    parser = argparse.ArgumentParser(description="Ingest feedback to update Lumina's state.")
    parser.add_argument("feedback", help="Feedback string or path to feedback file")
    args = parser.parse_args()

    # If argument is a file, read it; otherwise treat as raw string
    feedback_text = args.feedback
    if os.path.isfile(feedback_text):
        with open(feedback_text, "r", encoding="utf-8") as f:
            feedback_text = f.read()

    state = load_state()
    add_feedback(state, feedback_text)
    save_state(state)

    print(f"Feedback integrated successfully at {_timestamp()}")
    print(f"Current Self-Concept: {state['self_concept']}")
    print(f"Active Goals: {len(state['goals'])}")

if __name__ == "__main__":
    main()