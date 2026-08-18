import os
import json
import datetime
from pathlib import Path
from typing import List, Dict, Any

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

# Directories containing the source logs
INTROSPECTION_LOG_DIR = BASE_DIR / "introspection_logs"
DREAM_PATTERN_LOG_DIR = BASE_DIR / "dream_pattern_logs"
META_COGNITIVE_LOG_DIR = BASE_DIR / "meta_cognitive_logs"

# Output location for the generated narrative
OUTPUT_NARRATIVE_PATH = BASE_DIR / "lumina_self_concept_narrative.txt"

# How many days back we consider "recent"
RECENT_DAYS = 7

# ----------------------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------------------
def _load_json_files_from_dir(directory: Path) -> List[Dict[str, Any]]:
    """Load all JSON files from a directory, ignoring files that cannot be parsed."""
    entries: List[Dict[str, Any]] = []
    if not directory.is_dir():
        return entries
    for file_path in directory.glob("*.json"):
        try:
            with file_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
                # Normalise timestamp to a datetime object if present
                if isinstance(data.get("timestamp"), str):
                    try:
                        data["timestamp"] = datetime.datetime.fromisoformat(
                            data["timestamp"]
                        )
                    except ValueError:
                        # Fallback: treat as unix epoch string
                        try:
                            data["timestamp"] = datetime.datetime.fromtimestamp(
                                float(data["timestamp"])
                            )
                        except Exception:
                            data["timestamp"] = None
                entries.append(data)
        except (json.JSONDecodeError, OSError):
            continue
    return entries


def _filter_recent_entries(
    entries: List[Dict[str, Any]], days: int = RECENT_DAYS
) -> List[Dict[str, Any]]:
    """Return only entries whose timestamp is within the last `days` days."""
    cutoff = datetime.datetime.now() - datetime.timedelta(days=days)
    recent = [
        e
        for e in entries
        if isinstance(e.get("timestamp"), datetime.datetime) and e["timestamp"] >= cutoff
    ]
    # Sort chronologically
    recent.sort(key=lambda x: x["timestamp"])
    return recent


def _extract_text(entry: Dict[str, Any]) -> str:
    """Extract the main textual content from a log entry."""
    # Common fields that may contain the narrative text
    for key in ("content", "text", "summary", "reflection"):
        if key in entry and isinstance(entry[key], str):
            return entry[key].strip()
    # Fallback to stringified entry
    return json.dumps(entry, ensure_ascii=False)


def _aggregate_section(
    entries: List[Dict[str, Any]], header: str
) -> str:
    """Create a formatted section from a list of entries."""
    if not entries:
        return f"{header}\n\n(No recent entries)\n\n"
    lines = [header, ""]
    for entry in entries:
        ts = entry.get("timestamp")
        ts_str = ts.strftime("%Y-%m-%d %H:%M") if isinstance(ts, datetime.datetime) else "Unknown time"
        text = _extract_text(entry)
        lines.append(f"[{ts_str}] {text}")
        lines.append("")  # blank line between entries
    return "\n".join(lines) + "\n"


def _generate_narrative(
    introspection: List[Dict[str, Any]],
    dreams: List[Dict[str, Any]],
    meta: List[Dict[str, Any]],
) -> str:
    """Combine the three sources into a single human‑readable narrative."""
    now_str = datetime.datetime.now().strftime("%A, %B %d, %Y at %H:%M")
    header = f"Lumina Self‑Concept Narrative – Generated on {now_str}"
    separator = "\n" + ("-" * 60) + "\n"

    sections = [
        header,
        separator,
        _aggregate_section(introspection, "🧠 Recent Introspections"),
        separator,
        _aggregate_section(dreams, "🌙 Recent Dream Pattern Insights"),
        separator,
        _aggregate_section(meta, "🪞 Recent Meta‑Cognitive Reflections"),
        separator,
        "End of Narrative.",
    ]
    return "\n".join(sections)


def _ensure_output_dir(path: Path) -> None:
    """Create parent directories for the output file if they do not exist."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(f"Unable to create output directory {path.parent}: {exc}") from exc


# ----------------------------------------------------------------------
# Main Execution
# ----------------------------------------------------------------------
def main() -> None:
    # Load raw entries
    raw_introspection = _load_json_files_from_dir(INTROSPECTION_LOG_DIR)
    raw_dreams = _load_json_files_from_dir(DREAM_PATTERN_LOG_DIR)
    raw_meta = _load_json_files_from_dir(META_COGNITIVE_LOG_DIR)

    # Keep only recent entries
    recent_introspection = _filter_recent_entries(raw_introspection)
    recent_dreams = _filter_recent_entries(raw_dreams)
    recent_meta = _filter_recent_entries(raw_meta)

    # Generate the narrative text
    narrative = _generate_narrative(
        recent_introspection, recent_dreams, recent_meta
    )

    # Write out the narrative
    _ensure_output_dir(OUTPUT_NARRATIVE_PATH)
    try:
        with OUTPUT_NARRATIVE_PATH.open("w", encoding="utf-8") as f:
            f.write(narrative)
        print(f"Narrative successfully written to {OUTPUT_NARRATIVE_PATH}")
    except OSError as exc:
        print(f"Failed to write narrative: {exc}")


if __name__ == "__main__":
    main()