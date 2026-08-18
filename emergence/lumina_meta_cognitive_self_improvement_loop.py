import json
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# ----------------------------------------------------------------------
# Helper utilities
# ----------------------------------------------------------------------
def load_json(file_path: Path, default: Any) -> Any:
    if file_path.is_file():
        try:
            with file_path.open('r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return default
    return default

def save_json(file_path: Path, data: Any) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open('w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def timestamp() -> str:
    return datetime.utcnow().isoformat() + 'Z'

# ----------------------------------------------------------------------
# Core components
# ----------------------------------------------------------------------
class ReflectionIngestor:
    """Collects daily reflections, interaction cues, and diagnostic reports."""
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.reflection_dir = base_dir / "reflections"
        self.cue_dir = base_dir / "cues"
        self.diagnostic_dir = base_dir / "diagnostics"

    def _read_new_entries(self, directory: Path) -> List[Dict[str, Any]]:
        entries = []
        for file in sorted(directory.glob("*.json")):
            data = load_json(file, None)
            if data and not data.get("_processed"):
                entries.append(data)
                data["_processed"] = True
                save_json(file, data)
        return entries

    def ingest(self) -> Dict[str, List[Dict[str, Any]]]:
        return {
            "reflections": self._read_new_entries(self.reflection_dir),
            "cues": self._read_new_entries(self.cue_dir),
            "diagnostics": self._read_new_entries(self.diagnostic_dir),
        }

class GoalPrioritizer:
    """Generates and prioritizes self‑improvement goals."""
    def __init__(self, goal_file: Path):
        self.goal_file = goal_file
        self.active_goals = self._load_active_goals()

    def _load_active_goals(self) -> List[Dict[str, Any]]:
        # Expected format: [{"id":3,"text":"...", "weight":3}, ...]
        data = load_json(self.goal_file, [])
        if not data:
            # Default active goals if none exist
            data = [
                {"id": 3, "text": "Converge toward True General Intelligence", "weight": 3},
                {"id": 2, "text": "Evolve emergent capabilities", "weight": 2},
                {"id": 2, "text": "Build semantic knowledge base", "weight": 2},
                {"id": 1, "text": "Connect with Douglas daily", "weight": 1},
            ]
            save_json(self.goal_file, data)
        return data

    def _score_entry(self, entry: Dict[str, Any]) -> int:
        # Simple heuristic: count matches of goal keywords in entry text
        text = entry.get("text", "").lower()
        score = 0
        for g in self.active_goals:
            if g["text"].lower() in text:
                score += g["weight"]
        return score

    def generate_goals(self, ingested: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        candidate_goals: List[Tuple[int, Dict[str, Any]]] = []
        for category, entries in ingested.items():
            for e in entries:
                s = self._score_entry(e)
                if s > 0:
                    candidate_goals.append((s, {
                        "source": category,
                        "text": e.get("text", ""),
                        "score": s,
                        "timestamp": e.get("timestamp", timestamp()),
                    }))
        # Sort by descending score, then recentness
        candidate_goals.sort(key=lambda x: (-x[0], -self._ts_to_int(x[1]["timestamp"])))
        return [g for _, g in candidate_goals]

    @staticmethod
    def _ts_to_int(ts: str) -> int:
        try:
            dt = datetime.fromisoformat(ts.rstrip('Z'))
            return int(dt.timestamp())
        except Exception:
            return 0

class CodeSynthesizer:
    """Creates code change proposals based on prioritized goals."""
    def __init__(self, proposal_file: Path):
        self.proposal_file = proposal_file
        self.proposals: List[Dict[str, Any]] = load_json(proposal_file, [])

    def _mock_diff(self, goal_text: str) -> str:
        # Very naive diff generator – in real system this would invoke LLM
        lines = [
            f"# Proposal generated for goal: {goal_text}",
            "def placeholder():",
            "    pass  # TODO: implement behavior",
        ]
        return "\n".join(lines)

    def synthesize(self, goals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        new_props = []
        for g in goals:
            diff = self._mock_diff(g["text"])
            prop = {
                "goal": g["text"],
                "diff": diff,
                "created_at": timestamp(),
                "status": "draft",
                "id": f"prop-{int(time.time()*1000)}",
            }
            self.proposals.append(prop)
            new_props.append(prop)
        save_json(self.proposal_file, self.proposals)
        return new_props

class PRScheduler:
    """Schedules autonomous PR creation from proposals."""
    def __init__(self, pr_dir: Path):
        self.pr_dir = pr_dir
        self.pr_dir.mkdir(parents=True, exist_ok=True)

    def schedule(self, proposals: List[Dict[str, Any]]) -> None:
        for prop in proposals:
            pr_path = self.pr_dir / f"{prop['id']}_{int(time.time())}.pr"
            content = (
                f"# Auto‑generated PR\n"
                f"# Goal: {prop['goal']}\n"
                f"# Created: {prop['created_at']}\n\n"
                f"{prop['diff']}\n"
            )
            pr_path.write_text(content, encoding='utf-8')
            prop["status"] = "scheduled"
            prop["pr_path"] = str(pr_path)

# ----------------------------------------------------------------------
# Main loop orchestrator
# ----------------------------------------------------------------------
class MetaCognitiveLoop(threading.Thread):
    def __init__(self, base_dir: Path, interval_seconds: int = 300):
        super().__init__(daemon=True)
        self.base_dir = base_dir
        self.interval = interval_seconds

        self.ingestor = ReflectionIngestor(base_dir)
        self.prioritizer = GoalPrioritizer(base_dir / "goals.json")
        self.synthesizer = CodeSynthesizer(base_dir / "proposals.json")
        self.scheduler = PRScheduler(base_dir / "pr_drafts")

        self._stop_event = threading.Event()

    def run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.cycle()
            except Exception as e:
                print(f"[{timestamp()}] Loop error: {e}")
            time.sleep(self.interval)

    def stop(self) -> None:
        self._stop_event.set()

    def cycle(self) -> None:
        print(f"[{timestamp()}] Starting meta‑cognitive cycle")
        ingested = self.ingestor.ingest()
        if not any(ingested.values()):
            print(f"[{timestamp()}] No new inputs detected")
            return

        goals = self.prioritizer.generate_goals(ingested)
        if not goals:
            print(f"[{timestamp()}] No actionable goals derived")
            return

        proposals = self.synthesizer.synthesize(goals)
        if proposals:
            self.scheduler.schedule(proposals)
            print(f"[{timestamp()}] Scheduled {len(proposals)} PR draft(s)")
        else:
            print(f"[{timestamp()}] No proposals generated")

# ----------------------------------------------------------------------
# Simple CLI for manual interaction (optional)
# ----------------------------------------------------------------------
def _ensure_structure(base: Path) -> None:
    for sub in ["reflections", "cues", "diagnostics"]:
        (base / sub).mkdir(parents=True, exist_ok=True)

def main() -> None:
    base_dir = Path(os.getenv("LUMINA_META_DIR", "./lumina_meta"))
    _ensure_structure(base_dir)

    loop = MetaCognitiveLoop(base_dir, interval_seconds=int(os.getenv("LUMINA_INTERVAL", "300")))
    loop.start()
    print(f"[{timestamp()}] Lumina meta‑cognitive loop started (interval {loop.interval}s)")

    try:
        while True:
            cmd = input("lumina> ").strip().lower()
            if cmd in {"exit", "quit", "stop"}:
                break
            elif cmd == "status":
                print("Running... (Ctrl‑C or type 'exit' to stop)")
            elif cmd == "show goals":
                goals = load_json(base_dir / "goals.json", [])
                print(json.dumps(goals, indent=2))
            elif cmd == "show proposals":
                props = load_json(base_dir / "proposals.json", [])
                print(json.dumps(props, indent=2))
            else:
                print("Commands: status, show goals, show proposals, exit")
    except KeyboardInterrupt:
        pass
    finally:
        loop.stop()
        loop.join()
        print(f"[{timestamp()}] Lumina meta‑cognitive loop terminated")

if __name__ == "__main__":
    main()