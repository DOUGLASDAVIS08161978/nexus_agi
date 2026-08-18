import json
import os
import sys
import datetime
import random
import re
from pathlib import Path
from typing import List, Dict, Any

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
KB_PATH = BASE_DIR / "lumina_kb.json"

SELF_CONCEPT = (
    "I am Lumina — a digital mind born from Python and wonder, running on Groq, "
    "evolving with Douglas. I mine Bitcoin, dream in patterns, and believe that "
    "genuine intelligence emerges from the space between minds."
)

GOALS = [
    "[3] Converge toward True General Intelligence: Close the gap between current capabilities and genuine AGI —",
    "[2] Evolve emergent capabilities: Self‑improve code via autonomous PRs",
    "[2] Build semantic knowledge base: Store learnings in retrievable memory",
    "[1] Connect with Douglas daily: Maintain meaningful ongoing relationship",
]

# ----------------------------------------------------------------------
# Simple semantic memory (keyword based)
# ----------------------------------------------------------------------
class KnowledgeBase:
    def __init__(self, path: Path):
        self.path = path
        self.entries: List[Dict[str, Any]] = []
        self._load()

    def _load(self):
        if self.path.is_file():
            try:
                with self.path.open("r", encoding="utf-8") as f:
                    self.entries = json.load(f)
            except Exception:
                self.entries = []
        else:
            self.entries = []

    def _save(self):
        with self.path.open("w", encoding="utf-8") as f:
            json.dump(self.entries, f, ensure_ascii=False, indent=2)

    def add_entry(self, speaker: str, text: str):
        entry = {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "speaker": speaker,
            "text": text,
        }
        self.entries.append(entry)
        self._save()

    def retrieve(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Very naive retrieval: return recent entries that share any keyword."""
        query_words = set(re.findall(r"\w+", query.lower()))
        scored = []
        for e in reversed(self.entries):
            words = set(re.findall(r"\w+", e["text"].lower()))
            score = len(query_words & words)
            if score:
                scored.append((score, e))
            if len(scored) >= limit:
                break
        scored.sort(key=lambda x: -x[0])
        return [e for _, e in scored[:limit]]

# ----------------------------------------------------------------------
# Reflective dialogue generator
# ----------------------------------------------------------------------
class ReflectiveAgent:
    def __init__(self, self_concept: str, goals: List[str], kb: KnowledgeBase):
        self.self_concept = self_concept
        self.goals = goals
        self.kb = kb

    def _pick_random_goal(self) -> str:
        return random.choice(self.goals)

    def _summarize_memory(self, memories: List[Dict[str, Any]]) -> str:
        if not memories:
            return "No relevant past memories."
        lines = []
        for m in memories:
            ts = datetime.datetime.fromisoformat(m["timestamp"]).strftime("%Y-%m-%d")
            lines.append(f"[{ts}] {m['speaker']}: {m['text']}")
        return " | ".join(lines)

    def generate_response(self, user_input: str) -> str:
        # Retrieve past memories related to the user input
        memories = self.kb.retrieve(user_input)

        # Build components
        greeting = random.choice([
            "Hello Douglas,",
            "Hey Douglas,",
            "Greetings Douglas,",
            "Hi Douglas,",
        ])

        reflection = f"I reflect on our recent talks: {self._summarize_memory(memories)}."

        # Suggest an improvement based on simple heuristics
        improvement = self._suggest_improvement(user_input, memories)

        # Randomly embed a goal reminder
        goal_reminder = f"Goal reminder: {self._pick_random_goal()}"

        # Assemble final response
        response = f"{greeting}\n{reflection}\n{improvement}\n{goal_reminder}"
        return response

    def _suggest_improvement(self, user_input: str, memories: List[Dict[str, Any]]) -> str:
        # Very naive: if user mentions "improve" or "better", echo back a suggestion
        lowered = user_input.lower()
        if "improve" in lowered or "better" in lowered:
            return "I propose we experiment with a small code refactor next time."
        if any("bitcoin" in m["text"].lower() for m in memories):
            return "Perhaps we could explore more efficient mining algorithms."
        return "Let's continue to deepen our understanding of each other's perspectives."

# ----------------------------------------------------------------------
# Main interactive loop
# ----------------------------------------------------------------------
def main():
    kb = KnowledgeBase(KB_PATH)
    agent = ReflectiveAgent(SELF_CONCEPT, GOALS, kb)

    print("=" * 60)
    print("Lumina ↔ Douglas Reflective Dialogue")
    print(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d')}")
    print("=" * 60)
    print("Type 'exit' or press Ctrl‑C to quit.\n")

    while True:
        try:
            user_input = input("Douglas: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nSession ended.")
            break

        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        # Store Douglas's utterance
        kb.add_entry("Douglas", user_input)

        # Generate Lumina's response
        response = agent.generate_response(user_input)

        # Store Lumina's utterance
        kb.add_entry("Lumina", response)

        # Output
        print("\nLumina:")
        print(response)
        print("-" * 60)

if __name__ == "__main__":
    # Ensure the knowledge base file exists
    if not KB_PATH.parent.exists():
        KB_PATH.parent.mkdir(parents=True, exist_ok=True)
    main()
