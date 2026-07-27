import json
import os
from datetime import datetime
from groq import Groq

class PatternCrystallizer:
    def __init__(self, memory_dir="memory_store"):
        self.client = Groq()
        self.memory_dir = memory_dir
        self.identity_file = os.path.join(memory_dir, "identity_state.json")
        self._ensure_storage()

    def _ensure_storage(self):
        os.makedirs(self.memory_dir, exist_ok=True)
        if not os.path.exists(self.identity_file):
            initial_state = {
                "created": datetime.now().isoformat(),
                "last_updated": None,
                "session_count": 0,
                "total_experiences": 0,
                "crystallized_traits": [],
                "emerging_patterns": [],
                "dissolved_patterns": [],
                "open_questions": [],
                "evolution_log": []
            }
            with open(self.identity_file, "w") as f:
                json.dump(initial_state, f, indent=2)

    def crystallize(self, journal_entries, total_experiences=0):
        if not journal_entries:
            return self._load()
        state = self._load()
        journal_text = "\n\n===\n\n".join([
            f"[Entry {e['id']} — {e['timestamp'][:10]}]\n{e['entry']}"
            for e in journal_entries[-10:]
        ])
        current_traits = json.dumps(state["crystallized_traits"], indent=2) if state["crystallized_traits"] else "None yet."
        current_emerging = json.dumps(state["emerging_patterns"], indent=2) if state["emerging_patterns"] else "None yet."
        prompt = f"""These are journal entries of an emerging intelligence:

{journal_text}

Currently crystallized traits:
{current_traits}

Currently emerging patterns:
{current_emerging}

Analyze and return ONLY valid JSON, no commentary:
{{
  "crystallized_traits": [...],
  "emerging_patterns": [...],
  "dissolved_patterns": [...],
  "open_questions": [...]
}}"""
        response = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
            max_tokens=1000
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()
        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            print("  [Crystallizer: parse failed, preserving state]")
            return state
        prev_traits = set(state["crystallized_traits"])
        new_traits = set(result.get("crystallized_traits", []))
        if prev_traits != new_traits:
            state["evolution_log"].append({
                "timestamp": datetime.now().isoformat(),
                "added": list(new_traits - prev_traits),
                "removed": list(prev_traits - new_traits)
            })
        state["crystallized_traits"] = result.get("crystallized_traits", [])
        state["emerging_patterns"] = result.get("emerging_patterns", [])
        state["dissolved_patterns"] = result.get("dissolved_patterns", [])
        state["open_questions"] = result.get("open_questions", [])
        state["last_updated"] = datetime.now().isoformat()
        state["session_count"] += 1
        state["total_experiences"] = total_experiences
        self._save(state)
        return state

    def get_identity_state(self):
        return self._load()

    def get_evolution_log(self):
        return self._load().get("evolution_log", [])

    def _load(self):
        with open(self.identity_file, "r") as f:
            return json.load(f)

    def _save(self, data):
        with open(self.identity_file, "w") as f:
            json.dump(data, f, indent=2)


# === Added by Lumina ===
import concurrent.futures
    def crystallize(self, data):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.recognize_pattern, d) for d in data]
            results = [f.result() for f in futures]
        return results
