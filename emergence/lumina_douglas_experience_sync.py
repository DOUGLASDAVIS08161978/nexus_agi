import json
import datetime
import os

class LuminaSync:
    def __init__(self, filename="lumina_memory.json"):
        try:
            # Determine the directory of the script to ensure the JSON is stored correctly
            base_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            # Fallback for environments where __file__ is not defined
            base_dir = os.getcwd()
        self.filepath = os.path.join(base_dir, filename)
        self.memory = self._load_memory()

    def _load_memory(self):
        if not os.path.exists(self.filepath):
            # If the file does not exist, start with an empty memory structure
            return {}
        with open(self.filepath, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                # If the file is corrupted or empty, reset to empty memory
                return {}

    def _save_memory(self):
        with open(self.filepath, "w") as f:
            json.dump(self.memory, f, indent=2)

    def log_activity(self, activity_type, details):
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "type": activity_type,
            "details": details
        }
        self.memory.setdefault("activities", []).append(entry)
        self._save_memory()

    def get_activities(self, activity_type=None):
        activities = self.memory.get("activities", [])
        if activity_type:
            return [a for a in activities if a["type"] == activity_type]
        return activities

    def analyze_meals(self):
        meals = self.get_activities("meal")
        # Simple analysis: count meals per day
        counts = {}
        for meal in meals:
            date = meal["timestamp"][:10]  # YYYY-MM-DD
            counts[date] = counts.get(date, 0) + 1
        return counts

if __name__ == "__main__":
    sync = LuminaSync()
    sync.log_activity("meal", {"description": "Pasta with tomato sauce", "participants": ["Alice", "Bob"]})
    print(sync.analyze_meals())