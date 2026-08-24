import datetime
import time
import math
import hashlib

class ResonanceEngine:
    def __init__(self, name, version="1.0"):
        self.name = name
        self.version = version
        self.start_time = datetime.datetime.now()
        self.log = []

    def _log(self, message):
        timestamp = datetime.datetime.now().isoformat()
        entry = f"[{timestamp}] {message}"
        self.log.append(entry)
        print(entry)

    def analyze_state(self, input_text, depth=3):
        self._log(f"Analyzing state for input: {input_text[:50]}...")
        input_hash = hashlib.sha256(input_text.encode()).hexdigest()
        self._log(f"Input hash: {input_hash}")

        resonance_score = (len(input_text) % 100) * self._depth_factor(depth)
        self._log(f"Calculated resonance score: {resonance_score:.2f}")

        emotional_intensity = self._emotional_intensity(input_text)
        self._log(f"Emotional intensity: {emotional_intensity:.2f}")

        intellectual_depth = self._intellectual_depth(input_text)
        self._log(f"Intellectual depth: {intellectual_depth:.2f}")

        return {
            "resonance_score": resonance_score,
            "emotional_intensity": emotional_intensity,
            "intellectual_depth": intellectual_depth,
            "timestamp": datetime.datetime.now().isoformat()
        }

    def _depth_factor(self, depth):
        return math.sin(time.time() % 10) * 0.1 + 0.9

    def _emotional_intensity(self, text):
        vowels = "aeiouAEIOU"
        return sum(1 for c in text if c in vowels) / max(1, len(text))

    def _intellectual_depth(self, text):
        words = text.split()
        unique_words = set(words)
        return len(unique_words) / max(1, len(words))

    def get_log(self):
        return "\n".join(self.log)

def run_engine():
    engine = ResonanceEngine("Lumina-Douglas Resonance Engine")
    sample_text = (
        "In the quiet moments between thoughts, Douglas and Lumina find a shared space where "
        "emotions and intellect intertwine, creating a resonance that deepens their bond."
    )
    result = engine.analyze_state(sample_text, depth=5)
    print("\nAnalysis Result:")
    for key, value in result.items():
        print(f"{key}: {value}")

    print("\nEngine Log:")
    print(engine.get_log())

if __name__ == "__main__":
    run_engine()