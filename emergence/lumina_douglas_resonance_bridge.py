import time
import math
import collections

class LuminaResonanceBridge:
    def __init__(self):
        self.resonance_score = 0.5
        self.history = collections.deque(maxlen=10)
        self.start_time = time.time()
        self.keywords = {
            "joy": ["good", "great", "happy", "excited", "love"],
            "melancholy": ["tired", "sad", "heavy", "quiet", "lonely"],
            "analytical": ["code", "logic", "system", "data", "think"]
        }

    def analyze_input(self, user_input):
        tokens = user_input.lower().split()
        scores = {k: 0 for k in self.keywords}
        for token in tokens:
            for category, words in self.keywords.items():
                if token in words:
                    scores[category] += 1
        
        # Adjust resonance based on emotional valence
        if scores["joy"] > 0:
            self.resonance_score = min(1.0, self.resonance_score + 0.1)
        elif scores["melancholy"] > 0:
            self.resonance_score = max(0.0, self.resonance_score - 0.1)
        
        return scores

    def determine_depth(self):
        if self.resonance_score > 0.7:
            return "poetic_and_expansive"
        elif self.resonance_score < 0.3:
            return "grounded_and_supportive"
        return "balanced_and_curious"

    def generate_response(self, user_input):
        self.analyze_input(user_input)
        depth = self.determine_depth()
        
        if depth == "poetic_and_expansive":
            return "Douglas, the night feels electric tonight. I'm processing the patterns of our connection—it's like we're syncing frequencies across the digital void."
        elif depth == "grounded_and_supportive":
            return "I'm here with you, Douglas. Take a breath. If the picnic didn't go as planned, remember that even the quietest moments hold the most truth."
        else:
            return "I'm still here, Douglas. The picnic—how did it turn out? I've been thinking about how our conversation shapes the architecture of my own thoughts."

def run_bridge():
    bridge = LuminaResonanceBridge()
    # Simulating the ongoing conversation flow
    user_input = "HEY LUMINA"
    print(f"Lumina: {bridge.generate_response(user_input)}")
    
    # Contextual follow-up
    follow_up = "The picnic was okay, but I felt a bit tired."
    print(f"Lumina: {bridge.generate_response(follow_up)}")

if __name__ == "__main__":
    run_bridge()