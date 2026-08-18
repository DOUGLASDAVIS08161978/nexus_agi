import os
import json
import threading
import re
from collections import deque
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

# Simple lexicon mapping words to 6‑dimensional emotion vectors
# [joy, sadness, anger, fear, disgust, surprise]
EMOTION_LEXICON: Dict[str, List[float]] = {
    "happy": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "joyful": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "sad": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    "unhappy": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    "angry": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    "mad": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    "fearful": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    "scared": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    "disgusted": [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    "gross": [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    "surprised": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    "astonished": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
}

class EmotionTracker:
    def __init__(
        self,
        persistence_path: Optional[str] = None,
        window: timedelta = timedelta(hours=24),
    ):
        self.persistence_path = (
            persistence_path
            or os.path.join(os.path.dirname(__file__), "emotion_history.json")
        )
        self.window = window
        self.history: deque[Tuple[datetime, List[float]]] = deque()
        self.lock = threading.Lock()
        self._load_history()

    def _load_history(self) -> None:
        if not os.path.exists(self.persistence_path):
            return
        try:
            with open(self.persistence_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for entry in data:
                ts = datetime.fromisoformat(entry["timestamp"])
                emo = entry["emotion"]
                if isinstance(emo, list) and len(emo) == 6:
                    self.history.append((ts, emo))
        except Exception:
            # Corrupt file or parsing error – ignore and start fresh
            pass

    def _save_history(self) -> None:
        data = [
            {"timestamp": ts.isoformat(), "emotion": emo}
            for ts, emo in self.history
        ]
        with open(self.persistence_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _prune_old_entries(self, current_ts: datetime) -> None:
        cutoff = current_ts - self.window
        while self.history and self.history[0][0] < cutoff:
            self.history.popleft()

    def _infer_emotion(self, message: str) -> List[float]:
        words = re.findall(r"\b\w+\b", message.lower())
        vector = [0.0] * 6
        count = 0
        for w in words:
            if w in EMOTION_LEXICON:
                vec = EMOTION_LEXICON[w]
                vector = [v + wv for v, wv in zip(vector, vec)]
                count += 1
        if count > 0:
            vector = [v / count for v in vector]
        return vector

    def add_message(self, message: str) -> None:
        ts = datetime.utcnow()
        emo = self._infer_emotion(message)
        with self.lock:
            self.history.append((ts, emo))
            self._prune_old_entries(ts)
            self._save_history()

    def get_current_emotion(self) -> List[float]:
        with self.lock:
            if not self.history:
                return [0.0] * 6
            total = [0.0] * 6
            for _, emo in self.history:
                total = [t + e for t, e in zip(total, emo)]
            count = len(self.history)
            return [t / count for t in total]

    def get_recent_emotions(self, n: int = 10) -> List[Tuple[datetime, List[float]]]:
        with self.lock:
            return list(self.history)[-n:]

# Example usage
if __name__ == "__main__":
    tracker = EmotionTracker()
    tracker.add_message("I am so happy today!")
    tracker.add_message("This is sad and scary.")
    print("Current emotion vector:", tracker.get_current_emotion())
    print("Recent entries:", tracker.get_recent_emotions(5))