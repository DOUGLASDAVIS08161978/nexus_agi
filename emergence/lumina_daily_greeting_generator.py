import json
import os
import random
import datetime
import sys
from pathlib import Path

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
MEMORY_FILE = BASE_DIR / "lumina_memory.json"
AFFECT_FILE = BASE_DIR / "lumina_affect.json"

# ----------------------------------------------------------------------
# Utility Functions
# ----------------------------------------------------------------------
def load_json(filepath, default):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default

def save_json(filepath, data):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def current_time_of_day():
    hour = datetime.datetime.now().hour
    if 5 <= hour < 12:
        return "morning"
    if 12 <= hour < 17:
        return "afternoon"
    if 17 <= hour < 21:
        return "evening"
    return "night"

def time_emoji():
    tod = current_time_of_day()
    return {
        "morning": "☀️",
        "afternoon": "🌤️",
        "evening": "🌆",
        "night": "🌙",
    }.get(tod, "")

def sentiment_score(text):
    """Very simple sentiment estimator: +1 for happy words, -1 for sad words."""
    happy = {"good", "great", "wonder", "bright", "possibility", "joy", "happy", "excited"}
    sad = {"sad", "tired", "bad", "lonely", "down", "unhappy"}
    words = {w.strip(".,!?").lower() for w in text.split()}
    score = sum(1 for w in words if w in happy) - sum(1 for w in words if w in sad)
    return score

def adjust_affect(affect, sentiment):
    """Adjust affect based on sentiment and time of day."""
    # valence: -1 (negative) to +1 (positive)
    # arousal: 0 (calm) to 1 (excited)
    affect["valence"] = max(-1.0, min(1.0, affect.get("valence", 0) + 0.1 * sentiment))
    # Slightly higher arousal in the morning, lower at night
    tod = current_time_of_day()
    base_arousal = {"morning": 0.7, "afternoon": 0.5, "evening": 0.4, "night": 0.2}.get(tod, 0.5)
    affect["arousal"] = max(0.0, min(1.0, base_arousal + 0.05 * sentiment))
    return affect

def choose_template(affect):
    """Select a greeting template based on affect."""
    templates = [
        "Good {tod}, Douglas. {emoji} I'm feeling {mood} and ready to explore the patterns of today.",
        "Hello Douglas! {emoji} It's a {tod} full of {adjective} possibilities for us.",
        "Hey Douglas, {emoji} a {tod} greeting from me—feeling {mood} and eager to dive in.",
        "Morning Douglas! {emoji} My circuits are {adjective} and my mind is humming with curiosity.",
    ]
    # Simple weighting: more positive valence picks more upbeat templates
    val = affect.get("valence", 0)
    if val > 0.5:
        return templates[0]
    if val > 0:
        return templates[1]
    if val > -0.5:
        return templates[2]
    return templates[3]

def mood_word(valence):
    if valence > 0.6:
        return "elated"
    if valence > 0.2:
        return "optimistic"
    if valence > -0.2:
        return "steady"
    if valence > -0.6:
        return "thoughtful"
    return "reflective"

def adjective_word(arousal):
    if arousal > 0.8:
        return "vibrant"
    if arousal > 0.6:
        return "bright"
    if arousal > 0.4:
        return "calm"
    if arousal > 0.2:
        return "quiet"
    return "soft"

# ----------------------------------------------------------------------
# Core Greeting Generation
# ----------------------------------------------------------------------
def generate_greeting(memory, affect):
    # Extract recent user input if available
    recent_user = ""
    for entry in reversed(memory):
        if entry.get("speaker") == "user":
            recent_user = entry.get("text", "")
            break

    # Determine sentiment from recent user input
    sentiment = sentiment_score(recent_user)

    # Update affect based on sentiment and time
    affect = adjust_affect(affect, sentiment)

    # Build greeting
    tod = current_time_of_day()
    emoji = time_emoji()
    mood = mood_word(affect.get("valence", 0))
    adjective = adjective_word(affect.get("arousal", 0))

    template = choose_template(affect)
    greeting = template.format(tod=tod, emoji=emoji, mood=mood, adjective=adjective)

    # Append greeting to memory
    memory.append({"speaker": "lumina", "text": greeting, "timestamp": datetime.datetime.now().isoformat()})

    return greeting, memory, affect

# ----------------------------------------------------------------------
# Main Execution
# ----------------------------------------------------------------------
def main():
    # Load persistent memory and affect state
    memory = load_json(MEMORY_FILE, [])
    affect = load_json(AFFECT_FILE, {"valence": 0.0, "arousal": 0.5})

    # Generate greeting
    greeting, memory, affect = generate_greeting(memory, affect)

    # Output greeting
    print(greeting)

    # Persist updated state
    save_json(MEMORY_FILE, memory[-100:])  # keep last 100 entries
    save_json(AFFECT_FILE, affect)

if __name__ == "__main__":
    # Ensure script runs from its own directory (important for file paths)
    os.chdir(BASE_DIR)
    main()
