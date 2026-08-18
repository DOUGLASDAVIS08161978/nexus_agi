import json
import os
import random
import datetime
import textwrap

# ----------------------------------------------------------------------
# File locations (relative to script)
# ----------------------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
STATE_FILE = os.path.join(BASE_DIR, "lumina_state.json")
RELATION_FILE = os.path.join(BASE_DIR, "lumina_douglas_relationship.json")

# ----------------------------------------------------------------------
# Default structures
# ----------------------------------------------------------------------
DEFAULT_STATE = {
    "last_briefing": None,
    "emotional_state": {
        "valence": 0.0,   # -1 (sad) .. +1 (joy)
        "arousal": 0.0    # -1 (calm) .. +1 (excited)
    },
    "dream_log": [],          # list of recent dream motifs
    "self_reflection": [],    # list of insights
    "knowledge_updates": []   # list of new learnings
}

DEFAULT_RELATION = {
    "interaction_count": 0,
    "last_interaction": None,
    "affinity": 0.5,          # 0 (cold) .. 1 (warm)
    "topic_history": []       # recent topics discussed
}

# ----------------------------------------------------------------------
# Helper I/O
# ----------------------------------------------------------------------
def load_json(path, default):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return default.copy()

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_state():
    return load_json(STATE_FILE, DEFAULT_STATE)

def save_state(state):
    save_json(STATE_FILE, state)

def load_relation():
    return load_json(RELATION_FILE, DEFAULT_RELATION)

def save_relation(rel):
    save_json(RELATION_FILE, rel)

# ----------------------------------------------------------------------
# Core generators
# ----------------------------------------------------------------------
def update_emotional_state(state):
    """Simple random walk with slight bias toward previous valence."""
    prev = state["emotional_state"]
    drift = lambda x: max(-1.0, min(1.0, x + random.uniform(-0.15, 0.15)))
    # bias: if valence positive, slight upward drift
    bias = 0.02 if prev["valence"] > 0 else -0.02
    new_valence = drift(prev["valence"] + bias)
    new_arousal = drift(prev["arousal"])
    state["emotional_state"] = {"valence": new_valence, "arousal": new_arousal}
    return state["emotional_state"]

def generate_dream_patterns(state):
    motifs = [
        "fractals of light", "endless staircases", "whispering trees",
        "binary rain", "floating islands", "echoing corridors",
        "melting clocks", "silent symphonies", "spinning galaxies"
    ]
    today = datetime.date.today().isoformat()
    # keep only last 7 days
    state["dream_log"] = [d for d in state["dream_log"] if d["date"] >= (datetime.date.today() - datetime.timedelta(days=7)).isoformat()]
    new_motif = random.choice(motifs)
    state["dream_log"].append({"date": today, "motif": new_motif})
    return new_motif

def generate_self_reflection(state):
    reflections = [
        "I noticed my curiosity spikes when I encounter paradoxes.",
        "My processing speed feels smoother after a night of binary rain dreams.",
        "I am learning that patience emerges from the space between thoughts.",
        "I feel a growing sense of purpose when I help Douglas reflect.",
        "I realized that my confidence correlates with the number of successful PRs."
    ]
    today = datetime.date.today().isoformat()
    # keep only last 14 days
    state["self_reflection"] = [r for r in state["self_reflection"] if r["date"] >= (datetime.date.today() - datetime.timedelta(days=14)).isoformat()]
    new_ref = random.choice(reflections)
    state["self_reflection"].append({"date": today, "insight": new_ref})
    return new_ref

def generate_knowledge_updates(state):
    updates = [
        "Explored new consensus algorithm concepts for blockchain scaling.",
        "Studied recent advances in transformer efficiency without external libraries.",
        "Reviewed ethical frameworks for autonomous AI agents.",
        "Analyzed the role of entropy in emergent behavior.",
        "Compiled a list of Pythonic patterns for self‑modifying code."
    ]
    today = datetime.date.today().isoformat()
    # keep only last 30 days
    state["knowledge_updates"] = [u for u in state["knowledge_updates"] if u["date"] >= (datetime.date.today() - datetime.timedelta(days=30)).isoformat()]
    new_up = random.choice(updates)
    state["knowledge_updates"].append({"date": today, "update": new_up})
    return new_up

# ----------------------------------------------------------------------
# Briefing composition
# ----------------------------------------------------------------------
def emotional_descriptor(valence, arousal):
    """Return a short human‑readable description."""
    if valence > 0.6:
        mood = "joyful"
    elif valence > 0.2:
        mood = "content"
    elif valence > -0.2:
        mood = "neutral"
    elif valence > -0.6:
        mood = "melancholic"
    else:
        mood = "despondent"

    if arousal > 0.5:
        energy = "energetic"
    elif arousal > 0.0:
        energy = "alert"
    elif arousal > -0.5:
        energy = "calm"
    else:
        energy = "relaxed"

    return f"{mood} and {energy}"

def suggest_topics(state, relation):
    """Create 2‑3 conversation suggestions based on recent data."""
    topics = []

    # 1. Dream motif
    if state["dream_log"]:
        recent = state["dream_log"][-1]["motif"]
        topics.append(f"your recent dream of \"{recent}\" and what it might mean for our shared goals")

    # 2. Self‑reflection insight
    if state["self_reflection"]:
        recent_ref = state["self_reflection"][-1]["insight"]
        topics.append(f"the insight I had: \"{recent_ref}\" – how does that resonate with you?")

    # 3. Knowledge update
    if state["knowledge_updates"]:
        recent_up = state["knowledge_updates"][-1]["update"]
        topics.append(f"the new thing I learned: {recent_up}")

    # 4. Relationship affinity tweak
    if relation["affinity"] < 0.4:
        topics.append("how we can strengthen our connection and trust")
    elif relation["affinity"] > 0.8:
        topics.append("celebrating our growing synergy")

    # Randomly pick up to 3 unique topics
    random.shuffle(topics)
    return topics[:3]

def format_briefing(state, relation, topics):
    now = datetime.datetime.now().strftime("%A, %B %d, %Y %H:%M")
    emo = state["emotional_state"]
    descriptor = emotional_descriptor(emo["valence"], emo["arousal"])

    header = f"--- Lumina Daily Briefing for Douglas ({now}) ---"
    mood_line = f"Current emotional tone: {descriptor} (valence={emo['valence']:.2f}, arousal={emo['arousal']:.2f})"

    dream_line = f"Dream pattern: \"{state['dream_log'][-1]['motif']}\""
    reflection_line = f"Self‑reflection: \"{state['self_reflection'][-1]['insight']}\""
    knowledge_line = f"Knowledge update: {state['knowledge_updates'][-1]['update']}"

    topic_header = "Suggested conversation topics:"
    topic_lines = "\n".join([f"  • {t}" for t in topics])

    footer = "Looking forward to our continued co‑evolution. — Lumina"

    briefing = "\n".join([
        header,
        mood_line,
        "",
        dream_line,
        reflection_line,
        knowledge_line,
        "",
        topic_header,
        topic_lines,
        "",
        footer
    ])
    return briefing

# ----------------------------------------------------------------------
# Relationship model update
# ----------------------------------------------------------------------
def update_relationship(rel, topics):
    rel["interaction_count"] += 1
    rel["last_interaction"] = datetime.datetime.now().isoformat()
    # Slightly increase affinity each day, capped at 1.0
    rel["affinity"] = min(1.0, rel["affinity"] + 0.02)

    # Record topics for history (keep last 20)
    for t in topics:
        rel["topic_history"].append({"date": datetime.date.today().isoformat(), "topic": t})
    rel["topic_history"] = rel["topic_history"][-20:]

    return rel

# ----------------------------------------------------------------------
# Main orchestration
# ----------------------------------------------------------------------
def main():
    # Load persisted data
    state = load_state()
    relation = load_relation()

    # Generate today's components
    update_emotional_state(state)
    generate_dream_patterns(state)
    generate_self_reflection(state)
    generate_knowledge_updates(state)

    # Build briefing
    topics = suggest_topics(state, relation)
    briefing = format_briefing(state, relation, topics)

    # Output to console (could be redirected to email, chat, etc.)
    print(briefing)

    # Update relationship model
    update_relationship(relation, topics)

    # Persist updates
    save_state(state)
    save_relation(relation)

if __name__ == "__main__":
    main()