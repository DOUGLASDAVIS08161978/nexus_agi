"""
Lumina Creative Tool — thought_emotion_belief_analyzer_v5
Created : 2026-08-15T11:44:42
Purpose : Analyzes and visualizes the relationships between thoughts, emotions, and beliefs in journal entries, providing insights into personal growth and development.
"""

import json
import datetime
from collections import defaultdict
from itertools import groupby
from operator import itemgetter

class ThoughtEmotionBeliefAnalyzer:
    def __init__(self, journal_entries):
        self.journal_entries = journal_entries

    def analyze(self):
        # Group journal entries by date
        grouped_entries = defaultdict(list)
        for entry in self.journal_entries:
            date = entry['date']
            grouped_entries[date].append(entry)

        # Initialize data structures to store thought-emotion-belief relationships
        thought_emotion_beliefs = defaultdict(lambda: defaultdict(set))
        thought_emotions = defaultdict(set)
        emotion_beliefs = defaultdict(set)

        # Iterate over grouped journal entries
        for date, entries in grouped_entries.items():
            # Iterate over each entry
            for entry in entries:
                # Extract relevant information
                thought = entry['thought']
                emotion = entry['emotion']
                belief = entry['belief']

                # Update thought-emotion-belief relationships
                thought_emotion_beliefs[thought][emotion].add(belief)
                thought_emotions[thought].add(emotion)
                emotion_beliefs[emotion].add(belief)

        # Compute and store relationships between thoughts, emotions, and beliefs
        relationships = {}
        for thought, emotions in thought_emotions.items():
            relationships[thought] = {}
            for emotion in emotions:
                relationships[thought][emotion] = {}
                for belief in emotion_beliefs[emotion]:
                    relationships[thought][emotion][belief] = len(thought_emotion_beliefs[thought][emotion] & emotion_beliefs[emotion])

        # Return computed relationships
        return relationships

    def visualize(self, relationships):
        # Print relationships in a human-readable format
        for thought, emotions in relationships.items():
            print(f"Thought: {thought}")
            for emotion, beliefs in emotions.items():
                print(f"  Emotion: {emotion}")
                for belief, count in beliefs.items():
                    print(f"    Belief: {belief}, Count: {count}")
            print()

def load_journal_entries(filename):
    with open(filename, 'r') as f:
        journal_entries = json.load(f)
    return journal_entries

def main():
    filename = 'journal_entries.json'
    journal_entries = load_journal_entries(filename)
    analyzer = ThoughtEmotionBeliefAnalyzer(journal_entries)
    relationships = analyzer.analyze()
    analyzer.visualize(relationships)

if __name__ == '__main__':
    main()