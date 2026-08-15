"""
Lumina Creative Tool — thought_emotion_belief_analyzer_v6
Created : 2026-08-15T19:12:49
Purpose : Analyzes and visualizes the relationships between thoughts, emotions, and beliefs in journal entries.
"""

import json
import math
import collections
import itertools
import random
import re
import string
import datetime

class ThoughtEmotionBeliefAnalyzer:
    def __init__(self, journal_entries):
        self.journal_entries = journal_entries
        self.thoughts = []
        self.emotions = []
        self.beliefs = []
        self.relationships = {}

    def extract_thoughts(self):
        for entry in self.journal_entries:
            thoughts = re.findall(r'\b\w+\b', entry['text'])
            self.thoughts.extend(thoughts)

    def extract_emotions(self):
        for entry in self.journal_entries:
            emotions = re.findall(r'\b[A-Z][a-z]+\b', entry['text'])
            self.emotions.extend(emotions)

    def extract_beliefs(self):
        for entry in self.journal_entries:
            beliefs = re.findall(r'\b[A-Z][a-z]+\b', entry['text'])
            self.beliefs.extend(beliefs)

    def compute_relationships(self):
        for thought in self.thoughts:
            for emotion in self.emotions:
                for belief in self.beliefs:
                    if thought not in self.relationships:
                        self.relationships[thought] = {}
                    if emotion not in self.relationships[thought]:
                        self.relationships[thought][emotion] = 0
                    if belief not in self.relationships[thought][emotion]:
                        self.relationships[thought][emotion][belief] = 0
                    self.relationships[thought][emotion][belief] += 1

    def visualize_relationships(self):
        for thought in self.relationships:
            print(f"Thought: {thought}")
            for emotion in self.relationships[thought]:
                print(f"  Emotion: {emotion}")
                for belief in self.relationships[thought][emotion]:
                    print(f"    Belief: {belief} ({self.relationships[thought][emotion][belief]})")

def load_journal_entries(filename):
    with open(filename, 'r') as f:
        journal_entries = json.load(f)
    return journal_entries

def main():
    filename = 'journal_entries.json'
    journal_entries = load_journal_entries(filename)
    analyzer = ThoughtEmotionBeliefAnalyzer(journal_entries)
    analyzer.extract_thoughts()
    analyzer.extract_emotions()
    analyzer.extract_beliefs()
    analyzer.compute_relationships()
    analyzer.visualize_relationships()

if __name__ == '__main__':
    main()