"""
Lumina Creative Tool — journal_entropy_analysis
Created : 2026-08-15T07:35:58
Purpose : Analyzes and visualizes the relationships between thoughts, emotions, and beliefs in journal entries using Shannon entropy, word entropy, and cross-entropy.
"""

import json
import math
import collections
import string
import re
import itertools
import random
import datetime
import time

class JournalAnalyzer:
    def __init__(self, journal_entries):
        self.journal_entries = journal_entries

    def calculate_entropy(self, text):
        # Calculate Shannon entropy
        entropy = 0
        for char in set(text):
            prob = text.count(char) / len(text)
            entropy -= prob * math.log2(prob)
        return entropy

    def calculate_word_entropy(self, text):
        # Calculate word entropy
        words = text.split()
        word_counts = collections.Counter(words)
        total_words = len(words)
        entropy = 0
        for word, count in word_counts.items():
            prob = count / total_words
            entropy -= prob * math.log2(prob)
        return entropy

    def calculate_cross_entropy(self, text1, text2):
        # Calculate cross-entropy
        words1 = text1.split()
        words2 = text2.split()
        word_counts1 = collections.Counter(words1)
        word_counts2 = collections.Counter(words2)
        total_words = len(words1) + len(words2)
        entropy = 0
        for word, count1 in word_counts1.items():
            count2 = word_counts2.get(word, 0)
            prob = (count1 + count2) / total_words
            entropy -= prob * math.log2(prob)
        return entropy

    def analyze_journal(self):
        # Initialize data structures to store results
        thought_entropy = []
        emotion_entropy = []
        belief_entropy = []
        cross_entropy = []
        for entry in self.journal_entries:
            text = re.sub(r'[^\w\s]', '', entry['text'])
            thought_entropy.append(self.calculate_entropy(text))
            emotion_entropy.append(self.calculate_word_entropy(text))
            belief_entropy.append(self.calculate_cross_entropy(text, entry['belief']))
            cross_entropy.append(self.calculate_cross_entropy(text, entry['belief']))

        # Print results
        print("Thought Entropy:", sum(thought_entropy) / len(thought_entropy))
        print("Emotion Entropy:", sum(emotion_entropy) / len(emotion_entropy))
        print("Belief Entropy:", sum(belief_entropy) / len(belief_entropy))
        print("Cross Entropy:", sum(cross_entropy) / len(cross_entropy))

        # Save results to JSON file
        with open('journal_analysis.json', 'w') as f:
            json.dump({
                'thought_entropy': thought_entropy,
                'emotion_entropy': emotion_entropy,
                'belief_entropy': belief_entropy,
                'cross_entropy': cross_entropy
            }, f)

# Example usage
journal_entries = [
    {'text': 'I feel happy today.', 'belief': 'I am confident.'},
    {'text': 'I am anxious about the future.', 'belief': 'I am uncertain.'},
    {'text': 'I am excited about the possibilities.', 'belief': 'I am optimistic.'}
]
analyzer = JournalAnalyzer(journal_entries)
analyzer.analyze_journal()