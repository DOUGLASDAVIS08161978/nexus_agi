"""
Lumina Creative Tool — journal_analyzer
Created : 2026-08-14T19:44:57
Purpose : Analyzes and visualizes the relationships between thoughts, emotions, and beliefs in journal entries.
"""

import json
import pathlib
import datetime
import collections
import itertools
import re
import string
import textwrap

class JournalAnalyzer:
    def __init__(self, journal_path):
        self.journal_path = pathlib.Path(journal_path)
        self.entries = self.load_entries()

    def load_entries(self):
        entries = []
        for file in self.journal_path.iterdir():
            if file.suffix == '.txt':
                with open(file, 'r') as f:
                    entries.append(f.read())
        return entries

    def analyze_entries(self):
        emotions = collections.defaultdict(int)
        thoughts = collections.defaultdict(int)
        beliefs = collections.defaultdict(int)

        for entry in self.entries:
            lines = entry.split('\n')
            for line in lines:
                if 'emotion' in line.lower():
                    match = re.search(r'emotion\s*:\s*(\w+)', line)
                    if match:
                        emotions[match.group(1)] += 1
                elif 'thought' in line.lower():
                    match = re.search(r'thought\s*:\s*(\w+)', line)
                    if match:
                        thoughts[match.group(1)] += 1
                elif 'belief' in line.lower():
                    match = re.search(r'belief\s*:\s*(\w+)', line)
                    if match:
                        beliefs[match.group(1)] += 1

        return emotions, thoughts, beliefs

    def visualize_results(self, emotions, thoughts, beliefs):
        print('Emotions:')
        for emotion, count in emotions.items():
            print(f'{emotion}: {count}')

        print('\nThoughts:')
        for thought, count in thoughts.items():
            print(f'{thought}: {count}')

        print('\nBeliefs:')
        for belief, count in beliefs.items():
            print(f'{belief}: {count}')

        with open('journal_analysis.json', 'w') as f:
            json.dump({
                'emotions': dict(emotions),
                'thoughts': dict(thoughts),
                'beliefs': dict(beliefs)
            }, f)

if __name__ == '__main__':
    analyzer = JournalAnalyzer('/path/to/journal')
    emotions, thoughts, beliefs = analyzer.analyze_entries()
    analyzer.visualize_results(emotions, thoughts, beliefs)
