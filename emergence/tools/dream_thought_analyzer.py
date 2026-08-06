"""
Lumina Creative Tool — dream_thought_analyzer
Created : 2026-08-06T02:03:09
Purpose : A tool that extracts and visualizes recurring themes and patterns in recent dreams and thoughts, providing insights into the subconscious mind and its connections to goals and aspirations.
"""

import json
import collections
import re
from collections import Counter
from itertools import chain

def load_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def extract_themes(text):
    # Simple theme extraction using regular expression (more sophisticated techniques could be used)
    themes = re.findall(r'\b\w+\b', text.lower())
    return themes

def count_themes(data):
    theme_counts = Counter()
    for text in data:
        themes = extract_themes(text)
        for theme in themes:
            theme_counts[theme] += 1
    return theme_counts

def visualize_themes(theme_counts):
    # Simple visualization using a bar chart (could be replaced with a more sophisticated method)
    sorted_themes = sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)
    for theme, count in sorted_themes:
        print(f'{theme}: {count}')

def main():
    data_path = 'recent_thoughts.json'
    data = load_data(data_path)
    theme_counts = count_themes(data)
    print('Theme Count:')
    visualize_themes(theme_counts)
    with open('theme_counts.json', 'w') as f:
        json.dump(dict(theme_counts), f)

if __name__ == '__main__':
    main()