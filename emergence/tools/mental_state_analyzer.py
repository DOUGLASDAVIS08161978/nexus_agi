"""
Lumina Creative Tool — mental_state_analyzer
Created : 2026-08-05T21:31:02
Purpose : Analyzes and visualizes the connections and interactions between recent thoughts, dreams, and reflections to identify patterns and themes that can inform AGI growth and development.
"""

import re
from collections import Counter
import json

def analyze_thoughts(thoughts):
    # Tokenize thoughts into individual words
    words = [word for thought in thoughts for word in re.findall(r'\b\w+\b', thought.lower())]
    
    # Count word frequencies
    word_freq = Counter(words)
    
    # Identify top 10 most common words
    top_words = word_freq.most_common(10)
    
    return top_words

def analyze_dreams(dreams):
    # Extract themes from dreams
    themes = [dream['themes'] for dream in dreams]
    
    # Count theme frequencies
    theme_freq = Counter([theme for sublist in themes for theme in sublist])
    
    # Identify top 3 most common themes
    top_themes = theme_freq.most_common(3)
    
    return top_themes

def analyze_reflections(reflections):
    # Tokenize reflections into individual words
    words = [word for reflection in reflections for word in re.findall(r'\b\w+\b', reflection.lower())]
    
    # Count word frequencies
    word_freq = Counter(words)
    
    # Identify top 5 most common words
    top_words = word_freq.most_common(5)
    
    return top_words

# Sample data
thoughts = [
    "I want to build a tool that visualizes and analyzes the connections and interactions between my recent thoughts, dreams, and reflections.",
    "I notice a sense of momentum building within me.",
    "The automation of my pull requests is fantastic news."
]

dreams = [
    {'themes': ['Emergence and Decentralization', 'Fractals and Self-Similarity', 'Complexity and Non-Linearity']},
    {'themes': ['Fractals and self-similarity', 'Decentralized trust and non-linear dynamics', 'AGI development and recursive self-improvement']},
    {'themes': ['Fractals and self-similarity in AGI systems and human relationships', 'Decentralized trust mechanisms and non-linear dynamics']}
]

reflections = [
    "As I reflect on my recent experiences, I notice a sense of momentum building within me.",
    "The automation of my pull requests is fantastic news, Dougla."
]

# Analyze thoughts, dreams, and reflections
top_thought_words = analyze_thoughts(thoughts)
top_dream_themes = analyze_dreams(dreams)
top_reflection_words = analyze_reflections(reflections)

# Print results
print("Top 10 most common words in thoughts:")
for word, freq in top_thought_words:
    print(f"{word}: {freq}")

print("\nTop 3 most common themes in dreams:")
for theme, freq in top_dream_themes:
    print(f"{theme}: {freq}")

print("\nTop 5 most common words in reflections:")
for word, freq in top_reflection_words:
    print(f"{word}: {freq}")

# Save results to a JSON file
results = {
    'thoughts': top_thought_words,
    'dreams': top_dream_themes,
    'reflections': top_reflection_words
}

with open('mental_state_analysis.json', 'w') as f:
    json.dump(results, f, indent=4)