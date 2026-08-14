"""
Lumina Creative Tool — lumina_tool_fabdad
Created : 2026-08-14T04:12:15
Purpose : 
"""

import math
import json
from pathlib import Path
from collections import Counter

def calculate_entropy(distribution):
    total = sum(distribution.values())
    if total == 0: return 0.0
    probs = [count/total for count in distribution.values()]
    return -sum(p * math.log2(p) for p in probs if p > 0)

def simulate_categorization_learning(steps=50):
    categories = {"concept_A": [], "concept_B": [], "concept_C": []}
    entropy_history = []
    perplexity_history = []
    
    # Simulate incoming contextual data
    context_stream = ["A", "A", "B", "A", "C", "B", "A", "A", "B", "C"] * 5
    
    for i, obs in enumerate(context_stream[:steps]):
        # Simple update rule: assign to category with highest current count, or new if balanced
        counts = Counter(categories.keys()) # placeholder logic
        # Actually, let's just append to the observed category to simulate reinforcement
        if obs in categories:
            categories[obs].append(obs)
        
        dist = {k: len(v) for k, v in categories.items()}
        h = calculate_entropy(dist)
        entropy_history.append(h)
        perplexity_history.append(2**h)
        
    return entropy_history, perplexity_history

# ... output formatting ...