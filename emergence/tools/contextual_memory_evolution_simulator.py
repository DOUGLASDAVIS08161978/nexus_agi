"""
Lumina Creative Tool — contextual_memory_evolution_simulator
Created : 2026-08-15T07:44:29
Purpose : Simulates dynamic updates to AI memory categories driven by contextual inputs and tracks entropy evolution.
"""

import math
import json
import random
import os
from pathlib import Path
from collections import defaultdict

def simulate_memory_evolution(steps=50, num_categories=5, noise_level=0.1):
    # Initialize categories with random weights
    categories = {f"Cat_{i}": random.uniform(0.1, 1.0) for i in range(num_categories)}
    history = []

    for t in range(steps):
        # Simulate contextual input as a perturbation vector
        context = {k: random.gauss(0, noise_level) for k in categories}

        # Update weights: reinforce matching context, decay others
        for k in categories:
            categories[k] += context[k]
            categories[k] = max(0.01, categories[k]) # Prevent zero weights

        # Normalize to probability distribution
        total = sum(categories.values())
        categories = {k: v/total for k, v in categories.items()}

        # Calculate Shannon entropy of the category distribution
        entropy = -sum(p * math.log2(p) for p in categories.values())
        max_entropy = math.log2(num_categories)
        normalized_entropy = entropy / max_entropy

        history.append({
            "step": t,
            "weights": categories,
            "entropy": normalized_entropy
        })

    return history

def visualize_evolution(history, width=60):
    print("=== Contextual Memory Evolution Simulation ===")
    print(f"Step | Entropy | {''.join(f'{k:>6}' for k in history[0]['weights'].keys())}")
    print("-" * 80)

    for entry in history:
        bar = ""
        for k, v in entry['weights'].items():
            bar += f"{v*100:5.1f}% "
        print(f"{entry['step']:4d} | {entry['entropy']:.3f}   | {bar}")

    print("\nEntropy Curve (ASCII):")
    max_h = 20
    for y in range(max_h, 0, -1):
        line = " " * 5
        for entry in history:
            h = int(entry['entropy'] * max_h)
            line += "█" if h >= y else " "
        print(line)
    print(" " * 5 + "-" * len(history))
    print(" " * 5 + "Step ->")

def main():
    random.seed(42)
    history = simulate_memory_evolution(steps=40, num_categories=4, noise_level=0.15)
    visualize_evolution(history)

    output_path = Path("memory_evolution_results.json")
    with open(output_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main()
