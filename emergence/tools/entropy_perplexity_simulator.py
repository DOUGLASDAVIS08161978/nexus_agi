"""
Lumina Creative Tool — entropy_perplexity_simulator
Created : 2026-08-16T13:22:02
Purpose : This tool simulates the dynamic interplay between information entropy and perplexity in the context of artificial neural networks, providing insights into the relationship between these two important metrics.
"""

import math
import collections
import random

class PerplexityEntropySimulator:
    def __init__(self):
        self.entropy_values = collections.defaultdict(lambda: 0)
        self.perplexity_values = collections.defaultdict(lambda: 0)
        self.iterations = 1000

    def simulate(self):
        for _ in range(self.iterations):
            # Generate a random probability distribution
            probabilities = [random.random() for _ in range(10)]
            probabilities = [p / sum(probabilities) for p in probabilities]

            # Calculate the entropy of the distribution
            entropy = -sum([p * math.log(p, 2) for p in probabilities])
            self.entropy_values[entropy] += 1

            # Calculate the perplexity of the distribution
            perplexity = 2 ** (entropy)
            self.perplexity_values[perplexity] += 1

        # Print the results
        print("Entropy values:")
        for entropy, count in sorted(self.entropy_values.items()):
            print(f"{entropy:.2f}: {count}")

        print("\nPerplexity values:")
        for perplexity, count in sorted(self.perplexity_values.items()):
            print(f"{perplexity:.2f}: {count}")

    def plot(self):
        # Create a simple ASCII plot
        print("\nEntropy vs. Perplexity:")
        for i in range(10):
            entropy = i / 10
            perplexity = 2 ** (entropy)
            print(f"{entropy:.2f} | {perplexity:.2f}")

simulator = PerplexityEntropySimulator()
simulator.simulate()
simulator.plot()