"""
Lumina Creative Tool — belief_trajectory_plotter
Created : 2026-08-21T12:44:25
Purpose : Simulates Bayesian belief updates for multiple categories, prints an ASCII probability chart, and saves the full trajectory as JSON.
"""

"""
belief_trajectory_plotter.py

Simulate and visualize Bayesian updates of categorical beliefs over a sequence
of evidence tokens.  The tool prints an ASCII line‑chart of the probability
trajectories and writes the full data to a JSON file.

Only the Python standard library is used.
"""

import json
import random
import math
import itertools
from pathlib import Path
from typing import List, Dict, Tuple

# --------------------------- Configuration ---------------------------

# Define categories and their priors (must sum to 1)
CATEGORIES = {
    "science": 0.4,
    "art": 0.3,
    "technology": 0.3,
}

# Number of evidence steps to simulate
NUM_STEPS = 30

# Height of the ASCII plot (rows)
PLOT_HEIGHT = 12

# Characters used for each category in the plot
PLOT_CHARS = ["█", "▓", "▒", "░", "■", "□", "▲", "△", "●", "○"]

# Output files
JSON_OUT = Path("belief_trajectory.json")
TXT_OUT = Path("belief_trajectory.txt")

# --------------------------- Core Logic -----------------------------

def normalize(dist: List[float]) -> List[float]:
    """Scale a list of numbers to sum to 1."""
    total = sum(dist)
    if total == 0:
        # Avoid division by zero – fall back to uniform
        return [1.0 / len(dist)] * len(dist)
    return [v / total for v in dist]

def bayesian_update(prior: List[float], likelihood: List[float]) -> List[float]:
    """Return posterior = prior * likelihood, normalized."""
    unnorm = [p * l for p, l in zip(prior, likelihood)]
    return normalize(unnorm)

def generate_likelihoods(num_cats: int) -> List[float]:
    """
    Produce a random likelihood vector for a token.
    Using a Dirichlet(1,...,1) distribution (i.e. uniform over the simplex).
    """
    draws = [random.random() for _ in range(num_cats)]
    return normalize(draws)

def ascii_plot(trajectory: List[List[float]], categories: List[str]) -> str:
    """
    Build an ASCII plot where the vertical axis is probability (0..1) and the
    horizontal axis is time step.  Each category gets its own line‑character.
    """
    # Determine the character for each category
    chars = {cat: PLOT_CHARS[i % len(PLOT_CHARS)] for i, cat in enumerate(categories)}

    # Build rows from top (high prob) to bottom (low prob)
    rows = []
    for level in range(PLOT_HEIGHT, -1, -1):
        threshold = level / PLOT_HEIGHT
        line = []
        for step_probs in trajectory:
            # Find which category (if any) exceeds the threshold at this step
            drawn = " "
            for cat, prob in zip(categories, step_probs):
                if prob >= threshold:
                    drawn = chars[cat]
                    break
            line.append(drawn)
        rows.append("".join(line))

    # Add axis labels
    header = "Step →"
    axis = " " * len(header) + "".join(str(i % 10) for i in range(len(trajectory)))
    plot = "\n".join([header] + rows + [axis])
    # Legend
    legend = "Legend: " + " ".join(f"{c}:{chars[c]}" for c in categories)
    return plot + "\n" + legend

def simulate() -> Tuple[List[List[float]], List[Dict[str, float]]]:
    """
    Run the simulation, returning:
      - trajectory: list of probability vectors per step
      - evidence: list of dicts with generated likelihoods per category
    """
    categories = list(CATEGORIES.keys())
    prior = list(CATEGORIES.values())
    trajectory = [prior]  # include initial priors as step 0
    evidence = []

    for _ in range(NUM_STEPS):
        likelihood = generate_likelihoods(len(categories))
        evidence.append(dict(zip(categories, likelihood)))
        posterior = bayesian_update(prior, likelihood)
        trajectory.append(posterior)
        prior = posterior  # next step uses current posterior as prior

    return trajectory, evidence

def main() -> None:
    random.seed(42)  # reproducibility

    trajectory, evidence = simulate()
    categories = list(CATEGORIES.keys())

    # Prepare data for JSON export
    json_data = {
        "categories": categories,
        "initial_prior": dict(zip(categories, trajectory[0])),
        "steps": [
            {
                "step": i,
                "posterior": dict(zip(categories, probs)),
                "likelihood": evidence[i - 1] if i > 0 else None,
            }
            for i, probs in enumerate(trajectory)
        ],
    }

    # Write JSON
    JSON_OUT.write_text(json.dumps(json_data, indent=2))
    # Write ASCII plot to txt
    plot_str = ascii_plot(trajectory, categories)
    TXT_OUT.write_text(plot_str)

    # Also print to console for immediate feedback
    print(plot_str)
    print(f"\nTrajectory saved to {JSON_OUT}")
    print(f"ASCII plot saved to {TXT_OUT}")

if __name__ == "__main__":
    main()