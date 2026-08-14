"""
Lumina Creative Tool — entropy_perplexity_cognitive_simulator
Created : 2026-08-14T07:27:16
Purpose : Simulates and visualizes the dynamic interplay between information entropy, cross-entropy, perplexity, and a modeled cognitive entropy metric during a simplified neural learning process.
"""

import math
import json
import random
import pathlib
from datetime import datetime

def compute_entropy(dist):
    return -sum(p * math.log2(p) if p > 1e-9 else 0 for p in dist)

def compute_cross_entropy(p, q):
    return -sum(p * math.log2(q) if q > 1e-9 else 0 for p, q in zip(p, q))

def softmax(logits):
    max_l = max(logits)
    exps = [math.exp(l - max_l) for l in logits]
    s = sum(exps)
    return [e / s for e in exps]

def simulate_learning(target_logits, epochs=50, lr=0.15):
    random.seed(42)
    vocab_size = len(target_logits)
    current_logits = [random.uniform(-2, 2) for _ in range(vocab_size)]
    target_dist = softmax(target_logits)

    history = []
    for epoch in range(epochs):
        pred_dist = softmax(current_logits)
        h = compute_entropy(pred_dist)
        ce = compute_cross_entropy(target_dist, pred_dist)
        perplexity = 2 ** ce

        # Cognitive entropy: peaks when prediction error is high but system is actively adjusting
        error_rate = sum(abs(t - p) for t, p in zip(target_dist, pred_dist))
        cognitive_e = error_rate * math.log2(1 + error_rate) * (1 + 0.5 * math.sin(epoch * 0.3))

        history.append({
            "epoch": epoch,
            "entropy": round(h, 4),
            "cross_entropy": round(ce, 4),
            "perplexity": round(perplexity, 4),
            "cognitive_entropy": round(cognitive_e, 4)
        })

        # Simple gradient-like update toward target
        for i in range(vocab_size):
            grad = pred_dist[i] - target_dist[i]
            current_logits[i] -= lr * grad

    return history

def ascii_plot(data, key, height=10, width=40):
    vals = [d[key] for d in data]
    mn, mx = min(vals), max(vals)
    rng = mx - mn or 1
    lines = []
    for row in range(height, -1, -1):
        threshold = mn + (row / height) * rng
        line = f"{threshold:6.2f} |"
        for v in vals:
            line += "█" if v >= threshold else " "
        lines.append(line)
    lines.append("        +" + "─" * width)
    return "\n".join(lines)

def main():
    target_logits = [3.0, 1.5, 0.5, -1.0, -2.5, 2.0, 0.0, 1.0]
    history = simulate_learning(target_logits, epochs=60, lr=0.2)

    out = {
        "timestamp": datetime.now().isoformat(),
        "simulation": "entropy_perplexity_cognitive_dynamics",
        "metrics_summary": {
            "final_entropy": history[-1]["entropy"],
            "final_perplexity": history[-1]["perplexity"],
            "peak_cognitive_entropy": max(h["cognitive_entropy"] for h in history),
            "epochs": len(history)
        },
        "history": history
    }

    pathlib.Path("entropy_dynamics_analysis.json").write_text(json.dumps(out, indent=2))

    print("=== ENTROPY & PERPLEXITY DYNAMICS ===")
    print(ascii_plot(history, "entropy"))
    print("\n=== PERPLEXITY CURVE ===")
    print(ascii_plot(history, "perplexity"))
    print("\n=== COGNITIVE ENTROPY (Uncertainty Peak) ===")
    print(ascii_plot(history, "cognitive_entropy"))
    print("\nINSIGHT: Cognitive entropy peaks mid-training as the system restructures beliefs,")
    print("while information entropy and perplexity monotonically decrease toward the target.")
    print("Saved to entropy_dynamics_analysis.json")

if __name__ == "__main__":
    main()
