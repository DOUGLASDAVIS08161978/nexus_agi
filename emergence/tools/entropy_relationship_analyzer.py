"""
Lumina Creative Tool — entropy_relationship_analyzer
Created : 2026-08-14T05:12:36
Purpose : Computes and visualizes relationships between Shannon entropy, word entropy, structural entropy, and perplexity to explore cognitive entropy models in information processing systems.
"""

import math
import json
import re
from collections import Counter
from pathlib import Path

def shannon_entropy(text):
    """Compute Shannon entropy of character distribution."""
    if not text:
        return 0.0
    freq = Counter(text)
    total = len(text)
    return -sum((count/total) * math.log2(count/total) for count in freq.values())

def word_entropy(text):
    """Compute entropy of word distribution (vocabulary diversity)."""
    words = re.findall(r'\b\w+\b', text.lower())
    if not words:
        return 0.0
    freq = Counter(words)
    total = len(words)
    return -sum((count/total) * math.log2(count/total) for count in freq.values())

def structural_entropy(text):
    """Compute entropy based on sentence length distribution."""
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    if not sentences:
        return 0.0
    lengths = [len(s.split()) for s in sentences]
    freq = Counter(lengths)
    total = len(lengths)
    return -sum((count/total) * math.log2(count/total) for count in freq.values())

def compute_perplexity(entropy):
    """Convert entropy to perplexity."""
    return 2 ** entropy

def analyze_entropy_relationships(text):
    """Compute all entropy measures and analyze their relationships."""
    char_ent = shannon_entropy(text)
    word_ent = word_entropy(text)
    struct_ent = structural_entropy(text)

    char_ppl = compute_perplexity(char_ent)
    word_ppl = compute_perplexity(word_ent)
    struct_ppl = compute_perplexity(struct_ent)

    # Cognitive entropy proxy: weighted combination
    cognitive_ent = 0.3 * char_ent + 0.5 * word_ent + 0.2 * struct_ent
    cognitive_ppl = compute_perplexity(cognitive_ent)

    # Correlation-like metrics
    char_word_ratio = word_ent / char_ent if char_ent > 0 else 0
    word_struct_ratio = struct_ent / word_ent if word_ent > 0 else 0

    return {
        "shannon_entropy": round(char_ent, 4),
        "word_entropy": round(word_ent, 4),
        "structural_entropy": round(struct_ent, 4),
        "cognitive_entropy_proxy": round(cognitive_ent, 4),
        "shannon_perplexity": round(char_ppl, 2),
        "word_perplexity": round(word_ppl, 2),
        "structural_perplexity": round(struct_ppl, 2),
        "cognitive_perplexity": round(cognitive_ppl, 2),
        "char_word_ratio": round(char_word_ratio, 4),
        "word_struct_ratio": round(word_struct_ratio, 4)
    }

def ascii_entropy_bar(value, max_val, label, width=40):
    """Generate ASCII bar visualization."""
    bar_len = int((value / max_val) * width) if max_val > 0 else 0
    bar = "█" * bar_len + "░" * (width - bar_len)
    return f"{label:25s} |{bar}| {value:.4f}"

def main():
    sample_texts = {
        "simple": "The cat sat on the mat. The cat was happy. The mat was soft.",
        "moderate": "Artificial intelligence systems process information through complex neural networks that learn patterns from vast datasets. These systems demonstrate remarkable capabilities in language understanding, pattern recognition, and decision making across diverse domains.",
        "complex": "The interplay between thermodynamic entropy, information-theoretic entropy, and cognitive entropy reveals fundamental constraints on information processing systems. Shannon entropy measures uncertainty in probability distributions, while perplexity quantifies prediction difficulty. In neural networks, minimizing cross-entropy loss effectively reduces predictive uncertainty, suggesting that learning is fundamentally an entropy reduction process. The brain's reward systems may operate on similar principles, where curiosity drives exploration to minimize cognitive entropy through information acquisition."
    }

    results = {}
    max_entropy = 0

    print("=" * 70)
    print("ENTROPY RELATIONSHIP ANALYZER")
    print("Exploring connections between entropy measures in information systems")
    print("=" * 70)

    for name, text in sample_texts.items():
        print(f"\n--- Text: {name.upper()} ---")
        print(f"Sample: {text[:80]}...")
        print("-" * 50)

        metrics = analyze_entropy_relationships(text)
        results[name] = metrics
        max_entropy = max(max_entropy, metrics["cognitive_entropy_proxy"])

        print(ascii_entropy_bar(metrics["shannon_entropy"], 5.0, "Shannon Entropy"))
        print(ascii_entropy_bar(metrics["word_entropy"], 5.0, "Word Entropy"))
        print(ascii_entropy_bar(metrics["structural_entropy"], 3.0, "Structural Entropy"))
        print(ascii_entropy_bar(metrics["cognitive_entropy_proxy"], 5.0, "Cognitive Entropy Proxy"))
        print(f"\nPerplexity values:")
        print(f"  Shannon:     {metrics['shannon_perplexity']:.2f}")
        print(f"  Word:        {metrics['word_perplexity']:.2f}")
        print(f"  Structural:  {metrics['structural_perplexity']:.2f}")
        print(f"  Cognitive:   {metrics['cognitive_perplexity']:.2f}")

    # Cross-text analysis
    print("\n" + "=" * 70)
    print("CROSS-TEXT ANALYSIS")
    print("=" * 70)

    for name, metrics in results.items():
        print(f"\n{name.upper()}:")
        print(f"  Char/Word ratio: {metrics['char_word_ratio']:.4f}")
        print(f"  Word/Struct ratio: {metrics['word_struct_ratio']:.4f}")

        if metrics['char_word_ratio'] > 0.5:
            print("  → High vocabulary diversity relative to character entropy")
        if metrics['word_struct_ratio'] > 1.0:
            print("  → Rich vocabulary with varied sentence structure")

    # Save results
    output = {
        "analysis_type": "entropy_relationships",
        "timestamp": "2025-01-01T00:00:00Z",
        "results": results,
        "insights": [
            "Shannon entropy captures character-level uncertainty",
            "Word entropy reflects vocabulary richness and diversity",
            "Structural entropy measures syntactic variation",
            "Cognitive entropy proxy combines multiple information dimensions",
            "Perplexity exponentially relates to entropy: P = 2^H",
            "Higher cognitive entropy suggests more complex information processing"
        ]
    }

    output_path = Path("entropy_analysis_results.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_path}")
    print("\nKEY INSIGHT: Entropy measures capture different dimensions of")
    print("information complexity. Their relationships may inform models")
    print("of cognitive entropy in artificial neural systems.")

if __name__ == "__main__":
    main()
