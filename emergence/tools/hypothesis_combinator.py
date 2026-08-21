"""
Lumina Creative Tool — hypothesis_combinator
Created : 2026-08-21T06:34:09
Purpose : Generates interdisciplinary research hypotheses by mixing key phrases from tagged curiosity statements and saves them as JSON.
"""

"""
hypothesis_combinator.py

Generate interdisciplinary research hypotheses by mixing tagged curiosity statements.
Outputs a JSON file with generated ideas and prints a short summary.
"""

import json
import itertools
import re
import textwrap
from collections import defaultdict
from random import shuffle, seed
from datetime import datetime

# ----------------------------------------------------------------------
# Data: Curiosity statements (normally could be loaded from a file)
# Each line starts with a tag in square brackets.
STATEMENTS = [
    "[agi] Can large language models effectively utilize contextual information to update and refine their categorization models, thereby simulating the dynamic updating of long-term memories?",
    "[bitcoin] What are the specific constraints and limitations of 2-way interleaving in ARM SHA2 mining that might be overcome by new optimizations?",
    "[bitcoin] Can you provide examples of ARM SHA2 mining optimizations for specific use cases, such as low-power devices or high-performance servers?",
    "[consciousness] How does the relationship between entropy and perplexity specifically apply to the context of artificial neural networks, and can we leverage this connection to improve their performance?",
    "[consciousness] Can we develop a more refined mathematical model that captures the interplay between thermodynamic entropy, information‑theoretic entropy, and cognitive entropy in the context of intelligent systems?",
    "[consciousness] What are the specific brain regions identified in the general neuroscience literature as being involved in the processing of reward and curiosity?",
]

# ----------------------------------------------------------------------
def parse_statements(lines):
    """
    Returns a dict mapping tag -> list of raw statements (without the tag).
    """
    tag_map = defaultdict(list)
    tag_pattern = re.compile(r"^\[(?P<tag>[^\]]+)]\s*(?P<stmt>.+)$")
    for line in lines:
        m = tag_pattern.match(line.strip())
        if not m:
            continue
        tag = m.group("tag").strip().lower()
        stmt = m.group("stmt").strip()
        tag_map[tag].append(stmt)
    return tag_map

def extract_key_phrases(statement):
    """
    Very simple key‑phrase extractor:
    - Split on punctuation.
    - Keep phrases with at least two words.
    - Remove stop‑words (a small built‑in list).
    Returns a set of lower‑cased phrases.
    """
    stop = {"the", "and", "of", "to", "in", "for", "a", "an", "as", "by", "with", "on", "or", "that"}
    # Replace commas, semicolons, question marks with a period to split uniformly
    cleaned = re.sub(r"[?,;]", ".", statement)
    phrases = set()
    for part in cleaned.split("."):
        words = [w.lower() for w in re.findall(r"\b\w+\b", part) if w.lower() not in stop]
        if len(words) >= 2:
            phrases.add(" ".join(words))
    return phrases

def combine_phrases(phrases_a, phrases_b, max_combinations=10):
    """
    Randomly pair phrases from two sets and build a hypothesis sentence.
    Limits the number of generated combos to `max_combinations`.
    """
    combos = []
    # Shuffle to get varied output each run
    a = list(phrases_a)
    b = list(phrases_b)
    shuffle(a)
    shuffle(b)
    for pa, pb in itertools.islice(itertools.product(a, b), max_combinations):
        # Simple template: "Investigate how {pa} influences {pb}."
        sentence = f"Investigate how {pa} influences {pb}."
        combos.append(sentence)
    return combos

def generate_hypotheses(tag_map, max_per_pair=8):
    """
    For every unordered pair of distinct tags, combine their key phrases.
    Returns a list of hypothesis dicts:
        {"tags": [tag1, tag2], "hypothesis": "..."}
    """
    hypotheses = []
    tags = sorted(tag_map.keys())
    for tag_a, tag_b in itertools.combinations(tags, 2):
        # Gather key phrases from all statements of each tag
        phrases_a = set()
        for stmt in tag_map[tag_a]:
            phrases_a.update(extract_key_phrases(stmt))
        phrases_b = set()
        for stmt in tag_map[tag_b]:
            phrases_b.update(extract_key_phrases(stmt))

        if not phrases_a or not phrases_b:
            continue

        combos = combine_phrases(phrases_a, phrases_b, max_per_pair)
        for h in combos:
            hypotheses.append({
                "tags": [tag_a, tag_b],
                "hypothesis": h
            })
    return hypotheses

def save_json(data, filename):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def print_summary(hypotheses, limit=5):
    print("\nGenerated Hypotheses Summary")
    print("-" * 30)
    for i, hyp in enumerate(hypotheses[:limit], 1):
        tags = ", ".join(hyp["tags"])
        print(f"{i}. [{tags}] {hyp['hypothesis']}")
    if len(hypotheses) > limit:
        print(f"... (+{len(hypotheses)-limit} more)")

def main():
    seed(42)  # deterministic for reproducibility
    tag_map = parse_statements(STATEMENTS)
    hypotheses = generate_hypotheses(tag_map, max_per_pair=12)

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_file = f"generated_hypotheses_{timestamp}.json"
    save_json(hypotheses, out_file)

    print_summary(hypotheses, limit=7)
    print(f"\nAll {len(hypotheses)} hypotheses saved to '{out_file}'")

if __name__ == "__main__":
    main()
