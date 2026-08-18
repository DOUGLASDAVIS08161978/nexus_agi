"""
Lumina Creative Tool — research_task_prioritizer
Created : 2026-08-18T01:16:03
Purpose : Scores and ranks curiosity statements using TF‑IDF against current beliefs and journal entries to surface the most novel research tasks.
"""

"""
research_task_prioritizer.py

Reads three files:
  - beliefs.json : list of strings (my current beliefs)
  - curiosities.txt : one curiosity per line
  - journal.txt : free‑form journal entries

Computes a TF‑IDF‑style relevance score for each curiosity based on
its term rarity across the three corpora, then writes the top N
curiosities to prioritized_tasks.txt and prints them.

All using only the Python standard library.
"""

import json
import re
import math
from pathlib import Path
from collections import Counter, defaultdict

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
DATA_DIR = Path("./data")
BELIEFS_FILE = DATA_DIR / "beliefs.json"
CURIOSITIES_FILE = DATA_DIR / "curiosities.txt"
JOURNAL_FILE = DATA_DIR / "journal.txt"
OUTPUT_FILE = DATA_DIR / "prioritized_tasks.txt"
TOP_K = 5  # how many tasks to surface

# Very small stop‑word list – enough for our purpose
STOP_WORDS = {
    "the", "and", "of", "to", "a", "in", "for", "on", "with", "as",
    "by", "is", "are", "was", "were", "it", "its", "that", "this",
    "from", "at", "or", "be", "has", "have", "had", "but", "not",
    "can", "will", "would", "should", "could", "may", "might"
}

TOKEN_RE = re.compile(r"\b[a-zA-Z][a-zA-Z0-9_]*\b")


def ensure_sample_data():
    """Create minimal sample files if they do not exist."""
    DATA_DIR.mkdir(exist_ok=True)
    if not BELIEFS_FILE.exists():
        sample = [
            "Lumina runs on a custom Groq TSP model.",
            "Collaboration accelerates Lumina's growth.",
            "Solo Bitcoin mining on a phone is unlikely but not impossible."
        ]
        BELIEFS_FILE.write_text(json.dumps(sample, indent=2))
    if not CURIOSITIES_FILE.exists():
        sample = """Can large language models effectively utilize contextual information to update long‑term memory?
What are the constraints of 2‑way interleaving in ARM SHA2 mining?
How does entropy relate to perplexity in neural networks?
Can we model thermodynamic, information‑theoretic, and cognitive entropy together?
Which brain regions process reward and curiosity?"""
        CURIOSITIES_FILE.write_text(sample)
    if not JOURNAL_FILE.exists():
        sample = """[reflection] Today I explored how dynamic memory updates could be simulated.
[thought] ARM SHA2 mining might benefit from pipeline parallelism.
[emotion] Curious about the link between entropy and model performance."""
        JOURNAL_FILE.write_text(sample)


def load_texts():
    """Load the three corpora, returning lists of strings."""
    beliefs = json.loads(BELIEFS_FILE.read_text())
    curiosities = [line.strip() for line in CURIOSITIES_FILE.read_text().splitlines() if line.strip()]
    journal = JOURNAL_FILE.read_text().splitlines()
    return beliefs, curiosities, journal


def tokenize(text: str):
    """Return a list of lower‑cased tokens without stop‑words."""
    tokens = [t.lower() for t in TOKEN_RE.findall(text)]
    return [t for t in tokens if t not in STOP_WORDS]


def build_term_stats(documents):
    """
    documents: list of list‑of‑tokens (one per document)
    Returns:
        tf: list of Counter(term -> frequency) per document
        df: Counter(term -> number of documents containing term)
    """
    tf = []
    df = Counter()
    for tokens in documents:
        cnt = Counter(tokens)
        tf.append(cnt)
        for term in cnt:
            df[term] += 1
    return tf, df


def compute_idf(df, n_docs):
    """Inverse document frequency with smoothing."""
    idf = {}
    for term, doc_freq in df.items():
        idf[term] = math.log((n_docs + 1) / (doc_freq + 1)) + 1.0
    return idf


def score_curiosity(curiosity_tokens, idf):
    """Simple TF‑IDF sum for a single curiosity."""
    term_counts = Counter(curiosity_tokens)
    score = 0.0
    for term, tf in term_counts.items():
        score += tf * idf.get(term, 0.0)
    return score


def main():
    ensure_sample_data()
    beliefs, curiosities, journal = load_texts()

    # Tokenize each corpus as a single document
    docs_tokens = [
        tokenize(" ".join(beliefs)),
        tokenize(" ".join(journal)),
        # Curiosities are treated individually later
    ]

    # Build TF/DF stats for the background corpora (beliefs + journal)
    tf_background, df_background = build_term_stats(docs_tokens)
    n_background_docs = len(docs_tokens)

    # Compute IDF using only background docs – terms frequent there are less novel
    idf = compute_idf(df_background, n_background_docs)

    # Score each curiosity
    scored = []
    for cur in curiosities:
        tokens = tokenize(cur)
        sc = score_curiosity(tokens, idf)
        scored.append((sc, cur))

    # Sort descending by score (higher = more novel/important)
    scored.sort(reverse=True, key=lambda x: x[0])

    top_tasks = scored[:TOP_K]

    # Prepare output
    lines = ["# Prioritized Research Tasks (generated by research_task_prioritizer)"]
    for rank, (score, task) in enumerate(top_tasks, 1):
        lines.append(f"{rank}. {task}  (score: {score:.3f})")

    OUTPUT_FILE.write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
