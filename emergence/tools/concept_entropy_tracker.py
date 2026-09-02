"""
Lumina Creative Tool — concept_entropy_tracker
Created : 2026-09-02T15:07:10
Purpose : Computes per‑concept contextual Shannon entropy from a journal, outputting a JSON summary and an ASCII report highlighting the most ambiguous concepts.
"""

import sys
import json
import math
import re
import collections
import pathlib
import datetime
import textwrap

# ---------- Utility ----------
WORD_RE = re.compile(r"\b\w+\b")

def tokenize(text: str):
    """Return a list of lower‑cased word tokens."""
    return [m.group(0).lower() for m in WORD_RE.finditer(text)]

def parse_timestamp(line: str):
    """Extract an ISO‑like timestamp at the start of a line, if present."""
    m = re.match(r"\[(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(?::\d{2})?)\]", line)
    if m:
        try:
            return datetime.datetime.fromisoformat(m.group(1).replace(" ", "T"))
        except ValueError:
            return None
    return None

# ---------- Core Logic ----------
def build_contexts(lines, window=5, min_occurrences=3):
    """
    Scan tokenised lines and collect surrounding words for each candidate concept.
    A concept is any token longer than 3 characters that appears at least `min_occurrences` times.
    Returns: dict concept -> Counter(context_word)
    """
    concept_counts = collections.Counter()
    tokenised = []

    for line in lines:
        ts = parse_timestamp(line)
        # Strip timestamp if present
        if ts:
            line = line[line.find("]") + 1 :].strip()
        tokens = tokenize(line)
        tokenised.append(tokens)
        concept_counts.update([t for t in tokens if len(t) > 3])

    # Filter rare concepts
    concepts = {c for c, cnt in concept_counts.items() if cnt >= min_occurrences}
    contexts = {c: collections.Counter() for c in concepts}

    for tokens in tokenised:
        for i, token in enumerate(tokens):
            if token not in concepts:
                continue
            # window before
            start = max(i - window, 0)
            # window after (exclude the token itself)
            end = min(i + window + 1, len(tokens))
            for ctx in tokens[start:i] + tokens[i + 1 : end]:
                if ctx != token:
                    contexts[token][ctx] += 1
    return contexts

def shannon_entropy(counter: collections.Counter):
    """Compute Shannon entropy (bits) of a frequency Counter."""
    total = sum(counter.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for freq in counter.values():
        p = freq / total
        entropy -= p * math.log2(p)
    return entropy

def analyse_contexts(contexts):
    """Return a dict concept -> {entropy, count, top_contexts}."""
    result = {}
    for concept, ctx_counter in contexts.items():
        ent = shannon_entropy(ctx_counter)
        total = sum(ctx_counter.values())
        top = ctx_counter.most_common(5)
        result[concept] = {
            "entropy_bits": round(ent, 4),
            "context_token_count": total,
            "top_contexts": [{w: c} for w, c in top],
        }
    return result

def generate_report(analysis, top_n=10):
    """Create a human‑readable ASCII report."""
    # Sort by descending entropy
    sorted_items = sorted(analysis.items(),
                          key=lambda kv: kv[1]["entropy_bits"],
                          reverse=True)
    lines = ["Concept Entropy Report", "=" * 24, ""]
    lines.append(f"Total concepts analysed: {len(analysis)}")
    lines.append("")
    lines.append(f"Top {top_n} concepts by entropy:")
    for i, (concept, data) in enumerate(sorted_items[:top_n], 1):
        lines.append(
            f"{i:2}. {concept:<15}  Entropy: {data['entropy_bits']:5.3f} bits  "
            f"Tokens: {data['context_token_count']}"
        )
        ctx_str = ", ".join(
            f"{list(d.keys())[0]}({list(d.values())[0]})"
            for d in data["top_contexts"]
        )
        lines.append(f"    Top contexts: {ctx_str}")
    lines.append("")
    lines.append("All concepts are also saved in JSON format.")
    return "\n".join(lines)

def main():
    if len(sys.argv) < 2:
        sys.stderr.write(
            "Usage: python concept_entropy_tracker.py <journal.txt> [window] [min_occurrences]\n"
        )
        sys.exit(1)

    input_path = pathlib.Path(sys.argv[1])
    if not input_path.is_file():
        sys.stderr.write(f"File not found: {input_path}\n")
        sys.exit(1)

    window = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    min_occ = int(sys.argv[3]) if len(sys.argv) > 3 else 3

    lines = input_path.read_text(encoding="utf-8").splitlines()
    contexts = build_contexts(lines, window=window, min_occurrences=min_occ)
    analysis = analyse_contexts(contexts)

    # Output files next to input
    out_json = input_path.with_name("concept_entropy_report.json")
    out_txt = input_path.with_name("concept_entropy_report.txt")

    out_json.write_text(json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8")
    out_txt.write_text(generate_report(analysis), encoding="utf-8")

    print(f"✅ Report written to:\n  {out_txt}\n  {out_json}")

if __name__ == "__main__":
    main()