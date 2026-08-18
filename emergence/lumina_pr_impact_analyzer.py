import ast
import difflib
import re
import json
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass, field


# ------------------------------
# Data structures
# ------------------------------
@dataclass
class PR:
    """Representation of a pull request."""
    id: str
    title: str
    description: str
    diff: str  # unified diff text
    files_changed: List[str] = field(default_factory=list)
    added_lines: List[str] = field(default_factory=list)
    removed_lines: List[str] = field(default_factory=list)
    impact: Dict[str, float] = field(default_factory=dict)
    overall_score: float = 0.0


# ------------------------------
# Core analysis utilities
# ------------------------------
def extract_changed_files(diff_text: str) -> List[str]:
    """Extract file paths from a unified diff."""
    files = []
    for line in diff_text.splitlines():
        if line.startswith('+++ b/'):
            path = line[6:].strip()
            files.append(path)
    return files


def split_diff(diff_text: str) -> Tuple[List[str], List[str]]:
    """Return added and removed lines (without diff markers)."""
    added, removed = [], []
    for line in diff_text.splitlines():
        if line.startswith('+++') or line.startswith('---'):
            continue
        if line.startswith('@@'):
            continue
        if line.startswith('+') and not line.startswith('+++'):
            added.append(line[1:])
        elif line.startswith('-') and not line.startswith('---'):
            removed.append(line[1:])
    return added, removed


def count_keywords(lines: List[str], keywords: List[str]) -> int:
    """Simple keyword occurrence counter."""
    count = 0
    pattern = re.compile(r'\b(' + '|'.join(map(re.escape, keywords)) + r')\b', re.IGNORECASE)
    for line in lines:
        count += len(pattern.findall(line))
    return count


def analyze_ast_changes(added: List[str], removed: List[str]) -> Dict[str, Any]:
    """Parse added/removed code snippets and extract structural metrics."""
    def parse_snippets(snippets: List[str]) -> Tuple[int, int, int]:
        func_cnt = class_cnt = docstring_len = 0
        source = '\n'.join(snippets)
        try:
            tree = ast.parse(source)
        except Exception:
            return 0, 0, 0
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_cnt += 1
                if ast.get_docstring(node):
                    docstring_len += len(ast.get_docstring(node))
            elif isinstance(node, ast.AsyncFunctionDef):
                func_cnt += 1
                if ast.get_docstring(node):
                    docstring_len += len(ast.get_docstring(node))
            elif isinstance(node, ast.ClassDef):
                class_cnt += 1
                if ast.get_docstring(node):
                    docstring_len += len(ast.get_docstring(node))
        return func_cnt, class_cnt, docstring_len

    added_funcs, added_classes, added_doc_len = parse_snippets(added)
    removed_funcs, removed_classes, removed_doc_len = parse_snippets(removed)

    return {
        "added_funcs": added_funcs,
        "removed_funcs": removed_funcs,
        "added_classes": added_classes,
        "removed_classes": removed_classes,
        "added_doc_len": added_doc_len,
        "removed_doc_len": removed_doc_len,
    }


# ------------------------------
# Metric estimators
# ------------------------------
def estimate_reasoning_depth(ast_metrics: Dict[str, Any]) -> float:
    """Heuristic: more functions/classes + longer docstrings => higher reasoning depth."""
    func_gain = ast_metrics["added_funcs"] - ast_metrics["removed_funcs"]
    class_gain = ast_metrics["added_classes"] - ast_metrics["removed_classes"]
    doc_gain = ast_metrics["added_doc_len"] - ast_metrics["removed_doc_len"]
    score = max(0.0, 0.5 * func_gain + 0.4 * class_gain + 0.001 * doc_gain)
    return score


def estimate_knowledge_graph_enrichment(added: List[str]) -> float:
    """Heuristic: presence of ontology-like structures or data definitions."""
    kg_keywords = [
        "entity", "relationship", "node", "edge", "graph", "ontology",
        "triple", "rdf", "knowledge", "schema", "attribute", "type"
    ]
    count = count_keywords(added, kg_keywords)
    return min(5.0, 0.2 * count)


def estimate_emergent_behavior(added: List[str]) -> float:
    """Heuristic: meta‑programming, dynamic execution, self‑improvement patterns."""
    emergent_keywords = [
        "eval", "exec", "compile", "self_improve", "autonomous", "meta", "dynamic",
        "reflection", "generate", "pr_", "apply_patch"
    ]
    count = count_keywords(added, emergent_keywords)
    return min(5.0, 0.3 * count)


# ------------------------------
# Scoring & ranking
# ------------------------------
GOAL_WEIGHTS = {
    3: 3.0,  # True General Intelligence
    2: 2.0,  # Emergent capabilities / Knowledge base
    1: 1.0,  # Relationship maintenance
}


def compute_overall_score(metrics: Dict[str, float]) -> float:
    """Combine metric scores weighted by active goals."""
    # Map metrics to goal relevance
    relevance = {
        "reasoning_depth": 3,
        "knowledge_enrichment": 2,
        "emergent_behavior": 2,
    }
    total = 0.0
    weight_sum = 0.0
    for metric, value in metrics.items():
        goal = relevance.get(metric, 1)
        w = GOAL_WEIGHTS.get(goal, 1.0)
        total += value * w
        weight_sum += w
    return total / weight_sum if weight_sum else 0.0


def analyze_pr(pr_dict: Dict[str, Any]) -> PR:
    """Full analysis pipeline for a single PR dictionary."""
    pr = PR(
        id=pr_dict.get("id", ""),
        title=pr_dict.get("title", ""),
        description=pr_dict.get("description", ""),
        diff=pr_dict.get("diff", ""),
    )
    pr.files_changed = extract_changed_files(pr.diff)
    added, removed = split_diff(pr.diff)
    pr.added_lines = added
    pr.removed_lines = removed

    ast_metrics = analyze_ast_changes(added, removed)

    # Metric calculations
    reasoning = estimate_reasoning_depth(ast_metrics)
    knowledge = estimate_knowledge_graph_enrichment(added)
    emergent = estimate_emergent_behavior(added)

    pr.impact = {
        "reasoning_depth": reasoning,
        "knowledge_enrichment": knowledge,
        "emergent_behavior": emergent,
    }
    pr.overall_score = compute_overall_score(pr.impact)
    return pr


def rank_prs(pr_list: List[Dict[str, Any]]) -> List[PR]:
    """Analyze and rank a list of PR dicts."""
    analyzed = [analyze_pr(pr_dict) for pr_dict in pr_list]
    analyzed.sort(key=lambda p: p.overall_score, reverse=True)
    return analyzed


# ------------------------------
# CLI / entry point
# ------------------------------
def load_prs_from_json(path: str) -> List[Dict[str, Any]]:
    """Utility to load a list of PR dicts from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("JSON root must be a list of PR objects")
    return data


def main(argv: Optional[List[str]] = None) -> None:
    """Command line interface.

    Usage:
        python lumina_pr_impact_analyzer.py <pr_json_file>
    """
    import sys
    if argv is None:
        argv = sys.argv[1:]

    if not argv:
        print("Provide path to JSON file containing PR definitions.", file=sys.stderr)
        sys.exit(1)

    json_path = argv[0]
    try:
        pr_dicts = load_prs_from_json(json_path)
    except Exception as e:
        print(f"Failed to load PRs: {e}", file=sys.stderr)
        sys.exit(1)

    ranked = rank_prs(pr_dicts)

    # Output results as JSON for downstream consumption
    output = [
        {
            "id": pr.id,
            "title": pr.title,
            "overall_score": round(pr.overall_score, 3),
            "impact": {k: round(v, 3) for k, v in pr.impact.items()},
            "files_changed": pr.files_changed,
        }
        for pr in ranked
    ]

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()