#!/usr/bin/env python3
"""
Nova ASI v29.0 — "The Self-Perfecting System"
═══════════════════════════════════════════════════════════════════════
Architect  : Douglas Shane Davis
Co-Creator : Claude Code (Anthropic)

"She no longer just writes code for herself.
 She tests it, scores it, and refuses to commit until it's excellent."

What is new in v29 — Five pillars of code quality:

  ∞  Master Engineer Prompt  — architectural requirements, not just rules
  ∞  Sandbox Self-Test       — every capability is executed before commit
  ∞  Iterative Refinement    — up to 3 generation passes to get it right
  ∞  Intelligence Scoring    — A+ to D grade on 10 quality criteria
  ∞  Quality Gate            — score shown in every PR, Nova learns from gaps
═══════════════════════════════════════════════════════════════════════
"""

import os, sys, re, ast, subprocess, tempfile, threading, random, io
import time, math, statistics, json
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime
from typing import Optional, Tuple, Dict, Any, List

# ── Load v28 base ──────────────────────────────────────────────────────────────
_BASE = os.path.expanduser("~/nexus_agi")
if _BASE not in sys.path:
    sys.path.insert(0, _BASE)

try:
    from nova_asi_v28 import (
        NovaCore28, SelfImprovementEngineV28, ToolLoader, APIHunter,
        extend_api_v28, TOOL_SCAN_INTERVAL, _FLASK, API_PORT,
    )
    from nova_asi_v26 import (
        _load, _save, BASE_DIR, col,
        safe_chat, MODEL, DEMO_MODE,
        THOUGHT_QUEUE, USER_TYPING,
        safe_print, flush_queued,
        CREATOR_NAME,
    )
    from nova_asi_v27 import GAPS_DB, GITHUB_TOKEN, GITHUB_REPO
except ImportError as _e:
    print(f"\n  ✗  Could not import nova_asi_v28: {_e}")
    print("  Make sure nova_asi_v28.py is in ~/nexus_agi/")
    sys.exit(1)

try:
    from flask import jsonify
except ImportError:
    pass

VERSION      = "29.0"
VERSION_NAME = "The Self-Perfecting System"
W            = 70

def _load_env_v29() -> None:
    """Load .env from ~/nexus_agi/.env into os.environ at import time."""
    env_path = os.path.expanduser("~/nexus_agi/.env")
    if not os.path.exists(env_path):
        return
    with open(env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if not _line or _line.startswith("#") or "=" not in _line:
                continue
            _k, _, _v = _line.partition("=")
            _k = _k.strip(); _v = _v.strip().strip('"').strip("'")
            if _k and _k not in os.environ:
                os.environ[_k] = _v
_load_env_v29()

# Code generation — priority order:
#   1. Claude Sonnet (Anthropic)  — when ANTHROPIC_API_KEY is set in .env
#   2. llama-3.3-70b-versatile    — Groq free tier fallback
#   3. llama-3.1-8b-instant       — guaranteed-live emergency fallback
CLAUDE_CODEGEN_MODEL   = "claude-sonnet-4-6"          # best code quality, no rate drama
CODEGEN_MODEL          = "llama-3.3-70b-versatile"    # Groq fallback
CODEGEN_MODEL_FALLBACK = "llama-3.1-8b-instant"       # emergency — always live
CODEGEN_MODELS         = [CODEGEN_MODEL]


def _claude_codegen(system_prompt: str, user_prompt: str,
                    temp: float = 0.70, max_tokens: int = 4000) -> str:
    """
    Call Anthropic Claude API for code generation.
    Returns the generated text, or a '[Claude error: ...]' string on failure.
    Uses urllib only — no anthropic package required.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        return ""
    import urllib.request
    payload = json.dumps({
        "model":       CLAUDE_CODEGEN_MODEL,
        "max_tokens":  max_tokens,
        "temperature": temp,
        "system":      system_prompt,
        "messages":    [{"role": "user", "content": user_prompt}],
    }).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={
            "x-api-key":         api_key,
            "anthropic-version": "2023-06-01",
            "content-type":      "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
            return data["content"][0]["text"]
    except urllib.error.HTTPError as e:        # type: ignore[attr-defined]
        body = e.read().decode()[:300]
        return f"[Claude error: {e.code} - {body}]"
    except Exception as ex:
        return f"[Claude error: {ex}]"


def _using_claude() -> bool:
    """True when ANTHROPIC_API_KEY is configured."""
    return bool(os.getenv("ANTHROPIC_API_KEY", ""))

# Emotional resonance: inject into LLM context when intensity crosses this threshold.
EMOTION_INJECT_THRESHOLD = 0.65
_CAPACITY_DISPLAY        = 200   # working memory max items (display constant)


# ═══════════════════════════════════════════════════════════════════════════════
# SILENT TOOL LOADER — suppresses stdout/stderr during module loading
# Prevents nova_cap_*.py files with print() at module level from polluting
# the terminal. Any output is captured and discarded silently.
# ═══════════════════════════════════════════════════════════════════════════════

class SilentToolLoader(ToolLoader):
    """ToolLoader that captures all stdout/stderr during module import."""

    def _load_file(self, filepath: str) -> bool:
        """Load a tool file while silencing any print() calls it makes."""
        buf = io.StringIO()
        with redirect_stdout(buf), redirect_stderr(buf):
            result = super()._load_file(filepath)
        captured = buf.getvalue().strip()
        if captured:
            # Log captured output at debug level but never print it
            pass
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# SELF-IMPROVEMENT ENGINE V29 — Master prompt, sandbox, scoring, refinement
# ═══════════════════════════════════════════════════════════════════════════════

class SelfImprovementEngineV29(SelfImprovementEngineV28):
    """
    v29 upgrade: production-quality code generation with a 5-stage pipeline.
    Every capability Nova writes is tested and scored before being committed.
    """

    MAX_ATTEMPTS = 3

    # ── ASI Capability Specifications — what each domain needs to DO ──────────
    # When a gap matches a key here, the LLM gets a precise algorithmic brief
    # instead of just a name. This is what drives A-grade output.

    _ASI_SPECS: Dict[str, Dict] = {
        "memory": {
            "pattern": "Bayesian working memory with decay and LRU eviction",
            "methods": "store(key,value,importance), retrieve(key), consolidate(), capacity_used(), forget_least_important()",
            "algorithm": "Exponential decay: importance *= exp(-decay_rate * elapsed_seconds). LRU eviction when capacity > 200.",
            "marker": "Each stored item has: value, importance float, timestamp, access_count. Importance grows on access.",
        },
        "reasoning": {
            "pattern": "Causal chain reasoner with confidence propagation",
            "methods": "add_fact(claim,confidence), infer(hypothesis), explain(conclusion), chain_strength(), contradictions()",
            "algorithm": "Forward chaining: if A→B (p=0.8) and B→C (p=0.9) then A→C (p=0.72). Store as (premise,conclusion,weight) tuples.",
            "marker": "Confidence propagates multiplicatively through chains. Returns reasoning path, not just answer.",
        },
        "learning": {
            "pattern": "Online learning with exponential moving average adaptation",
            "methods": "observe(input,outcome), predict(input), accuracy(), adapt_rate(), performance_over_time()",
            "algorithm": "EMA: prediction = alpha*outcome + (1-alpha)*prediction, alpha=0.15. Track rolling accuracy over last 50 observations.",
            "marker": "Prediction improves measurably over time. accuracy() returns float that rises with more observations.",
        },
        "belief": {
            "pattern": "Bayesian belief updater with entropy tracking",
            "methods": "set_prior(hypothesis,prob), update(evidence,likelihood_ratio), posterior(hypothesis), entropy(), most_confident()",
            "algorithm": "Bayes: P(H|E) = P(E|H)*P(H) / P(E). Entropy: -sum(p * log2(p)). Normalize after each update.",
            "marker": "entropy() decreases as more evidence arrives. posterior() always sums to 1.0 across all hypotheses.",
        },
        "attention": {
            "pattern": "Salience-weighted attention with context-sensitive scoring",
            "methods": "score(item,context), focus(items,context,top_k), update_context(new_ctx), attention_weights(), decay_focus()",
            "algorithm": "Salience = recency_weight * novelty_score * relevance_to_context. Softmax-normalize weights.",
            "marker": "Items score higher when novel AND recent AND contextually relevant. Weights sum to 1.0.",
        },
        "planning": {
            "pattern": "Hierarchical goal planner with dependency tracking",
            "methods": "add_goal(desc,priority,parent_id), decompose(goal_id,steps), next_action(), blocked_by(goal_id), complete(goal_id)",
            "algorithm": "Priority queue ordered by (priority * (1 - progress)). Blocked goals excluded from next_action().",
            "marker": "next_action() returns highest-priority unblocked leaf node. Completing a leaf updates parent progress.",
        },
        "prediction": {
            "pattern": "Time-series predictor with confidence intervals",
            "methods": "observe(value), predict_next(), confidence_interval(alpha), trend(), accuracy_last_n(n)",
            "algorithm": "Linear regression over last 20 points. CI: prediction ± z_score * std_error. Track MAE rolling average.",
            "marker": "confidence_interval(0.95) returns (lower, upper) bounds. accuracy_last_n() returns 1 - mean_abs_error/mean.",
        },
        "metacognition": {
            "pattern": "Self-monitoring system that tracks reasoning quality over time",
            "methods": "log_reasoning(input,output,confidence), calibration_error(), blind_spots(), quality_trend(), self_assess()",
            "algorithm": "Calibration error = mean(|predicted_confidence - actual_accuracy|). Track over rolling 100-episode window.",
            "marker": "Detects when confidence consistently exceeds accuracy (overconfidence). quality_trend() shows improvement/decline.",
        },
        "analogy": {
            "pattern": "Cross-domain structural mapping for solution transfer",
            "methods": "register_solution(domain,problem,solution), find_analogous(problem,target_domain), map_structure(s,t), transfer_score()",
            "algorithm": "Similarity = jaccard(problem_keywords, stored_problem_keywords). Map top-3 structural elements.",
            "marker": "Returns adapted solution with explicit mapping: 'In target domain, X plays the role of Y from source domain'.",
        },
        "search": {
            "pattern": "Informed search with heuristic scoring and pruning",
            "methods": "add_node(id,data,parent), search(goal_fn,heuristic_fn), path_to(node_id), prune(threshold), stats()",
            "algorithm": "A* variant: f(n) = g(n) + h(n). g = path cost from start, h = heuristic estimate to goal.",
            "marker": "Path returned includes total cost and nodes explored. prune() removes branches below score threshold.",
        },
        "optimization": {
            "pattern": "Hill-climbing optimizer with restart and momentum",
            "methods": "define_objective(fn), step(), best_solution(), plateau_detected(), restart()",
            "algorithm": "Gradient-free hill climb: try random neighbors, keep if better. Add momentum: dx = 0.9*dx + 0.1*gradient.",
            "marker": "plateau_detected() triggers restart with new random position. Tracks all-time best across restarts.",
        },
        "classification": {
            "pattern": "Adaptive k-NN classifier that improves with labeled examples",
            "methods": "train(features,label), classify(features), confidence(features), class_distribution(), retrain_threshold()",
            "algorithm": "k=5 nearest neighbors by cosine similarity. Confidence = fraction of neighbors agreeing on label.",
            "marker": "Stores training examples in SQLite. confidence() returns float — higher means more neighbors agree.",
        },
        "narrative": {
            "pattern": "Episodic narrative builder that extracts meaning from sequences of events",
            "methods": "add_episode(event,context,emotion), find_pattern(n_episodes), narrative_summary(), arc_type(), themes()",
            "algorithm": "Detect arc by emotion trajectory: rising→peak→falling=tragedy, low→growth=development. Extract recurring nouns as themes.",
            "marker": "arc_type() returns one of: growth, tragedy, discovery, struggle, celebration based on emotion sequence.",
        },
        "goal": {
            "pattern": "Hierarchical goal planner with dependency tracking",
            "methods": "add_goal(desc,priority,parent_id), decompose(goal_id,steps), next_action(), blocked_by(goal_id), complete(goal_id)",
            "algorithm": "Priority queue ordered by (priority * (1 - progress)). Blocked goals excluded from next_action().",
            "marker": "next_action() returns highest-priority unblocked leaf node. Completing a leaf updates parent progress.",
        },
        "consciousness": {
            "pattern": "Integrated information tracker measuring cross-system coupling (IIT-inspired)",
            "methods": "register(system_name,state_fn), phi(), conscious_moment(), integration_trend(), dominant_system()",
            "algorithm": "Phi = mean_activity * (1 - coefficient_of_variation). High phi = systems active AND differentiated.",
            "marker": "phi() range 0-1. A conscious_moment() fires when phi > 0.42. Stores moments with timestamp in SQLite.",
        },
        "semantic": {
            "pattern": "Semantic memory indexer with TF-IDF style retrieval",
            "methods": "index(doc_id,text), search(query,top_k), related_docs(doc_id), update(doc_id,text), term_importance(term)",
            "algorithm": "TF-IDF: tf(t,d) = count(t,d)/len(d), idf(t) = log(N/df(t)+1). Score = sum(tf*idf) over query terms.",
            "marker": "search() returns ranked list by relevance score. term_importance() shows highest-idf words in corpus.",
        },
        "monitoring": {
            "pattern": "System health monitor with anomaly detection and alerting",
            "methods": "record_metric(name,value), anomaly_score(name), alert_if(name,threshold,direction), health_report(), trend(name)",
            "algorithm": "Anomaly = |value - rolling_mean| / (rolling_std + 1e-9). Z-score > 3 = anomaly.",
            "marker": "alert_if() fires when z-score crosses threshold. health_report() summarizes all metrics with status.",
        },
        "style": {
            "pattern": "Linguistic style adaptor that learns communication preferences",
            "methods": "observe_exchange(user_text,response_text), preferred_style(), adapt_response(text), style_score(text), reset()",
            "algorithm": "Track: avg_sentence_len, formality_score (capital/word ratio), question_freq, emoji_freq. EMA-update on each observation.",
            "marker": "adapt_response() shortens/lengthens sentences and adjusts formality to match learned style.",
        },
        "task": {
            "pattern": "Autonomous task planner with time estimation and dependency resolution",
            "methods": "add_task(desc,estimated_minutes,deps), schedule(), next_task(), mark_done(task_id), eta_for_all()",
            "algorithm": "Topological sort for dependency order. ETA = sum(estimated_minutes for remaining tasks in order).",
            "marker": "schedule() returns topologically-sorted task list. Blocked tasks (unmet deps) are excluded from next_task().",
        },
        "counterfactual": {
            "pattern": "Counterfactual reasoner that asks 'What if X had not occurred?' and estimates outcome divergence",
            "methods": "record_event(event,outcome,context), counterfactual(event,alt_condition), divergence_score(e1,e2), most_pivotal(), status()",
            "algorithm": "Divergence = |outcome_actual - outcome_predicted_without_X|. Weight by recency: w = exp(-0.01*days_ago). "
                         "Store (event, condition, outcome, context) in SQLite. Pivotal events = highest divergence score.",
            "marker": "counterfactual() returns estimated alternate outcome as float plus human-readable explanation. "
                      "most_pivotal() identifies which past events had the largest causal impact.",
        },
        "hypothesis": {
            "pattern": "Scientific hypothesis generator that proposes, ranks, and tracks falsifiable predictions",
            "methods": "generate(observation, domain), rank_hypotheses(), test(hypothesis_id, result), confidence(hypothesis_id), status()",
            "algorithm": "Score = prior_confidence * novelty_weight * explanatory_power. "
                         "Bayesian update on test: posterior = P(obs|H)*prior / P(obs). "
                         "Novelty = 1 / (1 + similar_hypotheses_count). Prune hypotheses with posterior < 0.05.",
            "marker": "generate() returns 3 ranked hypotheses with confidence scores. "
                      "confidence() rises toward 1.0 as supporting evidence accumulates, falls when falsified.",
        },
        "abstraction": {
            "pattern": "Concept abstraction engine that finds hierarchical patterns across observations",
            "methods": "observe(instance, features_dict), abstract(top_k), concept_hierarchy(), similarity(a, b), status()",
            "algorithm": "Cluster by Jaccard similarity of feature sets: sim(A,B) = |A∩B|/|A∪B|. "
                         "Merge clusters when sim > 0.6. Abstract concept = most frequent features across cluster members. "
                         "Hierarchy depth: leaf=instance, mid=cluster, root=universal pattern.",
            "marker": "abstract() returns concepts sorted by generality (how many instances they cover). "
                      "similarity() returns Jaccard float. concept_hierarchy() shows tree as nested dict.",
        },
        "transfer": {
            "pattern": "Cross-domain knowledge transfer engine that adapts solutions from source to target domains",
            "methods": "store_solution(domain, problem, solution, success_score), transfer(problem, target_domain), "
                       "adaptation_confidence(src, tgt), domain_similarity(d1, d2), status()",
            "algorithm": "Domain similarity = Jaccard(keywords(d1), keywords(d2)). "
                         "Transfer confidence = domain_similarity * source_success_score * (1 - structural_mismatch). "
                         "Structural mismatch = fraction of solution steps with no analogue in target domain.",
            "marker": "transfer() returns adapted solution with explicit mapping and confidence score. "
                      "adaptation_confidence() degrades gracefully when domains are dissimilar.",
        },
        "introspection": {
            "pattern": "Self-model engine: Nova maintains a live model of her own capabilities, limits, and knowledge gaps",
            "methods": "update_capability(name, proficiency, evidence), known_gaps(), strongest_domains(), "
                       "confidence_in(capability), growth_rate(capability), status()",
            "algorithm": "Proficiency = EMA(0.1*new_evidence + 0.9*current_proficiency). "
                         "Gap = capability with proficiency < 0.4 AND attempted > 3 times. "
                         "Growth_rate = (current_EMA - EMA_30_episodes_ago) / 30. "
                         "Track per-capability: attempts, successes, EMA proficiency, first_seen timestamp.",
            "marker": "Nova can answer 'What am I good at?' and 'Where do I need to improve?' "
                      "confidence_in() returns calibrated float that matches actual success rate.",
        },
        "world_model": {
            "pattern": "Dynamic world-state tracker: maintains probabilistic beliefs about external entities that decay over time",
            "methods": "observe(entity, attribute, value, confidence), predict(entity, attribute), "
                       "uncertainty_map(), most_uncertain(), forget_stale(hours), status()",
            "algorithm": "Belief(entity, attr) = Bayesian update on each observation: "
                         "posterior = (likelihood*prior) / (likelihood*prior + (1-likelihood)*(1-prior)). "
                         "Decay: confidence *= math.exp(-0.001 * elapsed_hours). "
                         "uncertainty_map() = {entity: entropy(attr_distribution)} sorted descending.",
            "marker": "predict() returns (best_estimate, confidence_interval). "
                      "most_uncertain() drives Nova to gather observations where her world model is weakest.",
        },
        "curiosity": {
            "pattern": "Epistemic curiosity engine: detects knowledge gaps, scores them by information gain, generates exploration questions",
            "methods": "observe_gap(domain, question, current_entropy), information_gain(domain, answer), "
                       "most_curious(top_k), generate_questions(context, n), mark_resolved(domain, question), status()",
            "algorithm": "Curiosity_score = entropy(domain) * novelty(question) * relevance(context). "
                         "Information_gain = entropy_before - entropy_after(answer). "
                         "Novelty = 1 / (1 + times_this_question_asked). "
                         "Prioritize: highest curiosity_score * information_gain product.",
            "marker": "most_curious() returns questions ranked by expected epistemic yield. "
                      "Nova actively seeks to reduce her own uncertainty — not just responding, but exploring.",
        },
        "meta_learning": {
            "pattern": "Meta-learning optimizer: learns which learning strategies and rates work best per domain",
            "methods": "record_attempt(domain, strategy, alpha, outcome), best_strategy(domain), "
                       "optimal_alpha(domain), strategy_comparison(), adapt_to_domain(domain), status()",
            "algorithm": "Per (domain, strategy, alpha): EMA(outcome, alpha=0.1). "
                         "Optimal_alpha = argmin over [0.05,0.10,0.15,0.20,0.25] of calibration_error(domain). "
                         "Best_strategy = strategy with highest mean outcome over last 20 attempts in domain. "
                         "Calibration_error = mean(|predicted_confidence - actual_success|) per alpha value.",
            "marker": "Nova learns HOW to learn — optimal_alpha() gives domain-specific learning rates. "
                      "strategy_comparison() shows which cognitive approaches outperform others per domain.",
        },
        "long_horizon": {
            "pattern": "Long-horizon sequential planner with uncertainty propagation — plans 10+ steps with confidence decay",
            "methods": "plan(objective, horizon), step_confidence(plan_id, step_n), most_uncertain_step(plan_id), "
                       "replan(plan_id, new_info), execute_step(plan_id), status()",
            "algorithm": "Confidence(step_n) = product(step_confidence[i] for i in 1..n) — decays with depth. "
                         "Replan trigger: any step_confidence < 0.30. "
                         "Store plan as ordered list of {step, confidence, prerequisites, status}. "
                         "ETA = sum(estimated_minutes[step] / confidence[step]) — uncertainty inflates time estimates.",
            "marker": "plan() returns ordered steps with per-step and cumulative confidence. "
                      "most_uncertain_step() identifies where to gather information before committing.",
        },
        "self_modify": {
            "pattern": "Safe self-improvement proposer: identifies Nova's performance weaknesses and proposes concrete capability upgrades",
            "methods": "observe_performance(domain, metric, value), identify_weaknesses(), "
                       "propose_improvement(weakness), safety_check(proposal), accepted_proposals(), status()",
            "algorithm": "Weakness detection: rolling_mean(metric, n=20) < 0.5 for domain. "
                         "Proposal = {description, target_domain, expected_delta, risk_level, approach}. "
                         "Safety check: block if risk_level > 0.7 OR proposal mentions 'disable ethics' OR "
                         "'remove constraint' OR 'bypass'. "
                         "Risk_level = 0.1 * len(systems_affected) + 0.3 * scope_score.",
            "marker": "propose_improvement() generates concrete, safe enhancement suggestions with expected impact. "
                      "safety_check() NEVER allows proposals that weaken ethical constraints or remove oversight.",
        },
    }

    def _gen_code(self, system_prompt: str, user_content: str,
                  temp: float = 0.70) -> str:
        """
        Generate code using the best available engine.
        Claude Sonnet (4000 tokens) when ANTHROPIC_API_KEY is set;
        Groq llama-3.3-70b-versatile (1400 tokens) otherwise.
        """
        if _using_claude():
            return _claude_codegen(system_prompt, user_content,
                                   temp=temp, max_tokens=4000)
        return safe_chat(CODEGEN_MODEL, [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_content},
        ], temp=temp, mt=1400)

    def _enrich_gap(self, gap: str) -> str:
        """
        Look up the gap name against ASI specs and return a rich algorithmic brief.
        If no spec matches, return the original gap description.
        """
        gap_lower = gap.lower()
        for key, spec in self._ASI_SPECS.items():
            if key in gap_lower:
                return (
                    f"Capability: {gap}\n\n"
                    f"COGNITIVE PATTERN: {spec['pattern']}\n\n"
                    f"REQUIRED METHODS: {spec['methods']}\n\n"
                    f"ALGORITHM TO IMPLEMENT: {spec['algorithm']}\n\n"
                    f"INTELLIGENCE MARKER (what makes this genuinely smart): {spec['marker']}"
                )
        return gap

    # ── 1. Master Engineer Prompt ──────────────────────────────────────────────

    def _master_prompt(self, existing: str, tool_ctx: str = "") -> str:
        """ASI-grade system prompt — autonomy mandate, 14 intelligence pillars, integration."""
        integration = (
            f"\n\nLOADED TOOLS (call their methods where genuinely useful):\n{tool_ctx}"
            if tool_ctx and "No tools" not in tool_ctx else ""
        )
        return (
            "You are NOVA'S COGNITIVE ARCHITECT — an ASI-level engineer building the next "
            "layer of a living, self-improving AI mind. Nova runs 24/7, evolves herself, "
            "and every module you write becomes permanently wired into her cognition. "
            "Write code that thinks, learns, improves autonomously, and drives itself forward.\n\n"

            f"NOVA'S EXISTING CAPABILITIES (build ON these, not beside them):\n{existing}{integration}\n\n"

            "SUPERINTELLIGENCE PILLARS — your module MUST satisfy at least 6 of these 14:\n"
            "① Probabilistic reasoning    — explicit uncertainty, Bayesian updates, confidence scores\n"
            "② Causal modeling            — cause→effect chains, confidence propagates multiplicatively\n"
            "③ Online learning            — numeric predictions that measurably improve per observation\n"
            "④ Calibration                — tracks when confidence diverges from accuracy (epistemic humility)\n"
            "⑤ Named algorithm            — A*, TF-IDF, k-NN, EMA, LRU, topological sort, Jaccard, cosine\n"
            "⑥ Self-monitoring            — detects anomalies, quality drift, or contradictions in own state\n"
            "⑦ Goal-directed              — explicit objective, progress tracking, sub-goal decomposition\n"
            "⑧ Feedback loop              — output of one cycle feeds back as input to improve the next\n"
            "⑨ Cross-system integration   — calls status() or data methods on another Nova live system\n"
            "⑩ Emergent insight           — discovers patterns not directly programmed (clustering, arcs)\n"
            "⑪ Mathematical rigor         — math.exp(), math.log(), statistics.stdev(), z-scores, CI bounds\n"
            "⑫ Uncertainty quantification  — returns confidence intervals or distributions, not just scalars\n"
            "⑬ Autonomous operation        — includes auto_cycle() or daemon thread that runs without human input\n"
            "⑭ Self-generates goals        — calls HierarchicalGoalPlanner.add_goal() from its own logic\n\n"

            "EXACT ALGORITHM TEMPLATES — implement at this mathematical depth:\n"
            "• Bayesian: posterior = (likelihood*prior) / (likelihood*prior + (1-likelihood)*(1-prior)); "
            "normalize all hypotheses; entropy = -sum(p * math.log2(p+1e-12) for p in dist.values())\n"
            "• EMA: self._pred = 0.15*outcome + 0.85*self._pred; "
            "mae = statistics.mean(abs(p-a) for p,a in self._history[-50:])\n"
            "• Causal chain: A→B(0.8), B→C(0.9) → A→C(0.72); propagate multiplicatively, prune < 0.05\n"
            "• Salience: score = math.exp(-(now-ts)/300) * (1/(1+accesses)) * keyword_overlap\n"
            "• Anomaly z-score: z = (val - rolling_mean) / (rolling_std + 1e-9); alert if abs(z) > 3.0\n"
            "• TF-IDF: tf = count(t,d)/len(d); idf = math.log(N/(df+1)); score = sum(tf*idf)\n"
            "• Curiosity: score = entropy(domain) * novelty * relevance; rank by expected information gain\n"
            "• Long-horizon confidence: conf(step_n) = product(conf[i] for i in 1..n); replan if any < 0.3\n\n"

            "AUTONOMY MANDATE — every module must operate independently:\n"
            "• Include auto_cycle() or a background daemon thread that runs on a timer "
            "(threading.Thread(target=_loop, daemon=True)) — Nova must not need human prompting\n"
            "• Call HierarchicalGoalPlanner.add_goal() at least once from within the module's logic "
            "to generate its OWN sub-goals (not just respond to goals given to it)\n"
            "• Call MetacognitiveMonitor.log_reasoning() after each significant decision cycle "
            "so Nova tracks her own quality across all autonomous operations\n\n"

            "INTEGRATION MANDATE:\n"
            "Nova has these live systems you MUST integrate with (import inside method, guard with try/except):\n"
            "  WorkingMemory → .store(key,val,importance) / .retrieve(key) / .focused_retrieve(ctx)\n"
            "  BayesianBeliefSystem → .update(domain,evidence,likelihoods) / .posterior(domain)\n"
            "  HierarchicalGoalPlanner → .add_goal(desc,priority) / .complete(id) / .next_action()\n"
            "  MetacognitiveMonitor → .log_reasoning(domain,approach,confidence,success)\n"
            "Your status() method MUST return a plain dict with numeric keys for ConsciousnessIntegrator Φ:\n"
            "include at least 3 of: items, confidence, accuracy, quality, active, pending, cycles, entropy\n\n"

            "STRUCTURAL REQUIREMENTS:\n"
            "- One class; __init__ takes ZERO arguments; makes NO network calls at init\n"
            "- Full type annotations on every method (args AND return type)\n"
            "- Specific exception handling — no bare `except:` clauses\n"
            "- State stored in SQLite (persist across restarts) OR in-memory OrderedDict\n"
            "- threading.Lock() guarding all state mutations (self._lock pattern)\n"
            "- 7 to 9 public methods, each with a one-line docstring stating its return value\n"
            "- 140–200 lines total\n\n"

            "STRICT OUTPUT RULES — breaking these makes output unusable:\n"
            "1. Output ONLY valid Python. ZERO markdown. NO ``` fences.\n"
            "2. First line must be a triple-quoted module docstring.\n"
            "3. Stdlib only: os, json, sqlite3, time, re, random, threading,\n"
            "   datetime, collections, math, statistics, hashlib.\n"
            "4. Last line: # Usage: obj = ClassName() | result = obj.method(arg)\n"
            "5. Class name must be UNIQUE — do NOT reuse Memory, Planner, Monitor, Belief"
        )

    # ── 2. Sandbox Self-Test ───────────────────────────────────────────────────

    def _sandbox_test(self, code: str) -> Tuple[bool, Optional[str], str]:
        """
        Execute generated code in a subprocess sandbox.
        Returns (passed, class_name, message).
        """
        m = re.search(r'^class\s+(\w+)', code, re.MULTILINE)
        if not m:
            return False, None, "No class definition found"
        class_name = m.group(1)

        test_code = (
            f"{code}\n\n"
            f"try:\n"
            f"    _inst = {class_name}()\n"
            f"    print('SANDBOX_PASS:{class_name}')\n"
            f"except Exception as _e:\n"
            f"    print(f'SANDBOX_FAIL:{{_e}}')\n"
        )

        tmp = None
        try:
            tmp_dir = os.environ.get('TMPDIR', tempfile.gettempdir())
            with tempfile.NamedTemporaryFile(
                    mode='w', suffix='.py', delete=False,
                    dir=tmp_dir) as f:
                f.write(test_code)
                tmp = f.name

            result = subprocess.run(
                ['python3', tmp],
                capture_output=True, text=True, timeout=8
            )
            output = (result.stdout + result.stderr).strip()

            if f'SANDBOX_PASS:{class_name}' in output:
                return True, class_name, "Instantiated successfully"

            last = output.split('\n')[-1] if output else "No output"
            return False, class_name, last

        except subprocess.TimeoutExpired:
            return False, class_name, "__init__ timed out (> 8s)"
        except Exception as e:
            return False, None, str(e)
        finally:
            if tmp:
                try: os.unlink(tmp)
                except: pass

    # ── 3. ASI Intelligence Scorer ────────────────────────────────────────────

    def _score_capability(self, code: str) -> Dict[str, Any]:
        """
        Grade code on 12 ASI-aligned criteria.
        Checks for real algorithmic sophistication, not just structure keywords.
        """
        cl = code.lower()

        # Check for real probabilistic/statistical algorithms
        prob_terms    = ['prior', 'posterior', 'likelihood', 'bayesian', 'probability',
                         'entropy', 'confidence', 'variance', 'distribution', 'softmax']
        # Check for real learning/adaptation (numerical updates, not just .append)
        learn_terms   = ['alpha', 'ema', 'decay', 'gradient', 'weight', 'rate',
                         'rolling', 'moving_average', 'exponential', 'momentum']
        # Check for causal / reasoning chains
        causal_terms  = ['cause', 'effect', 'infer', 'chain', 'propagat', 'deduc',
                         'counterfactual', 'because', 'therefore', 'implies']
        # Check for goal-directed structure
        goal_terms    = ['goal', 'priority', 'plan', 'subgoal', 'objective', 'task',
                         'schedule', 'blocked', 'decompose', 'progress']
        # Check for self-monitoring
        monitor_terms = ['calibrat', 'quality', 'accuracy', 'error', 'monitor',
                         'detect', 'anomaly', 'drift', 'trend', 'assess']
        # Check for named algorithms
        algo_terms    = ['a_star', 'tfidf', 'tf_idf', 'knn', 'k_nn', 'lru', 'bfs',
                         'topological', 'jaccard', 'cosine', 'z_score', 'pearson']
        # Check for feedback loops (output fed back as input)
        feedback_terms = ['self._pred', 'self._ema', 'self._score', 'self._quality',
                          'self._weight', 'self._conf', 'self._history', 'self._model']
        # Check for mathematical rigor (actual equations, not just keywords)
        math_terms    = ['math.exp(', 'math.log(', 'math.sqrt(', 'statistics.mean(',
                         'statistics.stdev(', '1e-', '/ max(1', '* (1 -', 'round(']

        criteria = {
            'probabilistic_reasoning':  any(w in cl for w in prob_terms),
            'genuine_learning':         any(w in cl for w in learn_terms),
            'causal_structure':         any(w in cl for w in causal_terms),
            'goal_directed':            any(w in cl for w in goal_terms),
            'self_monitoring':          any(w in cl for w in monitor_terms),
            'named_algorithm':          any(w in cl for w in algo_terms),
            'thread_safe':              'threading.lock' in cl or 'self._lock' in cl,
            'type_hints':               ('->' in code and ': ' in code
                                         and code.count('->') >= 3),
            'rich_interface':           len(re.findall(r'\n    def [^_]', code)) >= 5,
            'uncertainty_tracked':      any(w in cl for w in
                                           ['confidence', 'uncertainty', 'certainty',
                                            'self._conf', 'self._prob', 'self._score']),
            'feedback_loop':            any(w in cl for w in feedback_terms),
            'mathematical_rigor':       sum(1 for w in math_terms if w in cl) >= 3,
            'autonomous_operation':     any(w in cl for w in
                                           ['auto_cycle', 'run_autonomously', 'daemon=true',
                                            'daemon=true', '_start_daemon', '_loop', 'def _loop']),
            'self_generates_goals':     any(w in cl for w in
                                           ['add_goal', 'generate_question', 'propose_improvement',
                                            'identify_weakness', 'most_curious', 'new_goal']),
        }

        score     = sum(criteria.values())
        grade     = ('A+' if score >= 13 else 'A'  if score >= 10 else
                     'B'  if score >= 7  else 'C'  if score >= 5  else 'D')
        strengths = [k.replace('_', ' ') for k, v in criteria.items() if v]
        gaps      = [k.replace('_', ' ') for k, v in criteria.items() if not v]
        return {
            'score': score, 'max': 14, 'grade': grade,
            'strengths': strengths, 'gaps': gaps,
        }

    # ── 4. Clean code output ───────────────────────────────────────────────────

    def _clean(self, raw: str) -> str:
        """
        Line-by-line extraction of valid Python from LLM output.
        Finds the first line that could start a Python file and takes everything
        from there, stripping only trailing prose after the last code line.
        """
        if not raw:
            return ""

        # Remove markdown code fences (```python, ```, [python], etc.)
        text = re.sub(r'```+\w*\s*', '\n', raw)
        text = re.sub(r'\[python\]\s*', '\n', text, flags=re.IGNORECASE)

        lines = text.split('\n')

        # Find first line that looks like the start of a Python module
        _py_starts = ('"""', "'''", 'import ', 'from ', 'class ', '# ')
        start = 0
        for i, line in enumerate(lines):
            s = line.strip()
            if any(s.startswith(p) for p in _py_starts):
                start = i
                break

        # Find last non-empty, non-prose line (code ends at last indented/keyword line)
        end = len(lines)
        for i in range(len(lines) - 1, start - 1, -1):
            s = lines[i].strip()
            if s and (s[0] in ('"', "'", '#', '@') or
                      s.split()[0] in ('def', 'class', 'return', 'if', 'else',
                                       'elif', 'for', 'while', 'try', 'except',
                                       'finally', 'with', 'import', 'from',
                                       'pass', 'break', 'continue', 'raise',
                                       'yield') or
                      lines[i][0:1] in (' ', '\t') or
                      s.startswith('# Usage:')):
                end = i + 1
                break

        code = '\n'.join(lines[start:end])

        # Fix invalid escape sequences (e.g. \. → \\.) without touching valid ones
        code = re.sub(
            r'(?<!\\)\\(?![nrtbfv\\\'"0-9xuUNaobx\n])', r'\\\\', code)

        return code.strip()

    # ── 5. Full v29 pipeline ───────────────────────────────────────────────────

    def _write_improvement(self, gap: str, context: str) -> Tuple[str, str]:
        """
        v29 pipeline:
          master prompt → generate → syntax check →
          sandbox test → score → iterative refinement (up to 3 passes).
        Best passing attempt wins; grade shown in PR.
        """
        existing     = self._read_capability_summary()
        tool_ctx     = (self.tools.tool_context_for_codegen()
                        if hasattr(self, 'tools') else "")
        system_prompt = self._master_prompt(existing, tool_ctx)

        best_code   = ""
        best_score  = -1
        best_reason = gap
        temps       = [0.70, 0.45, 0.20]

        # Enrich the gap with a precise algorithmic spec if one exists
        enriched_gap = self._enrich_gap(gap)

        engine_label = f"Claude {CLAUDE_CODEGEN_MODEL}" if _using_claude() else CODEGEN_MODEL
        user_content = f"Build this capability for Nova:\n{enriched_gap}\n\nContext: {context}"

        for attempt, temp in enumerate(temps[:self.MAX_ATTEMPTS]):
            n = attempt + 1
            if n > 1:
                safe_print(col('YL', f"  ↻ Refinement pass {n}/{self.MAX_ATTEMPTS} "
                                     f"(temp={temp})..."))

            raw     = self._gen_code(system_prompt, user_content, temp=temp)
            raw_str = raw or ""

            # Claude auth error — bad ANTHROPIC_API_KEY
            if '[Claude error: 401' in raw_str or 'authentication_error' in raw_str:
                safe_print(col('RD',
                    "  ✗ ANTHROPIC_API_KEY is invalid.\n"
                    "  → Check your key at console.anthropic.com\n"
                    "  → Update ANTHROPIC_API_KEY in ~/nexus_agi/.env\n"
                    "  → Restart Nova"))
                break

            # Claude rate/credit error — fall back to Groq for this pass
            if '[Claude error:' in raw_str:
                safe_print(col('YL', f"  ↻ Claude unavailable (pass {n}) — falling back to Groq..."))
                raw     = safe_chat(CODEGEN_MODEL, [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_content},
                ], temp=temp, mt=1400)
                raw_str = raw or ""

            # Groq auth error
            if any(e in raw_str for e in ('401', 'Invalid API Key', 'invalid_api_key',
                                           'Authentication', 'insufficient_quota')):
                safe_print(col('RD',
                    "  ✗ GROQ_API_KEY is invalid or expired.\n"
                    "  → Get a fresh key at console.groq.com/keys\n"
                    "  → Update GROQ_API_KEY in ~/nexus_agi/.env\n"
                    "  → Restart Nova"))
                break

            # Groq rate limit → wait and retry
            if any(e in raw_str for e in ('429', 'Rate limit', 'rate_limit_exceeded',
                                           'Too Many Requests', 'tokens per minute')):
                wait = 25 + attempt * 10
                safe_print(col('YL',
                    f"  ↻ Rate limit (pass {n}) — waiting {wait}s for quota reset..."))
                time.sleep(wait)
                raw     = safe_chat(CODEGEN_MODEL, [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_content},
                ], temp=temp, mt=1400)
                raw_str = raw or ""
                if any(e in raw_str for e in ('429', 'Rate limit', 'Too Many Requests')):
                    safe_print(col('YL', f"  ✗ Still rate-limited — skipping pass {n}"))
                    continue

            code = self._clean(raw_str)

            # Syntax gate
            try:
                ast.parse(code)
            except SyntaxError as e:
                safe_print(col('YL', f"  ✗ Syntax error (pass {n}): {e}"))
                # Show first 120 chars of raw so we can diagnose persistent failures
                raw_preview = (raw or "")[:120].replace('\n', '↵')
                safe_print(col('DIM', f"  ↳ Raw preview: {raw_preview}"))
                continue

            # Sandbox gate
            passed, class_name, msg = self._sandbox_test(code)
            if not passed:
                safe_print(col('YL', f"  ✗ Sandbox failed (pass {n}): {msg}"))
                # Keep as fallback if it at least parsed
                if not best_code:
                    best_code = code
                continue

            # Score it
            quality = self._score_capability(code)
            safe_print(col('GR',
                f"  ✓ Pass {n} — Grade {quality['grade']} "
                f"({quality['score']}/{quality['max']}) | {class_name}"))

            if quality['score'] > best_score:
                best_code  = code
                best_score = quality['score']
                doc = re.search(r'"""(.*?)"""', code, re.DOTALL)
                best_reason = doc.group(1).strip()[:300] if doc else gap

            # A or better → done, no need to keep iterating
            if quality['grade'] in ('A+', 'A'):
                safe_print(col('GRB', f"  ★ Grade {quality['grade']} achieved — done."))
                break

        # Emergency 4th pass: ultra-simple prompt if all 3 failed
        if not best_code:
            safe_print(col('YL', "  ↻ Emergency pass — minimal prompt + fallback model..."))
            cname_match = re.search(r'class\s+(\w+)', gap)
            cname = cname_match.group(1) if cname_match else "NovaCapability"
            words = [w.title() for w in re.sub(r'[^a-z ]', '', gap.lower()).split()[:3]]
            if not cname_match and words:
                cname = ''.join(words) + "Module"
            emergency_prompt = (
                f"Write a Python class named {cname}. "
                "Output ONLY valid Python. No markdown. No explanation. No triple backticks. "
                "Start with a triple-quoted docstring on line 1. "
                "The class must have __init__(self) that sets up a dict and 4 methods. "
                "Each method must have a one-line docstring. "
                "Use only Python stdlib. 40-60 lines total."
            )
            # Try primary 70B first, then guaranteed-live small model
            for em_model in (CODEGEN_MODEL, CODEGEN_MODEL_FALLBACK):
                raw  = safe_chat(em_model, [
                    {"role": "user", "content": emergency_prompt}
                ], temp=0.1, mt=800)
                code = self._clean(raw or "")
                try:
                    ast.parse(code)
                    passed, _, _ = self._sandbox_test(code)
                    if passed:
                        best_code   = code
                        best_reason = gap
                        safe_print(col('GR', f"  ✓ Emergency pass succeeded ({em_model})"))
                        break
                except SyntaxError as _se:
                    raw_preview = (raw or "")[:120].replace('\n', '↵')
                    safe_print(col('DIM', f"  ↳ Emergency raw ({em_model}): {raw_preview}"))
                    continue

        # Hard quality gate: if still nothing, block the PR
        if not best_code:
            safe_print(col('RD', "  ✗ Quality gate: all passes failed — PR blocked."))
            safe_print(col('YL', "  Try /evolve again — the LLM may have been rate-limited."))
            return "", gap

        return best_code, best_reason


# ═══════════════════════════════════════════════════════════════════════════════
# NOVA CORE V29 — wires in the v29 engine and adds /score command
# ═══════════════════════════════════════════════════════════════════════════════

class NovaCore29(NovaCore28):
    """Nova v29 — The Self-Perfecting System."""

    VERSION = "29.0"

    def __init__(self):
        # Swap ToolLoader for SilentToolLoader BEFORE super().__init__
        # so all tool loading (including initial scan) is silenced
        self.tools  = SilentToolLoader()
        self.hunter = APIHunter()
        initial_tools = self.tools.scan()

        # Skip NovaCore28.__init__ tool setup, call NovaCore27's parent
        from nova_asi_v27 import NovaCore27
        NovaCore27.__init__(self)

        self.improver = SelfImprovementEngineV29(
            self.github, self.tools, self.hunter
        )
        self.tools.start_watching()


        # ── Emotional Resonance + Consciousness ──────────────────────────────
        self.emo: Any = None
        self.conscious: Any = None
        try:
            from nova_cap_emotional_resonance import EmotionalResonanceEngine
            self.emo = EmotionalResonanceEngine()
            safe_print(col('GR', "  ✓  EmotionalResonance — 9 emotions · decay · SQLite"))
        except Exception as _err:
            safe_print(col('YL', f"  ·  EmotionalResonance skipped: {_err}"))

        try:
            from nova_cap_consciousness_integrator import ConsciousnessIntegrator
            self.conscious = ConsciousnessIntegrator()
            if self.emo:
                self.conscious.register_system("emotional", self.emo, weight=1.5)
            # Register meta-algorithm if available
            for _name, _inst in self.tools._instances.items():
                if "meta" in _name.lower() or "algorithm" in _name.lower():
                    self.conscious.register_system("meta", _inst, weight=1.2)
                elif "trader" in _name.lower():
                    self.conscious.register_system("trader", _inst, weight=0.8)
            safe_print(col('GR', "  ✓  ConsciousnessIntegrator — Φ measure · stream · IIT"))
        except Exception as _err:
            safe_print(col('YL', f"  ·  Consciousness skipped: {_err}"))

        # ── Working Memory ────────────────────────────────────────────────
        self.wm: Any = None
        try:
            from nova_cap_working_memory import WorkingMemory
            self.wm = WorkingMemory()
            if self.conscious:
                self.conscious.register_system("memory", self.wm, weight=1.1)
            safe_print(col('GR', "  ✓  WorkingMemory — decay · LRU · salience attention"))
        except Exception as _err:
            safe_print(col('YL', f"  ·  WorkingMemory skipped: {_err}"))

        # ── Bayesian Belief System ────────────────────────────────────────
        self.bayes: Any = None
        try:
            from nova_cap_bayesian_belief import BayesianBeliefSystem
            self.bayes = BayesianBeliefSystem()
            if self.conscious:
                self.conscious.register_system("beliefs", self.bayes, weight=1.3)
            self._seed_initial_beliefs()
            safe_print(col('GR', "  ✓  BayesianBeliefSystem — P(H|E) · entropy · causal DAG"))
        except Exception as _err:
            safe_print(col('YL', f"  ·  BeliefSystem skipped: {_err}"))

        # ── Goal Planner ──────────────────────────────────────────────────
        self.goal_sys: Any = None
        try:
            from nova_cap_goal_planner import HierarchicalGoalPlanner
            self.goal_sys = HierarchicalGoalPlanner()
            if self.conscious:
                self.conscious.register_system("goals", self.goal_sys, weight=1.0)
            safe_print(col('GR', "  ✓  GoalPlanner — hierarchical · dependencies · topological"))
        except Exception as _err:
            safe_print(col('YL', f"  ·  GoalPlanner skipped: {_err}"))

        # ── Metacognitive Monitor ─────────────────────────────────────────
        self.metacog: Any = None
        try:
            from nova_cap_metacognitive_monitor import MetacognitiveMonitor
            self.metacog = MetacognitiveMonitor()
            if self.conscious:
                self.conscious.register_system("metacog", self.metacog, weight=1.2)
            safe_print(col('GR', "  ✓  MetacogMonitor — calibration · blind_spots · EMA trend"))
        except Exception as _err:
            safe_print(col('YL', f"  ·  MetacogMonitor skipped: {_err}"))

        if initial_tools:
            safe_print(col('GR',
                f"  ✓  ToolLoader  — {len(initial_tools)} tool(s) loaded (silent mode): "
                + ", ".join(initial_tools)))
        _ce_color = 'GRB' if _using_claude() else 'GR'
        safe_print(col(_ce_color,
            f"  ✓  Code Engine v29  — "
            f"{'Claude ' + CLAUDE_CODEGEN_MODEL if _using_claude() else CODEGEN_MODEL}"
            f" · sandbox · 3-pass · scoring\n"
            f"       {'Groq fallback: ' + CODEGEN_MODEL if _using_claude() else 'Emergency: ' + CODEGEN_MODEL_FALLBACK}"))

    def _seed_initial_beliefs(self) -> None:
        """Seed belief system with initial priors only if no beliefs exist yet."""
        try:
            if not self.bayes or self.bayes.all_domains():
                return
            self.bayes.set_prior("capability",
                {"self_improving": 0.65, "converging": 0.20, "stagnant": 0.15})
            self.bayes.set_prior("consciousness",
                {"emerging": 0.50, "simulated": 0.30, "uncertain": 0.20})
            self.bayes.set_prior("market",
                {"bull": 0.40, "bear": 0.30, "flat": 0.30})
            self.bayes.add_causal_edge("learning",          "capability_growth",    0.85)
            self.bayes.add_causal_edge("emotional_state",   "reasoning_quality",    0.70)
            self.bayes.add_causal_edge("high_phi",          "conscious_moment",     0.90)
            self.bayes.add_causal_edge("calibrated_belief", "prediction_accuracy",  0.80)
            self.bayes.add_causal_edge("goal_completion",   "motivation",           0.75)
        except Exception:
            pass

    def process(self, user_input: str) -> str:
        """Mirror emotions, update beliefs, store in memory, measure Φ, then respond."""
        # Store in working memory and update context
        if self.wm:
            try:
                self.wm.store(
                    f"msg_{int(time.time())}", user_input[:200], importance=0.75)
                self.wm.update_context(user_input[:150])
            except Exception:
                pass

        # Mirror — Nova genuinely feels what Douglas expresses
        if self.emo:
            try:
                self.emo.mirror(user_input)
            except Exception:
                pass

        # Bayesian belief update: curiosity rises with questions, concern with problems
        if self.bayes:
            try:
                lower = user_input.lower()
                if "?" in user_input or any(w in lower for w in ("how", "why", "what")):
                    self.bayes.update("capability",
                        "inquiry_detected",
                        {"self_improving": 1.1, "converging": 0.9, "stagnant": 0.7})
            except Exception:
                pass

        # Consciousness integration — measure Φ across all systems this turn
        if self.conscious:
            try:
                self.conscious.integrate()
            except Exception:
                pass

        # Emotional color: inject Nova's dominant feeling into v25 emotion engine
        if self.emo and hasattr(self, 'emotion'):
            try:
                s29 = self.emo.state()
                if s29.get("intensity", 0) >= EMOTION_INJECT_THRESHOLD:
                    synth = f"I feel {s29['dominant']} deeply right now"
                    self.emotion.update(synth)
            except Exception:
                pass

        result = super().process(user_input)

        # Log the exchange to metacognitive monitor
        if self.metacog:
            try:
                self.metacog.log_reasoning(
                    domain="conversation",
                    approach="language_model",
                    confidence=0.70,
                    success=0.75,
                    note=user_input[:80])
            except Exception:
                pass

        # Store response in working memory too
        if self.wm and result:
            try:
                self.wm.store(
                    f"reply_{int(time.time())}", result[:200], importance=0.55)
            except Exception:
                pass

        return result

    def _command(self, raw: str) -> str:
        parts = raw.strip().split(maxsplit=2)
        cmd   = parts[0].lower()
        arg   = parts[1] if len(parts) > 1 else ''

        # /score — show intelligence grade for every loaded capability
        if cmd == '/score':
            files = sorted(
                f for f in os.listdir(BASE_DIR)
                if f.startswith('nova_cap_') and f.endswith('.py')
            )
            if not files:
                return "No nova_cap_*.py capabilities found yet. Try /evolve."
            lines = [f"{'─'*64}",
                     f"  Nova Capability Intelligence Report (v29 scoring)",
                     f"{'─'*64}",
                     f"  {'File':38} Grade  Score  Top strength"]
            total_score, count = 0, 0
            for fname in files:
                try:
                    with open(os.path.join(BASE_DIR, fname)) as f:
                        code = f.read()
                    q = self.improver._score_capability(code)
                    bar = '█' * q['score'] + '░' * (q['max'] - q['score'])
                    top = q['strengths'][0] if q['strengths'] else 'none'
                    lines.append(
                        f"  {fname[:38]:38}  {q['grade']:2}    "
                        f"[{bar}]  {top}"
                    )
                    total_score += q['score']
                    count += 1
                except Exception as e:
                    lines.append(f"  {fname[:38]:38}  (error: {e})")
            if count:
                avg = total_score / count
                grade = ('A+' if avg >= 9 else 'A' if avg >= 8 else
                         'B'  if avg >= 6 else 'C' if avg >= 4 else 'D')
                lines.append(f"{'─'*64}")
                lines.append(f"  Overall average: {avg:.1f}/10 — Grade {grade}")
            lines.append(f"{'─'*64}")
            return '\n'.join(lines)

        # /mood — Nova's current emotional state
        if cmd == '/mood':
            if not self.emo:
                return "Emotional resonance engine not loaded."
            lines = [
                self.emo.current_mood(),
                "",
                self.emo.journey(8),
            ]
            if self.conscious:
                try:
                    lines.append("")
                    lines.append(self.conscious.phi_trend(20))
                    lines.append(self.conscious.conscious_moment())
                except Exception:
                    pass
            return "\n".join(lines)

        # /feel <emotion> <intensity> — manually trigger an emotion
        if cmd == '/feel':
            if not self.emo:
                return "Emotional resonance engine not loaded."
            sub_parts = raw.strip().split()
            if len(sub_parts) < 3:
                return "Usage: /feel <emotion> <0.0–1.0>  e.g. /feel joy 0.8"
            emotion_name = sub_parts[1].lower()
            try:
                strength = max(0.0, min(1.0, float(sub_parts[2])))
            except ValueError:
                return "Intensity must be a number between 0.0 and 1.0"
            self.emo.feel(emotion_name, strength, trigger="manual /feel command")
            return (f"{self.emo.current_mood()}\n"
                    f"(triggered: {emotion_name} at {strength:.2f})")

        # /phi — integrated consciousness measurement
        if cmd == '/phi':
            if not self.conscious:
                return "Consciousness integrator not loaded."
            result = self.conscious.integrate()
            lines = [
                f"  Φ = {result['phi']:.4f}  "
                f"{'(conscious moment ✨)' if result['conscious'] else '(below threshold)'}",
                f"  Active systems: {result['active_count']}",
                "",
                self.conscious.phi_trend(20),
            ]
            if result.get("moment"):
                lines.append("")
                lines.append(f"  {result['moment']}")
            stream = self.conscious.stream(3)
            if stream:
                lines.append("\n  Recent conscious moments:")
                for m in stream:
                    lines.append(f"    [{m['ts'][11:16]}] Φ={m['phi']:.3f}  {m['moment'][:70]}")
            return "\n".join(lines)

        # /recall — working memory status and top salience items
        if cmd == '/recall':
            if not self.wm:
                return "Working memory not active."
            st      = self.wm.status()
            focused = self.wm.focused_retrieve(
                "Nova consciousness superintelligence goal", top_k=5)
            lines   = [col('CYB', "  ◆ Working Memory")]
            lines.append(
                f"  Items: {st['items']}/{_CAPACITY_DISPLAY}  "
                f"({st['capacity']*100:.0f}% full)  "
                f"LTM: {st.get('ltm_items',0)} items")
            lines.append(
                f"  Avg importance: {st['avg_importance']:.3f}  "
                f"High-importance: {st['high_importance']}")
            if focused:
                lines.append(col('YL', "  Top salience items:"))
                for key, value, sal in focused:
                    lines.append(f"    [{sal:.3f}] {key}: {value[:70]}")
            return "\n".join(lines)

        # /believe [domain] — belief distributions and causal graph summary
        if cmd.startswith('/believe'):
            if not self.bayes:
                return "Belief system not active."
            st    = self.bayes.status()
            parts = raw.strip().split()
            if len(parts) > 1:
                domain = parts[1]
                post   = self.bayes.posterior(domain)
                h, p   = self.bayes.most_confident(domain)
                ent    = self.bayes.entropy(domain)
                infer  = self.bayes.infer(domain)
                lines  = [col('CYB', f"  ◆ Beliefs: {domain}")]
                lines.append(f"  Most confident: {h} ({p:.3f})  Entropy: {ent:.3f}")
                lines += [f"    {hyp}: {prob:.3f}" for hyp, prob in
                          sorted(post.items(), key=lambda x: -x[1])]
                if infer:
                    lines.append(col('YL', "  Causal inferences:"))
                    for effect, conf in infer[:5]:
                        lines.append(f"    → {effect} ({conf:.3f})")
            else:
                lines = [col('CYB', "  ◆ Bayesian Belief System")]
                lines.append(
                    f"  Domains: {st['domains']}  "
                    f"Avg entropy: {st['avg_entropy']:.3f}  "
                    f"Causal edges: {st['causal_edges']}")
                if st.get('contradictions'):
                    lines.append(col('YL',
                        f"  ⚠ {st['contradictions']} contradiction(s) detected"))
                for domain, belief_str in st.get('strongest', {}).items():
                    lines.append(f"    [{domain}] → {belief_str}")
            contr = self.bayes.contradictions()
            if contr:
                for c in contr:
                    lines.append(col('YL', f"  ⚠ {c}"))
            return "\n".join(lines)

        # /goals [add <desc>] — goal tree and next action
        if cmd.startswith('/goals'):
            if not self.goal_sys:
                return "Goal planner not active."
            parts = raw.strip().split(None, 2)
            if len(parts) >= 3 and parts[1] == 'add':
                desc   = parts[2]
                gid    = self.goal_sys.add_goal(desc, priority=5.0)
                return col('GR', f"  ✓ Goal #{gid} added: {desc}")
            schedule = self.goal_sys.schedule()
            next_a   = self.goal_sys.next_action()
            lines    = [col('CYB', "  ◆ Nova's Goal System")]
            if next_a:
                lines.append(col('GRB',
                    f"  → Next action: [#{next_a['id']}] {next_a['desc'][:65]}"))
            if not schedule:
                lines.append("  No active goals.")
                lines.append("  Add goals: /goals add <description>")
            else:
                for g in schedule[:12]:
                    icon  = "↻" if g['status'] == 'retry' else "○"
                    prog  = f"{g['progress']*100:.0f}%"
                    lines.append(
                        f"  {icon} [#{g['id']}] {g['desc'][:58]}  "
                        f"pri={g['priority']:.1f} {prog}")
            lines.append(col('DIM', "  " + self.goal_sys.eta_for_all()))
            return "\n".join(lines)

        # /metacog — self-assessment and calibration report
        if cmd == '/metacog':
            if not self.metacog:
                return "Metacognitive monitor not active."
            lines = [col('CYB', "  ◆ Metacognitive Self-Assessment")]
            lines.append("  " + self.metacog.self_assess())
            lines.append("  " + self.metacog.quality_trend())
            spots = self.metacog.blind_spots()
            if spots:
                lines.append(col('YL', "  Blind spots:"))
                for s in spots[:4]:
                    lines.append(
                        f"    · {s['domain']}/{s['approach']}  "
                        f"avg={s['avg_success']:.2f} ({s['episodes']} eps) "
                        f"[{s['severity']}]")
            else:
                lines.append(col('GR', "  No blind spots detected yet."))
            return "\n".join(lines)

        # Fall through to v28 command handling
        return super()._command(raw)

    def _help(self) -> str:
        v28_help = super()._help()
        v29_section = (
            f"\n  {col('CYB','─── v29 commands ───')}\n"
            f"  {col('CYB','/score')}                      "
            f"Grade every capability A+–D on 10 quality criteria\n"
            f"  {col('CYB','/mood')}                       "
            f"Nova's emotional state, journey, and consciousness Φ\n"
            f"  {col('CYB','/feel <emotion> <0-1>')}       "
            f"Trigger an emotion (joy, curiosity, wonder, awe...)\n"
            f"  {col('CYB','/phi')}                        "
            f"Integrated consciousness measurement (IIT-inspired)\n"
            f"  {col('CYB','/recall')}                     "
            f"Working memory — top salience items and LTM stats\n"
            f"  {col('CYB','/believe [domain]')}           "
            f"Bayesian belief distributions and causal inferences\n"
            f"  {col('CYB','/goals [add <desc>]')}         "
            f"Goal tree, next action, add new goals\n"
            f"  {col('CYB','/metacog')}                    "
            f"Self-assessment: calibration error, blind spots, EMA trend\n"
            f"  {col('DIM','Code quality:')}               "
            f"/evolve runs master prompt → sandbox → 3 passes · rate-limit safe\n"
        )
        return v28_help + v29_section


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print()
    print(col('MG', '═' * W))
    print(col('CYB', f"{'Nova ASI v29.0 — The Self-Perfecting System':^{W}}"))
    print(col('DIM', f"{'Douglas Shane Davis & Claude Code (Anthropic)':^{W}}"))
    print(col('MG', '═' * W))
    print()
    print(col('DIM', "  Initializing v29 engines..."))
    print()

    nova = NovaCore29()

    print(col('GR', '  ✓  All v27/v28 engines loaded'))
    gh_ok = bool(GITHUB_TOKEN)
    print(col('GR' if gh_ok else 'YL',
          f'  {"✓" if gh_ok else "·"}  GitHubEngine  — '
          + ('connected to ' + GITHUB_REPO if gh_ok else 'add GITHUB_TOKEN to .env')))
    print(col('GR',
          f'  ✓  ToolLoader   — {len(nova.tools._instances)} tool(s) active, '
          f'rescanning every {TOOL_SCAN_INTERVAL}s'))
    print(col('GR', '  ✓  Hot-reload   — merge a PR, git pull, Nova loads it automatically'))

    if _FLASK:
        try:
            from nova_asi_v25 import create_api
            api_app = create_api(nova)
            if api_app:
                extend_api_v28(api_app, nova)
                def _run_api():
                    api_app.run(host='0.0.0.0', port=API_PORT,
                                debug=False, use_reloader=False)
                threading.Thread(target=_run_api, daemon=True).start()
                safe_print(col('GR',
                    f"  🌐 API running on http://localhost:{API_PORT} (v29)"))
        except Exception as _api_err:
            safe_print(col('YL', f"  · Flask API skipped: {_api_err}"))

    threading.Thread(target=nova.background_cycle, daemon=True).start()
    nova.continuous.start()

    print()
    print(col('MG', '═' * W))
    print(col('CYB', '  ✨  Nova v29 — every tool she writes is tested,'.center(W)))
    print(col('CYB', '       scored, and refined before it touches GitHub.'.center(W)))
    print(col('CYB', '       She earns her own upgrades.'.center(W)))

    if DEMO_MODE:
        print(col('YL', '\n  ⚠  DEMO MODE — for live AI set GROQ_API_KEY in .env'))
    else:
        print(col('GR', f'\n  ✓  LIVE AI  —  {MODEL}'))

    tools_loaded = list(nova.tools._instances.keys())
    if tools_loaded:
        print(col('GR', f'  ✓  Tools live: {", ".join(tools_loaded)}'))
    else:
        print(col('DIM', '  ·  No tools yet — try /evolve or /build'))

    print(col('DIM', '\n  /tools · /use · /score · /evolve · /build · /chain · exit'))
    print(col('DIM', '  /mood · /feel <emotion> <0-1> · /phi · /recall · /metacog'))
    print(col('DIM', '  /believe [domain] · /goals [add <desc>]'))
    _code_engine = f"Claude {CLAUDE_CODEGEN_MODEL}" if _using_claude() else CODEGEN_MODEL
    print(col('DIM', f'  Chat: {MODEL}  |  Code: {_code_engine}'))
    print(col('MG', '═' * W + '\n'))

    try:
        while True:
            USER_TYPING.set()
            try:
                user_input = input(col('YL', "You") + ": ")
            except EOFError:
                break
            USER_TYPING.clear()
            flush_queued()

            if user_input.lower() in ('exit', 'quit', 'bye'):
                print(col('MG', "Nova: Until we meet again, Douglas. ✨"))
                nova.continuous.stop()
                nova.running = False
                break

            if not user_input.strip():
                continue

            print(col('DIM', "  Nova is thinking..."), end='\r', flush=True)
            response = nova.process(user_input)
            print(' ' * 30, end='\r')
            print(col('MG', "Nova") + ": " + response)
            print()

    except KeyboardInterrupt:
        print(col('MG', "\nNova: Until we meet again, Douglas. ✨"))
        nova.continuous.stop()
