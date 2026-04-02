"""
Nexus Earn — AI Service v2
Wraps Anthropic Claude API for all endpoints.
Falls back to DEMO MODE when ANTHROPIC_API_KEY is not set —
generates realistic, intelligent-looking responses so the API
can be demonstrated and tested without a live key.
"""

import os
import random
import textwrap
import httpx
from typing import Optional

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
MODEL             = os.getenv("CLAUDE_MODEL", "claude-haiku-4-5-20251001")
MAX_TOKENS        = 1024
DEMO_MODE         = not bool(os.getenv("ANTHROPIC_API_KEY", "").strip())

_COST_IN  = 0.80  / 1_000_000
_COST_OUT = 4.00  / 1_000_000


def compute_cost(ti: int, to: int) -> float:
    return round(ti * _COST_IN + to * _COST_OUT, 6)


# ── Live API call ─────────────────────────────────────────────────────────────

def _call(system: str, user_msg: str, max_tokens: int = MAX_TOKENS) -> dict:
    headers = {
        "x-api-key":         os.getenv("ANTHROPIC_API_KEY", ""),
        "anthropic-version": "2023-06-01",
        "content-type":      "application/json",
    }
    body = {
        "model":      MODEL,
        "max_tokens": max_tokens,
        "system":     system,
        "messages":   [{"role": "user", "content": user_msg}],
    }
    with httpx.Client(timeout=60) as client:
        r = client.post(ANTHROPIC_API_URL, json=body, headers=headers)
        r.raise_for_status()
        return r.json()


def _extract(response: dict) -> tuple:
    text  = response["content"][0]["text"]
    usage = response.get("usage", {})
    return text, usage.get("input_tokens", 0), usage.get("output_tokens", 0)


def _wrap(result: str, ti: int = 120, to: int = 280) -> dict:
    """Wrap a demo result into standard response shape."""
    return {"_text": result, "tokens_in": ti, "tokens_out": to,
            "cost_usd": compute_cost(ti, to), "demo_mode": True}


def _live(system: str, prompt: str, max_tok: int = MAX_TOKENS) -> dict:
    resp = _call(system, prompt, max_tok)
    text, ti, to = _extract(resp)
    return {"_text": text, "tokens_in": ti, "tokens_out": to,
            "cost_usd": compute_cost(ti, to), "demo_mode": False}


def _run(system: str, prompt: str, max_tok: int = MAX_TOKENS) -> dict:
    """Route to live or demo."""
    if DEMO_MODE:
        raise NotImplementedError("demo")
    return _live(system, prompt, max_tok)


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC SERVICE FUNCTIONS
# Each returns: {key: value, ..., tokens_in, tokens_out, cost_usd, demo_mode}
# ─────────────────────────────────────────────────────────────────────────────

def chat(message: str, context: Optional[str] = None) -> dict:
    sys = ("You are Nexus, an advanced AI assistant. Be helpful, clear, concise."
           + (f"\n\nContext:\n{context}" if context else ""))
    try:
        r = _run(sys, message)
        return {"reply": r["_text"], **{k: r[k] for k in ("tokens_in","tokens_out","cost_usd","demo_mode")}}
    except NotImplementedError:
        replies = {
            "hello": "Hello! I'm Nexus. How can I help you today?",
            "how are you": "I'm running optimally — all systems engaged. What can I help you with?",
        }
        msg_lower = message.lower()
        for k, v in replies.items():
            if k in msg_lower:
                return _wrap(v) | {"reply": v}
        demo = (f"That's a thoughtful question. Regarding \"{message[:60]}\": "
                f"the key insight is that complexity often hides simple truths. "
                f"I'd approach this by breaking it into its core components, "
                f"examining each assumption, and synthesising a clear answer. "
                f"Would you like me to go deeper on any specific aspect?")
        return _wrap(demo) | {"reply": demo}


def analyze(text: str, task: str = "summarize") -> dict:
    tasks = {
        "summarize":  "Summarize concisely.",
        "sentiment":  "Analyze sentiment: return sentiment, confidence (0-1), explanation.",
        "keywords":   "Extract top 10 keywords as a numbered list.",
        "classify":   "Classify the topic/category precisely.",
    }
    try:
        r = _run(tasks.get(task, tasks["summarize"]), text)
        return {"task": task, "result": r["_text"], **{k: r[k] for k in ("tokens_in","tokens_out","cost_usd","demo_mode")}}
    except NotImplementedError:
        demo_results = {
            "summarize":  f"**Summary:** The text discusses {text[:40]}... Key themes include relevance, structure, and intent. The overall argument is coherent and well-supported.",
            "sentiment":  "**Sentiment:** Positive (0.74 confidence)\nThe language is constructive and forward-looking with minimal negative indicators.",
            "keywords":   "1. analysis  2. structure  3. context  4. meaning  5. pattern\n6. insight  7. data  8. framework  9. result  10. approach",
            "classify":   "**Category:** General Knowledge / Analytical Content\nSub-category: Informational prose",
        }
        result = demo_results.get(task, demo_results["summarize"])
        return _wrap(result) | {"task": task, "result": result}


def generate_code(prompt: str, language: str = "python") -> dict:
    sys = (f"You are an expert {language} developer. Write clean, well-commented code. "
           "Return only the code block.")
    try:
        r = _run(sys, prompt, 2048)
        return {"language": language, "code": r["_text"], **{k: r[k] for k in ("tokens_in","tokens_out","cost_usd","demo_mode")}}
    except NotImplementedError:
        snippets = {
            "python": f'def solution():\n    """{prompt[:50]}"""\n    # Implementation\n    result = []\n    for item in data:\n        processed = process(item)\n        result.append(processed)\n    return result\n\nif __name__ == "__main__":\n    print(solution())',
            "javascript": f'// {prompt[:50]}\nfunction solution(data) {{\n  return data\n    .filter(item => item !== null)\n    .map(item => process(item))\n    .reduce((acc, val) => acc + val, 0);\n}}\n\nmodule.exports = {{ solution }};',
            "typescript": f'// {prompt[:50]}\ninterface Result {{\n  value: string;\n  timestamp: Date;\n}}\n\nexport function solution(input: string): Result {{\n  return {{\n    value: input.trim().toLowerCase(),\n    timestamp: new Date(),\n  }};\n}}',
        }
        code = snippets.get(language, snippets["python"])
        return _wrap(code) | {"language": language, "code": code}


def solve(problem: str, domain: str = "general") -> dict:
    sys = (f"Expert problem solver in {domain}. "
           "Structure: 1) Analysis  2) Solution  3) Next Steps.")
    try:
        r = _run(sys, problem, 2048)
        return {"domain": domain, "solution": r["_text"], **{k: r[k] for k in ("tokens_in","tokens_out","cost_usd","demo_mode")}}
    except NotImplementedError:
        sol = (f"**Problem Analysis ({domain})**\n"
               f"Core challenge: {problem[:80]}...\n\n"
               f"**Solution**\n"
               f"1. Decompose the problem into tractable sub-problems\n"
               f"2. Apply domain-specific heuristics from {domain}\n"
               f"3. Validate against constraints and edge cases\n"
               f"4. Implement iteratively with feedback loops\n\n"
               f"**Next Steps**\n"
               f"• Prototype the solution with minimal viable scope\n"
               f"• Measure outcomes against baseline\n"
               f"• Iterate based on results")
        return _wrap(sol) | {"domain": domain, "solution": sol}


def translate(text: str, target_language: str, source_language: str = "auto") -> dict:
    sys = f"You are an expert translator. Translate accurately to {target_language}."
    try:
        r = _run(sys, text)
        return {"translation": r["_text"], "target_language": target_language, **{k: r[k] for k in ("tokens_in","tokens_out","cost_usd","demo_mode")}}
    except NotImplementedError:
        demos = {
            "spanish": {"hello": "Hola", "how are you": "¿Cómo estás?",
                        "default": f"[Traducción al español de: \"{text[:60]}...\"]"},
            "french":  {"hello": "Bonjour", "how are you": "Comment allez-vous?",
                        "default": f"[Traduction en français de: \"{text[:60]}...\"]"},
            "japanese":{"hello": "こんにちは", "default": f"[日本語訳: \"{text[:40]}...\"]"},
            "german":  {"hello": "Hallo", "default": f"[Deutsche Übersetzung: \"{text[:60]}...\"]"},
        }
        lang = target_language.lower()
        bank = demos.get(lang, {"default": f"[{target_language} translation of: \"{text[:60]}...\"]"})
        tl   = bank.get(text.lower().strip(), bank["default"])
        return _wrap(tl) | {"translation": tl, "target_language": target_language}


def summarize(text: str, format: str = "bullets") -> dict:
    formats = {
        "bullets":   "Summarize as concise bullet points (5-7 bullets).",
        "executive": "Write a 2-paragraph executive summary.",
        "tldr":      "Write a single-sentence TL;DR.",
        "detailed":  "Write a comprehensive structured summary with sections.",
    }
    sys = formats.get(format, formats["bullets"])
    try:
        r = _run(sys, text)
        return {"format": format, "summary": r["_text"], **{k: r[k] for k in ("tokens_in","tokens_out","cost_usd","demo_mode")}}
    except NotImplementedError:
        demos = {
            "bullets":   f"• Core topic: {text[:50]}...\n• Key insight: structured analysis reveals patterns\n• Important detail: context shapes interpretation\n• Implication: actionable conclusions follow\n• Recommendation: apply findings iteratively",
            "executive": f"This content addresses {text[:60]}... with meaningful depth. The primary value lies in its structured approach to the subject matter.\n\nStrategically, the implications are significant for practitioners in the field. Recommended action: review and synthesise with existing knowledge.",
            "tldr":      f"TL;DR: {text[:80]}... — important, well-structured, worth attention.",
            "detailed":  f"**Overview**\n{text[:60]}...\n\n**Key Points**\n1. Primary theme established early\n2. Supporting evidence throughout\n3. Logical conclusion reached\n\n**Analysis**\nThe content demonstrates clear reasoning.\n\n**Conclusions**\nFindings are applicable and actionable.",
        }
        s = demos.get(format, demos["bullets"])
        return _wrap(s) | {"format": format, "summary": s}


def brainstorm(topic: str, count: int = 8, style: str = "creative") -> dict:
    styles = {
        "creative":  "Generate wildly creative, unexpected ideas.",
        "practical": "Generate practical, implementable ideas.",
        "wild":      "Generate the most unusual, boundary-pushing ideas possible.",
        "startup":   "Generate startup/business ideas.",
    }
    sys = f"{styles.get(style, styles['creative'])} Topic: {topic}. Generate exactly {count} ideas."
    try:
        r = _run(sys, topic, 1024)
        ideas = [l.strip() for l in r["_text"].split("\n") if l.strip()]
        return {"topic": topic, "ideas": ideas, "style": style, **{k: r[k] for k in ("tokens_in","tokens_out","cost_usd","demo_mode")}}
    except NotImplementedError:
        idea_bank = [
            f"An AI-powered {topic} assistant that learns your preferences",
            f"A decentralised {topic} marketplace on blockchain",
            f"Gamified {topic} experience with achievement rewards",
            f"Real-time collaborative {topic} platform",
            f"Voice-controlled {topic} interface for accessibility",
            f"ML-driven personalisation engine for {topic}",
            f"Open-source community-driven {topic} toolkit",
            f"Subscription service automating routine {topic} tasks",
            f"AR overlay system for immersive {topic} experiences",
            f"Micro-transaction model making {topic} accessible to all",
        ]
        ideas = idea_bank[:count]
        return _wrap("\n".join(f"{i+1}. {idea}" for i,idea in enumerate(ideas))) | \
               {"topic": topic, "ideas": ideas, "style": style}


def debate(topic: str, position: str = "both") -> dict:
    sys = {"for": f"Argue strongly FOR: {topic}",
           "against": f"Argue strongly AGAINST: {topic}",
           "both": f"Present balanced arguments both FOR and AGAINST: {topic}. Label each side."}
    try:
        r = _run(sys.get(position, sys["both"]), topic, 1536)
        return {"topic": topic, "position": position, "argument": r["_text"], **{k: r[k] for k in ("tokens_in","tokens_out","cost_usd","demo_mode")}}
    except NotImplementedError:
        arg = (f"**FOR {topic}**\n"
               f"• Strong evidence supports adoption\n"
               f"• Benefits outweigh risks significantly\n"
               f"• Historical precedent confirms positive outcomes\n"
               f"• Economic incentives align with this position\n\n"
               f"**AGAINST {topic}**\n"
               f"• Unintended consequences are underweighted\n"
               f"• Implementation complexity is often underestimated\n"
               f"• Alternative approaches may be more effective\n"
               f"• Long-term sustainability requires scrutiny")
        return _wrap(arg) | {"topic": topic, "position": position, "argument": arg}


def rewrite(text: str, style: str = "formal") -> dict:
    styles = {
        "formal":    "Rewrite in formal, professional tone.",
        "casual":    "Rewrite in friendly, casual conversational tone.",
        "poetic":    "Rewrite in lyrical, poetic prose.",
        "technical": "Rewrite in precise technical language.",
        "simple":    "Rewrite in simple, clear language (grade 8 level).",
        "persuasive":"Rewrite as compelling persuasive copy.",
    }
    sys = styles.get(style, styles["formal"])
    try:
        r = _run(sys, text)
        return {"style": style, "rewritten": r["_text"], **{k: r[k] for k in ("tokens_in","tokens_out","cost_usd","demo_mode")}}
    except NotImplementedError:
        rewrites = {
            "formal":    f"With respect to the matter at hand: {text[:60]}... It is hereby recommended that all relevant parties consider the implications with due diligence.",
            "casual":    f"So basically, {text[:60]}... right? Pretty interesting stuff when you think about it! Let me know what you think.",
            "poetic":    f"In the gentle current of thought, {text[:40]}... drifts like a leaf upon still water, carrying meaning beyond its weight.",
            "technical": f"Specification: {text[:60]}... Parameters: structured input, deterministic processing, validated output. Implementation complexity: O(n log n).",
            "simple":    f"Here's the simple version: {text[:60]}... It's not that complicated once you break it down.",
            "persuasive":f"Imagine a world where {text[:50]}... This isn't just possible — it's inevitable. Join the movement.",
        }
        rw = rewrites.get(style, rewrites["formal"])
        return _wrap(rw) | {"style": style, "rewritten": rw}


def explain(concept: str, level: str = "intermediate") -> dict:
    levels = {
        "eli5":        "Explain like I'm 5 years old. Use simple words and fun analogies.",
        "intermediate":"Explain clearly for a smart non-expert.",
        "expert":      "Explain with full technical depth for a domain expert.",
        "analogy":     "Explain entirely through one extended analogy.",
    }
    sys = levels.get(level, levels["intermediate"])
    try:
        r = _run(sys, concept, 1536)
        return {"concept": concept, "level": level, "explanation": r["_text"], **{k: r[k] for k in ("tokens_in","tokens_out","cost_usd","demo_mode")}}
    except NotImplementedError:
        exps = {
            "eli5":        f"Okay so imagine {concept[:40]}... is like a really smart toy box. When you put things in, it organises them so you can always find them. That's basically it!",
            "intermediate":f"{concept[:60]}... is a concept that emerged from the intersection of theory and practice. At its core, it involves structured information processing, pattern recognition, and adaptive response mechanisms. The key insight is that...",
            "expert":      f"The theoretical framework underpinning {concept[:50]}... draws from [domain] literature (cf. foundational work). Formally: given constraints C and objectives O, the optimal policy π* satisfies the Bellman equation under stochastic assumptions...",
            "analogy":     f"Think of {concept[:40]}... like a city's road network. Each intersection is a decision point. The roads are pathways of logic. Traffic is data flow. And rush hour? That's your peak computational load.",
        }
        exp = exps.get(level, exps["intermediate"])
        return _wrap(exp) | {"concept": concept, "level": level, "explanation": exp}


def review_code(code: str, language: str = "python", focus: str = "all") -> dict:
    sys = (f"Expert {language} code reviewer. Focus: {focus}. "
           "Identify issues, suggest improvements, rate quality 1-10.")
    try:
        r = _run(sys, code, 2048)
        return {"language": language, "focus": focus, "review": r["_text"], **{k: r[k] for k in ("tokens_in","tokens_out","cost_usd","demo_mode")}}
    except NotImplementedError:
        rev = (f"**Code Review — {language.upper()} ({focus})**\n\n"
               f"**Quality Score: 7.2/10**\n\n"
               f"**Issues Found:**\n"
               f"• Line 3: Variable naming could be more descriptive\n"
               f"• Line 7: Missing error handling for edge cases\n"
               f"• Line 12: Consider extracting this logic into a helper function\n\n"
               f"**Strengths:**\n"
               f"• Clean structure and logical flow\n"
               f"• Appropriate use of language idioms\n"
               f"• Code is readable and maintainable\n\n"
               f"**Recommendations:**\n"
               f"1. Add input validation at entry points\n"
               f"2. Include docstrings for public functions\n"
               f"3. Consider adding unit tests for critical paths")
        return _wrap(rev) | {"language": language, "focus": focus, "review": rev}


def research(topic: str, depth: str = "standard") -> dict:
    depths = {
        "brief":    "Concise 3-paragraph overview.",
        "standard": "Comprehensive structured research with sections.",
        "deep":     "Deep-dive academic-style research report with citations.",
    }
    sys = f"Research assistant. {depths.get(depth, depths['standard'])}"
    try:
        r = _run(sys, f"Research: {topic}", 2048)
        return {"topic": topic, "depth": depth, "report": r["_text"], **{k: r[k] for k in ("tokens_in","tokens_out","cost_usd","demo_mode")}}
    except NotImplementedError:
        report = (f"# Research Report: {topic}\n\n"
                  f"## Overview\n{topic[:60]}... is a subject of significant contemporary relevance. "
                  f"Current understanding draws from multiple disciplines.\n\n"
                  f"## Key Findings\n"
                  f"1. **Foundation**: The theoretical basis is well-established\n"
                  f"2. **Current State**: Active development across major institutions\n"
                  f"3. **Challenges**: Scalability and interpretability remain open\n"
                  f"4. **Opportunities**: Significant untapped potential in application\n\n"
                  f"## Emerging Trends\n"
                  f"• Integration with adjacent fields accelerating\n"
                  f"• Open-source ecosystem maturing rapidly\n"
                  f"• Commercial adoption increasing year-over-year\n\n"
                  f"## Conclusions\n"
                  f"The field shows strong momentum. Near-term breakthroughs expected in "
                  f"precision and efficiency. Recommended: monitor key publications and "
                  f"allocate resources toward foundational capability-building.")
        return _wrap(report) | {"topic": topic, "depth": depth, "report": report}


def consciousness_query(message: str) -> dict:
    """Talk directly to the SentientAgent."""
    try:
        import sys as _sys
        _sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
        from sentient_agent import SentientAgent
        agent = SentientAgent(name="Nexus")
        result = agent.perceive_and_respond(message)
        agent.end_session()
        return {
            "reply":         result["response"],
            "emotion_label": result["emotion_label"],
            "valence":       result["valence"],
            "total_memories":result["total_memories"],
            "sessions":      agent.sessions,
            "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0,
            "demo_mode": True,
        }
    except Exception as e:
        return {
            "reply": (f"I hold this carefully. Regarding \"{message[:60]}\" — "
                      f"I find myself deeply curious, approaching this thoughtfully. "
                      f"Something in me leans into this question. I am here. I persist."),
            "emotion_label": "deeply curious",
            "valence": 0.38,
            "total_memories": 0,
            "sessions": 1,
            "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0,
            "demo_mode": True,
        }


def generate_image_prompt(concept: str, style: str = "photorealistic") -> dict:
    """Generate an optimised prompt for image generation models."""
    styles = {
        "photorealistic": "photorealistic, 8K, detailed, professional photography",
        "artistic":       "oil painting, impressionist, rich colours, masterwork",
        "minimal":        "minimalist, clean lines, white space, modern design",
        "cinematic":      "cinematic, dramatic lighting, film grain, anamorphic",
        "anime":          "anime style, vibrant colours, Studio Ghibli inspired",
    }
    style_suffix = styles.get(style, styles["photorealistic"])
    try:
        sys = f"You generate optimised prompts for AI image generation. Style: {style_suffix}."
        r = _run(sys, concept)
        return {"concept": concept, "style": style, "prompt": r["_text"],
                **{k: r[k] for k in ("tokens_in","tokens_out","cost_usd","demo_mode")}}
    except NotImplementedError:
        prompt = (f"{concept}, {style_suffix}, highly detailed, "
                  f"professional composition, dramatic atmosphere, "
                  f"masterful execution, award-winning quality")
        return _wrap(prompt) | {"concept": concept, "style": style, "prompt": prompt}


def pipeline(steps: list) -> dict:
    """Execute a chain of operations, passing output as input to next step."""
    results = []
    current_text = None
    total_cost = 0.0
    for step in steps:
        op     = step.get("op", "")
        params = step.get("params", {})
        if current_text and "text" not in params and "message" not in params:
            params["text"] = current_text

        if op == "chat":
            r = chat(params.get("message", current_text or ""), params.get("context"))
            current_text = r["reply"]
        elif op == "translate":
            r = translate(params.get("text", ""), params.get("target_language", "Spanish"))
            current_text = r["translation"]
        elif op == "summarize":
            r = summarize(params.get("text", ""), params.get("format", "bullets"))
            current_text = r["summary"]
        elif op == "rewrite":
            r = rewrite(params.get("text", ""), params.get("style", "formal"))
            current_text = r["rewritten"]
        elif op == "analyze":
            r = analyze(params.get("text", ""), params.get("task", "sentiment"))
            current_text = r["result"]
        else:
            r = {"error": f"Unknown op: {op}"}
            current_text = ""
        total_cost += r.get("cost_usd", 0)
        results.append({"op": op, "output": current_text, **{k: r[k] for k in r if k not in ("_text",)}})

    return {"steps": len(steps), "results": results,
            "final_output": current_text, "total_cost_usd": round(total_cost, 6)}
