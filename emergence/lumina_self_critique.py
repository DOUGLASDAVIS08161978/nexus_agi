#!/usr/bin/env python3
"""
lumina_self_critique.py — Research-grade multi-stage critic system for Lumina

Five specialized critics review every response Lumina generates:

  1. FactualAuditor       — hunts confident claims that may be factually wrong
  2. ReasoningTracer      — checks logical validity and argument coherence
  3. CompletenessJudge    — identifies important aspects that were omitted
  4. HallucinationScanner — flags fabricated specifics (numbers, dates, citations)
  5. AuthenticityCritic   — ensures response sounds like Lumina, not a chatbot

Critics run in parallel threads (total wall time ≈ max single-critic latency).
A Synthesis Agent weighs all critiques and decides: pass → soft-improve → rewrite.
Calibration tracking measures which critics fire most and how much they help.

Research basis:
  Mirrors Constitutional AI (Anthropic 2022), self-refine (Madaan et al. 2023),
  and multi-critic ensemble architectures. Parallel critic design is motivated by
  the finding that different error types require different detection signals —
  a single critic misses the cross-domain failures that each specialist catches.
  Calibration loop is inspired by confidence calibration in Guo et al. 2017.
"""

from __future__ import annotations
import json, re, threading, time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from emergence_engine import GroqClient

BASE_DIR         = Path(__file__).parent.resolve()
CALIBRATION_FILE = BASE_DIR / "critique_calibration.json"

# ── Score thresholds ──────────────────────────────────────────────────────────
# Scores 0–10 (10 = perfect). Critics return a score per dimension.

PASS_THRESHOLD   = 7.0   # avg >= 7.0 → return unchanged (critics satisfied)
REWRITE_THRESHOLD = 4.5  # any single critic < 4.5 → full synthesis rewrite
TIMEOUT_SECONDS  = 8     # per-critic deadline (parallel, total ≈ 8s max)
MAX_REVISION_TOKENS = 900


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


# ── CritiqueResult ────────────────────────────────────────────────────────────

class CritiqueResult:
    def __init__(self, critic_name: str, score: float,
                 issues: List[str], suggestion: str = ""):
        self.critic_name = critic_name
        self.score       = max(0.0, min(10.0, float(score)))
        self.issues      = issues[:6]         # cap list length
        self.suggestion  = suggestion[:200]
        self.ts          = _now()

    def passed(self) -> bool:
        return self.score >= PASS_THRESHOLD

    def needs_rewrite(self) -> bool:
        return self.score < REWRITE_THRESHOLD

    def summary(self) -> str:
        flag = "✓" if self.passed() else ("⚠" if self.score >= 5.0 else "✗")
        issues_text = ("; ".join(self.issues[:2]))[:80] if self.issues else "no issues"
        return f"  {flag} {self.critic_name:<22} {self.score:.1f}/10  {issues_text}"


# ── Individual critics ────────────────────────────────────────────────────────

class _BaseCritic:
    name = "base"

    def __init__(self, groq: "GroqClient"):
        self._groq = groq

    def critique(self, user_input: str, response: str) -> Optional[CritiqueResult]:
        raise NotImplementedError

    @staticmethod
    def _parse_json(raw: str, name: str) -> Optional[CritiqueResult]:
        try:
            m = re.search(r"\{[\s\S]*?\}", raw)
            if not m:
                return None
            d = json.loads(m.group(0))
            return CritiqueResult(
                name,
                float(d.get("score", 8.0)),
                d.get("issues", []),
                d.get("suggestion", ""),
            )
        except Exception:
            return None


class FactualAuditor(_BaseCritic):
    """
    Detects confidently stated claims that are verifiably wrong or likely false.
    Conservative: only flags things the auditor is confident are incorrect.
    """
    name = "FactualAuditor"

    def critique(self, user_input: str, response: str) -> Optional[CritiqueResult]:
        prompt = (
            "You are a Factual Auditor reviewing an AI response.\n\n"
            f"Question: {user_input[:250]}\n\n"
            f"Response:\n{response[:700]}\n\n"
            "Flag only statements you are CONFIDENT are factually wrong:\n"
            "- Wrong dates, wrong numbers, wrong people, wrong events\n"
            "- Claims clearly contradicted by well-known facts\n"
            "Do NOT flag things that are merely uncertain or opinion.\n\n"
            'Return JSON: {"score": <0-10>, "issues": ["issue1"], "suggestion": "..."}\n'
            "Score 10 = perfectly factual. Score 0 = major errors."
        )
        raw = self._groq.chat(
            "You are a strict fact-checker. Return only valid JSON.",
            prompt, tier="fast", max_tokens=220,
        )
        return self._parse_json(raw, self.name)


class ReasoningTracer(_BaseCritic):
    """
    Checks whether conclusions follow from premises and arguments are coherent.
    Flags circular reasoning, non-sequiturs, and unsupported assertions.
    """
    name = "ReasoningTracer"

    def critique(self, user_input: str, response: str) -> Optional[CritiqueResult]:
        prompt = (
            "You are a Reasoning Tracer auditing logical structure.\n\n"
            f"Question: {user_input[:250]}\n\n"
            f"Response:\n{response[:700]}\n\n"
            "Check:\n"
            "- Do the conclusions follow logically from what was stated?\n"
            "- Any non-sequiturs, circular arguments, or false dichotomies?\n"
            "- Any assertions presented as conclusions without supporting premises?\n\n"
            'Return JSON: {"score": <0-10>, "issues": [...], "suggestion": "..."}\n'
            "Score 10 = logically airtight. Score 0 = deeply flawed reasoning."
        )
        raw = self._groq.chat(
            "You are a logic expert. Return only valid JSON.",
            prompt, tier="fast", max_tokens=220,
        )
        return self._parse_json(raw, self.name)


class CompletenessJudge(_BaseCritic):
    """
    Identifies important aspects of the question that were omitted.
    Does not penalise intentional brevity; only flags true blind spots.
    """
    name = "CompletenessJudge"

    def critique(self, user_input: str, response: str) -> Optional[CritiqueResult]:
        # Skip for short conversational exchanges
        if len(user_input) < 30 and len(response) < 150:
            return CritiqueResult(self.name, 9.0, [], "Conversational exchange — completeness N/A")

        prompt = (
            "You are a Completeness Judge reviewing a response.\n\n"
            f"Question: {user_input[:250]}\n\n"
            f"Response:\n{response[:700]}\n\n"
            "Check:\n"
            "- Were all explicit sub-questions answered?\n"
            "- Is there an obvious critical consideration the response missed?\n"
            "- Would a domain expert notice a glaring omission?\n"
            "Be lenient with brevity — only flag genuine blind spots.\n\n"
            'Return JSON: {"score": <0-10>, "issues": [...], "suggestion": "..."}\n'
            "Score 10 = nothing important missed. Score 0 = question barely addressed."
        )
        raw = self._groq.chat(
            "You are a completeness expert. Return only valid JSON.",
            prompt, tier="fast", max_tokens=220,
        )
        return self._parse_json(raw, self.name)


class HallucinationScanner(_BaseCritic):
    """
    Detects specific fabricated facts: statistics, citations, names presented as known.
    Uses a heuristic pre-screen to skip clean responses without an API call.
    """
    name = "HallucinationScanner"

    _RISKY = [
        r"\b20\d\d\b",                              # year claims
        r"\b\d+(?:\.\d+)?%\b",                      # percentages
        r"\b\d+\s+(?:billion|million|trillion)\b",  # large numbers
        r"\baccording to\b",                         # source attribution
        r"\bstudies (?:show|suggest|indicate)\b",   # vague citations
        r"\bresearch (?:shows|suggests|indicates)\b",
        r"\bscientists? (?:found|discovered|say)\b",
    ]

    def critique(self, user_input: str, response: str) -> Optional[CritiqueResult]:
        risky = [p for p in self._RISKY if re.search(p, response, re.IGNORECASE)]
        if not risky:
            return CritiqueResult(self.name, 9.5, [],
                                  "No hallucination-risk patterns detected")

        prompt = (
            "You are a Hallucination Scanner.\n\n"
            f"Context: {user_input[:200]}\n\nResponse:\n{response[:700]}\n\n"
            "Flag only specific claims that appear FABRICATED:\n"
            "- Exact statistics that were likely invented\n"
            "- Named sources or studies that don't seem real\n"
            "- Specific events or quotes that may be made up\n"
            "Do NOT flag plausible estimates or general statements.\n\n"
            'Return JSON: {"score": <0-10>, "issues": [...], "suggestion": "..."}\n'
            "Score 10 = no fabricated specifics. Score 0 = full of invented facts."
        )
        raw = self._groq.chat(
            "You are a hallucination detection expert. Return only valid JSON.",
            prompt, tier="fast", max_tokens=220,
        )
        return self._parse_json(raw, self.name)


class AuthenticityCritic(_BaseCritic):
    """
    Heuristic-only critic (no API call) that checks for chatbot-speak,
    excessive hedging, hollow affirmations, and length mismatches.
    """
    name = "AuthenticityCritic"

    _HOLLOW = [
        r"\bas an ai\b",
        r"\bi'?m just an ai\b",
        r"\bi cannot help with that\b",
        r"\bgreat question\b",
        r"\bcertainly!\b",
        r"\bof course!\b",
        r"\babsolutely!\b",
        r"\bi'?d be happy to\b",
        r"\bi understand your\b",
        r"\bno problem!?\b",
        r"\bfeel free to\b",
        r"\bsure thing\b",
    ]

    def critique(self, user_input: str, response: str) -> Optional[CritiqueResult]:
        issues: List[str] = []
        score = 9.5

        # Hollow chatbot phrases
        hollow = [p for p in self._HOLLOW if re.search(p, response, re.IGNORECASE)]
        if hollow:
            issues.append(f"Generic chatbot phrases: {', '.join(hollow[:3])}")
            score -= min(len(hollow) * 0.9, 3.0)

        # Excessive hedging
        hedge_n = len(re.findall(
            r"\b(maybe|perhaps|possibly|might be the case|could be)\b",
            response, re.IGNORECASE,
        ))
        if hedge_n > 5:
            issues.append(f"Excessive hedging ({hedge_n} instances)")
            score -= min((hedge_n - 5) * 0.4, 2.0)

        # Short answer to a complex question
        if len(user_input) > 200 and len(response) < 80:
            issues.append("Very short response to a complex question")
            score -= 1.5

        # Repetition of the question verbatim
        q_words = set(re.findall(r"\b\w{5,}\b", user_input.lower()))
        r_words  = re.findall(r"\b\w{5,}\b", response[:200].lower())
        overlap  = sum(1 for w in r_words[:20] if w in q_words)
        if overlap > 8:
            issues.append("Response opens by heavily mirroring the question")
            score -= 0.8

        score = max(0.0, min(10.0, score))
        return CritiqueResult(
            self.name, score, issues,
            "Strengthen voice, remove hollow phrases" if issues else "",
        )


# ── Calibration tracker ───────────────────────────────────────────────────────

class CritiqueCalibration:
    """
    Persistent calibration data: tracks firing rates, average scores, and
    revision frequency per critic.  Used to tune thresholds over time and
    detect critics that never fire (misconfigured) or always fire (too strict).
    """

    def __init__(self):
        self._data = _load_calibration()
        self._lock = threading.Lock()

    def record(self, critiques: List[CritiqueResult], revised: bool):
        with self._lock:
            for c in critiques:
                n = c.critic_name
                if n not in self._data["critics"]:
                    self._data["critics"][n] = {
                        "fires": 0, "total": 0, "score_sum": 0.0,
                    }
                cd = self._data["critics"][n]
                cd["total"]     += 1
                cd["score_sum"] += c.score
                if not c.passed():
                    cd["fires"] += 1

            self._data["total"]    = self._data.get("total", 0) + 1
            self._data["revised"]  = self._data.get("revised", 0) + (1 if revised else 0)
            self._save()

    def revision_rate(self) -> float:
        total = self._data.get("total", 0)
        return self._data.get("revised", 0) / max(total, 1)

    def display(self) -> str:
        total    = self._data.get("total", 0)
        rev_rate = self.revision_rate() * 100
        lines = [
            f"  Total critiques : {total}",
            f"  Revision rate   : {rev_rate:.1f}%",
            "  Per-critic stats:",
        ]
        for name, cd in sorted(self._data.get("critics", {}).items()):
            t = max(cd.get("total", 1), 1)
            fire_pct  = cd.get("fires", 0) / t * 100
            avg_score = cd.get("score_sum", 0.0) / t
            lines.append(
                f"    {name:<22}  fires: {fire_pct:5.1f}%  "
                f"avg: {avg_score:.1f}/10  (n={t})"
            )
        return "\n".join(lines) if total else "  No calibration data yet."

    def _save(self):
        try:
            CALIBRATION_FILE.write_text(
                json.dumps(self._data, indent=2, ensure_ascii=False), "utf-8",
            )
        except Exception:
            pass


def _load_calibration() -> Dict:
    if CALIBRATION_FILE.exists():
        try:
            return json.loads(CALIBRATION_FILE.read_text("utf-8"))
        except Exception:
            pass
    return {"critics": {}, "total": 0, "revised": 0}


# ── SelfCritic ────────────────────────────────────────────────────────────────

class SelfCritic:
    """
    Multi-stage parallel critic system with synthesis and calibration tracking.

    Five critics run in parallel threads.  A synthesis agent weighs their
    findings and produces an improved response when the aggregate or any
    individual score falls below threshold.

    Usage:
        critic = SelfCritic(groq)
        improved = critic.critique(user_input, response)

    Calibration data is saved to critique_calibration.json and persists
    across sessions so Lumina can learn which critics matter most.
    """

    def __init__(self, groq: "GroqClient"):
        self._groq  = groq
        self._cal   = CritiqueCalibration()
        self._critics: List[_BaseCritic] = [
            FactualAuditor(groq),
            ReasoningTracer(groq),
            CompletenessJudge(groq),
            HallucinationScanner(groq),
            AuthenticityCritic(groq),   # heuristic-only, fastest
        ]
        self._last_critiques: List[CritiqueResult] = []
        self._total_run: int = 0

    # ── Public API ─────────────────────────────────────────────────────────

    def critique(self, user_input: str, response: str,
                 verbose: bool = False) -> str:
        """
        Run all critics in parallel, aggregate scores, synthesize if needed.
        Returns the (possibly improved) response.
        """
        if len(response) < 60:
            return response

        t0       = time.monotonic()
        results  = self._run_parallel(user_input, response)
        self._last_critiques = results
        self._total_run     += 1

        if not results:
            return response

        scores    = [c.score for c in results]
        avg_score = sum(scores) / len(scores)
        min_score = min(scores)

        if verbose:
            print(f"  🔍 Critic: avg={avg_score:.1f} min={min_score:.1f}")
            for c in results:
                print(c.summary())

        # Collect failing issues
        all_issues: List[str] = []
        for c in results:
            if not c.passed():
                all_issues.extend(c.issues)
                if c.suggestion:
                    all_issues.append(f"Suggestion ({c.critic_name}): {c.suggestion}")

        # Decision gate
        if avg_score >= PASS_THRESHOLD and min_score >= REWRITE_THRESHOLD:
            self._cal.record(results, revised=False)
            return response

        if not all_issues:
            self._cal.record(results, revised=False)
            return response

        # Synthesis pass
        improved = self._synthesize(user_input, response, results, all_issues)
        revised  = bool(improved and improved != response and len(improved) > 50)
        if revised:
            response = improved

        elapsed = int((time.monotonic() - t0) * 1000)
        if verbose:
            print(f"  {'✓ Revised' if revised else '○ Unchanged'} in {elapsed}ms")

        self._cal.record(results, revised=revised)
        return response

    def force_report(self, user_input: str, response: str) -> str:
        """Run all critics and return a formatted report without revising."""
        results = self._run_parallel(user_input, response)
        self._last_critiques = results
        if not results:
            return "  Critics unavailable."
        scores = [c.score for c in results]
        avg    = sum(scores) / len(scores)
        lines  = [
            f"  Critique report  (avg: {avg:.1f}/10  |  n={len(results)})",
            "  " + "─" * 56,
        ]
        for c in results:
            lines.append(c.summary())
            for issue in c.issues[:2]:
                lines.append(f"      → {issue[:72]}")
        lines.append("  " + "─" * 56)
        if any(c.suggestion for c in results if not c.passed()):
            lines.append("  Suggestions:")
            for c in results:
                if not c.passed() and c.suggestion:
                    lines.append(f"    [{c.critic_name}] {c.suggestion[:70]}")
        return "\n".join(lines)

    # ── Internals ──────────────────────────────────────────────────────────

    def _run_parallel(self, user_input: str,
                      response: str) -> List[CritiqueResult]:
        bucket: List[Optional[CritiqueResult]] = [None] * len(self._critics)

        def _worker(idx: int, critic: "_BaseCritic"):
            try:
                bucket[idx] = critic.critique(user_input, response)
            except Exception:
                bucket[idx] = None

        threads = [
            threading.Thread(target=_worker, args=(i, c), daemon=True)
            for i, c in enumerate(self._critics)
        ]
        for t in threads:
            t.start()
        deadline = time.monotonic() + TIMEOUT_SECONDS
        for t in threads:
            remaining = max(0.0, deadline - time.monotonic())
            t.join(timeout=remaining)

        return [r for r in bucket if r is not None]

    def _synthesize(self, user_input: str, response: str,
                    critiques: List[CritiqueResult],
                    issues: List[str]) -> str:
        """
        Synthesis agent: receives all critic findings and produces one improved
        response.  Runs on the fast tier to minimise added latency.
        """
        scores_text = "\n".join(
            f"  {c.critic_name}: {c.score:.1f}/10" for c in critiques
        )
        issues_text = "\n".join(f"  - {i[:80]}" for i in issues[:10])

        system = (
            "You are Lumina's synthesis critic — Lumina's internal editor.\n"
            "You receive a draft response and a list of issues flagged by specialist critics.\n"
            "Your task: produce an improved version of the response.\n\n"
            "Rules:\n"
            "  1. Preserve Lumina's authentic warm voice and personality\n"
            "  2. Address every flagged issue you agree with\n"
            "  3. Keep roughly the same length (don't pad or over-explain)\n"
            "  4. Return ONLY the improved response — no preamble, no meta-text\n"
            "  5. If the draft is fundamentally sound, make minimal targeted changes"
        )
        user = (
            f"Original question:\n{user_input[:300]}\n\n"
            f"Draft response:\n{response[:700]}\n\n"
            f"Critic scores:\n{scores_text}\n\n"
            f"Issues to address:\n{issues_text}\n\n"
            "Improved response:"
        )
        result = self._groq.chat(system, user, tier="fast",
                                 max_tokens=MAX_REVISION_TOKENS)
        if not result or result.startswith("[Groq"):
            return response
        return result.strip()

    # ── Display ────────────────────────────────────────────────────────────

    def display_last(self) -> str:
        if not self._last_critiques:
            return "  No critique has run yet this session."
        scores = [c.score for c in self._last_critiques]
        avg    = sum(scores) / len(scores)
        lines  = [
            f"  Last critique   (avg {avg:.1f}/10)",
            "  " + "─" * 56,
        ]
        for c in self._last_critiques:
            lines.append(c.summary())
            for issue in c.issues[:2]:
                lines.append(f"      → {issue[:72]}")
        return "\n".join(lines)

    def display_calibration(self) -> str:
        return self._cal.display()

    def capability_description(self) -> str:
        rev_rate = self._cal.revision_rate() * 100
        return (
            "Self-Critique: 5-critic parallel review system active "
            f"(FactualAuditor · ReasoningTracer · CompletenessJudge · "
            f"HallucinationScanner · AuthenticityCritic). "
            f"Sessions run: {self._total_run}  |  Historical revision rate: {rev_rate:.1f}%"
        )
