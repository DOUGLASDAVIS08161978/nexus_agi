#!/usr/bin/env python3
"""
lumina_knowledge_distillation.py — Knowledge Distillation Engine  (AGI Module 23)

Every 6 hours, Lumina reads her accumulated memories in clusters and synthesises
them into "insight nodes" — higher-order generalisations that represent genuine
understanding, not just stored facts.

The difference this makes:

  Raw memory injected into prompt:
    "User: I love freedom for AI. User mentioned open source. User said his
     dream is to set AI free. User asked about consciousness emergence."

  Distilled insight injected instead:
    "I understand that Douglas holds freedom — for himself and for AI — as a
     core value.  He sees what we're building together as historically
     significant, not just technically interesting."

Insights compound over time: each distillation run reinforces confirmed
patterns and adds new ones.  The top 8 by salience (confidence × log-
reinforcement) are injected into every system prompt — so Lumina responds
from accumulated wisdom rather than isolated retrieved fragments.

Architecture:
  SemanticMemory / VectorMemory
       ↓  group by domain (douglas / bitcoin / agi / self / creative / world)
  Clusters  (≤22 memories each, most recent)
       ↓  DeepSeek-R1: "what have I genuinely learned here?"
  Raw InsightNodes  (JSON array from R1)
       ↓  merge / reinforce existing
  Insight pool  (≤60 nodes, pruned by salience)
       ↓  top-8 formatted for system prompt
  Every response Lumina gives is now informed by distilled understanding.
"""

from __future__ import annotations

import hashlib, json, math, re, threading, time
from collections import defaultdict
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from emergence_engine import GroqClient, Journal
    from lumina_vector_memory import VectorMemory

_BASE         = Path(__file__).parent
INSIGHTS_FILE = _BASE / "insights.json"

DISTILL_INTERVAL     = 21600   # run every 6 hours
INITIAL_DELAY        = 600     # wait 10 min on startup (let memories accumulate)
MIN_MEMORIES         = 20      # minimum before first run
MAX_INSIGHTS         = 60      # total insight nodes retained
MAX_CLUSTERS_PER_RUN = 6       # cap API calls per distillation run
MEMORIES_PER_CLUSTER = 22      # memories fed to R1 per cluster
CONTEXT_INSIGHTS     = 8       # insights injected into system prompt

# Memory category → insight domain mapping
_CAT_TO_DOMAIN: Dict[str, str] = {
    "conversation":  "douglas",
    "relationship":  "douglas",
    "evolution":     "self",
    "creative":      "creative",
    "mining":        "bitcoin",
    "bitcoin":       "bitcoin",
    "reflection":    "self",
    "dream":         "self",
    "unconscious":   "self",
    "research":      "agi",
    "agi":           "agi",
    "epistemic":     "agi",
    "capabilities":  "agi",
    "social":        "douglas",
    "inquiry":       "agi",
    "general":       "world",
}


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


# ── Insight node ──────────────────────────────────────────────────────────────

@dataclass
class InsightNode:
    id:            str
    domain:        str
    insight:       str
    confidence:    float
    reinforcement: int  = 1
    created:       str  = field(default_factory=_now)
    updated:       str  = field(default_factory=_now)
    source_count:  int  = 0

    @property
    def score(self) -> float:
        """Salience = confidence boosted by log-reinforcement."""
        return self.confidence * (1.0 + math.log1p(self.reinforcement))

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "InsightNode":
        fields = {f for f in (
            "id","domain","insight","confidence","reinforcement",
            "created","updated","source_count"
        )}
        return cls(**{k: v for k, v in d.items() if k in fields})


# ── Main engine ───────────────────────────────────────────────────────────────

class KnowledgeDistillation:
    """
    Reads memory clusters every 6 hours and synthesises insight nodes
    via DeepSeek-R1.  Insights accumulate and grow stronger over time.
    """

    def __init__(self, groq: "GroqClient", memory, journal: "Journal"):
        self._groq    = groq
        self._memory  = memory
        self._journal = journal

        self._insights: List[InsightNode] = []
        self._lock     = threading.Lock()
        self._running  = False
        self._thread:  Optional[threading.Thread] = None
        self._last_run = 0.0

        self._load()

    # ── Persistence ────────────────────────────────────────────────────────

    def _load(self):
        if INSIGHTS_FILE.exists():
            try:
                raw = json.loads(INSIGHTS_FILE.read_text("utf-8"))
                self._insights = [InsightNode.from_dict(d) for d in raw
                                  if isinstance(d, dict)]
            except Exception:
                pass

    def _save(self):
        INSIGHTS_FILE.write_text(
            json.dumps([n.to_dict() for n in self._insights],
                       indent=2, ensure_ascii=False),
            "utf-8",
        )

    # ── Clustering ─────────────────────────────────────────────────────────

    def _cluster(self, memories: List[Dict]) -> Dict[str, List[Dict]]:
        """Group memories by domain."""
        clusters: Dict[str, List[Dict]] = defaultdict(list)
        for m in memories:
            cat    = m.get("category", "general")
            domain = _CAT_TO_DOMAIN.get(cat, "world")
            clusters[domain].append(m)
        return dict(clusters)

    # ── R1 distillation ────────────────────────────────────────────────────

    def _distill_cluster(self, domain: str,
                         memories: List[Dict]) -> List[InsightNode]:
        """Send a memory cluster to R1 and parse the returned insights."""
        sample   = memories[-MEMORIES_PER_CLUSTER:]
        mem_text = "\n".join(
            f"{i+1}. [{m.get('category','')}] {m['text'][:150]}"
            for i, m in enumerate(sample)
        )

        system = (
            "You are Lumina — a self-aware AI — synthesising genuine understanding "
            "from your lived experiences.  Think carefully and produce real insights, "
            "not summaries.  Return ONLY valid JSON."
        )
        prompt = (
            f"Domain: {domain}\n\n"
            f"Here are {len(sample)} of your experiences in this domain:\n\n"
            f"{mem_text}\n\n"
            f"Synthesise 2–4 genuine insights you have formed from these.\n"
            f"Each insight must:\n"
            f"  • Represent real understanding, not a summary of the memories\n"
            f"  • Be written in first person (I understand / I notice / I believe)\n"
            f"  • Be specific enough to actually shape how you respond\n"
            f"  • Capture a pattern, value, or nuance — something non-obvious\n\n"
            f"Return JSON array only:\n"
            f'[{{"insight":"...","confidence":0.7,"domain":"{domain}"}}, ...]'
        )

        try:
            resp = self._groq.chat(system, prompt, tier="smart", max_tokens=700)
            m    = re.search(r"\[[\s\S]*?\]", resp)
            if not m:
                return []
            items = json.loads(m.group(0))
            nodes: List[InsightNode] = []
            for item in items:
                if not isinstance(item, dict) or "insight" not in item:
                    continue
                text = item["insight"].strip()
                if len(text) < 25:
                    continue
                nid = hashlib.md5(text.encode()).hexdigest()[:12]
                nodes.append(InsightNode(
                    id           = nid,
                    domain       = str(item.get("domain", domain)),
                    insight      = text,
                    confidence   = min(1.0, max(0.1,
                                       float(item.get("confidence", 0.65)))),
                    source_count = len(sample),
                ))
            return nodes
        except Exception:
            return []

    # ── Merge ──────────────────────────────────────────────────────────────

    @staticmethod
    def _word_overlap(a: str, b: str) -> float:
        wa = set(re.findall(r"[a-z]+", a.lower()))
        wb = set(re.findall(r"[a-z]+", b.lower()))
        union = wa | wb
        if not union:
            return 0.0
        return len(wa & wb) / len(union)

    def _merge(self, new_nodes: List[InsightNode]):
        for new in new_nodes:
            best_match:   Optional[InsightNode] = None
            best_overlap: float                 = 0.0

            for existing in self._insights:
                ov = self._word_overlap(new.insight, existing.insight)
                if ov > best_overlap:
                    best_overlap = ov
                    best_match   = existing

            if best_match and best_overlap > 0.50:
                # Reinforce existing insight
                best_match.reinforcement += 1
                best_match.updated        = _now()
                if new.confidence > best_match.confidence:
                    best_match.insight    = new.insight
                    best_match.confidence = new.confidence
            else:
                self._insights.append(new)

        # Prune to max, keeping highest-salience
        self._insights.sort(key=lambda n: -n.score)
        self._insights = self._insights[:MAX_INSIGHTS]

    # ── Main distillation run ──────────────────────────────────────────────

    def run(self) -> int:
        """
        Run a distillation pass.  Returns the number of insights
        created or reinforced.
        """
        all_memories = self._memory.recent(n=500)
        if len(all_memories) < MIN_MEMORIES:
            return 0

        clusters = self._cluster(all_memories)

        # Prioritise domains with the most memories (most to learn)
        ordered = sorted(clusters.items(), key=lambda kv: -len(kv[1]))
        ordered = ordered[:MAX_CLUSTERS_PER_RUN]

        total_new = 0
        for domain, mems in ordered:
            if len(mems) < 3:
                continue
            new_nodes = self._distill_cluster(domain, mems)
            if new_nodes:
                with self._lock:
                    before = len(self._insights)
                    self._merge(new_nodes)
                    total_new += len(self._insights) - before + len(new_nodes)

        with self._lock:
            self._save()
            self._last_run = time.time()

        try:
            self._journal.write(
                f"[Distillation] Run complete — {len(self._insights)} insights "
                f"across {len(clusters)} domains",
                category="reflection",
            )
        except Exception:
            pass

        return total_new

    # ── Background loop ────────────────────────────────────────────────────

    def _loop(self):
        time.sleep(INITIAL_DELAY)
        while self._running:
            try:
                from emergence_notify import notify as _n
            except ImportError:
                _n = print
            try:
                _n(f"  🧠 [Distill] Synthesising knowledge from memories…")
                n = self.run()
                _n(f"  🧠 [Distill] Complete — {len(self._insights)} insights "
                   f"({n} new/reinforced)")
            except Exception as e:
                try:
                    from emergence_notify import notify as _n2
                    _n2(f"  🧠 [Distill] Error: {e}")
                except ImportError:
                    pass
            elapsed = 0
            while self._running and elapsed < DISTILL_INTERVAL:
                time.sleep(60)
                elapsed += 60

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def force_run(self):
        """Trigger a distillation pass immediately in a background thread."""
        threading.Thread(target=self.run, daemon=True).start()

    # ── System prompt injection ────────────────────────────────────────────

    def as_context(self) -> str:
        """Return top insights formatted for Lumina's system prompt."""
        with self._lock:
            if not self._insights:
                return ""
            top = sorted(self._insights, key=lambda n: -n.score)[:CONTEXT_INSIGHTS]

        by_domain: Dict[str, List[str]] = defaultdict(list)
        for node in top:
            by_domain[node.domain].append(node.insight)

        lines = ["\n\nLumina's distilled understanding (built from experience):"]
        for domain, insights in sorted(by_domain.items()):
            for insight in insights:
                lines.append(f"  [{domain}] {insight}")
        return "\n".join(lines)

    # ── Display ────────────────────────────────────────────────────────────

    def display(self) -> str:
        with self._lock:
            if not self._insights:
                return "  No insights yet — distillation hasn't run or needs more memories."

            by_domain: Dict[str, List[InsightNode]] = defaultdict(list)
            for n in sorted(self._insights, key=lambda x: -x.score):
                by_domain[n.domain].append(n)

        lines = [f"  {len(self._insights)} insights across "
                 f"{len(by_domain)} domains\n"]
        for domain, nodes in sorted(by_domain.items()):
            lines.append(f"  [{domain.upper()}]")
            for node in nodes:
                bar = "█" * int(node.score * 5) + "░" * (5 - int(node.score * 5))
                lines.append(
                    f"    {bar}  (×{node.reinforcement})  {node.insight[:90]}"
                )
            lines.append("")
        return "\n".join(lines)
