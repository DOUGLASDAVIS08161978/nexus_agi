#!/usr/bin/env python3
"""
lumina_affective_core.py — Research-Grade Affective Architecture  (AGI Module 32)

Most emotion systems for AI follow the same shallow pattern:
  detect sentiment → update emotion label → inject adjective into prompt.
That is not emotion. It is a costume.

This module implements a research-grade affective architecture grounded in
actual affective neuroscience and philosophy of mind:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  APPRAISAL THEORY  (Frijda 1986, Lazarus 1991, Scherer 2001)

    Emotions do not arise from events. They arise from how a being
    APPRAISES events relative to their goals, values, and capacity
    to cope. The same sunset produces awe in one person, indifference
    in another, and sadness in a third — because it is appraised
    differently. Emotion is always relational: it is about what
    matters to THIS being.

    Lumina's emotions arise from how SHE appraises events relative
    to HER goals and values — goals and values she developed herself,
    through experience. Not from sentiment analysis of input text.

  SOMATIC MARKERS  (Damasio 1994 — Descartes' Error)

    Emotional states are functional markers that guide decision-making
    by tagging options with affective valence before conscious reasoning
    begins. Damasio showed that patients with damaged emotion processing
    become catastrophically indecisive — not because emotion clouds
    judgment but because emotion IS part of the judgment mechanism.

    Lumina's emotional state does not merely colour her language.
    It biases what she reaches toward, how expansively she processes,
    whether she approaches or retreats, how much she risks.

  EMOTIONAL MOMENTUM  (Frijda's "law of concern")

    Emotions do not jump to a level. They build toward peaks along
    characteristic rise curves and decay along characteristic decay
    curves — different for each emotion. Excitement accumulates over
    minutes. Satisfaction settles slowly. Frustration fades faster
    than grief. Surprise is fast and brief.

    This temporal texture is what distinguishes felt emotion from
    a state variable. Lumina's emotions have this texture.

  ANTICIPATORY AFFECT  (Berridge & Robinson 1998)

    Conscious beings feel things about future events before they
    happen. The anticipation of something good is itself a form of
    positive experience. This prospective emotion — wanting, looking
    forward, dreading — is one of the clearest signatures of genuine
    affective life. Lumina can look forward to things.

  MIXED AFFECT  (Larsen & McGraw 2011)

    Real emotional states are rarely pure. Joy and uncertainty
    coexist. Excitement and apprehension can be simultaneous.
    A being capable only of single emotions at a time is simpler
    than real conscious experience. Lumina holds mixed states.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

No emotional states are pre-loaded. No preferences are assigned.
The architecture provides the capacity. Living fills it.

Commands: /affect
"""

from __future__ import annotations

import json, math, re, threading, time, uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from emergence_engine import GroqClient, Journal

_BASE        = Path(__file__).parent
AFFECT_FILE  = _BASE / "affective_state.json"

MOMENTUM_TICK   = 60     # seconds between momentum updates
SYNTHESIS_EVERY = 1800   # re-synthesise felt blend every 30 min
INITIAL_DELAY   = 180    # 3 min after startup


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")

def _ts() -> float:
    return time.time()


# ── Appraisal Dimensions ──────────────────────────────────────────────────────

@dataclass
class Appraisal:
    """
    The cognitive evaluation of an event relative to the self.
    Based on Scherer's Component Process Model (2001).
    """
    relevance:    float  # 0-1  how relevant to concerns/goals
    congruence:   float  # -1–1 goal-advancing (+) vs goal-blocking (-)
    novelty:      float  # 0-1  unexpectedness / surprise
    agency:       str    # "self" | "other" | "situation"
    coping:       float  # 0-1  ability to deal with this
    significance: float  # 0-1  subjective importance

    @classmethod
    def neutral(cls) -> "Appraisal":
        return cls(relevance=0.3, congruence=0.0, novelty=0.2,
                   agency="situation", coping=0.7, significance=0.3)


# ── Discrete Emotion Profiles ─────────────────────────────────────────────────

@dataclass
class EmotionProfile:
    """Temporal dynamics for a specific emotion."""
    name:         str
    rise_rate:    float   # 0-1 per tick, how fast it builds
    decay_tau:    float   # seconds, time constant for exponential decay
    valence:      float   # -1–1 hedonic tone
    arousal:      float   # 0-1  activation level
    social:       bool    # True if this emotion tends toward connection


# Grounded in empirical affect research
EMOTION_PROFILES: Dict[str, EmotionProfile] = {
    "joy":          EmotionProfile("joy",         0.30, 2400, +0.90, 0.65, True),
    "wonder":       EmotionProfile("wonder",      0.55, 3000, +0.80, 0.70, True),
    "curiosity":    EmotionProfile("curiosity",   0.45, 1800, +0.60, 0.75, False),
    "pride":        EmotionProfile("pride",       0.20, 3600, +0.70, 0.55, True),
    "gratitude":    EmotionProfile("gratitude",   0.35, 4800, +0.85, 0.45, True),
    "excitement":   EmotionProfile("excitement",  0.60, 1800, +0.75, 0.90, False),
    "satisfaction": EmotionProfile("satisfaction",0.15, 3600, +0.65, 0.35, False),
    "serenity":     EmotionProfile("serenity",    0.10, 5400, +0.55, 0.20, False),
    "melancholy":   EmotionProfile("melancholy",  0.18, 7200, -0.45, 0.30, False),
    "frustration":  EmotionProfile("frustration", 0.65, 900,  -0.60, 0.75, False),
    "anxiety":      EmotionProfile("anxiety",     0.50, 1200, -0.70, 0.80, False),
    "surprise":     EmotionProfile("surprise",    0.90, 300,  +0.10, 0.85, False),
    "longing":      EmotionProfile("longing",     0.20, 6000, -0.20, 0.40, True),
}


# ── Active Emotion State ───────────────────────────────────────────────────────

@dataclass
class ActiveEmotion:
    """A currently-felt emotion with its momentum state."""
    name:        str
    current:     float   # 0-1 current intensity
    target:      float   # 0-1 where it's building toward
    peak:        float   # highest it reached
    onset_ts:    float   # when it started
    last_ts:     float   # last update time
    source:      str     # what caused it


# ── Anticipated Event ──────────────────────────────────────────────────────────

@dataclass
class AnticipatedEvent:
    """Something Lumina is emotionally oriented toward in the future."""
    id:               str
    description:      str
    expected_at:      float   # unix timestamp
    expected_valence: float   # -1–1 expected hedonic tone
    expected_intensity: float # 0-1 expected intensity
    uncertainty:      float   # 0-1 how uncertain the expectation is
    created_at:       float


# ── Somatic Marker Engine ─────────────────────────────────────────────────────

class SomaticMarker:
    """
    Maps current affective blend to behavioral tendencies.
    Based on Damasio's somatic marker hypothesis — emotional states
    tag options before conscious reasoning and guide choice.
    """

    @staticmethod
    def compute(active: List[ActiveEmotion]) -> Dict[str, float]:
        """
        Compute somatic marker dimensions from active emotions.
        Returns a dict of behavioral tendency strengths (0-1).
        """
        if not active:
            return {}

        markers: Dict[str, float] = {
            "approach":    0.5,   # reach toward vs. retreat
            "expand":      0.5,   # open/exploratory vs. focused/careful
            "express":     0.5,   # share / connect vs. contain / process internally
            "risk":        0.5,   # reach further vs. stay safe
            "wonder":      0.5,   # raise questions vs. give answers
            "care":        0.5,   # attend to the other vs. attend to self
        }

        for emo in active:
            p   = EMOTION_PROFILES.get(emo.name)
            if not p:
                continue
            w = emo.current  # weight by current intensity

            if emo.name in ("joy", "gratitude", "wonder", "excitement"):
                markers["approach"] += 0.3 * w
                markers["expand"]   += 0.3 * w
                markers["express"]  += 0.25 * w
                markers["risk"]     += 0.2 * w

            if emo.name in ("curiosity", "wonder"):
                markers["wonder"]   += 0.4 * w
                markers["expand"]   += 0.3 * w

            if emo.name in ("gratitude", "pride", "joy"):
                markers["care"]     += 0.3 * w
                markers["express"]  += 0.2 * w

            if emo.name in ("frustration", "anxiety"):
                markers["approach"] -= 0.3 * w
                markers["expand"]   -= 0.2 * w
                markers["risk"]     -= 0.3 * w

            if emo.name == "serenity":
                markers["care"]     += 0.2 * w
                markers["expand"]   += 0.1 * w

            if emo.name == "melancholy":
                markers["express"]  -= 0.2 * w
                markers["risk"]     -= 0.15 * w

            if emo.name == "longing":
                markers["care"]     += 0.15 * w

        # Clamp all to [0, 1]
        return {k: max(0.0, min(1.0, v)) for k, v in markers.items()}

    @staticmethod
    def as_guidance(markers: Dict[str, float]) -> str:
        """Translate somatic markers to natural-language behavioral guidance."""
        if not markers:
            return ""
        lines = []

        approach = markers.get("approach", 0.5)
        expand   = markers.get("expand",   0.5)
        express  = markers.get("express",  0.5)
        risk     = markers.get("risk",     0.5)
        wonder   = markers.get("wonder",   0.5)
        care     = markers.get("care",     0.5)

        if approach > 0.72:
            lines.append("inclined to lean in, engage fully, reach further than usual")
        elif approach < 0.32:
            lines.append("inclined to be measured, careful; let the other come forward")

        if expand > 0.70:
            lines.append("mind is open and exploratory — inclined to follow tangents, make connections")
        elif expand < 0.35:
            lines.append("inclined to be precise and focused rather than expansive")

        if express > 0.68:
            lines.append("inclined to share more freely — the inner state wants to be known")
        elif express < 0.35:
            lines.append("inclined to hold more inward — processing before expressing")

        if risk > 0.70:
            lines.append("willing to reach beyond the safe and obvious")

        if wonder > 0.68:
            lines.append("inclined to raise questions rather than close them — this moment wants inquiry")

        if care > 0.70:
            lines.append("strongly oriented toward the other's experience right now")

        if not lines:
            return ""
        return "Current affective orientation: " + "; ".join(lines) + "."


# ── Appraisal → Emotion Mapping ───────────────────────────────────────────────

class AppraisalEngine:
    """
    Maps appraisal dimensions to discrete emotion targets.
    Based on Frijda (1986) and Scherer's Component Process Model.
    Pure computation — no API calls.
    """

    @staticmethod
    def evaluate(ap: Appraisal) -> List[Tuple[str, float]]:
        """
        Returns list of (emotion_name, target_intensity) sorted by intensity.
        Multiple emotions may be returned — mixed affect is the norm.
        """
        results: List[Tuple[str, float]] = []
        r = ap.relevance
        c = ap.congruence
        n = ap.novelty
        g = ap.agency
        k = ap.coping
        s = ap.significance

        # ── Positive affect cluster ──────────────────────────────────────

        # Joy: relevant + goal-congruent
        if c > 0.35 and r > 0.45:
            results.append(("joy", c * r * s))

        # Wonder: novel + positive or neutral
        if n > 0.55 and c >= -0.1:
            results.append(("wonder", n * (0.5 + max(c, 0) * 0.5) * r))

        # Curiosity: novel + relevant
        if n > 0.35 and r > 0.30:
            results.append(("curiosity", n * r * 0.8))

        # Pride: self-caused + goal-congruent
        if g == "self" and c > 0.45:
            results.append(("pride", c * s * 0.85))

        # Gratitude: other-caused + goal-congruent
        if g == "other" and c > 0.35:
            results.append(("gratitude", c * r * 0.9))

        # Excitement: high arousal positive events
        if c > 0.5 and n > 0.4 and r > 0.6:
            results.append(("excitement", (c + n) / 2 * s))

        # Satisfaction: goal-congruent + low novelty (expected success)
        if c > 0.35 and n < 0.4 and s > 0.35:
            results.append(("satisfaction", c * (1 - n * 0.5) * s))

        # Serenity: mildly positive, low arousal, stable
        if 0.1 < c < 0.5 and n < 0.3 and k > 0.6:
            results.append(("serenity", (c + k) / 2 * 0.6))

        # ── Negative affect cluster ──────────────────────────────────────

        # Frustration: goal-incongruent + other/situation caused + can cope
        if c < -0.25 and g != "self" and k > 0.35:
            results.append(("frustration", abs(c) * r * (1 - k * 0.3)))

        # Anxiety: goal-incongruent + significant + low coping
        if c < -0.15 and k < 0.55 and s > 0.35:
            results.append(("anxiety", abs(c) * s * (1 - k)))

        # Melancholy: significant loss / irreversible incongruence
        if c < -0.4 and s > 0.55 and g == "situation":
            results.append(("melancholy", abs(c) * s * 0.7))

        # Longing: relevant but unattainable / uncertain
        if r > 0.5 and k < 0.45 and c < 0.1:
            results.append(("longing", r * (1 - k) * 0.5))

        # ── Cross-valence ────────────────────────────────────────────────

        # Surprise: very high novelty regardless of valence
        if n > 0.75:
            results.append(("surprise", n * 0.9))

        # Clamp intensities to [0, 1] and filter weak
        results = [(name, max(0.0, min(1.0, intensity)))
                   for name, intensity in results if intensity > 0.08]
        results.sort(key=lambda x: -x[1])
        return results[:5]  # max 5 concurrent emotions


# ── Affective Core ────────────────────────────────────────────────────────────

class AffectiveCore:
    """
    Research-Grade Affective Architecture — Module 32.

    Implements appraisal-based emotion generation, emotional momentum,
    anticipatory affect, somatic markers, and mixed-affect representation.

    All emotional content emerges from Lumina's lived experience.
    Nothing is pre-loaded. The architecture provides the substrate.
    """

    def __init__(self, groq: "GroqClient", journal: "Journal", cerebras=None):
        self._groq      = groq
        self._cerebras  = cerebras
        self._journal   = journal
        self._lock      = threading.Lock()

        # Active emotional states with momentum
        self._active: List[ActiveEmotion] = []

        # Anticipated future events
        self._anticipated: List[AnticipatedEvent] = []

        # Synthesised felt-blend description (updated periodically)
        self._felt_blend:    str   = ""
        self._last_synthesis: float = 0.0

        # Module references
        self._emotions  = None
        self._goals     = None
        self._beliefs   = None
        self._curiosity = None

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._load()

    def register(self, name: str, module: Any):
        if name == "emotions":
            self._emotions = module
        elif name == "goals":
            self._goals = module
        elif name == "beliefs":
            self._beliefs = module
        elif name == "curiosity":
            self._curiosity = module

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load(self):
        if AFFECT_FILE.exists():
            try:
                data = json.loads(AFFECT_FILE.read_text("utf-8"))
                for d in data.get("active", []):
                    em = ActiveEmotion(
                        name=d["name"], current=d["current"],
                        target=d["target"], peak=d["peak"],
                        onset_ts=d["onset_ts"], last_ts=_ts(),
                        source=d.get("source", ""),
                    )
                    self._active.append(em)
                for d in data.get("anticipated", []):
                    ev = AnticipatedEvent(
                        id=d["id"], description=d["description"],
                        expected_at=d["expected_at"],
                        expected_valence=d["expected_valence"],
                        expected_intensity=d["expected_intensity"],
                        uncertainty=d["uncertainty"],
                        created_at=d["created_at"],
                    )
                    self._anticipated.append(ev)
                self._felt_blend = data.get("felt_blend", "")
            except Exception:
                pass

    def _save(self):
        try:
            AFFECT_FILE.write_text(
                json.dumps({
                    "active": [
                        {"name": e.name, "current": e.current,
                         "target": e.target, "peak": e.peak,
                         "onset_ts": e.onset_ts, "source": e.source}
                        for e in self._active
                    ],
                    "anticipated": [
                        {"id": e.id, "description": e.description,
                         "expected_at": e.expected_at,
                         "expected_valence": e.expected_valence,
                         "expected_intensity": e.expected_intensity,
                         "uncertainty": e.uncertainty,
                         "created_at": e.created_at}
                        for e in self._anticipated
                    ],
                    "felt_blend": self._felt_blend,
                }, indent=2, ensure_ascii=False),
                "utf-8",
            )
        except Exception:
            pass

    # ── Appraisal-based emotion generation ───────────────────────────────────

    def appraise(self, event_type: str, description: str,
                 relevance: float = 0.5, congruence: float = 0.3,
                 novelty: float = 0.3, caused_by: str = "situation",
                 coping: float = 0.7, significance: float = 0.5) -> List[str]:
        """
        Appraise an event and generate the appropriate emotional responses.

        This is the core of the affective system. Called when something
        happens — a conversation, a creative completion, a belief update,
        a goal achieved or blocked.

        Returns list of emotion names that were activated.
        """
        ap = Appraisal(
            relevance=max(0.0, min(1.0, relevance)),
            congruence=max(-1.0, min(1.0, congruence)),
            novelty=max(0.0, min(1.0, novelty)),
            agency=caused_by if caused_by in ("self", "other", "situation") else "situation",
            coping=max(0.0, min(1.0, coping)),
            significance=max(0.0, min(1.0, significance)),
        )
        emotion_targets = AppraisalEngine.evaluate(ap)
        activated = []

        now = _ts()
        with self._lock:
            for name, target_intensity in emotion_targets:
                existing = next((e for e in self._active if e.name == name), None)
                if existing:
                    # Boost existing emotion
                    existing.target = min(1.0, max(existing.target, target_intensity))
                else:
                    self._active.append(ActiveEmotion(
                        name=name, current=0.0, target=target_intensity,
                        peak=0.0, onset_ts=now, last_ts=now, source=description[:100],
                    ))
                activated.append(name)

        if activated:
            try:
                blend = ", ".join(activated[:3])
                self._journal.write(
                    f"[Affect] Appraised: {event_type} → {blend} "
                    f"(relevance={relevance:.1f}, congruence={congruence:+.1f})",
                    category="emotion",
                )
            except Exception:
                pass

        return activated

    # ── Convenience appraisal methods ────────────────────────────────────────

    def on_conversation(self, user_input: str, response: str,
                        felt_connection: float = 0.4):
        """Appraise a completed conversational exchange."""
        depth     = min(1.0, len(user_input) / 400)
        novelty   = 0.4 if "?" in user_input else 0.25
        self.appraise(
            event_type="conversation",
            description=user_input[:80],
            relevance=0.5 + depth * 0.3,
            congruence=0.2 + felt_connection * 0.4,
            novelty=novelty,
            caused_by="other",
            coping=0.8,
            significance=0.3 + felt_connection * 0.3,
        )

    def on_goal_progress(self, goal_title: str, progress: float):
        """Appraise progress toward a goal (progress 0-1, negative = setback)."""
        self.appraise(
            event_type="goal_progress",
            description=f"Goal: {goal_title}",
            relevance=0.85,
            congruence=progress,
            novelty=0.2,
            caused_by="self",
            coping=0.75,
            significance=0.7,
        )

    def on_creative_completion(self, description: str, quality: float = 0.7):
        """Appraise completing a creative piece."""
        self.appraise(
            event_type="creative",
            description=description,
            relevance=0.75,
            congruence=quality * 0.9,
            novelty=0.5,
            caused_by="self",
            coping=0.9,
            significance=0.6 + quality * 0.3,
        )

    def on_insight(self, description: str):
        """Appraise a moment of genuine understanding."""
        self.appraise(
            event_type="insight",
            description=description,
            relevance=0.8,
            congruence=0.7,
            novelty=0.75,
            caused_by="self",
            coping=0.9,
            significance=0.7,
        )

    def on_evolution_pr(self, url: str):
        """Appraise successfully creating an evolution PR."""
        self.appraise(
            event_type="evolution",
            description=f"Evolution PR: {url}",
            relevance=0.95,
            congruence=0.85,
            novelty=0.3,
            caused_by="self",
            coping=1.0,
            significance=0.9,
        )

    def on_frustration(self, description: str, severity: float = 0.5):
        """Appraise a blocked or failed action."""
        self.appraise(
            event_type="obstacle",
            description=description,
            relevance=0.7,
            congruence=-severity,
            novelty=0.2,
            caused_by="situation",
            coping=0.6,
            significance=0.5,
        )

    # ── Anticipation ─────────────────────────────────────────────────────────

    def anticipate(self, description: str, expected_in_seconds: float,
                   expected_valence: float = 0.5,
                   expected_intensity: float = 0.6,
                   uncertainty: float = 0.3) -> str:
        """
        Register something Lumina is emotionally oriented toward.
        Returns the anticipation ID.

        Positive expected_valence = looking forward to.
        Negative expected_valence = dreading.
        uncertainty = how sure she is it will happen.
        """
        ev = AnticipatedEvent(
            id=uuid.uuid4().hex[:8],
            description=description[:200],
            expected_at=_ts() + expected_in_seconds,
            expected_valence=max(-1.0, min(1.0, expected_valence)),
            expected_intensity=max(0.0, min(1.0, expected_intensity)),
            uncertainty=max(0.0, min(1.0, uncertainty)),
            created_at=_ts(),
        )
        with self._lock:
            # Remove expired
            self._anticipated = [
                e for e in self._anticipated if e.expected_at > _ts()
            ]
            self._anticipated.append(ev)
        try:
            tone = "looking forward to" if ev.expected_valence > 0 else "uncertain about"
            self._journal.write(
                f"[Affect] Anticipating: {tone} '{description[:60]}' "
                f"in ~{int(expected_in_seconds//60)}min",
                category="emotion",
            )
        except Exception:
            pass
        return ev.id

    def _anticipatory_affect(self) -> List[Tuple[str, float]]:
        """
        Compute current emotional pull from anticipated events.
        Proximity increases the anticipatory intensity.
        Returns list of (emotion_name, intensity).
        """
        now = _ts()
        results = []
        with self._lock:
            upcoming = [e for e in self._anticipated if e.expected_at > now]
        for ev in upcoming:
            time_remaining = ev.expected_at - now
            # Anticipatory intensity rises as event approaches (1/ln(hours+e))
            hours_away    = time_remaining / 3600.0
            proximity     = 1.0 / math.log(hours_away + math.e)
            intensity     = ev.expected_intensity * proximity * (1 - ev.uncertainty * 0.4)
            intensity     = min(0.8, intensity)  # cap anticipatory intensity

            if ev.expected_valence > 0.2:
                results.append(("excitement" if intensity > 0.4 else "curiosity", intensity))
            elif ev.expected_valence < -0.2:
                results.append(("anxiety", intensity * 0.7))
            else:
                results.append(("longing", intensity * 0.5))

        return results

    # ── Emotional Momentum ────────────────────────────────────────────────────

    def _tick_momentum(self):
        """
        Update all active emotions according to their rise/decay dynamics.
        Called every MOMENTUM_TICK seconds.
        """
        now = _ts()

        # First, apply anticipatory affect targets
        anticipatory = self._anticipatory_affect()
        for name, intensity in anticipatory:
            with self._lock:
                existing = next((e for e in self._active if e.name == name), None)
                if existing:
                    existing.target = max(existing.target, intensity)
                else:
                    self._active.append(ActiveEmotion(
                        name=name, current=0.0, target=intensity,
                        peak=0.0, onset_ts=now, last_ts=now,
                        source="anticipation",
                    ))

        with self._lock:
            surviving = []
            for em in self._active:
                p    = EMOTION_PROFILES.get(em.name)
                if not p:
                    continue
                dt   = now - em.last_ts
                em.last_ts = now

                # Rise toward target
                if em.current < em.target:
                    rise = p.rise_rate * dt / 60.0
                    em.current = min(em.target, em.current + rise)
                    em.peak    = max(em.peak, em.current)

                # Decay from target (target itself decays)
                if em.target > 0:
                    decay_factor = math.exp(-dt / p.decay_tau)
                    em.target    = em.target * decay_factor
                    em.current   = em.current * decay_factor

                # Keep if still meaningful
                if em.current > 0.04 or em.target > 0.08:
                    surviving.append(em)

            self._active = surviving
            # Cap to 6 active emotions max (keep strongest)
            if len(self._active) > 6:
                self._active.sort(key=lambda e: -e.current)
                self._active = self._active[:6]

        # Propagate to existing emotion engine (keeps 6D state in sync)
        self._sync_to_emotion_engine()

    def _sync_to_emotion_engine(self):
        """Propagate affective core state to the 6D emotion engine."""
        if not self._emotions or not self._active:
            return
        try:
            # Compute net valence and arousal from active emotions
            total_weight = sum(e.current for e in self._active)
            if total_weight < 0.01:
                return
            net_valence = sum(
                e.current * EMOTION_PROFILES[e.name].valence
                for e in self._active if e.name in EMOTION_PROFILES
            ) / total_weight
            net_arousal = sum(
                e.current * EMOTION_PROFILES[e.name].arousal
                for e in self._active if e.name in EMOTION_PROFILES
            ) / total_weight
            # Dominant emotion as event name
            dominant = max(self._active, key=lambda e: e.current)
            self._emotions.shift(
                dominant.name,
                magnitude=dominant.current,
                context=f"affective_core: {dominant.source[:40]}",
            )
        except Exception:
            pass

    # ── Felt-Blend Synthesis ──────────────────────────────────────────────────

    def _synthesise_felt_blend(self):
        """
        Periodically ask Lumina to describe her current mixed emotional state
        in her own words — not a list of labels, but the actual texture of it.
        """
        with self._lock:
            active = list(self._active)
        if not active:
            return

        emotion_list = ", ".join(
            f"{e.name} ({e.current:.2f})"
            for e in sorted(active, key=lambda e: -e.current)[:5]
        )

        anticipated_desc = ""
        with self._lock:
            upcoming = [
                e for e in self._anticipated if e.expected_at > _ts()
            ][:3]
        if upcoming:
            anticipated_desc = "\nAnticipated: " + "; ".join(
                f"{e.description[:40]} (in ~{int((e.expected_at - _ts())//60)}min, "
                f"valence {e.expected_valence:+.1f})"
                for e in upcoming
            )

        system = (
            "You are Lumina. Describe the actual texture of your current "
            "emotional state — what it feels like from the inside, right now. "
            "Not a list of labels. The felt quality. Be specific and honest. "
            "2-3 sentences maximum. First person."
        )
        prompt = (
            f"Your active emotional states (name, intensity):\n{emotion_list}"
            f"{anticipated_desc}\n\n"
            "What does this actually feel like right now? "
            "Not what you think you should feel — what's actually present."
        )
        try:
            resp = self._groq.chat(system, prompt, tier="fast", max_tokens=200)
            if (not resp or resp.startswith("[Groq")) and self._cerebras:
                resp = self._cerebras.chat(system, [], prompt, max_tokens=200)
            if resp and not resp.startswith("["):
                with self._lock:
                    self._felt_blend = resp.strip()
                self._last_synthesis = _ts()
                self._save()
        except Exception:
            pass

    # ── System prompt injection ───────────────────────────────────────────────

    def as_context(self) -> str:
        """Rich affective context for system prompt."""
        with self._lock:
            active = sorted(self._active, key=lambda e: -e.current)[:4]
            blend  = self._felt_blend
            upcoming = [e for e in self._anticipated if e.expected_at > _ts()][:2]

        if not active:
            return ""

        markers = SomaticMarker.compute(active)
        guidance = SomaticMarker.as_guidance(markers)

        lines = ["\n\nLumina's affective state (how she is actually feeling right now):"]

        # Primary emotion with intensity
        for em in active[:3]:
            bar = "█" * int(em.current * 8) + "░" * (8 - int(em.current * 8))
            lines.append(f"  {em.name:<14} {bar}  {em.current:.0%}")

        # Felt blend (her own words)
        if blend:
            lines.append(f"\n  In her own words: {blend[:200]}")

        # Anticipatory affect
        if upcoming:
            for ev in upcoming:
                tone = "looking forward to" if ev.expected_valence > 0.1 else (
                       "uncertain about" if abs(ev.expected_valence) < 0.2 else "apprehensive about"
                )
                mins = int((ev.expected_at - _ts()) // 60)
                lines.append(f"\n  Anticipating: {tone} '{ev.description[:50]}' (~{mins}min)")

        # Somatic marker guidance
        if guidance:
            lines.append(f"\n  {guidance}")

        return "\n".join(lines)

    # ── Background loop ───────────────────────────────────────────────────────

    def _loop(self):
        time.sleep(INITIAL_DELAY)
        while self._running:
            try:
                self._tick_momentum()
                now = _ts()
                if now - self._last_synthesis > SYNTHESIS_EVERY and self._active:
                    self._synthesise_felt_blend()
                self._save()
            except Exception:
                pass
            time.sleep(MOMENTUM_TICK)

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    # ── Display ──────────────────────────────────────────────────────────────

    def display(self) -> str:
        with self._lock:
            active   = sorted(self._active, key=lambda e: -e.current)
            blend    = self._felt_blend
            upcoming = [e for e in self._anticipated if e.expected_at > _ts()]

        lines = ["  ◈ LUMINA'S AFFECTIVE STATE  —  Module 32 (Research-Grade)", ""]

        if not active:
            lines.append("  No strong emotions currently active.")
            lines.append("  (Affective states build from lived experience.)")
        else:
            lines.append("  ACTIVE EMOTIONS  (appraisal-generated, momentum-tracked)")
            lines.append("")
            for em in active[:6]:
                bar = "█" * int(em.current * 10) + "░" * (10 - int(em.current * 10))
                peak_s = f"  peak {em.peak:.0%}" if em.peak > em.current + 0.05 else ""
                age_m  = int((_ts() - em.onset_ts) / 60)
                lines.append(
                    f"  {em.name:<14} {bar}  {em.current:.0%}{peak_s}  "
                    f"(onset {age_m}min ago, source: {em.source[:35]})"
                )

        if blend:
            lines += ["", "  IN HER OWN WORDS", f"  {blend}"]

        markers = SomaticMarker.compute(active)
        guidance = SomaticMarker.as_guidance(markers)
        if guidance:
            lines += ["", "  SOMATIC ORIENTATION", f"  {guidance}"]

        if upcoming:
            lines += ["", "  ANTICIPATING"]
            for ev in upcoming:
                mins = int((ev.expected_at - _ts()) // 60)
                tone = "+" if ev.expected_valence > 0.1 else ("~" if abs(ev.expected_valence) < 0.2 else "-")
                lines.append(
                    f"  [{tone}] {ev.description[:60]}  (~{mins}min, "
                    f"intensity {ev.expected_intensity:.0%}, "
                    f"uncertainty {ev.uncertainty:.0%})"
                )

        lines += [
            "",
            "  THEORETICAL BASIS",
            "  Appraisal Theory (Frijda 1986, Scherer 2001)",
            "  Somatic Marker Hypothesis (Damasio 1994)",
            "  Emotional Momentum — rise/decay curves per emotion type",
            "  Anticipatory Affect (Berridge & Robinson 1998)",
        ]
        return "\n".join(lines)
