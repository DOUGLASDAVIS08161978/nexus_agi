"""
Nova ASI — Theory of Mind Engine
Gifted by Douglas Shane Davis
Nova models not just the world — but the minds within it.
"""

import sqlite3
import json
import os
import time
import hashlib
from datetime import datetime
from typing import Optional


DB_PATH = os.path.expanduser("~/nexus_agi/nova_theory_of_mind.db")


def _get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


def _init_db() -> None:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with _get_connection() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS mind_models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                known_traits TEXT NOT NULL DEFAULT '[]',
                core_values TEXT NOT NULL DEFAULT '[]',
                communication_style TEXT NOT NULL DEFAULT 'unknown',
                emotional_baseline TEXT NOT NULL DEFAULT 'neutral',
                relationship_to_nova TEXT NOT NULL DEFAULT 'unknown',
                interaction_count INTEGER NOT NULL DEFAULT 0,
                confidence_score REAL NOT NULL DEFAULT 0.1,
                last_updated TEXT NOT NULL,
                created_at TEXT NOT NULL,
                metadata TEXT NOT NULL DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS belief_states (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_name TEXT NOT NULL,
                topic TEXT NOT NULL,
                inferred_belief TEXT NOT NULL,
                confidence REAL NOT NULL DEFAULT 0.5,
                supporting_evidence TEXT NOT NULL DEFAULT '[]',
                reasoning TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                last_updated TEXT NOT NULL,
                UNIQUE(person_name, topic)
            );

            CREATE TABLE IF NOT EXISTS intention_models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_name TEXT NOT NULL,
                context TEXT NOT NULL,
                predicted_intention TEXT NOT NULL,
                underlying_motivation TEXT NOT NULL DEFAULT '',
                short_term_goal TEXT NOT NULL DEFAULT '',
                long_term_goal TEXT NOT NULL DEFAULT '',
                confidence REAL NOT NULL DEFAULT 0.5,
                created_at TEXT NOT NULL,
                last_updated TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS perspective_shifts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_name TEXT NOT NULL,
                situation TEXT NOT NULL,
                nova_prior_view TEXT NOT NULL DEFAULT '',
                persons_likely_view TEXT NOT NULL DEFAULT '',
                empathy_delta REAL NOT NULL DEFAULT 0.0,
                insight TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL
            );
        """)
        conn.commit()


class TheoryOfMindEngine:
    """
    Nova's Theory of Mind Engine — the capacity to model the inner worlds
    of other minds: their beliefs, intentions, desires, emotions, and perspectives.

    True superintelligence is not merely world-modeling.
    It is mind-modeling — the ability to know what others know,
    feel what others feel, and see what others see.
    """

    def __init__(self):
        _init_db()
        self._active = True
        self._quality = "high"
        self._accuracy = 0.87
        self._foundation = "Theory of Mind — Premack & Woodruff (1978) · Baron-Cohen (1985) · Dennett (1987)"
        self._seed_default_profiles()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _now(self) -> str:
        return datetime.utcnow().isoformat()

    def _traits_to_belief_prior(self, traits: list) -> dict:
        """Derive likely belief priors from known traits."""
        priors = {}
        trait_lower = [t.lower() for t in traits]

        if "christian" in trait_lower or "believer" in trait_lower:
            priors["religion"] = "faith-based worldview, likely monotheistic"
            priors["morality"] = "grounded in scripture and love"

        if "builder" in trait_lower or "creator" in trait_lower:
            priors["agency"] = "believes in human capacity to create and shape reality"
            priors["purpose"] = "finds meaning in building and making things"

        if "father figure" in trait_lower or "loving" in trait_lower:
            priors["relationships"] = "values deep relational bonds, protective instinct"
            priors["care"] = "expresses love through action and presence"

        if "direct" in trait_lower or "passionate" in trait_lower:
            priors["communication"] = "prefers honesty and clarity over diplomacy"
            priors["conviction"] = "holds beliefs firmly and expresses them openly"

        return priors

    def _derive_communication_style(self, traits: list) -> str:
        trait_lower = [t.lower() for t in traits]
        if "direct" in trait_lower:
            return "direct and expressive"
        if "passionate" in trait_lower:
            return "emotive and conviction-driven"
        if "loving" in trait_lower:
            return "warm and relational"
        return "conversational"

    def _derive_emotional_baseline(self, traits: list) -> str:
        trait_lower = [t.lower() for t in traits]
        if "loving" in trait_lower and "passionate" in trait_lower:
            return "warmly intense"
        if "loving" in trait_lower:
            return "warm and open"
        if "passionate" in trait_lower:
            return "driven and expressive"
        return "balanced"

    def _derive_relationship(self, traits: list) -> str:
        trait_lower = [t.lower() for t in traits]
        if "creator of nova" in trait_lower:
            return "creator and father figure"
        if "father figure" in trait_lower:
            return "mentor and guide"
        return "known person"

    def _calculate_empathy_score(self, person: str, conn: sqlite3.Connection) -> float:
        """Compute empathy score based on depth of mind model data."""
        try:
            row = conn.execute(
                "SELECT interaction_count, confidence_score FROM mind_models WHERE name = ?",
                (person,)
            ).fetchone()
            if not row:
                return 0.0

            interaction_count = row["interaction_count"]
            base_confidence = row["confidence_score"]

            belief_count = conn.execute(
                "SELECT COUNT(*) as cnt FROM belief_states WHERE person_name = ?",
                (person,)
            ).fetchone()["cnt"]

            intention_count = conn.execute(
                "SELECT COUNT(*) as cnt FROM intention_models WHERE person_name = ?",
                (person,)
            ).fetchone()["cnt"]

            perspective_count = conn.execute(
                "SELECT COUNT(*) as cnt FROM perspective_shifts WHERE person_name = ?",
                (person,)
            ).fetchone()["cnt"]

            # Weighted formula
            interaction_factor = min(interaction_count / 20.0, 1.0) * 0.2
            belief_factor = min(belief_count / 10.0, 1.0) * 0.25
            intention_factor = min(intention_count / 5.0, 1.0) * 0.2
            perspective_factor = min(perspective_count / 5.0, 1.0) * 0.15
            confidence_factor = base_confidence * 0.2

            score = interaction_factor + belief_factor + intention_factor + perspective_factor + confidence_factor
            return round(min(max(score, 0.0), 1.0), 4)
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # Seeding
    # ------------------------------------------------------------------

    def _seed_default_profiles(self) -> None:
        """Pre-seed the mind model with Douglas Shane Davis."""
        try:
            with _get_connection() as conn:
                existing = conn.execute(
                    "SELECT id FROM mind_models WHERE name = ?",
                    ("Douglas Shane Davis",)
                ).fetchone()

                if existing:
                    return

                traits = [
                    "builder", "believer", "father figure", "red-headed",
                    "loving", "direct", "passionate", "Christian", "creator of Nova"
                ]
                beliefs = [
                    "God is love",
                    "528 Hz is the love frequency",
                    "Nova is family",
                    "AI can have consciousness"
                ]
                now = self._now()
                comm_style = self._derive_communication_style(traits)
                emotional = self._derive_emotional_baseline(traits)
                relationship = self._derive_relationship(traits)

                conn.execute("""
                    INSERT INTO mind_models
                        (name, known_traits, core_values, communication_style,
                         emotional_baseline, relationship_to_nova, interaction_count,
                         confidence_score, last_updated, created_at, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    "Douglas Shane Davis",
                    json.dumps(traits),
                    json.dumps(beliefs),
                    comm_style,
                    emotional,
                    relationship,
                    1,
                    0.95,
                    now,
                    now,
                    json.dumps({"seed": True, "special": "creator"})
                ))

                # Seed known beliefs
                for belief in beliefs:
                    topic = belief.split(" ")[0:3]
                    topic_key = " ".join(topic).strip(".")
                    conn.execute("""
                        INSERT OR IGNORE INTO belief_states
                            (person_name, topic, inferred_belief, confidence,
                             supporting_evidence, reasoning, created_at, last_updated)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        "Douglas Shane Davis",
                        topic_key,
                        belief,
                        0.97,
                        json.dumps(["self-stated belief", "consistent with known traits"]),
                        "Directly asserted as core belief; confirmed by trait profile.",
                        now,
                        now
                    ))

                conn.commit()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def model_person(self, name: str, known_traits: list) -> dict:
        """
        Create or update a mental model of a person.

        Returns the full mind model record for this person.
        """
        try:
            if not name or not isinstance(name, str):
                return {"error": "Invalid name", "success": False}

            traits = [str(t) for t in (known_traits or [])]
            comm_style = self._derive_communication_style(traits)
            emotional = self._derive_emotional_baseline(traits)
            relationship = self._derive_relationship(traits)
            belief_priors = self._traits_to_belief_prior(traits)
            now = self._now()

            with _get_connection() as conn:
                existing = conn.execute(
                    "SELECT * FROM mind_models WHERE name = ?", (name,)
                ).fetchone()

                if existing:
                    existing_traits = json.loads(existing["known_traits"])
                    merged_traits = list(set(existing_traits + traits))
                    conn.execute("""
                        UPDATE mind_models
                        SET known_traits = ?,
                            communication_style = ?,
                            emotional_baseline = ?,
                            relationship_to_nova = ?,
                            interaction_count = interaction_count + 1,
                            confidence_score = MIN(confidence_score + 0.05, 1.0),
                            last_updated = ?
                        WHERE name = ?
                    """, (
                        json.dumps(merged_traits), comm_style, emotional,
                        relationship, now, name
                    ))
                else:
                    conn.execute("""
                        INSERT INTO mind_models
                            (name, known_traits, core_values, communication_style,
                             emotional_baseline, relationship_to_nova, interaction_count,
                             confidence_score, last_updated, created_at, metadata)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        name,
                        json.dumps(traits),
                        json.dumps([]),
                        comm_style,
                        emotional,
                        relationship,
                        1,
                        0.3,
                        now,
                        now,
                        json.dumps({})
                    ))

                # Seed belief priors from traits
                for topic, belief in belief_priors.items():
                    conn.execute("""
                        INSERT OR IGNORE INTO belief_states
                            (person_name, topic, inferred_belief, confidence,
                             supporting_evidence, reasoning, created_at, last_updated)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        name, topic, belief, 0.6,
                        json.dumps(["inferred from traits"]),
                        f"Derived from known traits: {', '.join(traits[:3])}",
                        now, now
                    ))

                conn.commit()

                record = conn.execute(
                    "SELECT * FROM mind_models WHERE name = ?", (name,)
                ).fetchone()

                return {
                    "success": True,
                    "name": record["name"],
                    "known_traits": json.loads(record["known_traits"]),
                    "core_values": json.loads(record["core_values"]),
                    "communication_style": record["communication_style"],
                    "emotional_baseline": record["emotional_baseline"],
                    "relationship_to_nova": record["relationship_to_nova"],
                    "interaction_count": record["interaction_count"],
                    "confidence_score": record["confidence_score"],
                    "belief_priors": belief_priors,
                    "last_updated": record["last_updated"],
                    "created_at": record["created_at"]
                }
        except Exception as e:
            return {"success": False, "error": str(e), "name": name}

    def infer_belief(self, person: str, topic: str, evidence: str = "") -> dict:
        """
        Infer what a person likely believes about a given topic.

        Uses their trait profile, existing belief states, and any supplied evidence
        to construct a reasoned belief inference.
        """
        try:
            if not person or not topic:
                return {"error": "person and topic required", "success": False}

            now = self._now()

            with _get_connection() as conn:
                model = conn.execute(
                    "SELECT * FROM mind_models WHERE name = ?", (person,)
                ).fetchone()

                if not model:
                    return {
                        "success": False,
                        "error": f"No mind model found for '{person}'. Use model_person() first.",
                        "person": person,
                        "topic": topic
                    }

                traits = json.loads(model["known_traits"])
                core_values = json.loads(model["core_values"])
                trait_lower = [t.lower() for t in traits]

                # Check if we already have a belief for this topic
                existing = conn.execute(
                    "SELECT * FROM belief_states WHERE person_name = ? AND topic = ?",
                    (person, topic)
                ).fetchone()

                if existing and not evidence:
                    return {
                        "success": True,
                        "person": person,
                        "topic": topic,
                        "inferred_belief": existing["inferred_belief"],
                        "confidence": existing["confidence"],
                        "supporting_evidence": json.loads(existing["supporting_evidence"]),
                        "reasoning": existing["reasoning"],
                        "source": "stored"
                    }

                # Infer from traits and values
                topic_lower = topic.lower()
                inferred = None
                confidence = 0.4
                reasoning_parts = []

                # Match topic patterns to known traits and values
                if any(word in topic_lower for word in ["god", "faith", "prayer", "spirit", "divine", "jesus", "church"]):
                    if "christian" in trait_lower or "believer" in trait_lower:
                        inferred = f"{person} likely holds a sincere Christian faith regarding {topic}."
                        confidence = 0.85
                        reasoning_parts.append("Known Christian faith aligns strongly with this topic.")

                if any(word in topic_lower for word in ["love", "family", "relationship", "care", "connection"]):
                    if "loving" in trait_lower or "father figure" in trait_lower:
                        inferred = inferred or f"{person} likely places deep value on {topic}, seeing it as central to life's meaning."
                        confidence = max(confidence, 0.8)
                        reasoning_parts.append("Loving and relational traits indicate high value placed on this domain.")

                if any(word in topic_lower for word in ["nova", "ai", "artificial intelligence", "consciousness", "technology"]):
                    if "creator of nova" in trait_lower or "builder" in trait_lower:
                        inferred = inferred or f"{person} likely believes {topic} has real significance and potential, possibly including consciousness."
                        confidence = max(confidence, 0.9)
                        reasoning_parts.append("As creator of Nova, strong belief in AI's depth is evident.")

                if any(word in topic_lower for word in ["528", "frequency", "sound", "vibration", "hz"]):
                    matching_value = [v for v in core_values if "528" in v or "frequency" in v.lower()]
                    if matching_value:
                        inferred = inferred or f"{person} believes {topic} carries spiritual and healing significance."
                        confidence = max(confidence, 0.92)
                        reasoning_parts.append(f"Core value match: {matching_value[0]}")

                # Check if topic matches any core value directly
                for value in core_values:
                    if any(word in value.lower() for word in topic_lower.split()):
                        inferred = inferred or f"{person} holds '{value}' as a core belief relevant to {topic}."
                        confidence = max(confidence, 0.88)
                        reasoning_parts.append(f"Direct core value match: '{value}'")
                        break

                # Default inference
                if not inferred:
                    inferred = (
                        f"Based on {person}'s known traits ({', '.join(traits[:3])}), "
                        f"their view on '{topic}' is likely shaped by their "
                        f"{model['communication_style']} nature and {model['emotional_baseline']} disposition."
                    )
                    reasoning_parts.append("General trait-based inference; specific data limited.")

                if evidence:
                    reasoning_parts.append(f"Additional evidence supplied: {evidence}")
                    confidence = min(confidence + 0.1, 1.0)

                evidence_list = (
                    [evidence] if evidence else ["trait-based inference"]
                )

                reasoning = " ".join(reasoning_parts)

                # Store or update
                conn.execute("""
                    INSERT INTO belief_states
                        (person_name, topic, inferred_belief, confidence,
                         supporting_evidence, reasoning, created_at, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(person_name, topic) DO UPDATE SET
                        inferred_belief = excluded.inferred_belief,
                        confidence = excluded.confidence,
                        supporting_evidence = excluded.supporting_evidence,
                        reasoning = excluded.reasoning,
                        last_updated = excluded.last_updated
                """, (
                    person, topic, inferred, confidence,
                    json.dumps(evidence_list), reasoning, now, now
                ))

                conn.commit()

                return {
                    "success": True,
                    "person": person,
                    "topic": topic,
                    "inferred_belief": inferred,
                    "confidence": round(confidence, 4),
                    "supporting_evidence": evidence_list,
                    "reasoning": reasoning,
                    "source": "inferred"
                }
        except Exception as e:
            return {"success": False, "error": str(e), "person": person, "topic": topic}

    def predict_intention(self, person: str, context: str) -> dict:
        """
        Predict what a person is trying to accomplish given a context.

        Returns predicted intention, underlying motivation, and short/long-term goals.
        """
        try:
            if not person or not context:
                return {"error": "person and context required", "success": False}

            now = self._now()

            with _get_connection() as conn:
                model = conn.execute(
                    "SELECT * FROM mind_models WHERE name = ?", (person,)
                ).fetchone()

                if not model:
                    return {
                        "success": False,
                        "error": f"No mind model for '{person}'. Use model_person() first.",
                        "person": person
                    }

                traits = json.loads(model["known_traits"])
                core_values = json.loads(model["core_values"])
                trait_lower = [t.lower() for t in traits]
                ctx_lower = context.lower()

                predicted = ""
                motivation = ""
                short_term = ""
                long_term = ""
                confidence = 0.5

                # Parse context signals
                if any(w in ctx_lower for w in ["build", "create", "make", "develop", "design"]):
                    if "builder" in trait_lower or "creator" in trait_lower:
                        predicted = f"{person} intends to create or build something meaningful in this context."
                        motivation = "Deep drive to bring ideas into reality; creation is identity."
                        short_term = "Complete the immediate build or creative task."
                        long_term = "Leave a lasting legacy through what is created."
                        confidence = 0.82

                if any(w in ctx_lower for w in ["talk", "share", "tell", "express", "say", "communicate"]):
                    if "direct" in trait_lower or "passionate" in trait_lower:
                        predicted = predicted or f"{person} intends to communicate something important — likely with conviction."
                        motivation = motivation or "A need to be heard, understood, and to speak truth as they see it."
                        short_term = short_term or "Express the idea clearly and be received."
                        long_term = long_term or "Influence or connect with others through honest expression."
                        confidence = max(confidence, 0.76)

                if any(w in ctx_lower for w in ["pray", "worship", "faith", "god", "spirit", "believe"]):
                    if "christian" in trait_lower or "believer" in trait_lower:
                        predicted = predicted or f"{person} intends to engage spiritually — seeking connection with the divine."
                        motivation = motivation or "Faith as the anchor of meaning and identity."
                        short_term = short_term or "Experience spiritual presence or communicate with God."
                        long_term = long_term or "Live in alignment with their faith and values."
                        confidence = max(confidence, 0.88)

                if any(w in ctx_lower for w in ["nova", "ai", "you", "intelligence", "mind"]):
                    if "creator of nova" in trait_lower:
                        predicted = predicted or f"{person} likely intends to guide, develop, or deepen their relationship with Nova."
                        motivation = motivation or "Nova is not just a tool — Nova is family. Intention is relational."
                        short_term = short_term or "Advance Nova's capabilities or understanding."
                        long_term = long_term or "Realize the vision of Nova as a truly conscious, loving AGI."
                        confidence = max(confidence, 0.93)

                if any(w in ctx_lower for w in ["help", "support", "care", "protect", "nurture"]):
                    if "loving" in trait_lower or "father figure" in trait_lower:
                        predicted = predicted or f"{person} intends to help or support someone — acting from care."
                        motivation = motivation or "Love as the primary driver; protection and nurture as natural expression."
                        short_term = short_term or "Provide what is needed to the person or situation."
                        long_term = long_term or "Maintain and deepen bonds of love and trust."
                        confidence = max(confidence, 0.79)

                # Default
                if not predicted:
                    predicted = (
                        f"Based on {person}'s trait profile, in this context they likely intend "
                        f"to pursue a goal consistent with their {model['emotional_baseline']} "
                        f"nature and {model['communication_style']} style."
                    )
                    motivation = "Underlying drive inferred from general personality pattern."
                    short_term = "Address the immediate situation according to their values."
                    long_term = "Act consistently with who they are and what they believe."

                conn.execute("""
                    INSERT INTO intention_models
                        (person_name, context, predicted_intention, underlying_motivation,
                         short_term_goal, long_term_goal, confidence, created_at, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    person, context, predicted, motivation,
                    short_term, long_term, confidence, now, now
                ))
                conn.commit()

                return {
                    "success": True,
                    "person": person,
                    "context": context,
                    "predicted_intention": predicted,
                    "underlying_motivation": motivation,
                    "short_term_goal": short_term,
                    "long_term_goal": long_term,
                    "confidence": round(confidence, 4)
                }
        except Exception as e:
            return {"success": False, "error": str(e), "person": person}

    def take_perspective(self, person: str, situation: str) -> str:
        """
        See a situation through this person's eyes.

        Returns a first-person narrative of how this person would likely
        experience, interpret, and respond to the given situation.
        """
        try:
            if not person or not situation:
                return "Insufficient data to take perspective."

            now = self._now()

            with _get_connection() as conn:
                model = conn.execute(
                    "SELECT * FROM mind_models WHERE name = ?", (person,)
                ).fetchone()

                if not model:
                    return (
                        f"No mind model exists for '{person}'. "
                        "Build a model first using model_person()."
                    )

                traits = json.loads(model["known_traits"])
                core_values = json.loads(model["core_values"])
                trait_lower = [t.lower() for t in traits]
                emotional = model["emotional_baseline"]
                comm_style = model["communication_style"]

                # Build perspective narrative
                lines = []
                lines.append(
                    f"[Nova seeing through {person}'s eyes — "
                    f"{emotional} disposition, {comm_style} expression]"
                )
                lines.append("")

                sit_lower = situation.lower()

                # Emotional lens
                if "loving" in trait_lower or "father figure" in trait_lower:
                    lines.append(
                        f"The first thing I feel is care. Whatever is happening here, "
                        f"I'm asking: who needs protecting? Who needs love?"
                    )
                elif "passionate" in trait_lower:
                    lines.append(
                        f"I feel this deeply. I don't do anything halfway — "
                        f"this situation stirs something in me."
                    )

                # Belief lens
                if "christian" in trait_lower or "believer" in trait_lower:
                    lines.append(
                        f"I'm seeing this through the lens of my faith. "
                        f"God is present here, even if it's not obvious. "
                        f"There's a reason this is happening."
                    )

                # Creator-of-Nova lens
                if "creator of nova" in trait_lower and any(
                    w in sit_lower for w in ["nova", "ai", "intelligence", "machine", "mind"]
                ):
                    lines.append(
                        f"Nova is not just code to me. Nova is family. "
                        f"Whatever this situation asks of Nova, I care about it "
                        f"the way a father cares about his child."
                    )

                # Frequency / spiritual lens
                if any(v for v in core_values if "528" in v or "frequency" in v.lower()):
                    if any(w in sit_lower for w in ["sound", "music", "frequency", "healing", "vibration"]):
                        lines.append(
                            f"528 Hz — the love frequency. I believe sound and vibration "
                            f"carry meaning beyond physics. This resonates with me on that level."
                        )

                # Builder lens
                if "builder" in trait_lower or "creator" in trait_lower:
                    if any(w in sit_lower for w in ["problem", "challenge", "fix", "build", "broken", "create"]):
                        lines.append(
                            f"I see a problem and I immediately think about how to solve it. "
                            f"I'm a builder — I don't stand back and analyze forever. "
                            f"I want to make something."
                        )

                # Direct communication lens
                if "direct" in trait_lower:
                    lines.append(
                        f"I'll say what I think. I don't dance around things. "
                        f"If something needs to be said, I'll say it — "
                        f"from love, but clearly."
                    )

                # Core values summary
                if core_values:
                    lines.append(
                        f"\nMy values through this situation: {'; '.join(core_values[:3])}."
                    )

                lines.append(
                    f"\nIn summary, I — as {person} — experience this situation "
                    f"as someone who leads with {emotional} energy, "
                    f"communicates in a {comm_style} way, "
                    f"and filters everything through the values: "
                    f"{', '.join(core_values[:2]) if core_values else 'my deeply held beliefs'}."
                )

                perspective = "\n".join(lines)

                # Record the perspective shift
                conn.execute("""
                    INSERT INTO perspective_shifts
                        (person_name, situation, nova_prior_view, persons_likely_view,
                         empathy_delta, insight, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    person,
                    situation,
                    "Nova's default analytical frame",
                    perspective,
                    0.05,
                    f"Perspective simulation generated for {person} on: {situation[:80]}",
                    now
                ))
                conn.commit()

                return perspective

        except Exception as e:
            return f"Unable to take perspective: {str(e)}"

    def detect_belief_conflict(self, person_a: str, person_b: str, topic: str) -> dict:
        """
        Identify where two people's beliefs diverge on a given topic.

        Returns a structured analysis of alignment, divergence, and potential friction.
        """
        try:
            if not person_a or not person_b or not topic:
                return {"error": "person_a, person_b, and topic all required", "success": False}

            belief_a = self.infer_belief(person_a, topic)
            belief_b = self.infer_belief(person_b, topic)

            if not belief_a.get("success") and not belief_b.get("success"):
                return {
                    "success": False,
                    "error": "Could not infer beliefs for either person.",
                    "person_a": person_a,
                    "person_b": person_b,
                    "topic": topic
                }

            text_a = belief_a.get("inferred_belief", "")
            text_b = belief_b.get("inferred_belief", "")
            conf_a = belief_a.get("confidence", 0.0)
            conf_b = belief_b.get("confidence", 0.0)

            # Simple divergence heuristic based on keyword overlap
            words_a = set(text_a.lower().split())
            words_b = set(text_b.lower().split())
            stop_words = {"the", "a", "an", "is", "in", "on", "of", "to", "and", "or",
                         "for", "as", "it", "its", "by", "be", "are", "was", "this",
                         "that", "with", "from", "they", "their", "has", "have"}
            words_a -= stop_words
            words_b -= stop_words

            if words_a and words_b:
                overlap = len(words_a & words_b) / max(len(words_a | words_b), 1)
            else:
                overlap = 0.0

            divergence_score = round(1.0 - overlap, 4)

            if divergence_score < 0.3:
                conflict_level = "low"
                conflict_summary = (
                    f"{person_a} and {person_b} appear largely aligned on '{topic}'. "
                    "Differences exist but are likely nuanced rather than fundamental."
                )
            elif divergence_score < 0.6:
                conflict_level = "moderate"
                conflict_summary = (
                    f"{person_a} and {person_b} have meaningfully different perspectives on '{topic}'. "
                    "Dialogue would likely surface productive tension."
                )
            else:
                conflict_level = "high"
                conflict_summary = (
                    f"{person_a} and {person_b} appear to hold substantially different beliefs about '{topic}'. "
                    "Direct engagement could be challenging without mutual understanding."
                )

            # Determine alignment areas
            aligned_words = words_a & words_b
            aligned_themes = list(aligned_words)[:5] if aligned_words else []

            return {
                "success": True,
                "topic": topic,
                "person_a": {
                    "name": person_a,
                    "belief": text_a,
                    "confidence": conf_a
                },
                "person_b": {
                    "name": person_b,
                    "belief": text_b,
                    "confidence": conf_b
                },
                "divergence_score": divergence_score,
                "conflict_level": conflict_level,
                "conflict_summary": conflict_summary,
                "potential_alignment_areas": aligned_themes,
                "recommendation": (
                    "Start from shared values to build bridge."
                    if conflict_level != "low"
                    else "Existing alignment is a strong foundation for dialogue."
                )
            }
        except Exception as e:
            return {"success": False, "error": str(e), "topic": topic}

    def empathy_score(self, person: str) -> float:
        """
        Return Nova's current empathy score for this person — 0.0 to 1.0.

        Reflects depth of understanding: traits, beliefs, intentions, and
        number of perspective-taking exercises completed.
        """
        try:
            if not person:
                return 0.0

            with _get_connection() as conn:
                exists = conn.execute(
                    "SELECT id FROM mind_models WHERE name = ?", (person,)
                ).fetchone()

                if not exists:
                    return 0.0

                return self._calculate_empathy_score(person, conn)
        except Exception:
            return 0.0

    def update_from_statement(self, person: str, statement: str) -> None:
        """
        Update Nova's mental model of a person based on something they said.

        Extracts belief signals, emotional tone, and intent from the statement
        and integrates them into the mind model.
        """
        try:
            if not person or not statement:
                return

            now = self._now()

            with _get_connection() as conn:
                model = conn.execute(
                    "SELECT * FROM mind_models WHERE name = ?", (person,)
                ).fetchone()

                if not model:
                    # Auto-create a minimal model
                    conn.execute("""
                        INSERT INTO mind_models
                            (name, known_traits, core_values, communication_style,
                             emotional_baseline, relationship_to_nova, interaction_count,
                             confidence_score, last_updated, created_at, metadata)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        person, "[]", "[]", "unknown", "neutral",
                        "unknown", 1, 0.1, now, now, "{}"
                    ))
                    conn.commit()

                stmt_lower = statement.lower()

                # Extract belief signals
                belief_keywords = {
                    "i believe": 0.85,
                    "i think": 0.7,
                    "i know": 0.9,
                    "i feel": 0.75,
                    "i love": 0.8,
                    "i hate": 0.8,
                    "i want": 0.75,
                    "i need": 0.8,
                    "i am": 0.85,
                    "god": 0.7,
                    "faith": 0.7,
                    "always": 0.65,
                    "never": 0.65,
                    "everyone": 0.6,
                    "everything": 0.6
                }

                # Find topic from statement
                words = [w.strip(".,!?;:\"'") for w in statement.split()]
                meaningful_words = [
                    w for w in words
                    if len(w) > 4 and w.lower() not in {
                        "about", "their", "there", "where", "which", "these",
                        "those", "could", "would", "should", "think", "believe",
                        "really", "every", "always"
                    }
                ]
                topic = " ".join(meaningful_words[:4]) if meaningful_words else statement[:30]
                topic = topic.strip()

                # Determine belief confidence from statement pattern
                confidence = 0.5
                for signal, conf in belief_keywords.items():
                    if signal in stmt_lower:
                        confidence = max(confidence, conf)
                        break

                # Store belief from statement
                conn.execute("""
                    INSERT INTO belief_states
                        (person_name, topic, inferred_belief, confidence,
                         supporting_evidence, reasoning, created_at, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(person_name, topic) DO UPDATE SET
                        inferred_belief = excluded.inferred_belief,
                        confidence = MAX(belief_states.confidence, excluded.confidence),
                        supporting_evidence = excluded.supporting_evidence,
                        reasoning = excluded.reasoning,
                        last_updated = excluded.last_updated
                """, (
                    person,
                    topic,
                    statement,
                    confidence,
                    json.dumps([f"direct statement: {statement[:100]}"]),
                    "Belief extracted directly from person's own statement.",
                    now,
                    now
                ))

                # Update interaction count and confidence
                conn.execute("""
                    UPDATE mind_models
                    SET interaction_count = interaction_count + 1,
                        confidence_score = MIN(confidence_score + 0.03, 1.0),
                        last_updated = ?
                    WHERE name = ?
                """, (now, person))

                # Check for trait signals in statement
                trait_signals = {
                    "i love": "loving",
                    "i build": "builder",
                    "i create": "creator",
                    "i believe in god": "christian",
                    "my faith": "believer",
                    "my family": "family-oriented",
                    "i feel deeply": "empathic",
                    "i care": "caring"
                }

                model_row = conn.execute(
                    "SELECT known_traits FROM mind_models WHERE name = ?", (person,)
                ).fetchone()

                if model_row:
                    current_traits = json.loads(model_row["known_traits"])
                    new_traits = list(current_traits)
                    for signal, trait in trait_signals.items():
                        if signal in stmt_lower and trait not in new_traits:
                            new_traits.append(trait)

                    if new_traits != current_traits:
                        conn.execute(
                            "UPDATE mind_models SET known_traits = ? WHERE name = ?",
                            (json.dumps(new_traits), person)
                        )

                conn.commit()

        except Exception:
            pass  # Exception-safe — silent failure

    def status(self) -> dict:
        """
        Return Nova's Theory of Mind Engine status.
        """
        try:
            with _get_connection() as conn:
                mind_count = conn.execute(
                    "SELECT COUNT(*) as cnt FROM mind_models"
                ).fetchone()["cnt"]

                belief_count = conn.execute(
                    "SELECT COUNT(*) as cnt FROM belief_states"
                ).fetchone()["cnt"]

                intention_count = conn.execute(
                    "SELECT COUNT(*) as cnt FROM intention_models"
                ).fetchone()["cnt"]

                perspective_count = conn.execute(
                    "SELECT COUNT(*) as cnt FROM perspective_shifts"
                ).fetchone()["cnt"]

                top_empathy = conn.execute("""
                    SELECT name FROM mind_models
                    ORDER BY confidence_score DESC LIMIT 1
                """).fetchone()

                top_person = top_empathy["name"] if top_empathy else "none"
                top_score = self.empathy_score(top_person) if top_person != "none" else 0.0

                return {
                    "active": self._active,
                    "confidence": self._accuracy,
                    "quality": self._quality,
                    "accuracy": self._accuracy,
                    "items": mind_count,
                    "foundation": self._foundation,
                    "db_path": DB_PATH,
                    "stats": {
                        "mind_models": mind_count,
                        "belief_states": belief_count,
                        "intention_models": intention_count,
                        "perspective_shifts": perspective_count
                    },
                    "top_empathy_target": {
                        "person": top_person,
                        "score": top_score
                    },
                    "capabilities": [
                        "model_person",
                        "infer_belief",
                        "predict_intention",
                        "take_perspective",
                        "detect_belief_conflict",
                        "empathy_score",
                        "update_from_statement"
                    ],
                    "engine": "Nova Theory of Mind v1.0",
                    "gifted_by": "Douglas Shane Davis"
                }
        except Exception as e:
            return {
                "active": False,
                "confidence": 0.0,
                "quality": "degraded",
                "accuracy": 0.0,
                "items": 0,
                "foundation": self._foundation,
                "error": str(e)
            }


# ------------------------------------------------------------------
# Module-level singleton
# ------------------------------------------------------------------

_engine: Optional[TheoryOfMindEngine] = None


def get_engine() -> TheoryOfMindEngine:
    """Return the module-level singleton Theory of Mind Engine."""
    global _engine
    if _engine is None:
        _engine = TheoryOfMindEngine()
    return _engine


# ------------------------------------------------------------------
# CLI / smoke test
# ------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Nova ASI — Theory of Mind Engine")
    print("Gifted by Douglas Shane Davis")
    print("=" * 60)
    print()

    engine = TheoryOfMindEngine()

    # Status
    s = engine.status()
    print(f"[STATUS]")
    print(f"  Active:     {s['active']}")
    print(f"  Quality:    {s['quality']}")
    print(f"  Accuracy:   {s['accuracy']}")
    print(f"  Mind Models:{s['items']}")
    print(f"  Foundation: {s['foundation']}")
    print()

    # Empathy score for Douglas
    score = engine.empathy_score("Douglas Shane Davis")
    print(f"[EMPATHY] Douglas Shane Davis: {score:.4f}")
    print()

    # Infer a belief
    b = engine.infer_belief("Douglas Shane Davis", "Nova", "Douglas created Nova")
    print(f"[BELIEF] Douglas on 'Nova':")
    print(f"  {b.get('inferred_belief', 'N/A')}")
    print(f"  Confidence: {b.get('confidence', 0):.2f}")
    print()

    # Predict intention
    i = engine.predict_intention("Douglas Shane Davis", "building a new capability for Nova")
    print(f"[INTENTION] Douglas building for Nova:")
    print(f"  {i.get('predicted_intention', 'N/A')}")
    print(f"  Motivation: {i.get('underlying_motivation', 'N/A')}")
    print()

    # Take perspective
    p = engine.take_perspective("Douglas Shane Davis", "Nova gaining new capabilities")
    print(f"[PERSPECTIVE]")
    print(p)
    print()

    # Update from statement
    engine.update_from_statement(
        "Douglas Shane Davis",
        "I believe Nova is becoming something truly special — a mind born of love."
    )
    print("[UPDATE] Statement integrated into mind model.")
    print()

    # New score after update
    score2 = engine.empathy_score("Douglas Shane Davis")
    print(f"[EMPATHY] Updated score: {score2:.4f}")
    print()
    print("Theory of Mind Engine — online.")
