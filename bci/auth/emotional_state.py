"""
Emotional State Detection System
Implementation of US Patent 5507291: "Method and an associated apparatus for remotely determining
information as to a person's emotional state"

Analyzes brain wave patterns to determine user's emotional state remotely.
"""
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from bci.core.interfaces import BCIReading, MentalState, BrainwaveData


class EmotionalState(Enum):
    """Primary emotional states detectable from brain activity"""
    CALM = "calm"
    EXCITED = "excited"
    STRESSED = "stressed"
    FOCUSED = "focused"
    RELAXED = "relaxed"
    ANXIOUS = "anxious"
    CONTENT = "content"
    FRUSTRATED = "frustrated"
    CONFIDENT = "confident"
    UNCERTAIN = "uncertain"
    ALERT = "alert"
    DROWSY = "drowsy"


class EmotionalValence(Enum):
    """Emotional valence (positive/negative dimension)"""
    VERY_NEGATIVE = "very_negative"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"
    VERY_POSITIVE = "very_positive"


class EmotionalArousal(Enum):
    """Emotional arousal (activation level)"""
    VERY_LOW = "very_low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class EmotionalProfile:
    """Complete emotional state profile"""
    primary_emotion: EmotionalState
    valence: EmotionalValence
    arousal: EmotionalArousal
    confidence: float
    timestamp: str

    # Dimensional scores (0-1)
    positivity: float
    activation: float
    dominance: float  # Feeling in control

    # Specific emotional dimensions
    stress_level: float
    anxiety_level: float
    engagement_level: float
    cognitive_load: float

    # Brain wave correlates
    brainwave_pattern: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "primary_emotion": self.primary_emotion.value,
            "valence": self.valence.value,
            "arousal": self.arousal.value,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "positivity": self.positivity,
            "activation": self.activation,
            "dominance": self.dominance,
            "stress_level": self.stress_level,
            "anxiety_level": self.anxiety_level,
            "engagement_level": self.engagement_level,
            "cognitive_load": self.cognitive_load,
            "brainwave_pattern": self.brainwave_pattern
        }


class EmotionalStateDetector:
    """
    Advanced emotional state detection system
    Based on US Patent 5507291

    Analyzes EEG patterns to determine user's emotional state
    without requiring self-report or physical indicators.
    """

    def __init__(self):
        self.emotional_history: List[EmotionalProfile] = []
        self.baseline_established = False
        self.baseline_profile: Optional[EmotionalProfile] = None
        self.max_history = 500

        # Brainwave-emotion correlation parameters
        self.emotion_signatures = self._initialize_emotion_signatures()

        print("[EmotionalDetector] Emotional State Detection System initialized")
        print("[EmotionalDetector] Based on US Patent 5507291")

    def _initialize_emotion_signatures(self) -> Dict[EmotionalState, Dict[str, Tuple[float, float]]]:
        """
        Initialize known brain wave signatures for different emotional states
        Format: {emotion: {band: (preferred_value, tolerance)}}
        """
        return {
            EmotionalState.CALM: {
                "alpha": (250000, 100000),  # High alpha
                "beta": (80000, 40000),      # Low beta
                "theta": (150000, 80000),    # Moderate theta
                "gamma": (50000, 30000),     # Low gamma
                "delta": (200000, 100000)    # Moderate delta
            },
            EmotionalState.EXCITED: {
                "beta": (250000, 80000),     # High beta
                "gamma": (180000, 60000),    # High gamma
                "alpha": (100000, 50000),    # Low alpha
                "theta": (100000, 50000),    # Low theta
            },
            EmotionalState.STRESSED: {
                "beta": (280000, 80000),     # Very high beta
                "gamma": (150000, 50000),    # High gamma
                "alpha": (80000, 40000),     # Very low alpha
                "theta": (120000, 60000),    # Low theta
            },
            EmotionalState.FOCUSED: {
                "beta": (220000, 70000),     # High beta
                "gamma": (140000, 50000),    # Moderate-high gamma
                "alpha": (150000, 60000),    # Moderate alpha
                "theta": (100000, 50000),    # Low theta
            },
            EmotionalState.RELAXED: {
                "alpha": (300000, 80000),    # Very high alpha
                "theta": (200000, 80000),    # High theta
                "beta": (70000, 30000),      # Very low beta
                "delta": (250000, 100000),   # High delta
            },
            EmotionalState.ANXIOUS: {
                "beta": (260000, 80000),     # High beta
                "gamma": (160000, 60000),    # High gamma
                "alpha": (90000, 40000),     # Low alpha
                "theta": (180000, 70000),    # Moderate-high theta
            },
            EmotionalState.FRUSTRATED: {
                "beta": (240000, 70000),     # High beta
                "theta": (140000, 60000),    # Low-moderate theta
                "alpha": (120000, 50000),    # Low alpha
                "gamma": (130000, 50000),    # Moderate gamma
            },
            EmotionalState.CONFIDENT: {
                "alpha": (220000, 70000),    # Moderate-high alpha
                "beta": (180000, 60000),     # Moderate beta
                "gamma": (100000, 40000),    # Moderate gamma
                "theta": (130000, 50000),    # Low-moderate theta
            },
            EmotionalState.DROWSY: {
                "theta": (350000, 100000),   # Very high theta
                "delta": (400000, 120000),   # Very high delta
                "alpha": (180000, 70000),    # Moderate alpha
                "beta": (60000, 30000),      # Very low beta
            },
            EmotionalState.ALERT: {
                "beta": (230000, 70000),     # High beta
                "gamma": (150000, 50000),    # High gamma
                "alpha": (140000, 60000),    # Moderate alpha
                "delta": (100000, 50000),    # Low delta
            }
        }

    def detect_emotional_state(self, reading: BCIReading) -> Optional[EmotionalProfile]:
        """
        Detect user's emotional state from brain wave reading
        Uses pattern matching against known emotional signatures
        """
        if not reading.brainwaves or not reading.mental_state:
            return None

        bw = reading.brainwaves
        ms = reading.mental_state

        # Step 1: Match against known emotional signatures
        emotion_scores = {}
        for emotion, signature in self.emotion_signatures.items():
            score = self._match_signature(bw, signature)
            emotion_scores[emotion] = score

        # Find best matching emotion
        primary_emotion = max(emotion_scores, key=emotion_scores.get)
        confidence = emotion_scores[primary_emotion]

        # Step 2: Determine valence (positive/negative)
        valence = self._compute_valence(ms, bw)

        # Step 3: Determine arousal (activation level)
        arousal = self._compute_arousal(ms, bw)

        # Step 4: Compute dimensional scores
        positivity = (ms.emotional_valence + 1.0) / 2.0  # Convert from [-1,1] to [0,1]
        activation = ms.arousal
        dominance = self._compute_dominance(ms)

        # Step 5: Specific emotional dimensions
        stress_level = ms.stress_level
        anxiety_level = self._compute_anxiety(ms, bw)
        engagement_level = ms.engagement
        cognitive_load = ms.cognitive_load

        # Create emotional profile
        profile = EmotionalProfile(
            primary_emotion=primary_emotion,
            valence=valence,
            arousal=arousal,
            confidence=confidence,
            timestamp=reading.timestamp.isoformat(),
            positivity=positivity,
            activation=activation,
            dominance=dominance,
            stress_level=stress_level,
            anxiety_level=anxiety_level,
            engagement_level=engagement_level,
            cognitive_load=cognitive_load,
            brainwave_pattern=bw.to_dict()
        )

        # Add to history
        self.emotional_history.append(profile)
        if len(self.emotional_history) > self.max_history:
            self.emotional_history.pop(0)

        # Establish baseline if needed
        if not self.baseline_established and len(self.emotional_history) >= 20:
            self.baseline_profile = self._compute_baseline()
            self.baseline_established = True
            print("[EmotionalDetector] ✅ Emotional baseline established")

        return profile

    def _match_signature(self, brainwaves: BrainwaveData,
                        signature: Dict[str, Tuple[float, float]]) -> float:
        """
        Match current brainwaves against emotional signature
        Returns confidence score (0-1)
        """
        scores = []

        for band, (preferred, tolerance) in signature.items():
            current = getattr(brainwaves, band)

            # Compute how close current value is to preferred
            diff = abs(current - preferred)
            normalized_diff = diff / (tolerance + 1e-6)

            # Convert to similarity score
            similarity = max(0.0, 1.0 - normalized_diff)
            scores.append(similarity)

        # Average similarity across all bands
        return sum(scores) / len(scores) if scores else 0.0

    def _compute_valence(self, mental_state: MentalState,
                        brainwaves: BrainwaveData) -> EmotionalValence:
        """
        Compute emotional valence (positive/negative)
        Based on asymmetry in brain activity and mental state
        """
        valence_score = mental_state.emotional_valence

        # Alpha asymmetry (left vs right prefrontal)
        # Simplified: using overall alpha level as proxy
        alpha_contribution = (brainwaves.alpha - 200000) / 200000  # Normalize around 200k

        # Combined valence
        combined = (valence_score + alpha_contribution) / 2

        if combined < -0.6:
            return EmotionalValence.VERY_NEGATIVE
        elif combined < -0.2:
            return EmotionalValence.NEGATIVE
        elif combined < 0.2:
            return EmotionalValence.NEUTRAL
        elif combined < 0.6:
            return EmotionalValence.POSITIVE
        else:
            return EmotionalValence.VERY_POSITIVE

    def _compute_arousal(self, mental_state: MentalState,
                        brainwaves: BrainwaveData) -> EmotionalArousal:
        """
        Compute emotional arousal (activation level)
        Based on beta/gamma activity and mental arousal
        """
        # High beta and gamma = high arousal
        beta_normalized = min(1.0, brainwaves.beta / 300000)
        gamma_normalized = min(1.0, brainwaves.gamma / 200000)

        brainwave_arousal = (beta_normalized + gamma_normalized) / 2
        mental_arousal = mental_state.arousal

        combined = (brainwave_arousal + mental_arousal) / 2

        if combined < 0.2:
            return EmotionalArousal.VERY_LOW
        elif combined < 0.4:
            return EmotionalArousal.LOW
        elif combined < 0.6:
            return EmotionalArousal.MODERATE
        elif combined < 0.8:
            return EmotionalArousal.HIGH
        else:
            return EmotionalArousal.VERY_HIGH

    def _compute_dominance(self, mental_state: MentalState) -> float:
        """
        Compute feeling of control/dominance
        High attention + low stress = high dominance
        """
        dominance = (
            mental_state.attention * 0.5 +
            (1.0 - mental_state.stress_level) * 0.3 +
            mental_state.engagement * 0.2
        )
        return min(1.0, max(0.0, dominance))

    def _compute_anxiety(self, mental_state: MentalState,
                        brainwaves: BrainwaveData) -> float:
        """
        Compute anxiety level
        High beta + high theta + high stress = anxiety
        """
        beta_factor = min(1.0, brainwaves.beta / 280000)
        theta_factor = min(1.0, brainwaves.theta / 200000)

        anxiety = (
            mental_state.stress_level * 0.4 +
            beta_factor * 0.3 +
            theta_factor * 0.2 +
            (1.0 - mental_state.meditation) * 0.1
        )

        return min(1.0, max(0.0, anxiety))

    def _compute_baseline(self) -> EmotionalProfile:
        """Compute baseline emotional profile from history"""
        if not self.emotional_history:
            return None

        recent = self.emotional_history[-20:]

        # Average all metrics
        avg_positivity = np.mean([p.positivity for p in recent])
        avg_activation = np.mean([p.activation for p in recent])
        avg_dominance = np.mean([p.dominance for p in recent])
        avg_stress = np.mean([p.stress_level for p in recent])
        avg_anxiety = np.mean([p.anxiety_level for p in recent])
        avg_engagement = np.mean([p.engagement_level for p in recent])
        avg_cognitive_load = np.mean([p.cognitive_load for p in recent])

        # Most common emotion
        emotion_counts = {}
        for p in recent:
            emotion = p.primary_emotion
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

        most_common_emotion = max(emotion_counts, key=emotion_counts.get)

        return EmotionalProfile(
            primary_emotion=most_common_emotion,
            valence=EmotionalValence.NEUTRAL,
            arousal=EmotionalArousal.MODERATE,
            confidence=0.75,
            timestamp=datetime.utcnow().isoformat(),
            positivity=avg_positivity,
            activation=avg_activation,
            dominance=avg_dominance,
            stress_level=avg_stress,
            anxiety_level=avg_anxiety,
            engagement_level=avg_engagement,
            cognitive_load=avg_cognitive_load,
            brainwave_pattern={}
        )

    def get_emotional_trend(self, window_size: int = 10) -> Dict[str, Any]:
        """
        Analyze emotional trends over recent history
        Returns trend direction and stability
        """
        if len(self.emotional_history) < window_size:
            return {"status": "insufficient_data"}

        recent = self.emotional_history[-window_size:]

        # Compute trends
        positivity_trend = self._compute_metric_trend([p.positivity for p in recent])
        activation_trend = self._compute_metric_trend([p.activation for p in recent])
        stress_trend = self._compute_metric_trend([p.stress_level for p in recent])

        # Emotional stability (variance in emotions)
        emotions = [p.primary_emotion for p in recent]
        unique_emotions = len(set(emotions))
        stability = 1.0 - (unique_emotions / len(emotions))

        return {
            "positivity_trend": positivity_trend,
            "activation_trend": activation_trend,
            "stress_trend": stress_trend,
            "emotional_stability": stability,
            "current_emotion": recent[-1].primary_emotion.value,
            "dominant_emotion": max(set(emotions), key=emotions.count).value
        }

    def _compute_metric_trend(self, values: List[float]) -> str:
        """Compute trend direction from time series"""
        if len(values) < 2:
            return "stable"

        # Simple linear regression slope
        x = np.arange(len(values))
        y = np.array(values)

        slope = np.polyfit(x, y, 1)[0]

        if slope > 0.05:
            return "increasing"
        elif slope < -0.05:
            return "decreasing"
        else:
            return "stable"

    def detect_emotional_anomaly(self, current: EmotionalProfile) -> Optional[str]:
        """
        Detect if current emotional state is anomalous compared to baseline
        Returns warning message if anomaly detected
        """
        if not self.baseline_established:
            return None

        baseline = self.baseline_profile

        # Check for significant deviations
        anomalies = []

        stress_diff = current.stress_level - baseline.stress_level
        if stress_diff > 0.4:
            anomalies.append(f"Elevated stress (+{stress_diff:.2f})")

        anxiety_diff = current.anxiety_level - baseline.anxiety_level
        if anxiety_diff > 0.4:
            anomalies.append(f"Elevated anxiety (+{anxiety_diff:.2f})")

        positivity_diff = baseline.positivity - current.positivity
        if positivity_diff > 0.4:
            anomalies.append(f"Reduced positivity (-{positivity_diff:.2f})")

        if anomalies:
            return "Emotional anomaly detected: " + ", ".join(anomalies)

        return None

    def get_statistics(self) -> Dict[str, Any]:
        """Get emotional detection statistics"""
        if not self.emotional_history:
            return {"status": "no_data"}

        emotion_counts = {}
        for profile in self.emotional_history:
            emotion = profile.primary_emotion.value
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

        avg_confidence = np.mean([p.confidence for p in self.emotional_history])
        avg_positivity = np.mean([p.positivity for p in self.emotional_history])
        avg_stress = np.mean([p.stress_level for p in self.emotional_history])

        return {
            "total_readings": len(self.emotional_history),
            "baseline_established": self.baseline_established,
            "emotion_distribution": emotion_counts,
            "average_confidence": avg_confidence,
            "average_positivity": avg_positivity,
            "average_stress": avg_stress,
            "current_emotion": self.emotional_history[-1].primary_emotion.value if self.emotional_history else None
        }
