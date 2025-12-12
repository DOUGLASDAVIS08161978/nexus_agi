"""
BCI-MindControl Integration Layer
Bridges BCI systems with Nexus AGI MindControl decision-making
"""
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime
import uuid

from bci.core.interfaces import BCIReading, MentalState, MotorIntent
from bci.orchestrator import UniversalBCIOrchestrator


class BCIMindControlBridge:
    """
    Integration bridge between BCI systems and MindControl
    Translates neural signals into AGI decision inputs
    """

    def __init__(self, orchestrator: UniversalBCIOrchestrator, mindcontrol_conductor=None):
        self.orchestrator = orchestrator
        self.mindcontrol = mindcontrol_conductor
        self.active = False

        # Register BCI reading callback
        self.orchestrator.on_reading(self.process_bci_reading)

        # Neural state history for temporal analysis
        self.mental_state_history: List[MentalState] = []
        self.max_history = 100

        print("[BCI-MindControl] Integration bridge initialized")

    async def start(self):
        """Start BCI-MindControl integration"""
        self.active = True
        print("[BCI-MindControl] ✓ Integration active")

    async def stop(self):
        """Stop integration"""
        self.active = False
        print("[BCI-MindControl] Integration stopped")

    async def process_bci_reading(self, reading: BCIReading):
        """
        Process BCI reading and integrate with MindControl decision-making
        """
        if not self.active:
            return

        # Extract mental state
        if reading.mental_state:
            self._add_mental_state(reading.mental_state)

            # Generate cognitive context for MindControl
            cognitive_context = self._generate_cognitive_context(reading)

            # If MindControl is available, enhance decision-making
            if self.mindcontrol:
                await self._enhance_mindcontrol_decision(cognitive_context, reading)

    def _generate_cognitive_context(self, reading: BCIReading) -> Dict[str, Any]:
        """
        Generate cognitive context from BCI data for AGI decision-making
        """
        mental_state = reading.mental_state

        # Calculate cognitive metrics
        context = {
            "timestamp": reading.timestamp.isoformat(),
            "device_id": reading.device_id,
            "device_type": reading.device_type.value,

            # Core cognitive metrics
            "user_attention": mental_state.attention if mental_state else 0.5,
            "cognitive_load": mental_state.cognitive_load if mental_state else 0.5,
            "stress_level": mental_state.stress_level if mental_state else 0.3,
            "engagement": mental_state.engagement if mental_state else 0.5,

            # Derived metrics
            "mental_capacity": self._calculate_mental_capacity(mental_state),
            "decision_readiness": self._calculate_decision_readiness(mental_state),
            "risk_tolerance": self._calculate_risk_tolerance(mental_state),

            # Temporal analysis
            "attention_trend": self._calculate_trend("attention"),
            "stress_trend": self._calculate_trend("stress_level"),

            # Signal quality as confidence
            "signal_confidence": self._signal_quality_to_confidence(reading.signal_quality)
        }

        # Add motor intent if available
        if reading.motor_intent:
            context["motor_intent"] = {
                "type": reading.motor_intent.movement_type,
                "confidence": reading.motor_intent.confidence
            }

        return context

    async def _enhance_mindcontrol_decision(self, cognitive_context: Dict[str, Any],
                                           reading: BCIReading):
        """
        Enhance MindControl decisions with BCI cognitive context
        """
        # Create task with BCI-enhanced parameters
        task = {
            "id": str(uuid.uuid4()),
            "type": "bci_enhanced_task",
            "importance": cognitive_context["engagement"],
            "time_sensitivity": cognitive_context["user_attention"],

            # BCI-specific enhancements
            "user_trust_score": cognitive_context["decision_readiness"],
            "data_sensitivity": 1.0 - cognitive_context["risk_tolerance"],

            # Cognitive state
            "cognitive_context": cognitive_context,

            # Adaptive parameters based on user state
            "max_complexity": cognitive_context["mental_capacity"],
            "require_confirmation": cognitive_context["cognitive_load"] > 0.7,

            "capability_tag": "general"
        }

        # Let MindControl make decision with BCI context
        if hasattr(self.mindcontrol, 'decide'):
            decision = self.mindcontrol.decide(task)

            # Log integration
            print(f"[BCI-MindControl] Decision enhanced with neural context:")
            print(f"  User Attention: {cognitive_context['user_attention']:.2f}")
            print(f"  Cognitive Load: {cognitive_context['cognitive_load']:.2f}")
            print(f"  Decision Readiness: {cognitive_context['decision_readiness']:.2f}")
            print(f"  Risk Tolerance: {cognitive_context['risk_tolerance']:.2f}")
            print(f"  Chosen Agent: {decision.chosen_agent}")
            print(f"  Risk Level: {decision.risk.value}")

    def _calculate_mental_capacity(self, mental_state: Optional[MentalState]) -> float:
        """
        Calculate available mental capacity for task execution
        Higher is better (0-1)
        """
        if not mental_state:
            return 0.5

        # High attention + low cognitive load + low stress = high capacity
        capacity = (
            mental_state.attention * 0.4 +
            (1.0 - mental_state.cognitive_load) * 0.4 +
            (1.0 - mental_state.stress_level) * 0.2
        )
        return max(0.0, min(1.0, capacity))

    def _calculate_decision_readiness(self, mental_state: Optional[MentalState]) -> float:
        """
        Calculate readiness for making decisions
        Based on attention, engagement, and absence of drowsiness
        """
        if not mental_state:
            return 0.5

        readiness = (
            mental_state.attention * 0.35 +
            mental_state.engagement * 0.35 +
            (1.0 - mental_state.drowsiness) * 0.2 +
            (1.0 - mental_state.stress_level) * 0.1
        )
        return max(0.0, min(1.0, readiness))

    def _calculate_risk_tolerance(self, mental_state: Optional[MentalState]) -> float:
        """
        Calculate user's current risk tolerance
        Higher stress and cognitive load = lower risk tolerance
        """
        if not mental_state:
            return 0.5

        # Lower stress + lower cognitive load + positive valence = higher risk tolerance
        tolerance = (
            (1.0 - mental_state.stress_level) * 0.4 +
            (1.0 - mental_state.cognitive_load) * 0.3 +
            (mental_state.emotional_valence + 1.0) / 2.0 * 0.3
        )
        return max(0.0, min(1.0, tolerance))

    def _calculate_trend(self, metric: str) -> str:
        """
        Calculate trend for a mental state metric
        Returns: "increasing", "decreasing", "stable"
        """
        if len(self.mental_state_history) < 5:
            return "stable"

        recent_values = [
            getattr(state, metric, 0.5)
            for state in self.mental_state_history[-5:]
        ]

        # Simple linear trend
        avg_early = sum(recent_values[:2]) / 2
        avg_late = sum(recent_values[-2:]) / 2

        diff = avg_late - avg_early

        if diff > 0.1:
            return "increasing"
        elif diff < -0.1:
            return "decreasing"
        else:
            return "stable"

    def _signal_quality_to_confidence(self, signal_quality) -> float:
        """Convert signal quality enum to confidence score"""
        quality_map = {
            "excellent": 1.0,
            "good": 0.8,
            "fair": 0.6,
            "poor": 0.3,
            "no_signal": 0.0
        }
        return quality_map.get(signal_quality.value, 0.5)

    def _add_mental_state(self, mental_state: MentalState):
        """Add mental state to history"""
        self.mental_state_history.append(mental_state)
        if len(self.mental_state_history) > self.max_history:
            self.mental_state_history.pop(0)

    def get_current_cognitive_state(self) -> Optional[Dict[str, Any]]:
        """Get current aggregated cognitive state"""
        if not self.mental_state_history:
            return None

        # Average recent mental states
        recent = self.mental_state_history[-5:]

        return {
            "attention": sum(s.attention for s in recent) / len(recent),
            "meditation": sum(s.meditation for s in recent) / len(recent),
            "cognitive_load": sum(s.cognitive_load for s in recent) / len(recent),
            "stress_level": sum(s.stress_level for s in recent) / len(recent),
            "engagement": sum(s.engagement for s in recent) / len(recent),
            "decision_readiness": sum(
                self._calculate_decision_readiness(s) for s in recent
            ) / len(recent),
            "mental_capacity": sum(
                self._calculate_mental_capacity(s) for s in recent
            ) / len(recent)
        }
