"""
Brain Wave Communication System
Implementation of US Patent 6011991: "Communication system and method including brain wave analysis
and/or use of brain activity"

Enables direct brain-to-system communication through decoded neural patterns.
"""
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from bci.core.interfaces import BCIReading, MentalState, MotorIntent


class CommunicationIntent(Enum):
    """Types of communication intents detectable from brain waves"""
    YES = "yes"
    NO = "no"
    HELP = "help"
    STOP = "stop"
    CONTINUE = "continue"
    AUTHORIZE = "authorize"
    DENY = "deny"
    ALERT = "alert"
    QUERY = "query"
    CONFIRM = "confirm"
    CANCEL = "cancel"
    UNKNOWN = "unknown"


class CommunicationChannel(Enum):
    """Communication channels based on brain wave patterns"""
    P300_EVENT_RELATED = "p300"  # Event-related potential
    SSVEP_FREQUENCY = "ssvep"  # Steady-state visually evoked potential
    MOTOR_IMAGERY = "motor_imagery"  # Imagined movements
    MENTAL_STATE = "mental_state"  # Cognitive state changes
    ATTENTION_MODULATION = "attention"  # Attention level changes


@dataclass
class BrainwaveMessage:
    """A decoded message from brain wave patterns"""
    intent: CommunicationIntent
    channel: CommunicationChannel
    confidence: float
    timestamp: str
    raw_pattern: Dict[str, Any]
    user_id: Optional[str] = None
    context: Optional[str] = None


@dataclass
class CommunicationProtocol:
    """Protocol definition for brain wave communication"""
    name: str
    channel: CommunicationChannel
    intent_mapping: Dict[str, CommunicationIntent]
    confidence_threshold: float
    response_time_ms: int


class BrainwaveCommunicationSystem:
    """
    Advanced brain wave communication system
    Based on US Patent 6011991

    Decodes user intent and commands directly from brain activity
    without physical input devices.
    """

    def __init__(self):
        self.active_protocols: List[CommunicationProtocol] = []
        self.message_queue: List[BrainwaveMessage] = []
        self.communication_history: List[BrainwaveMessage] = []
        self.max_history = 1000

        # Initialize default protocols
        self._initialize_protocols()

        print("[BrainComm] Brain Wave Communication System initialized")
        print(f"[BrainComm] Active protocols: {len(self.active_protocols)}")

    def _initialize_protocols(self):
        """Initialize communication protocols"""

        # Protocol 1: P300-based yes/no communication
        p300_protocol = CommunicationProtocol(
            name="P300 Binary Communication",
            channel=CommunicationChannel.P300_EVENT_RELATED,
            intent_mapping={
                "high_alpha_spike": CommunicationIntent.YES,
                "beta_suppression": CommunicationIntent.NO,
                "gamma_burst": CommunicationIntent.CONFIRM,
                "theta_increase": CommunicationIntent.QUERY
            },
            confidence_threshold=0.75,
            response_time_ms=300
        )

        # Protocol 2: Mental state-based communication
        mental_state_protocol = CommunicationProtocol(
            name="Mental State Communication",
            channel=CommunicationChannel.MENTAL_STATE,
            intent_mapping={
                "high_attention": CommunicationIntent.AUTHORIZE,
                "high_stress": CommunicationIntent.ALERT,
                "high_meditation": CommunicationIntent.CONFIRM,
                "low_engagement": CommunicationIntent.CANCEL
            },
            confidence_threshold=0.70,
            response_time_ms=1000
        )

        # Protocol 3: Motor imagery communication
        motor_protocol = CommunicationProtocol(
            name="Motor Imagery Communication",
            channel=CommunicationChannel.MOTOR_IMAGERY,
            intent_mapping={
                "right_hand_imagery": CommunicationIntent.YES,
                "left_hand_imagery": CommunicationIntent.NO,
                "foot_imagery": CommunicationIntent.CONTINUE,
                "tongue_imagery": CommunicationIntent.STOP
            },
            confidence_threshold=0.80,
            response_time_ms=1500
        )

        # Protocol 4: Attention modulation
        attention_protocol = CommunicationProtocol(
            name="Attention Modulation",
            channel=CommunicationChannel.ATTENTION_MODULATION,
            intent_mapping={
                "attention_spike": CommunicationIntent.AUTHORIZE,
                "attention_drop": CommunicationIntent.DENY,
                "sustained_attention": CommunicationIntent.CONFIRM
            },
            confidence_threshold=0.65,
            response_time_ms=2000
        )

        self.active_protocols = [
            p300_protocol,
            mental_state_protocol,
            motor_protocol,
            attention_protocol
        ]

    def decode_message(self, reading: BCIReading, user_id: Optional[str] = None) -> Optional[BrainwaveMessage]:
        """
        Decode communication intent from brain wave reading
        Returns the highest confidence message across all protocols
        """
        if not reading:
            return None

        decoded_messages = []

        # Try each protocol
        for protocol in self.active_protocols:
            message = self._decode_with_protocol(reading, protocol, user_id)
            if message:
                decoded_messages.append(message)

        if not decoded_messages:
            return None

        # Return highest confidence message
        best_message = max(decoded_messages, key=lambda m: m.confidence)

        if best_message.confidence >= 0.60:  # Minimum overall confidence
            self.message_queue.append(best_message)
            self.communication_history.append(best_message)

            # Maintain history limit
            if len(self.communication_history) > self.max_history:
                self.communication_history.pop(0)

            print(f"[BrainComm] 📡 Message decoded: {best_message.intent.value}")
            print(f"[BrainComm]    Channel: {best_message.channel.value}")
            print(f"[BrainComm]    Confidence: {best_message.confidence:.2%}")

            return best_message

        return None

    def _decode_with_protocol(self, reading: BCIReading,
                             protocol: CommunicationProtocol,
                             user_id: Optional[str]) -> Optional[BrainwaveMessage]:
        """Decode message using specific protocol"""

        if protocol.channel == CommunicationChannel.P300_EVENT_RELATED:
            return self._decode_p300(reading, protocol, user_id)

        elif protocol.channel == CommunicationChannel.MENTAL_STATE:
            return self._decode_mental_state(reading, protocol, user_id)

        elif protocol.channel == CommunicationChannel.MOTOR_IMAGERY:
            return self._decode_motor_imagery(reading, protocol, user_id)

        elif protocol.channel == CommunicationChannel.ATTENTION_MODULATION:
            return self._decode_attention(reading, protocol, user_id)

        return None

    def _decode_p300(self, reading: BCIReading,
                    protocol: CommunicationProtocol,
                    user_id: Optional[str]) -> Optional[BrainwaveMessage]:
        """Decode P300 event-related potentials"""
        if not reading.brainwaves:
            return None

        bw = reading.brainwaves

        # Detect patterns in brainwave bands
        patterns = {}

        # High alpha spike (relaxed affirmation)
        if bw.alpha > 300000:
            patterns["high_alpha_spike"] = min(1.0, bw.alpha / 400000)

        # Beta suppression (negative response)
        if bw.beta < 100000:
            patterns["beta_suppression"] = 1.0 - (bw.beta / 100000)

        # Gamma burst (strong confirmation)
        if bw.gamma > 150000:
            patterns["gamma_burst"] = min(1.0, bw.gamma / 200000)

        # Theta increase (questioning)
        if bw.theta > 400000:
            patterns["theta_increase"] = min(1.0, bw.theta / 600000)

        # Find best match
        for pattern_name, confidence in patterns.items():
            if pattern_name in protocol.intent_mapping:
                intent = protocol.intent_mapping[pattern_name]

                if confidence >= protocol.confidence_threshold:
                    return BrainwaveMessage(
                        intent=intent,
                        channel=protocol.channel,
                        confidence=confidence,
                        timestamp=reading.timestamp.isoformat(),
                        raw_pattern={"pattern": pattern_name, "bands": bw.to_dict()},
                        user_id=user_id
                    )

        return None

    def _decode_mental_state(self, reading: BCIReading,
                            protocol: CommunicationProtocol,
                            user_id: Optional[str]) -> Optional[BrainwaveMessage]:
        """Decode mental state-based communication"""
        if not reading.mental_state:
            return None

        ms = reading.mental_state
        patterns = {}

        # High attention = authorization/approval
        if ms.attention > 0.80:
            patterns["high_attention"] = ms.attention

        # High stress = alert/warning
        if ms.stress_level > 0.70:
            patterns["high_stress"] = ms.stress_level

        # High meditation = calm confirmation
        if ms.meditation > 0.75:
            patterns["high_meditation"] = ms.meditation

        # Low engagement = cancellation/withdrawal
        if ms.engagement < 0.30:
            patterns["low_engagement"] = 1.0 - ms.engagement

        for pattern_name, confidence in patterns.items():
            if pattern_name in protocol.intent_mapping:
                intent = protocol.intent_mapping[pattern_name]

                if confidence >= protocol.confidence_threshold:
                    return BrainwaveMessage(
                        intent=intent,
                        channel=protocol.channel,
                        confidence=confidence,
                        timestamp=reading.timestamp.isoformat(),
                        raw_pattern={"pattern": pattern_name, "mental_state": ms.to_dict()},
                        user_id=user_id
                    )

        return None

    def _decode_motor_imagery(self, reading: BCIReading,
                             protocol: CommunicationProtocol,
                             user_id: Optional[str]) -> Optional[BrainwaveMessage]:
        """Decode motor imagery communication"""
        if not reading.motor_intent:
            return None

        mi = reading.motor_intent

        # Map motor movements to intents
        movement_map = {
            "reach": "right_hand_imagery",
            "grasp": "right_hand_imagery",
            "move_right": "right_hand_imagery",
            "move_left": "left_hand_imagery",
            "move_down": "foot_imagery",
            "click": "right_hand_imagery"
        }

        movement_type = mi.movement_type
        if movement_type in movement_map:
            pattern_name = movement_map[movement_type]

            if pattern_name in protocol.intent_mapping:
                intent = protocol.intent_mapping[pattern_name]

                if mi.confidence >= protocol.confidence_threshold:
                    return BrainwaveMessage(
                        intent=intent,
                        channel=protocol.channel,
                        confidence=mi.confidence,
                        timestamp=reading.timestamp.isoformat(),
                        raw_pattern={"pattern": pattern_name, "motor": mi.to_dict()},
                        user_id=user_id
                    )

        return None

    def _decode_attention(self, reading: BCIReading,
                         protocol: CommunicationProtocol,
                         user_id: Optional[str]) -> Optional[BrainwaveMessage]:
        """Decode attention modulation communication"""
        if not reading.mental_state:
            return None

        ms = reading.mental_state
        patterns = {}

        # Attention spike (sudden increase)
        if ms.attention > 0.85:
            patterns["attention_spike"] = ms.attention

        # Attention drop (disengagement)
        if ms.attention < 0.30:
            patterns["attention_drop"] = 1.0 - ms.attention

        # Sustained attention (confirmation through focus)
        if 0.70 <= ms.attention <= 0.90 and ms.engagement > 0.75:
            patterns["sustained_attention"] = (ms.attention + ms.engagement) / 2

        for pattern_name, confidence in patterns.items():
            if pattern_name in protocol.intent_mapping:
                intent = protocol.intent_mapping[pattern_name]

                if confidence >= protocol.confidence_threshold:
                    return BrainwaveMessage(
                        intent=intent,
                        channel=protocol.channel,
                        confidence=confidence,
                        timestamp=reading.timestamp.isoformat(),
                        raw_pattern={"pattern": pattern_name, "attention": ms.attention},
                        user_id=user_id
                    )

        return None

    def get_pending_messages(self) -> List[BrainwaveMessage]:
        """Get all pending messages in queue"""
        messages = self.message_queue.copy()
        self.message_queue.clear()
        return messages

    def wait_for_intent(self, expected_intent: CommunicationIntent,
                       timeout_seconds: float = 10.0) -> Optional[BrainwaveMessage]:
        """
        Wait for specific communication intent from user
        Used for confirmation dialogs, authorization requests, etc.
        """
        import time
        start_time = time.time()

        print(f"[BrainComm] Waiting for brain wave intent: {expected_intent.value}")
        print(f"[BrainComm] Timeout: {timeout_seconds}s")

        while time.time() - start_time < timeout_seconds:
            if self.message_queue:
                for message in self.message_queue:
                    if message.intent == expected_intent:
                        self.message_queue.remove(message)
                        print(f"[BrainComm] ✅ Received expected intent: {expected_intent.value}")
                        return message

            time.sleep(0.1)

        print(f"[BrainComm] ⏱️  Timeout waiting for intent: {expected_intent.value}")
        return None

    def get_statistics(self) -> Dict[str, Any]:
        """Get communication statistics"""
        intent_counts = {}
        for message in self.communication_history:
            intent = message.intent.value
            intent_counts[intent] = intent_counts.get(intent, 0) + 1

        channel_counts = {}
        for message in self.communication_history:
            channel = message.channel.value
            channel_counts[channel] = channel_counts.get(channel, 0) + 1

        avg_confidence = (
            sum(m.confidence for m in self.communication_history) /
            len(self.communication_history)
            if self.communication_history else 0.0
        )

        return {
            "total_messages": len(self.communication_history),
            "pending_messages": len(self.message_queue),
            "intent_distribution": intent_counts,
            "channel_distribution": channel_counts,
            "average_confidence": avg_confidence,
            "active_protocols": len(self.active_protocols)
        }
