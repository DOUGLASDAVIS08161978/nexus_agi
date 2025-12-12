"""
BCI Authorization and Patent Systems
Implements patented technologies for secure BCI access
"""
from bci.auth.biometric_auth import BiometricEEGAuthenticator, AuthenticationResult, BiometricProfile
from bci.auth.brainwave_communication import BrainwaveCommunicationSystem, CommunicationIntent, BrainwaveMessage
from bci.auth.emotional_state import EmotionalStateDetector, EmotionalState, EmotionalProfile
from bci.auth.authorization_system import NexusAGIAuthorizationSystem, AccessLevel, SecurityEvent

__all__ = [
    "BiometricEEGAuthenticator",
    "AuthenticationResult",
    "BiometricProfile",
    "BrainwaveCommunicationSystem",
    "CommunicationIntent",
    "BrainwaveMessage",
    "EmotionalStateDetector",
    "EmotionalState",
    "EmotionalProfile",
    "NexusAGIAuthorizationSystem",
    "AccessLevel",
    "SecurityEvent"
]
