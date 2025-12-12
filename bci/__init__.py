"""
Universal BCI Controller Package
High-performance brain-computer interface system for Nexus AGI
"""

from bci.orchestrator import UniversalBCIOrchestrator
from bci.mindcontrol_integration import BCIMindControlBridge

from bci.core.interfaces import (
    IBCIDevice,
    BCIReading,
    BCICommand,
    BCIDeviceType,
    SignalQuality,
    BrainwaveData,
    MentalState,
    MotorIntent
)

from bci.devices.neurosky_adapter import NeuroSkyAdapter
from bci.devices.neuralink_adapter import NeuralinkAdapter
from bci.devices.stentrode_adapter import StentrodeAdapter
from bci.devices.multi_device_adapter import OpenBCIAdapter, EmotivEPOCAdapter

from bci.processing.signal_processor import UniversalSignalProcessor
from bci.safety.safety_monitor import BCISafetyMonitor

from bci.auth import (
    NexusAGIAuthorizationSystem,
    BiometricEEGAuthenticator,
    BrainwaveCommunicationSystem,
    EmotionalStateDetector,
    AccessLevel,
    SecurityEvent
)

__version__ = "1.0.0"
__author__ = "Nexus AGI Team"

__all__ = [
    "UniversalBCIOrchestrator",
    "BCIMindControlBridge",
    "NeuroSkyAdapter",
    "NeuralinkAdapter",
    "StentrodeAdapter",
    "OpenBCIAdapter",
    "EmotivEPOCAdapter",
    "UniversalSignalProcessor",
    "BCISafetyMonitor",
    "BCIReading",
    "BCICommand",
    "BCIDeviceType",
    "SignalQuality",
    "BrainwaveData",
    "MentalState",
    "MotorIntent",
    "NexusAGIAuthorizationSystem",
    "BiometricEEGAuthenticator",
    "BrainwaveCommunicationSystem",
    "EmotionalStateDetector",
    "AccessLevel",
    "SecurityEvent"
]
