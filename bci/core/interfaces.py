"""
Universal BCI Interface Abstractions
Provides base classes and protocols for all brain-computer interface devices
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from enum import Enum
import numpy as np


class BCIDeviceType(Enum):
    """Types of BCI devices supported"""
    NEUROSKY_MINDWAVE = "neurosky_mindwave"
    NEURALINK = "neuralink"
    STENTRODE = "stentrode"
    EMOTIV_EPOC = "emotiv_epoc"
    OPENBCI = "openbci"
    KERNEL_FLOW = "kernel_flow"
    NEXTMIND = "nextmind"
    CTRL_LABS = "ctrl_labs"
    GENERIC_EEG = "generic_eeg"
    GENERIC_INVASIVE = "generic_invasive"


class SignalQuality(Enum):
    """Signal quality levels"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    NO_SIGNAL = "no_signal"


class BCISignalType(Enum):
    """Types of neural signals"""
    EEG = "eeg"  # Electroencephalography
    ECOG = "ecog"  # Electrocorticography
    LFP = "lfp"  # Local Field Potential
    SPIKE = "spike"  # Single/Multi-unit activity
    BOLD = "bold"  # Blood-oxygen-level-dependent (fMRI)
    NIRS = "nirs"  # Near-infrared spectroscopy
    MEG = "meg"  # Magnetoencephalography


@dataclass
class BrainwaveData:
    """Raw brainwave measurements"""
    delta: float = 0.0  # 0.5-4 Hz - deep sleep
    theta: float = 0.0  # 4-8 Hz - drowsiness, meditation
    alpha: float = 0.0  # 8-13 Hz - relaxed, calm
    beta: float = 0.0  # 13-30 Hz - alert, active thinking
    gamma: float = 0.0  # 30-100 Hz - cognitive processing

    def to_dict(self) -> Dict[str, float]:
        return {
            "delta": self.delta,
            "theta": self.theta,
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma
        }


@dataclass
class MentalState:
    """Interpreted mental state metrics"""
    attention: float = 0.0  # 0-1
    meditation: float = 0.0  # 0-1
    cognitive_load: float = 0.0  # 0-1
    stress_level: float = 0.0  # 0-1
    engagement: float = 0.0  # 0-1
    drowsiness: float = 0.0  # 0-1
    emotional_valence: float = 0.0  # -1 to 1 (negative to positive)
    arousal: float = 0.0  # 0-1

    def to_dict(self) -> Dict[str, float]:
        return {
            "attention": self.attention,
            "meditation": self.meditation,
            "cognitive_load": self.cognitive_load,
            "stress_level": self.stress_level,
            "engagement": self.engagement,
            "drowsiness": self.drowsiness,
            "emotional_valence": self.emotional_valence,
            "arousal": self.arousal
        }


@dataclass
class MotorIntent:
    """Decoded motor intentions"""
    movement_type: str = "none"  # click, grab, move, etc.
    direction: Optional[np.ndarray] = None  # 3D vector
    velocity: float = 0.0
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "movement_type": self.movement_type,
            "direction": self.direction.tolist() if self.direction is not None else None,
            "velocity": self.velocity,
            "confidence": self.confidence
        }


@dataclass
class BCIReading:
    """Complete BCI reading snapshot"""
    timestamp: datetime
    device_id: str
    device_type: BCIDeviceType
    signal_quality: SignalQuality

    # Raw data
    raw_channels: Dict[str, np.ndarray] = field(default_factory=dict)
    sample_rate: float = 0.0

    # Processed data
    brainwaves: Optional[BrainwaveData] = None
    mental_state: Optional[MentalState] = None
    motor_intent: Optional[MotorIntent] = None

    # Additional metrics
    contact_quality: Dict[str, float] = field(default_factory=dict)
    battery_level: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "device_id": self.device_id,
            "device_type": self.device_type.value,
            "signal_quality": self.signal_quality.value,
            "sample_rate": self.sample_rate,
            "brainwaves": self.brainwaves.to_dict() if self.brainwaves else None,
            "mental_state": self.mental_state.to_dict() if self.mental_state else None,
            "motor_intent": self.motor_intent.to_dict() if self.motor_intent else None,
            "contact_quality": self.contact_quality,
            "battery_level": self.battery_level,
            "metadata": self.metadata
        }


@dataclass
class BCICommand:
    """Command to be sent to BCI device"""
    command_type: str  # calibrate, stimulate, configure, etc.
    parameters: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)


class IBCIDevice(ABC):
    """Universal BCI Device Interface"""

    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to the device"""
        pass

    @abstractmethod
    async def disconnect(self) -> bool:
        """Disconnect from the device"""
        pass

    @abstractmethod
    async def start_streaming(self) -> bool:
        """Start continuous data streaming"""
        pass

    @abstractmethod
    async def stop_streaming(self) -> bool:
        """Stop data streaming"""
        pass

    @abstractmethod
    async def read(self) -> Optional[BCIReading]:
        """Read current BCI data"""
        pass

    @abstractmethod
    async def send_command(self, command: BCICommand) -> bool:
        """Send command to device"""
        pass

    @abstractmethod
    def get_device_info(self) -> Dict[str, Any]:
        """Get device information and capabilities"""
        pass

    @abstractmethod
    async def calibrate(self) -> bool:
        """Perform device calibration"""
        pass

    @abstractmethod
    def get_signal_quality(self) -> SignalQuality:
        """Get current signal quality"""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if device is connected"""
        pass


class IBCIProcessor(ABC):
    """Signal processing interface"""

    @abstractmethod
    def process_raw_signal(self, raw_data: np.ndarray, sample_rate: float) -> Dict[str, Any]:
        """Process raw neural signals"""
        pass

    @abstractmethod
    def extract_features(self, processed_data: Dict[str, Any]) -> np.ndarray:
        """Extract features from processed signals"""
        pass

    @abstractmethod
    def decode_intent(self, features: np.ndarray) -> Dict[str, Any]:
        """Decode user intent from features"""
        pass


class IBCIDecoder(ABC):
    """Intent decoder interface"""

    @abstractmethod
    def decode_motor_intent(self, bci_reading: BCIReading) -> MotorIntent:
        """Decode motor intentions"""
        pass

    @abstractmethod
    def decode_mental_state(self, bci_reading: BCIReading) -> MentalState:
        """Decode mental state"""
        pass

    @abstractmethod
    def decode_communication_intent(self, bci_reading: BCIReading) -> Optional[str]:
        """Decode communication intent (P300, SSVEP, etc.)"""
        pass


class IBCISafetyMonitor(ABC):
    """Safety monitoring interface"""

    @abstractmethod
    def validate_reading(self, reading: BCIReading) -> bool:
        """Validate reading is safe and within bounds"""
        pass

    @abstractmethod
    def check_stimulation_safety(self, command: BCICommand) -> bool:
        """Check if stimulation command is safe"""
        pass

    @abstractmethod
    def get_safety_status(self) -> Dict[str, Any]:
        """Get current safety status"""
        pass
