"""
Multi-Device BCI Adapters
Includes: OpenBCI, Emotiv EPOC, and generic EEG adapters
"""
import asyncio
import numpy as np
from typing import Optional, Dict, Any, List
from datetime import datetime
import random

from bci.core.interfaces import (
    IBCIDevice, BCIReading, BCICommand, BCIDeviceType,
    SignalQuality, BrainwaveData, MentalState
)


class OpenBCIAdapter(IBCIDevice):
    """Adapter for OpenBCI devices (Cyton, Ganglion, etc.)"""

    def __init__(self, device_id: str = "openbci_001", board_type: str = "cyton"):
        self.device_id = device_id
        self.board_type = board_type
        self.connected = False
        self.streaming = False
        self._last_reading = None

        # Board specifications
        if board_type == "cyton":
            self.num_channels = 8
            self.sample_rate = 250.0
        elif board_type == "ganglion":
            self.num_channels = 4
            self.sample_rate = 200.0
        else:  # daisy (cyton + daisy)
            self.num_channels = 16
            self.sample_rate = 250.0

    async def connect(self) -> bool:
        """Connect to OpenBCI board"""
        try:
            print(f"[OpenBCI] Connecting to {self.board_type} board {self.device_id}...")
            await asyncio.sleep(1.0)
            self.connected = True
            print(f"[OpenBCI] Connected - {self.num_channels} channels @ {self.sample_rate} Hz")
            return True
        except Exception as e:
            print(f"[OpenBCI] Connection failed: {e}")
            return False

    async def disconnect(self) -> bool:
        self.streaming = False
        self.connected = False
        print(f"[OpenBCI] Disconnected")
        return True

    async def start_streaming(self) -> bool:
        if not self.connected:
            return False
        self.streaming = True
        print(f"[OpenBCI] Streaming started")
        return True

    async def stop_streaming(self) -> bool:
        self.streaming = False
        print(f"[OpenBCI] Streaming stopped")
        return True

    async def read(self) -> Optional[BCIReading]:
        if not self.connected or not self.streaming:
            return None

        # Simulate multi-channel EEG data
        raw_channels = {}
        channel_names = ["Fp1", "Fp2", "C3", "C4", "P7", "P8", "O1", "O2"][:self.num_channels]

        for ch_name in channel_names:
            signal = np.random.randn(int(self.sample_rate)) * 50  # 1 second
            raw_channels[ch_name] = signal

        brainwaves = BrainwaveData(
            delta=random.uniform(150000, 700000),
            theta=random.uniform(80000, 500000),
            alpha=random.uniform(40000, 300000),
            beta=random.uniform(25000, 250000),
            gamma=random.uniform(15000, 180000)
        )

        mental_state = MentalState(
            attention=random.uniform(0.4, 0.85),
            meditation=random.uniform(0.3, 0.75),
            cognitive_load=random.uniform(0.3, 0.8),
            stress_level=random.uniform(0.2, 0.6),
            engagement=random.uniform(0.5, 0.9),
            drowsiness=random.uniform(0.0, 0.4),
            emotional_valence=random.uniform(-0.3, 0.4),
            arousal=random.uniform(0.4, 0.85)
        )

        signal_quality = random.choice([
            SignalQuality.GOOD,
            SignalQuality.GOOD,
            SignalQuality.FAIR
        ])

        contact_quality = {ch: random.uniform(0.6, 0.95) for ch in channel_names}

        reading = BCIReading(
            timestamp=datetime.utcnow(),
            device_id=self.device_id,
            device_type=BCIDeviceType.OPENBCI,
            signal_quality=signal_quality,
            raw_channels=raw_channels,
            sample_rate=self.sample_rate,
            brainwaves=brainwaves,
            mental_state=mental_state,
            contact_quality=contact_quality,
            battery_level=random.uniform(0.6, 1.0),
            metadata={
                "board_type": self.board_type,
                "num_channels": self.num_channels,
                "firmware_version": "v3.1.2"
            }
        )

        self._last_reading = reading
        return reading

    async def send_command(self, command: BCICommand) -> bool:
        if not self.connected:
            return False

        if command.command_type == "calibrate":
            await self.calibrate()
            return True
        elif command.command_type == "set_channels":
            print(f"[OpenBCI] Configuring channels: {command.parameters}")
            return True

        return False

    def get_device_info(self) -> Dict[str, Any]:
        return {
            "device_id": self.device_id,
            "device_type": BCIDeviceType.OPENBCI.value,
            "manufacturer": "OpenBCI",
            "model": self.board_type.capitalize(),
            "channels": self.num_channels,
            "sample_rate": self.sample_rate,
            "capabilities": ["eeg", "research", "open_source"],
            "invasive": False,
            "wireless": True if self.board_type == "ganglion" else False
        }

    async def calibrate(self) -> bool:
        print(f"[OpenBCI] Calibrating...")
        await asyncio.sleep(2.0)
        print(f"[OpenBCI] Calibration complete")
        return True

    def get_signal_quality(self) -> SignalQuality:
        if self._last_reading:
            return self._last_reading.signal_quality
        return SignalQuality.NO_SIGNAL

    def is_connected(self) -> bool:
        return self.connected


class EmotivEPOCAdapter(IBCIDevice):
    """Adapter for Emotiv EPOC/EPOC+ headsets"""

    def __init__(self, device_id: str = "emotiv_001"):
        self.device_id = device_id
        self.connected = False
        self.streaming = False
        self._last_reading = None
        self.num_channels = 14  # 14-channel EEG
        self.sample_rate = 128.0  # 128 Hz (EPOC) or 256 Hz (EPOC+)

    async def connect(self) -> bool:
        try:
            print(f"[Emotiv] Connecting to EPOC device {self.device_id}...")
            await asyncio.sleep(0.8)
            self.connected = True
            print(f"[Emotiv] Connected - 14 channels active")
            return True
        except Exception as e:
            print(f"[Emotiv] Connection failed: {e}")
            return False

    async def disconnect(self) -> bool:
        self.streaming = False
        self.connected = False
        print(f"[Emotiv] Disconnected")
        return True

    async def start_streaming(self) -> bool:
        if not self.connected:
            return False
        self.streaming = True
        print(f"[Emotiv] Streaming started")
        return True

    async def stop_streaming(self) -> bool:
        self.streaming = False
        print(f"[Emotiv] Streaming stopped")
        return True

    async def read(self) -> Optional[BCIReading]:
        if not self.connected or not self.streaming:
            return None

        # Emotiv EPOC 14-channel layout
        channel_names = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
                        "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]

        raw_channels = {}
        for ch_name in channel_names:
            signal = np.random.randn(128) * 40  # 1 second
            raw_channels[ch_name] = signal

        # Emotiv provides performance metrics
        brainwaves = BrainwaveData(
            delta=random.uniform(100000, 600000),
            theta=random.uniform(70000, 450000),
            alpha=random.uniform(35000, 280000),
            beta=random.uniform(20000, 220000),
            gamma=random.uniform(12000, 160000)
        )

        # Emotiv's proprietary performance metrics
        mental_state = MentalState(
            attention=random.uniform(0.5, 0.9),  # Focus
            meditation=random.uniform(0.3, 0.75),  # Relaxation
            cognitive_load=random.uniform(0.3, 0.85),  # Mental effort
            stress_level=random.uniform(0.1, 0.6),
            engagement=random.uniform(0.5, 0.95),  # Engagement
            drowsiness=random.uniform(0.0, 0.5),
            emotional_valence=random.uniform(-0.4, 0.5),  # Excitement
            arousal=random.uniform(0.4, 0.9)
        )

        signal_quality = random.choice([
            SignalQuality.GOOD,
            SignalQuality.GOOD,
            SignalQuality.FAIR,
            SignalQuality.EXCELLENT
        ])

        contact_quality = {ch: random.uniform(0.5, 0.95) for ch in channel_names}

        reading = BCIReading(
            timestamp=datetime.utcnow(),
            device_id=self.device_id,
            device_type=BCIDeviceType.EMOTIV_EPOC,
            signal_quality=signal_quality,
            raw_channels=raw_channels,
            sample_rate=self.sample_rate,
            brainwaves=brainwaves,
            mental_state=mental_state,
            contact_quality=contact_quality,
            battery_level=random.uniform(0.5, 1.0),
            metadata={
                "model": "EPOC+",
                "num_channels": 14,
                "performance_metrics": {
                    "focus": mental_state.attention,
                    "engagement": mental_state.engagement,
                    "excitement": mental_state.arousal,
                    "relaxation": mental_state.meditation
                }
            }
        )

        self._last_reading = reading
        return reading

    async def send_command(self, command: BCICommand) -> bool:
        if not self.connected:
            return False

        if command.command_type == "calibrate":
            await self.calibrate()
            return True

        return False

    def get_device_info(self) -> Dict[str, Any]:
        return {
            "device_id": self.device_id,
            "device_type": BCIDeviceType.EMOTIV_EPOC.value,
            "manufacturer": "Emotiv",
            "model": "EPOC+",
            "channels": 14,
            "sample_rate": self.sample_rate,
            "capabilities": ["eeg", "performance_metrics", "facial_expressions"],
            "invasive": False,
            "wireless": True
        }

    async def calibrate(self) -> bool:
        print(f"[Emotiv] Starting calibration...")
        print(f"[Emotiv] Please remain still and relaxed")
        await asyncio.sleep(3.0)
        print(f"[Emotiv] Calibration complete")
        return True

    def get_signal_quality(self) -> SignalQuality:
        if self._last_reading:
            return self._last_reading.signal_quality
        return SignalQuality.NO_SIGNAL

    def is_connected(self) -> bool:
        return self.connected
