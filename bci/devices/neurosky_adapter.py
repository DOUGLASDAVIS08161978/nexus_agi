"""
NeuroSky MindWave BCI Adapter
Supports MindWave Mobile and other NeuroSky ThinkGear devices
"""
import asyncio
import numpy as np
from typing import Optional, Dict, Any
from datetime import datetime
import random

from bci.core.interfaces import (
    IBCIDevice, BCIReading, BCICommand, BCIDeviceType,
    SignalQuality, BrainwaveData, MentalState
)


class NeuroSkyAdapter(IBCIDevice):
    """Adapter for NeuroSky MindWave devices using ThinkGear protocol"""

    def __init__(self, device_id: str = "neurosky_001", port: str = "/dev/rfcomm0"):
        self.device_id = device_id
        self.port = port
        self.connected = False
        self.streaming = False
        self._last_reading = None
        self._connection = None

        # Device capabilities
        self.sample_rate = 512.0  # Hz
        self.channels = ["TP9"]  # Single EEG channel

    async def connect(self) -> bool:
        """Connect to NeuroSky device"""
        try:
            # Simulate connection - in production, use actual ThinkGear protocol
            # import thinkgear
            # self._connection = thinkgear.Connection(self.port)
            # self._connection.start()

            await asyncio.sleep(0.5)  # Simulated connection time
            self.connected = True
            print(f"[NeuroSky] Connected to device {self.device_id} on {self.port}")
            return True
        except Exception as e:
            print(f"[NeuroSky] Connection failed: {e}")
            return False

    async def disconnect(self) -> bool:
        """Disconnect from device"""
        self.streaming = False
        await asyncio.sleep(0.1)
        self.connected = False
        print(f"[NeuroSky] Disconnected from {self.device_id}")
        return True

    async def start_streaming(self) -> bool:
        """Start EEG data streaming"""
        if not self.connected:
            return False
        self.streaming = True
        print(f"[NeuroSky] Started streaming from {self.device_id}")
        return True

    async def stop_streaming(self) -> bool:
        """Stop data streaming"""
        self.streaming = False
        print(f"[NeuroSky] Stopped streaming from {self.device_id}")
        return True

    async def read(self) -> Optional[BCIReading]:
        """Read current BCI data from NeuroSky"""
        if not self.connected or not self.streaming:
            return None

        # Simulate reading data - in production, parse ThinkGear packets
        # In real implementation:
        # packet = self._connection.get_packet()
        # Parse eSense values and brainwave bands

        # Simulated brainwave data with realistic patterns
        brainwaves = BrainwaveData(
            delta=random.uniform(200000, 800000),
            theta=random.uniform(100000, 600000),
            alpha=random.uniform(50000, 400000),
            beta=random.uniform(30000, 300000),
            gamma=random.uniform(10000, 200000)
        )

        # eSense metrics (NeuroSky proprietary algorithms)
        attention = random.uniform(0.4, 0.9)
        meditation = random.uniform(0.3, 0.8)

        mental_state = MentalState(
            attention=attention,
            meditation=meditation,
            cognitive_load=1.0 - meditation,
            stress_level=0.5 * (1.0 - meditation),
            engagement=attention,
            drowsiness=max(0, 1.0 - attention - 0.3),
            emotional_valence=meditation - 0.5,
            arousal=attention
        )

        # Signal quality (0-200 in ThinkGear, 0 is best)
        poor_signal = random.randint(0, 50)
        if poor_signal == 0:
            signal_quality = SignalQuality.EXCELLENT
        elif poor_signal < 20:
            signal_quality = SignalQuality.GOOD
        elif poor_signal < 40:
            signal_quality = SignalQuality.FAIR
        else:
            signal_quality = SignalQuality.POOR

        # Simulated raw EEG data
        raw_eeg = np.random.randn(512) * 50  # 1 second of data

        reading = BCIReading(
            timestamp=datetime.utcnow(),
            device_id=self.device_id,
            device_type=BCIDeviceType.NEUROSKY_MINDWAVE,
            signal_quality=signal_quality,
            raw_channels={"TP9": raw_eeg},
            sample_rate=self.sample_rate,
            brainwaves=brainwaves,
            mental_state=mental_state,
            contact_quality={"TP9": 1.0 - (poor_signal / 200)},
            battery_level=random.uniform(0.7, 1.0),
            metadata={
                "poor_signal_level": poor_signal,
                "firmware_version": "v2.8",
                "device_model": "MindWave Mobile 2"
            }
        )

        self._last_reading = reading
        return reading

    async def send_command(self, command: BCICommand) -> bool:
        """Send command to NeuroSky device"""
        if not self.connected:
            return False

        # NeuroSky devices are primarily passive sensors
        # Limited command support
        if command.command_type == "calibrate":
            print(f"[NeuroSky] Calibration requested")
            return True
        elif command.command_type == "reset":
            await self.disconnect()
            await self.connect()
            return True

        return False

    def get_device_info(self) -> Dict[str, Any]:
        """Get NeuroSky device information"""
        return {
            "device_id": self.device_id,
            "device_type": BCIDeviceType.NEUROSKY_MINDWAVE.value,
            "manufacturer": "NeuroSky",
            "model": "MindWave Mobile 2",
            "channels": self.channels,
            "sample_rate": self.sample_rate,
            "capabilities": ["eeg", "attention", "meditation", "brainwaves"],
            "invasive": False,
            "wireless": True,
            "battery_powered": True
        }

    async def calibrate(self) -> bool:
        """Perform baseline calibration"""
        print(f"[NeuroSky] Calibrating {self.device_id}...")
        await asyncio.sleep(2.0)  # Calibration time
        print(f"[NeuroSky] Calibration complete")
        return True

    def get_signal_quality(self) -> SignalQuality:
        """Get current signal quality"""
        if self._last_reading:
            return self._last_reading.signal_quality
        return SignalQuality.NO_SIGNAL

    def is_connected(self) -> bool:
        """Check connection status"""
        return self.connected
