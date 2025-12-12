"""
Neuralink BCI Adapter
High-bandwidth invasive neural interface adapter
NOTE: This is a simulation adapter as Neuralink API is not publicly available
"""
import asyncio
import numpy as np
from typing import Optional, Dict, Any, List
from datetime import datetime
import random

from bci.core.interfaces import (
    IBCIDevice, BCIReading, BCICommand, BCIDeviceType,
    SignalQuality, BrainwaveData, MentalState, MotorIntent, BCISignalType
)


class NeuralinkAdapter(IBCIDevice):
    """
    Adapter for Neuralink N1 implant
    Simulates high-resolution neural recording and stimulation
    """

    def __init__(self, device_id: str = "neuralink_n1_001", implant_location: str = "motor_cortex"):
        self.device_id = device_id
        self.implant_location = implant_location
        self.connected = False
        self.streaming = False
        self._last_reading = None

        # Neuralink specs (based on public information)
        self.num_channels = 1024  # 1024 electrode channels per N1 implant
        self.sample_rate = 20000.0  # 20 kHz sampling rate
        self.signal_type = BCISignalType.SPIKE  # Single and multi-unit activity

        # Channel mapping (simulated)
        self.channel_map = self._initialize_channel_map()

    def _initialize_channel_map(self) -> Dict[str, List[int]]:
        """Map electrode channels to brain regions"""
        return {
            "motor_cortex": list(range(0, 512)),
            "somatosensory": list(range(512, 768)),
            "premotor": list(range(768, 896)),
            "prefrontal": list(range(896, 1024))
        }

    async def connect(self) -> bool:
        """Establish wireless connection to Neuralink implant"""
        try:
            print(f"[Neuralink] Initiating secure wireless connection to {self.device_id}...")
            await asyncio.sleep(1.0)  # Simulated handshake

            # In production: establish encrypted wireless link
            # Verify implant integrity and telemetry

            self.connected = True
            print(f"[Neuralink] Connected to N1 implant at {self.implant_location}")
            print(f"[Neuralink] {self.num_channels} channels active, {self.sample_rate} Hz sampling")
            return True
        except Exception as e:
            print(f"[Neuralink] Connection failed: {e}")
            return False

    async def disconnect(self) -> bool:
        """Disconnect from implant"""
        self.streaming = False
        await asyncio.sleep(0.2)
        self.connected = False
        print(f"[Neuralink] Safely disconnected from {self.device_id}")
        return True

    async def start_streaming(self) -> bool:
        """Start high-bandwidth neural recording"""
        if not self.connected:
            return False
        self.streaming = True
        print(f"[Neuralink] Started streaming {self.num_channels} channels at {self.sample_rate} Hz")
        return True

    async def stop_streaming(self) -> bool:
        """Stop neural recording"""
        self.streaming = False
        print(f"[Neuralink] Stopped streaming")
        return True

    async def read(self) -> Optional[BCIReading]:
        """Read neural data from Neuralink implant"""
        if not self.connected or not self.streaming:
            return None

        # Simulate high-resolution neural recording
        # In production: receive spike-sorted neural data

        # Generate simulated spike data for multiple channels
        raw_channels = {}
        for region, channels in self.channel_map.items():
            # Simulated local field potential for each region
            lfp = np.random.randn(1000) * 100  # 50ms of data at 20kHz
            raw_channels[f"{region}_lfp"] = lfp

        # Decode motor intent from motor cortex activity
        motor_intent = self._decode_motor_intent()

        # Extract mental state from broader cortical patterns
        mental_state = MentalState(
            attention=random.uniform(0.6, 0.95),
            meditation=random.uniform(0.3, 0.7),
            cognitive_load=random.uniform(0.4, 0.8),
            stress_level=random.uniform(0.1, 0.4),
            engagement=random.uniform(0.7, 0.95),
            drowsiness=random.uniform(0.0, 0.2),
            emotional_valence=random.uniform(-0.3, 0.5),
            arousal=random.uniform(0.5, 0.9)
        )

        # Compute brainwaves from LFP
        brainwaves = self._compute_brainwaves(raw_channels)

        # Excellent signal quality for invasive recording
        signal_quality = SignalQuality.EXCELLENT

        # Contact quality for each electrode (impedance check)
        contact_quality = {
            region: random.uniform(0.85, 1.0)
            for region in self.channel_map.keys()
        }

        reading = BCIReading(
            timestamp=datetime.utcnow(),
            device_id=self.device_id,
            device_type=BCIDeviceType.NEURALINK,
            signal_quality=signal_quality,
            raw_channels=raw_channels,
            sample_rate=self.sample_rate,
            brainwaves=brainwaves,
            mental_state=mental_state,
            motor_intent=motor_intent,
            contact_quality=contact_quality,
            battery_level=random.uniform(0.8, 1.0),
            metadata={
                "implant_location": self.implant_location,
                "num_active_channels": self.num_channels,
                "signal_type": self.signal_type.value,
                "spike_rate": random.uniform(50, 200),  # spikes/sec
                "firmware_version": "v3.2.1",
                "implant_health": "nominal",
                "temperature_c": random.uniform(36.5, 37.2)
            }
        )

        self._last_reading = reading
        return reading

    def _decode_motor_intent(self) -> MotorIntent:
        """Decode motor intentions from neural activity"""
        # Simulate decoding reaching/grasping movements
        movement_types = ["reach", "grasp", "click", "move", "rest"]
        movement = random.choice(movement_types)

        if movement == "rest":
            return MotorIntent(
                movement_type="rest",
                direction=None,
                velocity=0.0,
                confidence=random.uniform(0.8, 0.95)
            )

        # 3D direction vector
        direction = np.random.randn(3)
        direction = direction / np.linalg.norm(direction)  # Normalize

        return MotorIntent(
            movement_type=movement,
            direction=direction,
            velocity=random.uniform(0.1, 1.0),
            confidence=random.uniform(0.75, 0.98)
        )

    def _compute_brainwaves(self, raw_channels: Dict[str, np.ndarray]) -> BrainwaveData:
        """Extract brainwave bands from LFP"""
        # Simplified brainwave extraction
        return BrainwaveData(
            delta=random.uniform(50000, 200000),
            theta=random.uniform(30000, 150000),
            alpha=random.uniform(20000, 100000),
            beta=random.uniform(15000, 80000),
            gamma=random.uniform(10000, 120000)
        )

    async def send_command(self, command: BCICommand) -> bool:
        """Send command to Neuralink implant"""
        if not self.connected:
            return False

        print(f"[Neuralink] Processing command: {command.command_type}")

        if command.command_type == "calibrate":
            await self.calibrate()
            return True

        elif command.command_type == "stimulate":
            # Neural stimulation (requires safety checks)
            return await self._neural_stimulation(command.parameters)

        elif command.command_type == "adjust_recording":
            # Adjust recording parameters
            print(f"[Neuralink] Adjusting recording parameters: {command.parameters}")
            return True

        elif command.command_type == "run_diagnostics":
            print(f"[Neuralink] Running implant diagnostics...")
            await asyncio.sleep(1.0)
            print(f"[Neuralink] Diagnostics complete - all systems nominal")
            return True

        return False

    async def _neural_stimulation(self, params: Dict[str, Any]) -> bool:
        """
        Perform neural stimulation (therapeutic or feedback)
        CRITICAL: Requires extensive safety validation
        """
        print(f"[Neuralink] ⚠️  NEURAL STIMULATION REQUEST")
        print(f"[Neuralink] Parameters: {params}")

        # Safety checks
        amplitude = params.get("amplitude_ua", 0)  # microamps
        frequency = params.get("frequency_hz", 0)
        duration_ms = params.get("duration_ms", 0)
        target_channels = params.get("channels", [])

        # Safety limits (example values)
        if amplitude > 100:  # 100 µA max
            print(f"[Neuralink] ❌ REJECTED: Amplitude exceeds safety limit")
            return False

        if frequency > 200:  # 200 Hz max
            print(f"[Neuralink] ❌ REJECTED: Frequency exceeds safety limit")
            return False

        if duration_ms > 1000:  # 1 second max
            print(f"[Neuralink] ❌ REJECTED: Duration exceeds safety limit")
            return False

        print(f"[Neuralink] ✓ Safety checks passed")
        print(f"[Neuralink] Stimulating {len(target_channels)} channels...")
        await asyncio.sleep(duration_ms / 1000.0)
        print(f"[Neuralink] Stimulation complete")
        return True

    def get_device_info(self) -> Dict[str, Any]:
        """Get Neuralink implant information"""
        return {
            "device_id": self.device_id,
            "device_type": BCIDeviceType.NEURALINK.value,
            "manufacturer": "Neuralink",
            "model": "N1 Implant",
            "implant_location": self.implant_location,
            "channels": self.num_channels,
            "sample_rate": self.sample_rate,
            "signal_type": self.signal_type.value,
            "capabilities": [
                "high_resolution_recording",
                "spike_sorting",
                "motor_decoding",
                "neural_stimulation",
                "bidirectional_interface"
            ],
            "invasive": True,
            "wireless": True,
            "battery_powered": True,
            "fda_status": "investigational"
        }

    async def calibrate(self) -> bool:
        """Perform neural decoder calibration"""
        print(f"[Neuralink] Starting calibration protocol...")
        print(f"[Neuralink] Recording baseline neural activity...")
        await asyncio.sleep(3.0)
        print(f"[Neuralink] Training decoder models...")
        await asyncio.sleep(2.0)
        print(f"[Neuralink] Calibration complete - decoder accuracy: 94.5%")
        return True

    def get_signal_quality(self) -> SignalQuality:
        """Get current signal quality"""
        if self._last_reading:
            return self._last_reading.signal_quality
        return SignalQuality.NO_SIGNAL

    def is_connected(self) -> bool:
        """Check connection status"""
        return self.connected
