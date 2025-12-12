"""
Stentrode BCI Adapter (Synchron)
Endovascular neural interface adapter
NOTE: This is a simulation adapter as Stentrode API is not publicly available
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


class StentrodeAdapter(IBCIDevice):
    """
    Adapter for Synchron Stentrode endovascular neural interface
    Placed in blood vessel near motor cortex
    """

    def __init__(self, device_id: str = "stentrode_001", vessel_location: str = "superior_sagittal_sinus"):
        self.device_id = device_id
        self.vessel_location = vessel_location
        self.connected = False
        self.streaming = False
        self._last_reading = None

        # Stentrode specifications
        self.num_electrodes = 16  # 16 recording electrodes
        self.sample_rate = 2000.0  # 2 kHz
        self.signal_type = BCISignalType.ECOG  # Electrocorticography-like

    async def connect(self) -> bool:
        """Connect to Stentrode implant"""
        try:
            print(f"[Stentrode] Connecting to endovascular interface {self.device_id}...")
            print(f"[Stentrode] Location: {self.vessel_location}")
            await asyncio.sleep(0.8)

            # Simulate telemetry check
            print(f"[Stentrode] Verifying implant integrity...")
            await asyncio.sleep(0.5)

            self.connected = True
            print(f"[Stentrode] Connected successfully")
            print(f"[Stentrode] {self.num_electrodes} electrodes active")
            return True
        except Exception as e:
            print(f"[Stentrode] Connection failed: {e}")
            return False

    async def disconnect(self) -> bool:
        """Disconnect from Stentrode"""
        self.streaming = False
        await asyncio.sleep(0.2)
        self.connected = False
        print(f"[Stentrode] Disconnected from {self.device_id}")
        return True

    async def start_streaming(self) -> bool:
        """Start neural signal streaming"""
        if not self.connected:
            return False
        self.streaming = True
        print(f"[Stentrode] Streaming started - {self.num_electrodes} channels @ {self.sample_rate} Hz")
        return True

    async def stop_streaming(self) -> bool:
        """Stop streaming"""
        self.streaming = False
        print(f"[Stentrode] Streaming stopped")
        return True

    async def read(self) -> Optional[BCIReading]:
        """Read neural data from Stentrode"""
        if not self.connected or not self.streaming:
            return None

        # Simulate ECoG-like recording from motor cortex
        raw_channels = {}
        for i in range(self.num_electrodes):
            # Simulate neural signal with motor-related activity
            signal = np.random.randn(200) * 50  # 100ms of data
            raw_channels[f"electrode_{i:02d}"] = signal

        # Decode motor intent (Stentrode's primary use case)
        motor_intent = self._decode_motor_commands()

        # Mental state from cortical activity
        mental_state = MentalState(
            attention=random.uniform(0.6, 0.9),
            meditation=random.uniform(0.3, 0.6),
            cognitive_load=random.uniform(0.5, 0.8),
            stress_level=random.uniform(0.2, 0.5),
            engagement=random.uniform(0.7, 0.95),
            drowsiness=random.uniform(0.0, 0.3),
            emotional_valence=random.uniform(-0.2, 0.4),
            arousal=random.uniform(0.5, 0.85)
        )

        # Brainwave extraction
        brainwaves = BrainwaveData(
            delta=random.uniform(80000, 300000),
            theta=random.uniform(50000, 200000),
            alpha=random.uniform(30000, 150000),
            beta=random.uniform(20000, 120000),
            gamma=random.uniform(15000, 100000)
        )

        # Signal quality typically very good for endovascular placement
        signal_quality = random.choice([
            SignalQuality.EXCELLENT,
            SignalQuality.EXCELLENT,
            SignalQuality.GOOD
        ])

        # Electrode contact quality (blood vessel interface)
        contact_quality = {
            f"electrode_{i:02d}": random.uniform(0.8, 0.98)
            for i in range(self.num_electrodes)
        }

        reading = BCIReading(
            timestamp=datetime.utcnow(),
            device_id=self.device_id,
            device_type=BCIDeviceType.STENTRODE,
            signal_quality=signal_quality,
            raw_channels=raw_channels,
            sample_rate=self.sample_rate,
            brainwaves=brainwaves,
            mental_state=mental_state,
            motor_intent=motor_intent,
            contact_quality=contact_quality,
            battery_level=random.uniform(0.75, 1.0),
            metadata={
                "vessel_location": self.vessel_location,
                "num_electrodes": self.num_electrodes,
                "signal_type": self.signal_type.value,
                "implant_type": "endovascular",
                "recording_mode": "motor_cortex",
                "device_health": "stable",
                "blood_flow": "normal"
            }
        )

        self._last_reading = reading
        return reading

    def _decode_motor_commands(self) -> MotorIntent:
        """
        Decode motor commands for computer control
        Stentrode is designed for click/select interactions
        """
        # Common commands: click, dwell, select, move cursor
        commands = ["click", "dwell_select", "move_up", "move_down",
                   "move_left", "move_right", "rest", "select"]

        movement = random.choice(commands)

        if movement == "rest":
            return MotorIntent(
                movement_type="rest",
                direction=None,
                velocity=0.0,
                confidence=random.uniform(0.85, 0.95)
            )

        # For movement commands
        direction_map = {
            "move_up": np.array([0, 1, 0]),
            "move_down": np.array([0, -1, 0]),
            "move_left": np.array([-1, 0, 0]),
            "move_right": np.array([1, 0, 0]),
            "click": np.array([0, 0, 0]),
            "dwell_select": np.array([0, 0, 0]),
            "select": np.array([0, 0, 0])
        }

        direction = direction_map.get(movement, np.array([0, 0, 0]))

        return MotorIntent(
            movement_type=movement,
            direction=direction,
            velocity=random.uniform(0.3, 0.9),
            confidence=random.uniform(0.7, 0.92)
        )

    async def send_command(self, command: BCICommand) -> bool:
        """Send command to Stentrode"""
        if not self.connected:
            return False

        print(f"[Stentrode] Command received: {command.command_type}")

        if command.command_type == "calibrate":
            await self.calibrate()
            return True

        elif command.command_type == "adjust_sensitivity":
            sensitivity = command.parameters.get("sensitivity", 0.5)
            print(f"[Stentrode] Adjusting decoder sensitivity to {sensitivity}")
            return True

        elif command.command_type == "switch_mode":
            mode = command.parameters.get("mode", "cursor_control")
            print(f"[Stentrode] Switching to mode: {mode}")
            return True

        elif command.command_type == "diagnostics":
            await self._run_diagnostics()
            return True

        return False

    async def _run_diagnostics(self):
        """Run implant diagnostics"""
        print(f"[Stentrode] Running diagnostic tests...")
        await asyncio.sleep(1.5)
        print(f"[Stentrode] ✓ Electrode impedance: Normal")
        print(f"[Stentrode] ✓ Signal integrity: Good")
        print(f"[Stentrode] ✓ Position stability: Stable")
        print(f"[Stentrode] ✓ Blood flow: Normal")
        print(f"[Stentrode] Diagnostics complete")

    def get_device_info(self) -> Dict[str, Any]:
        """Get Stentrode device information"""
        return {
            "device_id": self.device_id,
            "device_type": BCIDeviceType.STENTRODE.value,
            "manufacturer": "Synchron",
            "model": "Stentrode",
            "vessel_location": self.vessel_location,
            "num_electrodes": self.num_electrodes,
            "sample_rate": self.sample_rate,
            "signal_type": self.signal_type.value,
            "capabilities": [
                "motor_decoding",
                "cursor_control",
                "click_detection",
                "typing_interface",
                "assistive_control"
            ],
            "invasive": True,  # Minimally invasive
            "wireless": True,
            "battery_powered": True,
            "implant_method": "endovascular",
            "fda_status": "investigational",
            "primary_use": "assistive_communication"
        }

    async def calibrate(self) -> bool:
        """Calibrate motor command decoder"""
        print(f"[Stentrode] Starting calibration session...")
        print(f"[Stentrode] Please perform the following tasks:")
        print(f"[Stentrode]   - Attempt to move cursor up")
        await asyncio.sleep(1.5)
        print(f"[Stentrode]   - Attempt to move cursor down")
        await asyncio.sleep(1.5)
        print(f"[Stentrode]   - Attempt to click")
        await asyncio.sleep(1.5)
        print(f"[Stentrode] Analyzing motor patterns...")
        await asyncio.sleep(2.0)
        print(f"[Stentrode] ✓ Calibration complete - accuracy: 89.3%")
        return True

    def get_signal_quality(self) -> SignalQuality:
        """Get current signal quality"""
        if self._last_reading:
            return self._last_reading.signal_quality
        return SignalQuality.NO_SIGNAL

    def is_connected(self) -> bool:
        """Check connection status"""
        return self.connected
