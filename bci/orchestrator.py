"""
Universal BCI Controller Orchestrator
Main coordination layer for all brain-computer interface devices
Integrates with Nexus AGI MindControl system
"""
import asyncio
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from dataclasses import dataclass, field
import json

from bci.core.interfaces import (
    IBCIDevice, BCIReading, BCICommand, BCIDeviceType,
    SignalQuality, MentalState
)
from bci.processing.signal_processor import UniversalSignalProcessor
from bci.safety.safety_monitor import BCISafetyMonitor, SafetyLevel


@dataclass
class BCISession:
    """Active BCI session tracking"""
    session_id: str
    start_time: datetime
    active_devices: List[str] = field(default_factory=list)
    total_readings: int = 0
    safety_alerts: int = 0
    status: str = "active"


class UniversalBCIOrchestrator:
    """
    Master orchestrator for universal BCI control
    Coordinates multiple devices, ensures safety, and provides unified interface
    """

    def __init__(self):
        self.devices: Dict[str, IBCIDevice] = {}
        self.signal_processor = UniversalSignalProcessor()
        self.safety_monitor = BCISafetyMonitor()

        self.active_session: Optional[BCISession] = None
        self.reading_callbacks: List[Callable] = []
        self.alert_callbacks: List[Callable] = []

        self.streaming_tasks: Dict[str, asyncio.Task] = {}
        self.running = False

        print("[Orchestrator] Universal BCI Controller initialized")

    # ========== Device Management ==========

    def register_device(self, device_id: str, device: IBCIDevice):
        """Register a BCI device with the orchestrator"""
        self.devices[device_id] = device
        device_info = device.get_device_info()
        print(f"[Orchestrator] Registered {device_info['device_type']} device: {device_id}")
        print(f"[Orchestrator]   Manufacturer: {device_info.get('manufacturer')}")
        print(f"[Orchestrator]   Capabilities: {device_info.get('capabilities')}")

    def get_device(self, device_id: str) -> Optional[IBCIDevice]:
        """Get device by ID"""
        return self.devices.get(device_id)

    def list_devices(self) -> List[Dict[str, Any]]:
        """List all registered devices"""
        return [
            {
                "id": device_id,
                "info": device.get_device_info(),
                "connected": device.is_connected(),
                "signal_quality": device.get_signal_quality().value
            }
            for device_id, device in self.devices.items()
        ]

    # ========== Session Management ==========

    async def start_session(self) -> str:
        """Start new BCI session"""
        if self.active_session:
            print("[Orchestrator] Session already active")
            return self.active_session.session_id

        session_id = f"session_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        self.active_session = BCISession(
            session_id=session_id,
            start_time=datetime.utcnow()
        )

        print(f"[Orchestrator] ✓ Started BCI session: {session_id}")
        self.running = True
        return session_id

    async def end_session(self):
        """End active BCI session"""
        if not self.active_session:
            return

        print(f"\n[Orchestrator] Ending session {self.active_session.session_id}")

        # Stop all streaming
        await self.stop_all_streaming()

        # Disconnect all devices
        for device_id, device in self.devices.items():
            if device.is_connected():
                await device.disconnect()

        # Session summary
        duration = datetime.utcnow() - self.active_session.start_time
        print(f"[Orchestrator] Session Summary:")
        print(f"  Duration: {duration}")
        print(f"  Total Readings: {self.active_session.total_readings}")
        print(f"  Safety Alerts: {self.active_session.safety_alerts}")
        print(f"  Active Devices: {len(self.active_session.active_devices)}")

        self.active_session = None
        self.running = False

    # ========== Connection Management ==========

    async def connect_device(self, device_id: str) -> bool:
        """Connect to a specific device"""
        device = self.devices.get(device_id)
        if not device:
            print(f"[Orchestrator] Device {device_id} not found")
            return False

        print(f"[Orchestrator] Connecting to {device_id}...")
        success = await device.connect()

        if success:
            if self.active_session:
                self.active_session.active_devices.append(device_id)
            print(f"[Orchestrator] ✓ {device_id} connected")
        else:
            print(f"[Orchestrator] ✗ Failed to connect {device_id}")

        return success

    async def connect_all_devices(self) -> Dict[str, bool]:
        """Connect to all registered devices"""
        print("[Orchestrator] Connecting to all devices...")
        results = {}

        tasks = [
            (device_id, device.connect())
            for device_id, device in self.devices.items()
        ]

        for device_id, task in tasks:
            try:
                success = await task
                results[device_id] = success
                if success and self.active_session:
                    self.active_session.active_devices.append(device_id)
            except Exception as e:
                print(f"[Orchestrator] Error connecting {device_id}: {e}")
                results[device_id] = False

        connected = sum(1 for v in results.values() if v)
        print(f"[Orchestrator] Connected {connected}/{len(self.devices)} devices")
        return results

    # ========== Data Streaming ==========

    async def start_streaming(self, device_id: str):
        """Start data streaming from a device"""
        device = self.devices.get(device_id)
        if not device or not device.is_connected():
            print(f"[Orchestrator] Cannot stream from {device_id} - not connected")
            return False

        await device.start_streaming()

        # Start background streaming task
        task = asyncio.create_task(self._stream_device_data(device_id))
        self.streaming_tasks[device_id] = task

        print(f"[Orchestrator] ✓ Streaming started for {device_id}")
        return True

    async def start_all_streaming(self):
        """Start streaming from all connected devices"""
        print("[Orchestrator] Starting streaming from all connected devices...")

        for device_id, device in self.devices.items():
            if device.is_connected():
                await self.start_streaming(device_id)

    async def stop_streaming(self, device_id: str):
        """Stop streaming from a device"""
        device = self.devices.get(device_id)
        if device:
            await device.stop_streaming()

        # Cancel streaming task
        if device_id in self.streaming_tasks:
            self.streaming_tasks[device_id].cancel()
            del self.streaming_tasks[device_id]

        print(f"[Orchestrator] ✓ Streaming stopped for {device_id}")

    async def stop_all_streaming(self):
        """Stop streaming from all devices"""
        print("[Orchestrator] Stopping all streaming...")

        for device_id in list(self.streaming_tasks.keys()):
            await self.stop_streaming(device_id)

    async def _stream_device_data(self, device_id: str):
        """Background task to continuously stream data from a device"""
        device = self.devices[device_id]

        try:
            while self.running and device.is_connected():
                # Read from device
                reading = await device.read()

                if reading:
                    # Safety validation
                    is_safe = self.safety_monitor.validate_reading(reading)

                    if not is_safe:
                        print(f"[Orchestrator] ⚠️  Safety issue detected on {device_id}")
                        self.active_session.safety_alerts += 1

                    # Update session stats
                    if self.active_session:
                        self.active_session.total_readings += 1

                    # Trigger callbacks
                    for callback in self.reading_callbacks:
                        try:
                            await callback(reading)
                        except Exception as e:
                            print(f"[Orchestrator] Callback error: {e}")

                # Streaming rate control
                await asyncio.sleep(0.1)  # 10 Hz orchestrator polling

        except asyncio.CancelledError:
            print(f"[Orchestrator] Streaming cancelled for {device_id}")
        except Exception as e:
            print(f"[Orchestrator] Streaming error on {device_id}: {e}")

    # ========== Command Execution ==========

    async def send_command(self, device_id: str, command: BCICommand) -> bool:
        """Send command to specific device with safety checks"""
        device = self.devices.get(device_id)
        if not device:
            print(f"[Orchestrator] Device {device_id} not found")
            return False

        # Safety check for stimulation commands
        if command.command_type == "stimulate":
            if not self.safety_monitor.check_stimulation_safety(command):
                print(f"[Orchestrator] ❌ Command rejected by safety monitor")
                return False

        print(f"[Orchestrator] Sending command to {device_id}: {command.command_type}")
        success = await device.send_command(command)

        return success

    # ========== Callbacks ==========

    def on_reading(self, callback: Callable):
        """Register callback for new BCI readings"""
        self.reading_callbacks.append(callback)

    def on_alert(self, callback: Callable):
        """Register callback for safety alerts"""
        self.alert_callbacks.append(callback)

    # ========== Status and Reporting ==========

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            "session": {
                "active": self.active_session is not None,
                "session_id": self.active_session.session_id if self.active_session else None,
                "start_time": self.active_session.start_time.isoformat() if self.active_session else None,
                "total_readings": self.active_session.total_readings if self.active_session else 0
            },
            "devices": {
                "total": len(self.devices),
                "connected": sum(1 for d in self.devices.values() if d.is_connected()),
                "streaming": len(self.streaming_tasks),
                "list": self.list_devices()
            },
            "safety": self.safety_monitor.get_safety_status(),
            "timestamp": datetime.utcnow().isoformat()
        }

    def print_status(self):
        """Print formatted system status"""
        status = self.get_system_status()

        print("\n" + "="*70)
        print("UNIVERSAL BCI CONTROLLER - SYSTEM STATUS")
        print("="*70)

        print(f"\n📊 SESSION:")
        if status["session"]["active"]:
            print(f"  Status: ACTIVE")
            print(f"  Session ID: {status['session']['session_id']}")
            print(f"  Total Readings: {status['session']['total_readings']}")
        else:
            print(f"  Status: INACTIVE")

        print(f"\n🔌 DEVICES:")
        print(f"  Total Registered: {status['devices']['total']}")
        print(f"  Connected: {status['devices']['connected']}")
        print(f"  Streaming: {status['devices']['streaming']}")

        print(f"\n🛡️  SAFETY:")
        safety = status["safety"]
        print(f"  Overall Status: {safety['overall_status'].upper()}")
        print(f"  Recent Alerts: {safety['recent_alerts']}")
        print(f"  Critical Alerts: {safety['critical_alerts']}")
        print(f"  Warning Alerts: {safety['warning_alerts']}")

        print("\n" + "="*70 + "\n")
