"""
BCI Safety Monitoring System
Critical safety checks and validation for brain-computer interfaces
"""
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

from bci.core.interfaces import (
    IBCISafetyMonitor, BCIReading, BCICommand,
    SignalQuality, BCIDeviceType
)


class SafetyLevel(Enum):
    """Safety status levels"""
    SAFE = "safe"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class SafetyAlert:
    """Safety alert structure"""
    level: SafetyLevel
    message: str
    timestamp: datetime
    device_id: str
    recommended_action: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class BCISafetyMonitor(IBCISafetyMonitor):
    """
    Comprehensive safety monitoring for BCI systems
    Ensures user safety and device operation within safe parameters
    """

    def __init__(self):
        self.alerts: List[SafetyAlert] = []
        self.reading_history: List[BCIReading] = []
        self.max_history = 1000

        # Safety thresholds
        self.thresholds = {
            "max_signal_amplitude": 5000,  # µV
            "min_signal_quality_score": 0.3,
            "max_artifact_percentage": 0.5,
            "max_stimulation_amplitude": 100,  # µA for invasive
            "max_stimulation_frequency": 200,  # Hz
            "max_stimulation_duration": 1000,  # ms
            "max_temperature": 39.0,  # °C
            "min_battery_level": 0.15,  # 15%
            "max_consecutive_poor_signals": 10
        }

        self.consecutive_poor_signals = 0

    def validate_reading(self, reading: BCIReading) -> bool:
        """
        Validate that reading is safe and within acceptable parameters
        """
        self._add_to_history(reading)

        # Check 1: Signal quality
        if reading.signal_quality == SignalQuality.NO_SIGNAL:
            self._create_alert(
                SafetyLevel.CRITICAL,
                f"No signal detected from device {reading.device_id}",
                reading.device_id,
                "Check device connection and electrode contact"
            )
            return False

        if reading.signal_quality == SignalQuality.POOR:
            self.consecutive_poor_signals += 1
            if self.consecutive_poor_signals >= self.thresholds["max_consecutive_poor_signals"]:
                self._create_alert(
                    SafetyLevel.WARNING,
                    f"Persistent poor signal quality on {reading.device_id}",
                    reading.device_id,
                    "Adjust electrodes or check device placement"
                )
        else:
            self.consecutive_poor_signals = 0

        # Check 2: Signal amplitude (saturation check)
        for channel, data in reading.raw_channels.items():
            max_amp = abs(data).max() if len(data) > 0 else 0
            if max_amp > self.thresholds["max_signal_amplitude"]:
                self._create_alert(
                    SafetyLevel.WARNING,
                    f"Signal saturation detected on channel {channel}",
                    reading.device_id,
                    "Check electrode impedance and connection"
                )

        # Check 3: Contact quality
        poor_contacts = [
            ch for ch, quality in reading.contact_quality.items()
            if quality < self.thresholds["min_signal_quality_score"]
        ]
        if len(poor_contacts) > len(reading.contact_quality) / 2:
            self._create_alert(
                SafetyLevel.WARNING,
                f"Multiple poor electrode contacts: {poor_contacts}",
                reading.device_id,
                "Clean and reposition electrodes"
            )

        # Check 4: Battery level
        if reading.battery_level < self.thresholds["min_battery_level"]:
            self._create_alert(
                SafetyLevel.WARNING,
                f"Low battery: {reading.battery_level * 100:.0f}%",
                reading.device_id,
                "Charge or replace battery soon"
            )

        # Check 5: Temperature (for implants)
        if "temperature_c" in reading.metadata:
            temp = reading.metadata["temperature_c"]
            if temp > self.thresholds["max_temperature"]:
                self._create_alert(
                    SafetyLevel.CRITICAL,
                    f"High device temperature: {temp}°C",
                    reading.device_id,
                    "STOP USE IMMEDIATELY - Potential overheating"
                )
                return False

        # Check 6: Device-specific safety checks
        if not self._device_specific_checks(reading):
            return False

        return True

    def check_stimulation_safety(self, command: BCICommand) -> bool:
        """
        Critical safety check for neural stimulation commands
        MUST be called before any stimulation
        """
        if command.command_type != "stimulate":
            return True  # Not a stimulation command

        print("[Safety] ⚠️  STIMULATION SAFETY CHECK INITIATED")

        params = command.parameters

        # Extract stimulation parameters
        amplitude = params.get("amplitude_ua", 0)
        frequency = params.get("frequency_hz", 0)
        duration = params.get("duration_ms", 0)
        target_channels = params.get("channels", [])
        waveform = params.get("waveform", "unknown")

        safety_checks = []

        # Check 1: Amplitude limit
        if amplitude > self.thresholds["max_stimulation_amplitude"]:
            safety_checks.append(f"❌ Amplitude {amplitude}µA exceeds limit {self.thresholds['max_stimulation_amplitude']}µA")
        else:
            safety_checks.append(f"✓ Amplitude within safe range")

        # Check 2: Frequency limit
        if frequency > self.thresholds["max_stimulation_frequency"]:
            safety_checks.append(f"❌ Frequency {frequency}Hz exceeds limit {self.thresholds['max_stimulation_frequency']}Hz")
        else:
            safety_checks.append(f"✓ Frequency within safe range")

        # Check 3: Duration limit
        if duration > self.thresholds["max_stimulation_duration"]:
            safety_checks.append(f"❌ Duration {duration}ms exceeds limit {self.thresholds['max_stimulation_duration']}ms")
        else:
            safety_checks.append(f"✓ Duration within safe range")

        # Check 4: Target specification
        if not target_channels:
            safety_checks.append(f"❌ No target channels specified")
        else:
            safety_checks.append(f"✓ Targets: {len(target_channels)} channels")

        # Check 5: Charge balance (for safety)
        charge_balanced = params.get("charge_balanced", False)
        if not charge_balanced:
            safety_checks.append(f"⚠️  Warning: Stimulation not charge-balanced")
        else:
            safety_checks.append(f"✓ Charge-balanced waveform")

        # Print all checks
        for check in safety_checks:
            print(f"[Safety]   {check}")

        # Determine if safe
        failed_checks = [c for c in safety_checks if "❌" in c]
        if failed_checks:
            print(f"[Safety] ❌ STIMULATION REJECTED - {len(failed_checks)} safety violation(s)")
            self._create_alert(
                SafetyLevel.CRITICAL,
                f"Unsafe stimulation parameters detected",
                "system",
                "Review and adjust stimulation parameters"
            )
            return False

        warnings = [c for c in safety_checks if "⚠️" in c]
        if warnings:
            print(f"[Safety] ⚠️  {len(warnings)} warning(s) - proceed with caution")

        print("[Safety] ✓ STIMULATION APPROVED")
        return True

    def get_safety_status(self) -> Dict[str, Any]:
        """Get comprehensive safety status"""
        recent_alerts = [
            a for a in self.alerts
            if a.timestamp > datetime.utcnow() - timedelta(minutes=5)
        ]

        critical_alerts = [a for a in recent_alerts if a.level == SafetyLevel.CRITICAL]
        warning_alerts = [a for a in recent_alerts if a.level == SafetyLevel.WARNING]

        overall_status = SafetyLevel.SAFE
        if critical_alerts:
            overall_status = SafetyLevel.CRITICAL
        elif warning_alerts:
            overall_status = SafetyLevel.WARNING

        return {
            "overall_status": overall_status.value,
            "recent_alerts": len(recent_alerts),
            "critical_alerts": len(critical_alerts),
            "warning_alerts": len(warning_alerts),
            "consecutive_poor_signals": self.consecutive_poor_signals,
            "readings_in_history": len(self.reading_history),
            "alerts": [
                {
                    "level": a.level.value,
                    "message": a.message,
                    "timestamp": a.timestamp.isoformat(),
                    "action": a.recommended_action
                }
                for a in recent_alerts[-5:]  # Last 5 alerts
            ]
        }

    def clear_old_alerts(self, hours: int = 24):
        """Clear alerts older than specified hours"""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        self.alerts = [a for a in self.alerts if a.timestamp > cutoff]

    def _device_specific_checks(self, reading: BCIReading) -> bool:
        """Device-specific safety validations"""

        # Invasive devices require extra monitoring
        if reading.device_type in [BCIDeviceType.NEURALINK, BCIDeviceType.STENTRODE]:
            # Check implant health
            if "implant_health" in reading.metadata:
                health = reading.metadata["implant_health"]
                if health not in ["nominal", "stable", "good"]:
                    self._create_alert(
                        SafetyLevel.CRITICAL,
                        f"Implant health issue: {health}",
                        reading.device_id,
                        "Contact medical professional immediately"
                    )
                    return False

            # Check for infection indicators (temperature, impedance changes)
            if "temperature_c" in reading.metadata:
                temp = reading.metadata["temperature_c"]
                if temp > 38.0:  # Elevated temp
                    self._create_alert(
                        SafetyLevel.WARNING,
                        f"Elevated temperature: {temp}°C",
                        reading.device_id,
                        "Monitor for infection signs"
                    )

        return True

    def _create_alert(self, level: SafetyLevel, message: str,
                     device_id: str, action: str):
        """Create and log safety alert"""
        alert = SafetyAlert(
            level=level,
            message=message,
            timestamp=datetime.utcnow(),
            device_id=device_id,
            recommended_action=action
        )
        self.alerts.append(alert)

        # Print critical alerts immediately
        if level in [SafetyLevel.CRITICAL, SafetyLevel.EMERGENCY]:
            print(f"\n{'='*60}")
            print(f"🚨 {level.value.upper()} ALERT 🚨")
            print(f"Device: {device_id}")
            print(f"Message: {message}")
            print(f"Action: {action}")
            print(f"{'='*60}\n")

    def _add_to_history(self, reading: BCIReading):
        """Add reading to history with size limit"""
        self.reading_history.append(reading)
        if len(self.reading_history) > self.max_history:
            self.reading_history.pop(0)
