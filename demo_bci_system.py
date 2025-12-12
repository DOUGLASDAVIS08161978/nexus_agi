#!/usr/bin/env python3
"""
Universal BCI Controller System - Comprehensive Demo
Demonstrates the most powerful universal BCI control system
"""
import asyncio
import sys
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, '/home/user/nexus_agi')

from bci import (
    UniversalBCIOrchestrator,
    BCIMindControlBridge,
    NeuroSkyAdapter,
    NeuralinkAdapter,
    StentrodeAdapter,
    OpenBCIAdapter,
    EmotivEPOCAdapter,
    BCICommand
)

# Mock MindControl components for demonstration
from mindcontrol.controller import MindControlConductor, RiskLevel


class MockRegistry:
    """Mock registry for demo"""
    def list_agents(self):
        return [
            {
                "id": "agent_alpha",
                "capabilities": ["general", "analysis"],
                "load": 0.3,
                "avg_reward": 0.8,
                "trust_score": 0.9
            },
            {
                "id": "agent_beta",
                "capabilities": ["general", "control"],
                "load": 0.5,
                "avg_reward": 0.7,
                "trust_score": 0.85
            }
        ]


class MockPolicyEngine:
    """Mock policy engine for demo"""
    def evaluate(self, task):
        class PolicyResult:
            def __init__(self):
                self.risk = RiskLevel.LOW
                self.flags = ["approved"]
                self.risk_score = 0.2

        return PolicyResult()


class MockLogger:
    """Mock logger for demo"""
    def info(self, message, extra=None):
        pass


async def demo_basic_operation(orchestrator: UniversalBCIOrchestrator):
    """Demo 1: Basic BCI operations"""
    print("\n" + "="*80)
    print("DEMO 1: BASIC BCI DEVICE OPERATIONS")
    print("="*80)

    # List registered devices
    print("\n📋 Registered Devices:")
    devices = orchestrator.list_devices()
    for dev in devices:
        print(f"  • {dev['id']}: {dev['info']['manufacturer']} {dev['info']['model']}")
        print(f"    Capabilities: {', '.join(dev['info']['capabilities'])}")

    # Connect all devices
    print("\n🔌 Connecting to all devices...")
    await orchestrator.connect_all_devices()
    await asyncio.sleep(1)

    # Calibrate devices
    print("\n🎯 Calibrating devices...")
    for device_id, device in orchestrator.devices.items():
        if device.is_connected():
            await device.calibrate()

    print("\n✓ Basic operations complete")


async def demo_neural_streaming(orchestrator: UniversalBCIOrchestrator):
    """Demo 2: Neural signal streaming"""
    print("\n" + "="*80)
    print("DEMO 2: NEURAL SIGNAL STREAMING")
    print("="*80)

    # Start streaming from all devices
    print("\n📡 Starting neural signal streaming...")
    await orchestrator.start_all_streaming()

    # Collect data for a few seconds
    print("\n📊 Collecting neural data for 5 seconds...")
    print("(Simulating real-time brain activity monitoring)\n")

    readings_collected = 0

    async def count_readings(reading):
        nonlocal readings_collected
        readings_collected += 1
        if readings_collected % 5 == 0:
            print(f"  [{reading.device_id}] Reading #{readings_collected}")
            if reading.mental_state:
                print(f"    Attention: {reading.mental_state.attention:.2f}")
                print(f"    Cognitive Load: {reading.mental_state.cognitive_load:.2f}")
                print(f"    Engagement: {reading.mental_state.engagement:.2f}")
            if reading.motor_intent and reading.motor_intent.movement_type != "rest":
                print(f"    Motor Intent: {reading.motor_intent.movement_type}")

    orchestrator.on_reading(count_readings)

    await asyncio.sleep(5)

    print(f"\n✓ Collected {readings_collected} neural readings")


async def demo_safety_system(orchestrator: UniversalBCIOrchestrator):
    """Demo 3: Safety monitoring and validation"""
    print("\n" + "="*80)
    print("DEMO 3: ADVANCED SAFETY MONITORING")
    print("="*80)

    # Test safe stimulation
    print("\n🛡️  Testing SAFE stimulation parameters...")
    safe_command = BCICommand(
        command_type="stimulate",
        parameters={
            "amplitude_ua": 50,  # 50 µA (safe)
            "frequency_hz": 100,  # 100 Hz (safe)
            "duration_ms": 500,  # 500 ms (safe)
            "channels": [0, 1, 2],
            "charge_balanced": True
        }
    )

    # This should pass
    await orchestrator.send_command("neuralink_n1_001", safe_command)

    print("\n🛡️  Testing UNSAFE stimulation parameters...")
    unsafe_command = BCICommand(
        command_type="stimulate",
        parameters={
            "amplitude_ua": 150,  # 150 µA (UNSAFE - exceeds limit)
            "frequency_hz": 300,  # 300 Hz (UNSAFE - exceeds limit)
            "duration_ms": 500,
            "channels": [0, 1, 2],
            "charge_balanced": False
        }
    )

    # This should be rejected
    await orchestrator.send_command("neuralink_n1_001", unsafe_command)

    # Show safety status
    print("\n📊 Safety System Status:")
    safety_status = orchestrator.safety_monitor.get_safety_status()
    print(f"  Overall Status: {safety_status['overall_status']}")
    print(f"  Recent Alerts: {safety_status['recent_alerts']}")
    print(f"  Critical Alerts: {safety_status['critical_alerts']}")

    print("\n✓ Safety system demonstration complete")


async def demo_mindcontrol_integration(orchestrator: UniversalBCIOrchestrator,
                                      bridge: BCIMindControlBridge):
    """Demo 4: BCI-MindControl integration"""
    print("\n" + "="*80)
    print("DEMO 4: BCI-MINDCONTROL INTEGRATION")
    print("="*80)

    print("\n🧠 Activating BCI-MindControl bridge...")
    await bridge.start()

    print("\n🔄 Integration will enhance AGI decisions with neural context...")
    print("(Monitoring for 5 seconds)\n")

    await asyncio.sleep(5)

    # Show cognitive state
    cog_state = bridge.get_current_cognitive_state()
    if cog_state:
        print("\n📊 Current Cognitive State Analysis:")
        print(f"  Attention Level: {cog_state['attention']:.2f}")
        print(f"  Cognitive Load: {cog_state['cognitive_load']:.2f}")
        print(f"  Stress Level: {cog_state['stress_level']:.2f}")
        print(f"  Decision Readiness: {cog_state['decision_readiness']:.2f}")
        print(f"  Mental Capacity: {cog_state['mental_capacity']:.2f}")

    print("\n✓ Integration demonstration complete")


async def demo_multi_device_fusion(orchestrator: UniversalBCIOrchestrator):
    """Demo 5: Multi-device data fusion"""
    print("\n" + "="*80)
    print("DEMO 5: MULTI-DEVICE NEURAL FUSION")
    print("="*80)

    print("\n🔀 Fusing data from multiple BCI devices...")
    print("(Non-invasive + Invasive hybrid control)")

    device_readings = {}

    async def collect_device_data(reading):
        device_type = reading.device_type.value
        device_readings[device_type] = reading

    orchestrator.on_reading(collect_device_data)

    await asyncio.sleep(3)

    print(f"\n📊 Fusion Analysis from {len(device_readings)} device types:")

    for device_type, reading in device_readings.items():
        print(f"\n  {device_type.upper()}:")
        if reading.mental_state:
            print(f"    Attention: {reading.mental_state.attention:.2f}")
            print(f"    Engagement: {reading.mental_state.engagement:.2f}")
        print(f"    Signal Quality: {reading.signal_quality.value}")

    print("\n✓ Multi-device fusion complete")


async def demo_advanced_commands(orchestrator: UniversalBCIOrchestrator):
    """Demo 6: Advanced device commands"""
    print("\n" + "="*80)
    print("DEMO 6: ADVANCED DEVICE COMMANDS")
    print("="*80)

    # Stentrode mode switching
    print("\n🎮 Stentrode: Switching to cursor control mode...")
    await orchestrator.send_command(
        "stentrode_001",
        BCICommand(
            command_type="switch_mode",
            parameters={"mode": "cursor_control"}
        )
    )

    # Neuralink diagnostics
    print("\n🔍 Neuralink: Running implant diagnostics...")
    await orchestrator.send_command(
        "neuralink_n1_001",
        BCICommand(command_type="run_diagnostics")
    )

    # OpenBCI channel configuration
    print("\n⚙️  OpenBCI: Configuring custom channel layout...")
    await orchestrator.send_command(
        "openbci_001",
        BCICommand(
            command_type="set_channels",
            parameters={"active_channels": [0, 1, 2, 3, 4, 5]}
        )
    )

    print("\n✓ Advanced commands executed")


async def main():
    """Main demonstration program"""
    print("\n" + "="*80)
    print("🧠 UNIVERSAL BCI CONTROLLER SYSTEM - FULL DEMONSTRATION 🧠")
    print("="*80)
    print("\nThe Most Powerful Universal Brain-Computer Interface Control System")
    print("Supporting: NeuroSky, Neuralink, Stentrode, OpenBCI, Emotiv, and more")
    print("\nIntegrated with Nexus AGI MindControl System")
    print("="*80)

    # Initialize orchestrator
    print("\n🚀 Initializing Universal BCI Orchestrator...")
    orchestrator = UniversalBCIOrchestrator()

    # Register all BCI devices
    print("\n📝 Registering BCI devices...")

    # Non-invasive devices
    orchestrator.register_device("neurosky_001", NeuroSkyAdapter("neurosky_001"))
    orchestrator.register_device("openbci_001", OpenBCIAdapter("openbci_001", "cyton"))
    orchestrator.register_device("emotiv_001", EmotivEPOCAdapter("emotiv_001"))

    # Invasive devices
    orchestrator.register_device("neuralink_n1_001", NeuralinkAdapter("neuralink_n1_001", "motor_cortex"))
    orchestrator.register_device("stentrode_001", StentrodeAdapter("stentrode_001"))

    # Initialize MindControl integration
    print("\n🧠 Initializing MindControl integration...")
    mock_registry = MockRegistry()
    mock_policy = MockPolicyEngine()
    mock_logger = MockLogger()

    mindcontrol = MindControlConductor(
        registry=mock_registry,
        policy_engine=mock_policy,
        logger=mock_logger
    )

    bridge = BCIMindControlBridge(orchestrator, mindcontrol)

    # Start session
    print("\n📋 Starting BCI session...")
    session_id = await orchestrator.start_session()
    print(f"Session ID: {session_id}")

    try:
        # Run demonstrations
        await demo_basic_operation(orchestrator)
        await demo_neural_streaming(orchestrator)
        await demo_safety_system(orchestrator)
        await demo_mindcontrol_integration(orchestrator, bridge)
        await demo_multi_device_fusion(orchestrator)
        await demo_advanced_commands(orchestrator)

        # Final system status
        print("\n" + "="*80)
        print("FINAL SYSTEM STATUS")
        print("="*80)
        orchestrator.print_status()

        # Performance summary
        print("\n" + "="*80)
        print("🏆 DEMONSTRATION COMPLETE - SYSTEM PERFORMANCE SUMMARY")
        print("="*80)
        print(f"\n✓ Successfully integrated {len(orchestrator.devices)} BCI devices")
        print(f"✓ Processed {orchestrator.active_session.total_readings} neural readings")
        print(f"✓ Safety monitoring: {orchestrator.active_session.safety_alerts} alerts")
        print(f"✓ All systems operational")
        print("\n🧠 Universal BCI Controller: FULLY OPERATIONAL")
        print("   Ready for advanced neural control and AGI integration")
        print("\n" + "="*80)

    finally:
        # Cleanup
        print("\n🛑 Shutting down session...")
        await orchestrator.end_session()

    print("\n✅ All demonstrations completed successfully!\n")


if __name__ == "__main__":
    print("\n🚀 Starting Universal BCI Controller System Demo...\n")
    asyncio.run(main())
