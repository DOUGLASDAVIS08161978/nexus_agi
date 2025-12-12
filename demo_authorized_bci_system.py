#!/usr/bin/env python3
"""
Nexus AGI BCI - Authorized Access System Demonstration
Shows Douglas Shane Davis exclusive authentication via patented technologies

US Patents Implemented:
- US Patent 3951134: Remote brain wave monitoring and biometric authentication
- US Patent 6011991: Brain wave communication system
- US Patent 5507291: Emotional state determination
"""
import asyncio
import sys

sys.path.insert(0, '/home/user/nexus_agi')

from bci import UniversalBCIOrchestrator
from bci.devices import NeuroSkyAdapter, NeuralinkAdapter, OpenBCIAdapter
from bci.auth import NexusAGIAuthorizationSystem


async def demo_enrollment(auth_system: NexusAGIAuthorizationSystem,
                          orchestrator: UniversalBCIOrchestrator):
    """Demo 1: User enrollment - only Douglas Shane Davis"""
    print("\n" + "="*80)
    print("DEMO 1: USER ENROLLMENT (US Patent 3951134 - Biometric EEG)")
    print("="*80)

    # Start device for calibration
    device_id = "neurosky_001"
    device = orchestrator.get_device(device_id)

    print(f"\n📡 Connecting to BCI device for calibration...")
    await device.connect()
    await device.start_streaming()

    # Collect calibration samples
    print(f"\n🎯 Collecting calibration samples...")
    print(f"Please remain still and relaxed during calibration...")
    calibration_readings = []

    for i in range(15):
        reading = await device.read()
        if reading:
            calibration_readings.append(reading)
            if (i + 1) % 5 == 0:
                print(f"  Calibration progress: {i+1}/15 samples")
        await asyncio.sleep(0.2)

    print(f"\n✅ Collected {len(calibration_readings)} calibration samples")

    # Attempt enrollment for authorized user (Douglas Shane Davis)
    print(f"\n🔐 Enrolling Authorized User...")
    success = auth_system.enroll_authorized_user(
        user_id="user_dsd_001",
        full_name="Douglas Shane Davis",
        calibration_readings=calibration_readings
    )

    if success:
        print(f"\n✅ Douglas Shane Davis enrolled successfully!")
    else:
        print(f"\n❌ Enrollment failed")

    # Try enrolling unauthorized user (should fail)
    print(f"\n🚫 Testing security: Attempting to enroll unauthorized user...")
    unauthorized_success = auth_system.enroll_authorized_user(
        user_id="user_unauthorized",
        full_name="John Doe",  # Not authorized
        calibration_readings=calibration_readings
    )

    if not unauthorized_success:
        print(f"\n✅ Security test passed: Unauthorized user rejected")

    await device.stop_streaming()
    return success


async def demo_authentication(auth_system: NexusAGIAuthorizationSystem,
                             orchestrator: UniversalBCIOrchestrator):
    """Demo 2: Multi-factor biometric authentication"""
    print("\n" + "="*80)
    print("DEMO 2: MULTI-FACTOR BIOMETRIC AUTHENTICATION")
    print("="*80)

    device_id = "neurosky_001"
    device = orchestrator.get_device(device_id)

    await device.start_streaming()

    # Get current reading for authentication
    print(f"\n🔍 Capturing biometric EEG pattern...")
    reading = None
    for _ in range(5):
        reading = await device.read()
        if reading and reading.brainwaves and reading.mental_state:
            break
        await asyncio.sleep(0.2)

    if not reading:
        print(f"❌ Could not capture BCI reading")
        return None

    print(f"\n🔓 Authenticating Douglas Shane Davis...")
    print(f"   Biometric verification...")
    print(f"   Emotional state analysis...")
    print(f"   Brain wave communication check...")

    session_token = auth_system.authenticate_user(
        user_id="user_dsd_001",
        reading=reading
    )

    if session_token:
        print(f"\n✅ AUTHENTICATION SUCCESSFUL!")
        print(f"   Session token issued: {session_token[:32]}...")

        # Show session info
        session_info = auth_system.get_session_info(session_token)
        print(f"\n📊 Session Information:")
        print(f"   User: {session_info['full_name']}")
        print(f"   Access Level: {session_info['access_level']}")
        print(f"   Biometric Confidence: {session_info['biometric_confidence']:.2%}")
        print(f"   Emotional State: {session_info['emotional_state']}")
        print(f"   Expires: {session_info['expires']}")

        return session_token
    else:
        print(f"\n❌ AUTHENTICATION FAILED")
        return None


async def demo_brain_wave_communication(auth_system: NexusAGIAuthorizationSystem,
                                       orchestrator: UniversalBCIOrchestrator,
                                       session_token: str):
    """Demo 3: Brain wave communication (US Patent 6011991)"""
    print("\n" + "="*80)
    print("DEMO 3: BRAIN WAVE COMMUNICATION (US Patent 6011991)")
    print("="*80)

    device_id = "neurosky_001"
    device = orchestrator.get_device(device_id)

    print(f"\n📡 Monitoring brain wave communication...")
    print(f"   Detecting user intent from neural patterns...")

    messages_detected = 0
    for i in range(20):
        reading = await device.read()
        if reading:
            # Decode brain wave message
            message = auth_system.brainwave_comm.decode_message(
                reading,
                user_id="user_dsd_001"
            )

            if message:
                messages_detected += 1
                print(f"\n   📨 Brain Wave Message #{messages_detected}:")
                print(f"      Intent: {message.intent.value}")
                print(f"      Channel: {message.channel.value}")
                print(f"      Confidence: {message.confidence:.2%}")

        await asyncio.sleep(0.3)

    print(f"\n✅ Detected {messages_detected} brain wave communication messages")

    # Show communication statistics
    comm_stats = auth_system.brainwave_comm.get_statistics()
    print(f"\n📊 Communication Statistics:")
    print(f"   Total Messages: {comm_stats['total_messages']}")
    print(f"   Average Confidence: {comm_stats['average_confidence']:.2%}")
    print(f"   Active Protocols: {comm_stats['active_protocols']}")


async def demo_emotional_monitoring(auth_system: NexusAGIAuthorizationSystem,
                                   orchestrator: UniversalBCIOrchestrator,
                                   session_token: str):
    """Demo 4: Emotional state detection (US Patent 5507291)"""
    print("\n" + "="*80)
    print("DEMO 4: EMOTIONAL STATE DETECTION (US Patent 5507291)")
    print("="*80)

    device_id = "neurosky_001"
    device = orchestrator.get_device(device_id)

    print(f"\n🧠 Monitoring emotional state remotely via EEG...")

    emotional_states = []
    for i in range(15):
        reading = await device.read()
        if reading:
            profile = auth_system.emotional_detector.detect_emotional_state(reading)

            if profile:
                emotional_states.append(profile)

                if len(emotional_states) % 5 == 0:
                    print(f"\n   📊 Emotional State #{len(emotional_states)}:")
                    print(f"      Primary Emotion: {profile.primary_emotion.value}")
                    print(f"      Valence: {profile.valence.value}")
                    print(f"      Arousal: {profile.arousal.value}")
                    print(f"      Stress Level: {profile.stress_level:.2%}")
                    print(f"      Positivity: {profile.positivity:.2%}")
                    print(f"      Confidence: {profile.confidence:.2%}")

        await asyncio.sleep(0.3)

    print(f"\n✅ Monitored {len(emotional_states)} emotional state readings")

    # Show emotional trends
    if len(emotional_states) >= 10:
        trend = auth_system.emotional_detector.get_emotional_trend(window_size=10)
        print(f"\n📈 Emotional Trends:")
        print(f"   Positivity Trend: {trend['positivity_trend']}")
        print(f"   Activation Trend: {trend['activation_trend']}")
        print(f"   Stress Trend: {trend['stress_trend']}")
        print(f"   Emotional Stability: {trend['emotional_stability']:.2%}")


async def demo_continuous_monitoring(auth_system: NexusAGIAuthorizationSystem,
                                    orchestrator: UniversalBCIOrchestrator,
                                    session_token: str):
    """Demo 5: Continuous security monitoring during active session"""
    print("\n" + "="*80)
    print("DEMO 5: CONTINUOUS SECURITY MONITORING")
    print("="*80)

    device_id = "neurosky_001"
    device = orchestrator.get_device(device_id)

    print(f"\n🔒 Continuous biometric and emotional monitoring active...")
    print(f"   Verifying Douglas Shane Davis identity continuously...")

    monitoring_count = 0
    for i in range(10):
        reading = await device.read()
        if reading:
            # Continuous verification
            is_valid = auth_system.continuous_monitoring(session_token, reading)

            monitoring_count += 1

            if monitoring_count % 3 == 0:
                print(f"   ✓ Continuous verification {monitoring_count}/10: {'PASS' if is_valid else 'FAIL'}")

            if not is_valid:
                print(f"\n   🚨 SECURITY ALERT: Biometric mismatch detected!")
                print(f"   Session terminated for security")
                return False

        await asyncio.sleep(0.4)

    print(f"\n✅ Continuous monitoring complete: All checks passed")
    return True


async def demo_unauthorized_access_attempt(auth_system: NexusAGIAuthorizationSystem,
                                          orchestrator: UniversalBCIOrchestrator):
    """Demo 6: Demonstrate security against unauthorized access"""
    print("\n" + "="*80)
    print("DEMO 6: SECURITY TESTING - UNAUTHORIZED ACCESS ATTEMPT")
    print("="*80)

    device_id = "neurosky_001"
    device = orchestrator.get_device(device_id)

    # Get a reading
    reading = await device.read()

    print(f"\n🚫 Simulating unauthorized access attempt...")
    print(f"   Attempting authentication as unauthorized user...")

    # Try to authenticate as unauthorized user
    session_token = auth_system.authenticate_user(
        user_id="user_unauthorized_999",  # Not Douglas Shane Davis
        reading=reading
    )

    if not session_token:
        print(f"\n✅ SECURITY TEST PASSED")
        print(f"   Unauthorized user was correctly denied access")
        print(f"   Only Douglas Shane Davis can access this system")
    else:
        print(f"\n❌ SECURITY BREACH - This should not happen!")


async def main():
    """Main demonstration program"""
    print("\n" + "="*100)
    print("🔐 NEXUS AGI BCI - AUTHORIZED ACCESS DEMONSTRATION 🔐")
    print("="*100)
    print("\nDouglas Shane Davis - Exclusive Access System")
    print("\nPatented Technologies:")
    print("  • US Patent 3951134: Remote brain wave monitoring apparatus (Biometric EEG)")
    print("  • US Patent 6011991: Brain wave communication system")
    print("  • US Patent 5507291: Emotional state determination")
    print("\n" + "="*100)

    # Initialize systems
    print("\n🚀 Initializing systems...")

    # Create orchestrator
    orchestrator = UniversalBCIOrchestrator()

    # Register BCI devices
    print(f"📝 Registering BCI devices...")
    orchestrator.register_device("neurosky_001", NeuroSkyAdapter("neurosky_001"))
    orchestrator.register_device("neuralink_001", NeuralinkAdapter("neuralink_001"))
    orchestrator.register_device("openbci_001", OpenBCIAdapter("openbci_001"))

    # Create authorization system
    print(f"🔐 Initializing authorization system...")
    auth_system = NexusAGIAuthorizationSystem()

    # Start session
    await orchestrator.start_session()

    # Connect device
    print(f"\n🔌 Connecting to BCI device...")
    await orchestrator.connect_device("neurosky_001")

    try:
        # Demo 1: Enrollment
        enrollment_success = await demo_enrollment(auth_system, orchestrator)

        if enrollment_success:
            # Demo 2: Authentication
            session_token = await demo_authentication(auth_system, orchestrator)

            if session_token:
                # Demo 3: Brain wave communication
                await demo_brain_wave_communication(auth_system, orchestrator, session_token)

                # Demo 4: Emotional monitoring
                await demo_emotional_monitoring(auth_system, orchestrator, session_token)

                # Demo 5: Continuous monitoring
                await demo_continuous_monitoring(auth_system, orchestrator, session_token)

                # Revoke session
                print(f"\n🔒 Ending session...")
                auth_system.revoke_session(session_token, "demo_complete")

        # Demo 6: Security testing
        await demo_unauthorized_access_attempt(auth_system, orchestrator)

        # Final status
        print("\n" + "="*100)
        print("SYSTEM STATUS SUMMARY")
        print("="*100)
        auth_system.print_status()

        # Security audit
        print("\n" + "="*100)
        print("SECURITY AUDIT LOG (Last 10 events)")
        print("="*100)
        audit_logs = auth_system.get_security_audit(limit=10)
        for i, log in enumerate(audit_logs, 1):
            print(f"\n{i}. {log['event'].upper()}")
            print(f"   Timestamp: {log['timestamp']}")
            print(f"   User: {log['user_id']}")
            print(f"   Success: {'✅' if log['success'] else '❌'}")
            print(f"   Details: {log['details']}")

        print("\n" + "="*100)
        print("✅ ALL DEMONSTRATIONS COMPLETE")
        print("="*100)
        print(f"\n🔐 System Security Status: OPERATIONAL")
        print(f"👤 Authorized User: Douglas Shane Davis")
        print(f"🛡️  Protection Level: MAXIMUM")
        print(f"📊 Patent Technologies: ACTIVE")
        print("\n" + "="*100)

    finally:
        # Cleanup
        print(f"\n🛑 Shutting down systems...")
        await orchestrator.end_session()

    print(f"\n✅ Demonstration complete!\n")


if __name__ == "__main__":
    print("\n🚀 Starting Authorized BCI System Demo...\n")
    asyncio.run(main())
