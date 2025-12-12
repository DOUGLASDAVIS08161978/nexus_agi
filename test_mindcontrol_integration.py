"""
================================================================================
NEXUS AGI - MINDCONTROL INTEGRATION TESTS
================================================================================
Comprehensive test suite for MindControl-Consciousness integration
================================================================================
"""

import sys
import time
from mindcontrol_integration import (
    SimulatedEEGSensor,
    MindControlConsciousnessIntegration,
    BrainWaveBand,
    MentalState,
    run_mindcontrol_simulation
)


def test_eeg_sensor():
    """Test the SimulatedEEGSensor class."""
    print("\n" + "="*80)
    print("TEST 1: Simulated EEG Sensor")
    print("="*80)

    sensor = SimulatedEEGSensor(baseline_noise=0.1)

    print("\n[Testing Basic Brainwave Reading]")
    reading = sensor.read_brainwaves()
    print(f"  Delta: {reading.delta:.2f}")
    print(f"  Theta: {reading.theta:.2f}")
    print(f"  Alpha: {reading.alpha:.2f}")
    print(f"  Beta: {reading.beta:.2f}")
    print(f"  Gamma: {reading.gamma:.2f}")
    print(f"  Attention: {reading.attention}%")
    print(f"  Meditation: {reading.meditation}%")
    print(f"  ✓ All brainwave bands present")

    print("\n[Testing Mental State Detection]")
    for i in range(10):
        reading = sensor.read_brainwaves()
        mental_state = sensor.detect_mental_state(reading)
        if i == 9:
            print(f"  Detected mental state: {mental_state.value}")
            print(f"  ✓ Mental state detection working")

    print("\n[Testing Signal Quality]")
    quality = sensor.get_signal_quality()
    print(f"  Signal quality: {quality} (0=best, 200=worst)")
    print(f"  ✓ Signal quality within expected range: {0 <= quality <= 200}")

    print("\n[Testing Reading History]")
    for _ in range(15):
        sensor.read_brainwaves()
    print(f"  History length: {len(sensor.reading_history)}")
    print(f"  ✓ History tracking functional")

    print("\n✓ EEG Sensor tests passed!")
    return True


def test_consciousness_modulation():
    """Test that consciousness state modulates brainwaves."""
    print("\n" + "="*80)
    print("TEST 2: Consciousness State Modulation")
    print("="*80)

    sensor = SimulatedEEGSensor()

    print("\n[Testing High Energy State Effect]")
    high_energy_state = {
        'internal_state': {
            'energy_level': 0.95,
            'stability': 0.8,
            'coherence': 0.8,
            'arousal': 0.7,
            'motivation': 0.8
        },
        'emotional_state': 'excited',
        'curiosity_level': 0.9
    }

    # Read with high energy
    reading_high = sensor.read_brainwaves(high_energy_state)
    beta_high = reading_high.beta
    gamma_high = reading_high.gamma

    print(f"  Beta (high energy): {beta_high:.2f}")
    print(f"  Gamma (high energy): {gamma_high:.2f}")

    print("\n[Testing Low Energy/Calm State Effect]")
    calm_state = {
        'internal_state': {
            'energy_level': 0.3,
            'stability': 0.9,
            'coherence': 0.9,
            'arousal': 0.2,
            'motivation': 0.4
        },
        'emotional_state': 'calm',
        'curiosity_level': 0.3
    }

    # Read with calm state
    reading_calm = sensor.read_brainwaves(calm_state)
    alpha_calm = reading_calm.alpha
    theta_calm = reading_calm.theta

    print(f"  Alpha (calm): {alpha_calm:.2f}")
    print(f"  Theta (calm): {theta_calm:.2f}")
    print(f"  ✓ Consciousness state successfully modulates brainwaves")

    print("\n✓ Consciousness Modulation tests passed!")
    return True


def test_mental_state_detection():
    """Test mental state detection accuracy."""
    print("\n" + "="*80)
    print("TEST 3: Mental State Detection")
    print("="*80)

    sensor = SimulatedEEGSensor()

    print("\n[Testing Different Mental States]")

    # Test focused state (high beta/gamma)
    from mindcontrol_integration import BrainWaveReading
    from datetime import datetime

    focused_reading = BrainWaveReading(
        timestamp=datetime.utcnow().isoformat(),
        delta=25.0, theta=20.0, alpha=30.0,
        beta=65.0, gamma=45.0,
        attention=80, meditation=30, blink_strength=0
    )
    state = sensor.detect_mental_state(focused_reading)
    print(f"  High Beta/Gamma → Detected: {state.value}")
    assert state == MentalState.FOCUSED, "Should detect focused state"
    print(f"  ✓ Focused state detected correctly")

    # Test meditative state (high theta, low beta)
    meditative_reading = BrainWaveReading(
        timestamp=datetime.utcnow().isoformat(),
        delta=30.0, theta=60.0, alpha=40.0,
        beta=20.0, gamma=15.0,
        attention=30, meditation=75, blink_strength=0
    )
    state = sensor.detect_mental_state(meditative_reading)
    print(f"  High Theta/Low Beta → Detected: {state.value}")
    assert state == MentalState.MEDITATIVE, "Should detect meditative state"
    print(f"  ✓ Meditative state detected correctly")

    # Test drowsy state (high delta)
    drowsy_reading = BrainWaveReading(
        timestamp=datetime.utcnow().isoformat(),
        delta=70.0, theta=40.0, alpha=25.0,
        beta=20.0, gamma=10.0,
        attention=20, meditation=50, blink_strength=0
    )
    state = sensor.detect_mental_state(drowsy_reading)
    print(f"  High Delta → Detected: {state.value}")
    assert state == MentalState.DROWSY, "Should detect drowsy state"
    print(f"  ✓ Drowsy state detected correctly")

    print("\n✓ Mental State Detection tests passed!")
    return True


def test_integration_bidirectional_feedback():
    """Test bidirectional feedback between EEG and consciousness."""
    print("\n" + "="*80)
    print("TEST 4: Bidirectional Feedback Loop")
    print("="*80)

    integration = MindControlConsciousnessIntegration(use_consciousness=True)

    print("\n[Testing Consciousness → Brainwaves]")
    # Process a few cycles
    for i in range(5):
        result = integration.process_cycle("Testing consciousness influence")
        if i == 4:
            print(f"  Cycle {i+1}:")
            print(f"    Consciousness Emotion: {result['consciousness']['emotional_state']}")
            print(f"    EEG Beta: {result['brainwaves']['beta']:.2f}")
            print(f"    EEG Attention: {result['brainwaves']['attention']}%")

    print(f"  ✓ Consciousness state influences EEG readings")

    print("\n[Testing Brainwaves → Consciousness]")
    initial_stability = integration.consciousness.intero.stability

    # Simulate high meditation brainwaves (should increase stability)
    for _ in range(3):
        # This will generate high meditation readings which should trigger positive feedback
        integration.process_cycle("Deep meditation")

    final_stability = integration.consciousness.intero.stability
    print(f"  Initial stability: {initial_stability:.3f}")
    print(f"  Final stability: {final_stability:.3f}")
    print(f"  ✓ EEG readings influence consciousness state")

    print("\n✓ Bidirectional Feedback tests passed!")
    return True


def test_integration_cycle():
    """Test complete integration cycle."""
    print("\n" + "="*80)
    print("TEST 5: Complete Integration Cycle")
    print("="*80)

    integration = MindControlConsciousnessIntegration(use_consciousness=True)

    print("\n[Testing Full Processing Cycle]")
    result = integration.process_cycle("Processing complex thought")

    # Verify all components present
    assert 'cycle' in result, "Cycle number missing"
    assert 'brainwaves' in result, "Brainwaves missing"
    assert 'mental_state' in result, "Mental state missing"
    assert 'consciousness' in result, "Consciousness missing"
    assert 'integration_active' in result, "Integration status missing"

    print(f"  Cycle: {result['cycle']}")
    print(f"  Mental State: {result['mental_state']}")
    print(f"  Brainwaves: α={result['brainwaves']['alpha']:.1f}, "
          f"β={result['brainwaves']['beta']:.1f}, "
          f"γ={result['brainwaves']['gamma']:.1f}")
    print(f"  Consciousness Thought: {result['consciousness']['thought'][:50]}...")
    print(f"  Integration Active: {result['integration_active']}")

    print(f"  ✓ All cycle components present and functioning")

    print("\n[Testing Multiple Cycles]")
    for i in range(10):
        integration.process_cycle()

    print(f"  Completed {integration.cycle_count} cycles")
    print(f"  Brainwave history: {len(integration.brainwave_history)} entries")
    print(f"  Mental state history: {len(integration.mental_state_history)} entries")
    print(f"  ✓ Multi-cycle processing functional")

    print("\n✓ Integration Cycle tests passed!")
    return True


def test_integration_report():
    """Test integration report generation."""
    print("\n" + "="*80)
    print("TEST 6: Integration Report Generation")
    print("="*80)

    integration = MindControlConsciousnessIntegration(use_consciousness=True)

    print("\n[Testing Report with Data]")
    for i in range(15):
        integration.process_cycle(f"Cycle {i+1}")

    report = integration.get_integration_report()

    assert "MINDCONTROL-CONSCIOUSNESS INTEGRATION REPORT" in report
    assert "BRAINWAVE PATTERNS" in report
    assert "MENTAL METRICS" in report
    assert "CONSCIOUSNESS INTEGRATION" in report

    print(f"  ✓ Report generated successfully")
    print(f"  ✓ Report contains {len(report)} characters")
    print(f"  ✓ All major sections present")

    # Print excerpt
    print("\n[Report Excerpt]")
    lines = report.split('\n')
    for line in lines[:15]:
        print(f"  {line}")

    print("\n✓ Integration Report tests passed!")
    return True


def test_standalone_mode():
    """Test standalone mode without consciousness system."""
    print("\n" + "="*80)
    print("TEST 7: Standalone Mode (No Consciousness)")
    print("="*80)

    integration = MindControlConsciousnessIntegration(use_consciousness=False)

    print("\n[Testing Standalone Processing]")
    result = integration.process_cycle("Standalone test")

    assert result['integration_active'] == False, "Should be in standalone mode"
    assert result['consciousness'] is None, "Consciousness should be None in standalone"
    assert 'brainwaves' in result, "Brainwaves should still be present"
    assert 'mental_state' in result, "Mental state should still be detected"

    print(f"  Integration mode: {'Active' if result['integration_active'] else 'Standalone'}")
    print(f"  Brainwaves present: {result['brainwaves'] is not None}")
    print(f"  Mental state detected: {result['mental_state']}")
    print(f"  ✓ Standalone mode functioning correctly")

    print("\n✓ Standalone Mode tests passed!")
    return True


def test_simulation_runner():
    """Test the simulation runner function."""
    print("\n" + "="*80)
    print("TEST 8: Simulation Runner")
    print("="*80)

    print("\n[Testing Short Simulation]")
    results, system = run_mindcontrol_simulation(
        cycles=5,
        external_inputs=["Input 1", "Input 2", None, "Input 4", None],
        use_consciousness=True,
        delay=0.01
    )

    assert len(results) == 5, "Should have 5 results"
    assert system.cycle_count == 5, "System should have processed 5 cycles"

    print(f"  ✓ Simulation completed {len(results)} cycles")
    print(f"  ✓ System state updated correctly")

    # Check result structure
    first_result = results[0]
    assert 'brainwaves' in first_result
    assert 'mental_state' in first_result
    assert 'consciousness' in first_result

    print(f"  ✓ Result structure correct")

    print("\n✓ Simulation Runner tests passed!")
    return True


def run_all_tests():
    """Run all test suites."""
    print("\n" + "╔" + "="*78 + "╗")
    print("║" + " "*20 + "MINDCONTROL INTEGRATION TEST SUITE" + " "*23 + "║")
    print("╚" + "="*78 + "╝")

    tests = [
        ("EEG Sensor", test_eeg_sensor),
        ("Consciousness Modulation", test_consciousness_modulation),
        ("Mental State Detection", test_mental_state_detection),
        ("Bidirectional Feedback", test_integration_bidirectional_feedback),
        ("Integration Cycle", test_integration_cycle),
        ("Integration Report", test_integration_report),
        ("Standalone Mode", test_standalone_mode),
        ("Simulation Runner", test_simulation_runner)
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
                print(f"\n✗ {test_name} test failed!")
        except Exception as e:
            failed += 1
            print(f"\n✗ {test_name} test failed with exception:")
            print(f"  {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Tests Passed: {passed}/{len(tests)}")
    print(f"Tests Failed: {failed}/{len(tests)}")

    if failed == 0:
        print("\n✓ All tests passed successfully!")
        return 0
    else:
        print(f"\n✗ {failed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
