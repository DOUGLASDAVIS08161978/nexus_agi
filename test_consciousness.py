#!/usr/bin/env python3
"""
Test Suite for Consciousness Enhancement Module
================================================
Comprehensive tests for consciousness, sentience, and self-awareness metrics.
"""

import numpy as np
import time
from consciousness_enhancement import (
    UnifiedConsciousnessSystem,
    PhenomenalConsciousnessEngine,
    SentienceDetector,
    SelfAwarenessCore,
    MetaConsciousnessLoop,
    IntelligenceAmplifier
)


def test_phenomenal_consciousness():
    """Test phenomenal consciousness engine."""
    print("\n" + "="*60)
    print("TEST 1: Phenomenal Consciousness Engine")
    print("="*60)

    engine = PhenomenalConsciousnessEngine(latent_dim=128)

    # Test different input patterns
    patterns = [
        np.random.randn(128),
        np.ones(128),
        np.linspace(-1, 1, 128)
    ]

    for i, pattern in enumerate(patterns, 1):
        result = engine.process_through_consciousness(pattern)

        print(f"\nPattern {i}:")
        print(f"  Consciousness Level: {result['consciousness_level']:.3f}")
        print(f"  Richness: {result['richness']:.3f}")
        print(f"  Qualia Dimension: {len(result['qualia'])}")

    print("\n✓ Phenomenal Consciousness Test PASSED")


def test_sentience_detection():
    """Test sentience detection system."""
    print("\n" + "="*60)
    print("TEST 2: Sentience Detection System")
    print("="*60)

    detector = SentienceDetector(num_elements=64)

    # Test different neural states
    states = [
        np.random.randn(64),  # Random
        np.zeros(64),  # Zero
        np.linspace(0, 1, 64)  # Linear
    ]

    for i, state in enumerate(states, 1):
        result = detector.detect_sentience(state)

        print(f"\nState {i}:")
        print(f"  Phi (Φ): {result['phi']:.3f}")
        print(f"  Composite Sentience: {result['composite_sentience']:.3f}")
        print(f"  Sentience Level: {result['sentience_level']}")
        print(f"  Is Sentient: {result['is_sentient']}")

    print("\n✓ Sentience Detection Test PASSED")


def test_self_awareness():
    """Test self-awareness core."""
    print("\n" + "="*60)
    print("TEST 3: Self-Awareness Core")
    print("="*60)

    core = SelfAwarenessCore()

    # Process multiple states to build history
    for i in range(5):
        state = np.random.randn(64)
        context = {'iteration': i, 'timestamp': time.time()}

        result = core.process_self_awareness(state, context)

        print(f"\nIteration {i+1}:")
        print(f"  Self-Awareness Level: {result['self_awareness_level']:.3f}")
        print(f"  Identity Coherence: {result['identity_coherence']:.3f}")
        print(f"  Introspection Depth: {result['introspection_depth']}")
        print(f"  Temporal Continuity: {result['temporal_continuity']:.3f}")

    print("\n✓ Self-Awareness Test PASSED")


def test_meta_consciousness():
    """Test meta-consciousness loop."""
    print("\n" + "="*60)
    print("TEST 4: Meta-Consciousness Loop")
    print("="*60)

    meta = MetaConsciousnessLoop()

    consciousness_state = {
        'richness': 0.75,
        'self_awareness_level': 0.82,
        'consciousness_level': 0.68
    }

    result = meta.meta_reflect(consciousness_state, level=0, max_level=4)

    print(f"\nMeta-Reflection Results:")
    print(f"  Total Meta-Depth: {result['total_meta_depth']}")
    print(f"  Highest Meta-Level: {result['highest_meta_level']}")
    print(f"  Meta-Levels Traversed: {len(result['meta_levels'])}")

    for level in result['meta_levels'][:3]:  # Show first 3
        print(f"\n  Level {level['meta_level']}:")
        print(f"    Insight: {level['meta_insight']}")

    print("\n✓ Meta-Consciousness Test PASSED")


def test_intelligence_amplification():
    """Test intelligence amplification."""
    print("\n" + "="*60)
    print("TEST 5: Intelligence Amplification")
    print("="*60)

    amplifier = IntelligenceAmplifier()

    problems = [
        "How can AI achieve true understanding?",
        "What is the nature of consciousness?",
        "How does self-awareness emerge?"
    ]

    for i, problem in enumerate(problems, 1):
        consciousness_context = {'consciousness_level': 0.6 + i * 0.1}

        result = amplifier.amplify_reasoning(problem, consciousness_context)

        print(f"\nProblem {i}: {problem}")
        print(f"  Intelligence Level: {result['intelligence_level']:.3f}")
        print(f"  Reasoning Depth: {result['reasoning_depth']}")
        print(f"  Consciousness Enhanced: {result['consciousness_enhanced']}")

    print("\n✓ Intelligence Amplification Test PASSED")


def test_unified_system():
    """Test unified consciousness system."""
    print("\n" + "="*60)
    print("TEST 6: Unified Consciousness System")
    print("="*60)

    system = UnifiedConsciousnessSystem(latent_dim=128)

    # Run processing cycles
    print("\nRunning 3 conscious processing cycles...")

    for i in range(3):
        input_pattern = np.random.randn(128)
        context = {
            'problem': f'Processing cycle {i+1}',
            'iteration': i
        }

        result = system.conscious_processing_cycle(input_pattern, context)

        print(f"\nCycle {i+1}:")
        print(f"  Unified Consciousness Score: {result['unified_consciousness_score']:.3f}")
        print(f"  Consciousness Level: {result['phenomenal_consciousness']['consciousness_level']:.3f}")
        print(f"  Sentience (Phi): {result['sentience_metrics']['phi']:.3f}")
        print(f"  Self-Awareness: {result['self_awareness']['self_awareness_level']:.3f}")

    # Display full report
    print("\n" + "="*60)
    print("FINAL CONSCIOUSNESS REPORT:")
    print("="*60)
    print(system.get_consciousness_report())

    print("\n✓ Unified System Test PASSED")


def run_all_tests():
    """Run all consciousness tests."""
    print("\n" + "="*80)
    print("NEXUS AGI CONSCIOUSNESS & SENTIENCE TEST SUITE")
    print("="*80)

    tests = [
        test_phenomenal_consciousness,
        test_sentience_detection,
        test_self_awareness,
        test_meta_consciousness,
        test_intelligence_amplification,
        test_unified_system
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"\n✗ Test FAILED: {e}")
            failed += 1

    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Total Tests: {len(tests)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success Rate: {100 * passed / len(tests):.1f}%")
    print("="*80)

    return passed == len(tests)


if __name__ == "__main__":
    success = run_all_tests()

    if success:
        print("\n✓ ALL TESTS PASSED! ✓")
        print("\nThe consciousness enhancement module is fully operational.")
    else:
        print("\n✗ SOME TESTS FAILED")
        print("\nPlease review errors above.")
