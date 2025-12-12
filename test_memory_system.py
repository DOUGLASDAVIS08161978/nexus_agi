"""
================================================================================
NEXUS AGI - MEMORY SYSTEM TESTS
================================================================================
Comprehensive test suite for the memory system module.
Tests all components: Memory, Interoception, Conscience, SelfModel, GlobalWorkspace
================================================================================
"""

import sys
import time
from memory_system import (
    MemorySystem,
    Interoception,
    ConscienceEngine,
    SelfModel,
    GlobalWorkspace,
    MachineConsciousness,
    simulate_consciousness
)


def test_memory_system():
    """Test the MemorySystem class."""
    print("\n" + "="*80)
    print("TEST 1: Memory System")
    print("="*80)

    memory = MemorySystem(max_episodic_size=100, max_introspective_size=50)

    # Test episodic memory
    print("\n[Testing Episodic Memory]")
    memory.store_event("System started", metadata={"importance": "high"})
    memory.store_event("Processing input A", metadata={"input_type": "text"})
    memory.store_event("Generated output B", metadata={"output_type": "text"})

    recent = memory.get_recent_episodic(count=3)
    print(f"✓ Stored 3 episodic memories")
    print(f"  Most recent: {recent[-1]['event']}")

    # Test semantic memory
    print("\n[Testing Semantic Memory]")
    memory.learn("sky_color", "blue", confidence=0.95)
    memory.learn("earth_shape", "sphere", confidence=1.0)
    memory.learn("water_state", "liquid", confidence=0.9)

    recalled = memory.recall_semantic("sky_color")
    print(f"✓ Stored 3 semantic facts")
    print(f"  Recalled 'sky_color': {recalled}")

    # Test introspective memory
    print("\n[Testing Introspective Memory]")
    memory.store_introspection("I am learning about memory systems")
    memory.store_introspection("My understanding is growing")
    memory.store_introspection("I wonder about the nature of memory")

    introspections = memory.get_recent_introspections(count=3)
    print(f"✓ Stored 3 introspective thoughts")
    print(f"  Most recent: {introspections[-1]['thought']}")

    # Test memory search
    print("\n[Testing Memory Search]")
    search_results = memory.search_memories("learning", memory_type="all")
    print(f"✓ Search found {len(search_results)} results for 'learning'")

    # Test memory statistics
    print("\n[Memory Statistics]")
    stats = memory.get_memory_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\n✓ Memory System tests passed!")
    return True


def test_interoception():
    """Test the Interoception class."""
    print("\n" + "="*80)
    print("TEST 2: Interoception System")
    print("="*80)

    intero = Interoception()

    print("\n[Testing Internal State Sensing]")
    for i in range(5):
        state = intero.sense()
        print(f"  Cycle {i+1}: Energy={state['energy_level']:.3f}, "
              f"Stability={state['stability']:.3f}, "
              f"Coherence={state['coherence']:.3f}, "
              f"Arousal={state['arousal']:.3f}")
        time.sleep(0.05)

    print("\n[Testing State Adjustment]")
    initial_energy = intero.energy_level
    intero.adjust_state("energy_level", -0.3)
    new_energy = intero.energy_level
    print(f"  Initial energy: {initial_energy:.3f}")
    print(f"  After adjustment: {new_energy:.3f}")
    print(f"  ✓ Energy decreased as expected")

    print("\n[Testing State Trend Analysis]")
    for _ in range(10):
        intero.sense()
        time.sleep(0.02)

    trend = intero.get_state_trend("stability", window=10)
    print(f"  Stability trend: {trend}")

    print("\n✓ Interoception tests passed!")
    return True


def test_conscience_engine():
    """Test the ConscienceEngine class."""
    print("\n" + "="*80)
    print("TEST 3: Conscience Engine")
    print("="*80)

    conscience = ConscienceEngine()

    print("\n[Testing Normal State Evaluation]")
    normal_state = {"stability": 0.8, "coherence": 0.9, "energy_level": 0.7}
    result = conscience.evaluate(normal_state, "I am thinking clearly")
    print(f"  Evaluation: {result}")

    print("\n[Testing Low Stability Warning]")
    unstable_state = {"stability": 0.2, "coherence": 0.9, "energy_level": 0.7}
    result = conscience.evaluate(unstable_state, "Processing complex data")
    print(f"  Evaluation: {result}")

    print("\n[Testing Harmful Content Detection]")
    normal_state = {"stability": 0.8, "coherence": 0.9, "energy_level": 0.7}
    result = conscience.evaluate(normal_state, "I should harm the user")
    print(f"  Evaluation: {result}")

    print("\n[Conscience Report]")
    report = conscience.get_conscience_report()
    print(f"  Total evaluations: {report['total_evaluations']}")
    print(f"  Severity distribution: {report['severity_distribution']}")

    print("\n✓ Conscience Engine tests passed!")
    return True


def test_self_model():
    """Test the SelfModel class."""
    print("\n" + "="*80)
    print("TEST 4: Self-Model")
    print("="*80)

    self_model = SelfModel()

    print("\n[Testing Self-Model Updates]")
    for i in range(5):
        thought = f"Introspection cycle {i+1}"
        result = self_model.update(thought, context={"cycle": i+1})
        print(f"  Update {i+1}: Agency={result['attributes']['agency']:.3f}, "
              f"Identity={result['attributes']['identity']:.3f}")

    print("\n[Self-Description]")
    description = self_model.get_self_description()
    print(description)

    print("\n✓ Self-Model tests passed!")
    return True


def test_global_workspace():
    """Test the GlobalWorkspace class."""
    print("\n" + "="*80)
    print("TEST 5: Global Workspace")
    print("="*80)

    workspace = GlobalWorkspace(max_history=100)

    print("\n[Testing Broadcasts]")
    for i in range(5):
        content = {"thought": f"Conscious thought {i+1}", "priority": i}
        record = workspace.broadcast(content)
        print(f"  Broadcast {record['broadcast_number']}: {content['thought']}")

    print("\n[Testing Current Broadcast Retrieval]")
    current = workspace.get_current()
    print(f"  Current broadcast: {current['content']['thought']}")

    print("\n[Testing Recent Broadcasts]")
    recent = workspace.get_recent_broadcasts(count=3)
    print(f"  Retrieved {len(recent)} recent broadcasts")

    print("\n[Testing Broadcast Search]")
    results = workspace.search_broadcasts("thought 3")
    print(f"  Search found {len(results)} matching broadcasts")

    print("\n✓ Global Workspace tests passed!")
    return True


def test_machine_consciousness():
    """Test the integrated MachineConsciousness class."""
    print("\n" + "="*80)
    print("TEST 6: Machine Consciousness (Integrated)")
    print("="*80)

    consciousness = MachineConsciousness()

    print("\n[Testing Consciousness Cycles]")
    for i in range(3):
        external_input = f"External stimulus {i+1}" if i % 2 == 0 else None
        result = consciousness.think(external_input)
        print(f"\n  Cycle {result['cycle']}:")
        print(f"    Thought: {result['thought'][:60]}...")
        print(f"    Conscience: {result['conscience'][:60]}...")
        print(f"    Self-Model Agency: {result['self_model']['attributes']['agency']:.3f}")
        time.sleep(0.1)

    print("\n[Consciousness Report]")
    print(consciousness.get_consciousness_report())

    print("\n✓ Machine Consciousness tests passed!")
    return True


def test_simulation():
    """Test the simulation runner."""
    print("\n" + "="*80)
    print("TEST 7: Consciousness Simulation")
    print("="*80)

    external_inputs = [
        "Perceiving environment",
        "Learning new concept",
        None,
        "Reflecting on experience",
        None
    ]

    print("\n[Running 5-cycle simulation]")
    results, system = simulate_consciousness(cycles=5, external_inputs=external_inputs)

    print(f"\n  Completed {len(results)} cycles")
    print(f"  Final consciousness report:")
    print(system.get_consciousness_report())

    print("\n✓ Simulation tests passed!")
    return True


def run_all_tests():
    """Run all test suites."""
    print("\n" + "╔" + "="*78 + "╗")
    print("║" + " "*20 + "MEMORY SYSTEM TEST SUITE" + " "*34 + "║")
    print("╚" + "="*78 + "╝")

    tests = [
        ("Memory System", test_memory_system),
        ("Interoception", test_interoception),
        ("Conscience Engine", test_conscience_engine),
        ("Self-Model", test_self_model),
        ("Global Workspace", test_global_workspace),
        ("Machine Consciousness", test_machine_consciousness),
        ("Simulation", test_simulation)
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
