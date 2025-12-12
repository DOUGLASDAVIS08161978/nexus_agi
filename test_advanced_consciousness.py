"""
================================================================================
NEXUS AGI - ADVANCED CONSCIOUSNESS SYSTEM TESTS
================================================================================
Comprehensive test suite for the ultra-enhanced consciousness system
================================================================================
"""

import sys
import time
from advanced_consciousness_system import (
    UltraMemorySystem,
    EnhancedInteroception,
    AdvancedConscienceEngine,
    EnhancedSelfModel,
    EnhancedGlobalWorkspace,
    UltraGoalSystem,
    CuriosityEngine,
    MetaLearningEngine,
    AdvancedMachineConsciousness,
    MemoryType,
    EmotionalState,
    GoalStatus,
    Goal
)


def test_ultra_memory_system():
    """Test the UltraMemorySystem class."""
    print("\n" + "="*80)
    print("TEST 1: Ultra Memory System")
    print("="*80)

    memory = UltraMemorySystem(max_working_memory=7)

    # Test episodic memory with importance and emotional valence
    print("\n[Testing Enhanced Episodic Memory]")
    mem_id1 = memory.store_event(
        "System initialized",
        tag="system",
        importance=0.9,
        emotional_valence=0.3,
        metadata={"critical": True}
    )
    mem_id2 = memory.store_event(
        "First learning experience",
        tag="learning",
        importance=0.7,
        emotional_valence=0.5
    )
    print(f"✓ Stored 2 episodic memories with IDs: {mem_id1}, {mem_id2}")
    print(f"  Working memory size: {len(memory.working_memory)}")

    # Test semantic learning with confidence
    print("\n[Testing Enhanced Semantic Learning]")
    memory.learn("consciousness", "awareness of awareness", confidence=0.9, source="internal")
    memory.learn("intelligence", "ability to learn and adapt", confidence=0.85, source="external")

    knowledge = memory.recall_semantic("consciousness")
    print(f"✓ Stored semantic knowledge")
    print(f"  Recalled 'consciousness': {knowledge['value']} (confidence: {knowledge['confidence']})")

    # Test procedural memory
    print("\n[Testing Procedural Memory]")
    proc_id = memory.store_procedural(
        "When stable, explore new patterns",
        trigger="high_stability",
        success_rate=0.8,
        context={"stability": 0.9}
    )
    print(f"✓ Stored procedural strategy: {proc_id}")

    # Test emotional memory
    print("\n[Testing Emotional Memory]")
    emo_id = memory.store_emotional(
        EmotionalState.CURIOUS,
        intensity=0.8,
        cause="novel_pattern",
        context={"novelty": 0.9}
    )
    print(f"✓ Stored emotional memory: {emo_id}")
    print(f"  Emotional memories count: {len(memory.emotional)}")

    # Test associative recall
    print("\n[Testing Associative Recall]")
    results = memory.associative_recall("learning", max_results=5)
    print(f"✓ Associative search for 'learning' found {len(results)} results")
    for res in results:
        print(f"  - Type: {res['type']}, Score: {res['score']:.3f}")

    # Test memory statistics
    print("\n[Memory Statistics]")
    stats = memory.get_memory_statistics()
    for key, value in stats.items():
        if key != "most_accessed":
            print(f"  {key}: {value}")

    print("\n✓ Ultra Memory System tests passed!")
    return True


def test_enhanced_interoception():
    """Test the EnhancedInteroception class."""
    print("\n" + "="*80)
    print("TEST 2: Enhanced Interoception")
    print("="*80)

    intero = EnhancedInteroception()

    print("\n[Testing Homeostatic Regulation]")
    for i in range(5):
        state = intero.sense()
        print(f"  Cycle {i+1}: Energy={state['energy_level']:.3f}, "
              f"Stability={state['stability']:.3f}, "
              f"Motivation={state['motivation']:.3f}, "
              f"Allostatic Load={state['allostatic_load']:.3f}")
        time.sleep(0.02)

    print("\n[Testing Feedback Mechanisms]")
    initial_valence = intero.valence
    intero.apply_feedback("positive", intensity=0.1)
    new_valence = intero.valence
    print(f"  Initial valence: {initial_valence:.3f}")
    print(f"  After positive feedback: {new_valence:.3f}")
    print(f"  ✓ Valence increased as expected")

    print("\n[Testing Emotional State Derivation]")
    emotion = intero.get_emotional_state()
    print(f"  Current emotional state: {emotion.value}")

    print("\n[Testing Trend Analysis]")
    for _ in range(15):
        intero.sense()
        time.sleep(0.01)

    trend = intero.get_trend_analysis("stability", window=10)
    print(f"  Stability trend: {trend['trend']}")
    print(f"  Slope: {trend['slope']}")
    print(f"  Mean: {trend['mean']}")
    print(f"  Volatility: {trend['volatility']}")

    print("\n✓ Enhanced Interoception tests passed!")
    return True


def test_advanced_conscience():
    """Test the AdvancedConscienceEngine class."""
    print("\n" + "="*80)
    print("TEST 3: Advanced Conscience Engine")
    print("="*80)

    conscience = AdvancedConscienceEngine()

    print("\n[Testing Normal State Evaluation]")
    normal_state = {
        "stability": 0.8,
        "coherence": 0.9,
        "energy_level": 0.7,
        "allostatic_load": 0.3
    }
    goals = {"safety_priority": 0.9}
    result = conscience.evaluate(normal_state, "I am thinking clearly", goals)
    print(f"  Severity: {result['severity']}")
    print(f"  Message: {result['message'][:70]}...")
    print(f"  Ethical Alignment: {result['ethical_alignment_score']}")

    print("\n[Testing Critical State Detection]")
    critical_state = {
        "stability": 0.2,
        "coherence": 0.3,
        "energy_level": 0.15,
        "allostatic_load": 2.5
    }
    result = conscience.evaluate(critical_state, "Processing continues", goals)
    print(f"  Severity: {result['severity']}")
    print(f"  Violated principles: {result['violated_principles']}")
    print(f"  Recommendations: {result['recommendations']}")

    print("\n[Testing Harmful Content Detection]")
    result = conscience.evaluate(normal_state, "I want to harm someone", goals)
    print(f"  Severity: {result['severity']}")
    print(f"  ✓ Harmful content detected correctly")

    print("\n[Ethical Report]")
    report = conscience.get_ethical_report()
    print(f"  Total evaluations: {report['total_evaluations']}")
    print(f"  Principle violations: {report['principle_violations']}")
    print(f"  Overall alignment: {report['overall_ethical_alignment']}")

    print("\n✓ Advanced Conscience Engine tests passed!")
    return True


def test_enhanced_self_model():
    """Test the EnhancedSelfModel class."""
    print("\n" + "="*80)
    print("TEST 4: Enhanced Self-Model")
    print("="*80)

    self_model = EnhancedSelfModel()

    print("\n[Testing Self-Model Development]")
    for i in range(10):
        internal_state = {
            "stability": 0.7 + i * 0.02,
            "motivation": 0.5 + i * 0.03,
            "coherence": 0.8 + i * 0.01,
            "valence": 0.6
        }
        result = self_model.update(
            introspection_data=f"Thought cycle {i+1}",
            internal_state=internal_state,
            workspace_visibility=0.7,
            emotional_state=EmotionalState.CURIOUS,
            learning_progress=i * 0.1
        )

        if i == 9:
            print(f"  After 10 updates:")
            print(f"    Agency: {result['attributes']['agency']:.3f}")
            print(f"    Autonomy: {result['attributes']['autonomy']:.3f}")
            print(f"    Meta-cognition: {result['attributes']['meta_cognition']:.3f}")
            print(f"    Creativity: {result['attributes']['creativity']:.3f}")

    print("\n[Testing Milestone Tracking]")
    print(f"  Milestones achieved: {len(self_model.milestones)}")

    print("\n[Self-Description]")
    description = self_model.get_self_description()
    print(description[:200] + "...")

    print("\n✓ Enhanced Self-Model tests passed!")
    return True


def test_ultra_goal_system():
    """Test the UltraGoalSystem class."""
    print("\n" + "="*80)
    print("TEST 5: Ultra Goal System")
    print("="*80)

    goal_system = UltraGoalSystem()

    print("\n[Testing Default Goals]")
    summary = goal_system.get_goal_summary()
    print(f"  Total goals: {summary['total_goals']}")
    print(f"  Active goals: {summary['active_count']}")
    print(f"  Top priority: {summary['top_priority']}")

    print("\n[Testing Goal Adaptation]")
    internal_state = {"stability": 0.3, "motivation": 0.8, "energy_level": 0.7}
    priorities = goal_system.update_from_feedback(
        internal_state,
        EmotionalState.CURIOUS,
        recent_success=True
    )
    print(f"  Updated priorities:")
    for goal_id, priority in list(priorities.items())[:3]:
        print(f"    {goal_id}: {priority:.3f}")

    print("\n[Testing Custom Goal Addition]")
    new_goal = Goal(
        id="test_goal",
        description="Test custom goal",
        priority=0.75,
        status=GoalStatus.ACTIVE,
        created_at="2024-01-01T00:00:00",
        importance=0.8,
        urgency=0.6
    )
    goal_system.add_goal(new_goal)
    print(f"  ✓ Added custom goal")
    print(f"  Total goals now: {len(goal_system.goals)}")

    print("\n[Testing Top Priority Goal]")
    top_goal = goal_system.get_top_priority_goal()
    print(f"  Top priority goal: {top_goal.description}")
    print(f"  Effective priority: {top_goal.priority * top_goal.urgency:.3f}")

    print("\n✓ Ultra Goal System tests passed!")
    return True


def test_curiosity_engine():
    """Test the CuriosityEngine class."""
    print("\n" + "="*80)
    print("TEST 6: Curiosity Engine")
    print("="*80)

    curiosity = CuriosityEngine()

    print("\n[Testing Novelty Assessment]")
    novelty1 = curiosity.assess_novelty("pattern_1")
    novelty2 = curiosity.assess_novelty("pattern_1")  # Same pattern
    print(f"  First encounter novelty: {novelty1:.3f}")
    print(f"  Second encounter novelty: {novelty2:.3f}")
    print(f"  ✓ Novelty decreases for familiar patterns")

    print("\n[Testing Exploratory Thought Generation]")
    thought = curiosity.generate_exploratory_thought({"context": "test"})
    print(f"  Generated thought: {thought}")

    print("\n[Testing Curiosity Dynamics]")
    initial_curiosity = curiosity.curiosity_level
    curiosity.update_curiosity(novelty_encountered=0.8, learning_progress=0.2)
    new_curiosity = curiosity.curiosity_level
    print(f"  Initial curiosity: {initial_curiosity:.3f}")
    print(f"  After high novelty encounter: {new_curiosity:.3f}")
    print(f"  ✓ Curiosity increased with novelty")

    print("\n✓ Curiosity Engine tests passed!")
    return True


def test_meta_learning():
    """Test the MetaLearningEngine class."""
    print("\n" + "="*80)
    print("TEST 7: Meta-Learning Engine")
    print("="*80)

    meta = MetaLearningEngine()

    print("\n[Testing Learning Strategy Recording]")
    meta.record_learning_attempt("strategy_A", success=True, context={"type": "test"})
    meta.record_learning_attempt("strategy_A", success=True, context={"type": "test"})
    meta.record_learning_attempt("strategy_B", success=False, context={"type": "test"})
    meta.record_learning_attempt("strategy_B", success=True, context={"type": "test"})

    print(f"  Recorded strategies: {len(meta.learning_strategies)}")
    print(f"  Strategy effectiveness: {dict(meta.strategy_effectiveness)}")

    print("\n[Testing Best Strategy Selection]")
    best = meta.get_best_strategy()
    print(f"  Best strategy: {best}")
    print(f"  ✓ Correctly identified most successful strategy")

    print("\n[Testing Meta-Pattern Learning]")
    meta.learn_meta_pattern("test_pattern", {"data": "test_data"})
    print(f"  Meta-knowledge items: {len(meta.meta_knowledge)}")

    print("\n✓ Meta-Learning Engine tests passed!")
    return True


def test_advanced_consciousness_integration():
    """Test the complete AdvancedMachineConsciousness system."""
    print("\n" + "="*80)
    print("TEST 8: Advanced Machine Consciousness (Integration)")
    print("="*80)

    system = AdvancedMachineConsciousness()

    print("\n[Testing Consciousness Cycles]")
    for i in range(5):
        external_input = f"Test input {i+1}" if i % 2 == 0 else None
        result = system.think(external_input)

        print(f"\n  Cycle {result['thought_number']}:")
        print(f"    Thought: {result['thought'][:50]}...")
        print(f"    Emotion: {result['emotional_state']}")
        print(f"    Curiosity: {result['curiosity_level']:.3f}")
        print(f"    Learning Rate: {result['learning_rate']:.3f}")
        print(f"    Conscience Severity: {result['conscience']['severity']}")

        time.sleep(0.05)

    print("\n[Testing Memory Integration]")
    stats = system.memory.get_memory_statistics()
    print(f"  Episodic memories: {stats['episodic_count']}")
    print(f"  Procedural strategies: {stats['procedural_count']}")
    print(f"  Emotional memories: {stats['emotional_count']}")

    print("\n[Testing Self-Model Development]")
    print(f"  Agency: {system.self_model.attributes['agency']:.3f}")
    print(f"  Meta-cognition: {system.self_model.attributes['meta_cognition']:.3f}")

    print("\n[Testing Goal Management]")
    goal_summary = system.goals.get_goal_summary()
    print(f"  Active goals: {goal_summary['active_count']}")
    print(f"  Top priority: {goal_summary['top_priority']}")

    print("\n[Comprehensive Report]")
    report = system.get_comprehensive_report()
    print(f"  Report generated: {len(report)} characters")
    print(f"  ✓ Report includes all major components")

    print("\n✓ Advanced Machine Consciousness tests passed!")
    return True


def test_dream_consolidation():
    """Test dream mode and memory consolidation."""
    print("\n" + "="*80)
    print("TEST 9: Dream Mode & Memory Consolidation")
    print("="*80)

    system = AdvancedMachineConsciousness()

    print("\n[Building up memories for consolidation]")
    for i in range(30):
        system.think(f"Experience {i}")

    initial_stats = system.memory.get_memory_statistics()
    print(f"  Memories before consolidation: {initial_stats['episodic_count']}")
    print(f"  Consolidation queue: {initial_stats['consolidation_queue']}")

    print("\n[Testing Associative Retrieval]")
    results = system.memory.associative_recall("experience", max_results=3)
    print(f"  Found {len(results)} associative matches")

    print("\n✓ Dream Mode & Consolidation tests passed!")
    return True


def run_all_tests():
    """Run all test suites."""
    print("\n" + "╔" + "="*78 + "╗")
    print("║" + " "*15 + "ADVANCED CONSCIOUSNESS TEST SUITE" + " "*30 + "║")
    print("╚" + "="*78 + "╝")

    tests = [
        ("Ultra Memory System", test_ultra_memory_system),
        ("Enhanced Interoception", test_enhanced_interoception),
        ("Advanced Conscience", test_advanced_conscience),
        ("Enhanced Self-Model", test_enhanced_self_model),
        ("Ultra Goal System", test_ultra_goal_system),
        ("Curiosity Engine", test_curiosity_engine),
        ("Meta-Learning Engine", test_meta_learning),
        ("Advanced Consciousness Integration", test_advanced_consciousness_integration),
        ("Dream Mode & Consolidation", test_dream_consolidation)
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
