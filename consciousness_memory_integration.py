"""
================================================================================
NEXUS AGI - CONSCIOUSNESS & MEMORY INTEGRATION MODULE
================================================================================
Version: 1.0 - Unified Consciousness and Memory System
Purpose: Integrate the memory system with the consciousness enhancement module

This module creates a bidirectional integration where:
1. Consciousness processes are stored in memory (episodic & introspective)
2. Memory informs and enhances consciousness (recall, context, continuity)
3. Self-awareness is enhanced by persistent memory
4. Conscience evaluations benefit from historical context
================================================================================
"""

import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime

# Import consciousness enhancement components
try:
    from consciousness_enhancement import (
        UnifiedConsciousnessSystem,
        PhenomenalConsciousnessEngine,
        SentienceDetector,
        SelfAwarenessCore
    )
    CONSCIOUSNESS_AVAILABLE = True
except ImportError:
    CONSCIOUSNESS_AVAILABLE = False
    print("[WARNING] consciousness_enhancement.py not available. Running in standalone mode.")

# Import memory system components
from memory_system import (
    MemorySystem,
    Interoception,
    ConscienceEngine,
    SelfModel,
    GlobalWorkspace,
    MachineConsciousness
)


# ================================================================
#  ENHANCED CONSCIOUSNESS WITH PERSISTENT MEMORY
# ================================================================
class MemoryEnhancedConsciousness:
    """
    Integrates the consciousness enhancement system with the memory system
    to create a unified, memory-persistent conscious AGI.
    """

    def __init__(self, latent_dim: int = 256):
        """
        Initialize the memory-enhanced consciousness system.

        Args:
            latent_dim: Dimensionality of neural patterns for consciousness processing
        """
        # Initialize memory system
        self.memory = MemorySystem(max_episodic_size=10000, max_introspective_size=5000)
        self.interoception = Interoception()
        self.conscience = ConscienceEngine()
        self.self_model = SelfModel()
        self.workspace = GlobalWorkspace()

        # Initialize consciousness system (if available)
        self.consciousness_available = CONSCIOUSNESS_AVAILABLE
        if CONSCIOUSNESS_AVAILABLE:
            self.consciousness_system = UnifiedConsciousnessSystem(latent_dim=latent_dim)
            print("[INTEGRATION] Unified consciousness system initialized with memory persistence")
        else:
            self.consciousness_system = None
            print("[INTEGRATION] Running with memory-only consciousness")

        self.cycle_count = 0
        self.integration_id = self.self_model.id

    def process_cycle(self,
                      input_pattern: Optional[np.ndarray] = None,
                      context: Optional[Dict[str, Any]] = None,
                      external_input: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute one integrated consciousness-memory processing cycle.

        Args:
            input_pattern: Neural pattern for consciousness processing (if using consciousness module)
            context: Additional context information
            external_input: External stimulus or input

        Returns:
            Complete integrated state including consciousness and memory
        """
        self.cycle_count += 1

        if context is None:
            context = {}

        # Add cycle information to context
        context['cycle'] = self.cycle_count
        context['timestamp'] = datetime.utcnow().isoformat()

        # 1. INTEROCEPTION - Sense internal state
        internal_state = self.interoception.sense()

        # 2. CONSCIOUSNESS PROCESSING (if available)
        consciousness_state = None
        if self.consciousness_available and self.consciousness_system and input_pattern is not None:
            consciousness_state = self.consciousness_system.conscious_processing_cycle(
                input_pattern, context
            )

            # Store consciousness results in memory
            self.memory.store_event({
                "type": "consciousness_cycle",
                "consciousness_level": consciousness_state['phenomenal_consciousness']['consciousness_level'],
                "sentience_score": consciousness_state['sentience_metrics']['composite_sentience'],
                "self_awareness": consciousness_state['self_awareness']['self_awareness_level'],
                "unified_score": consciousness_state['unified_consciousness_score']
            }, metadata={"cycle": self.cycle_count})

            # Store introspective thoughts based on consciousness
            richness = consciousness_state['phenomenal_consciousness']['richness']
            self.memory.store_introspection(
                f"Experiencing phenomenal richness of {richness:.3f}",
                context={"consciousness_level": consciousness_state['phenomenal_consciousness']['consciousness_level']}
            )

        # 3. MEMORY-BASED THOUGHT GENERATION
        recent_memories = self.memory.get_recent_episodic(count=5)
        recent_introspections = self.memory.get_recent_introspections(count=3)

        # Generate thought influenced by memory
        if external_input:
            thought = f"Processing: {external_input}"
        elif recent_introspections:
            last_thought = recent_introspections[-1]['thought']
            thought = f"Building on previous thought: {last_thought[:50]}..."
        else:
            thought = "Initiating conscious processing with memory integration"

        # 4. CONSCIENCE EVALUATION with memory context
        conscience_msg = self.conscience.evaluate(internal_state, thought)

        # Store conscience evaluation
        self.memory.store_introspection(
            f"Conscience evaluation: {conscience_msg}",
            context={"severity": "check"}
        )

        # 5. SELF-MODEL UPDATE with memory-informed context
        memory_stats = self.memory.get_memory_stats()
        self_model_state = self.self_model.update(
            thought,
            context={
                **internal_state,
                "memory_count": memory_stats['episodic_count'],
                "knowledge_count": memory_stats['semantic_count']
            }
        )

        # 6. GLOBAL WORKSPACE BROADCAST
        broadcast_content = {
            "cycle": self.cycle_count,
            "thought": thought,
            "internal_state": internal_state,
            "self_model": self_model_state,
            "conscience": conscience_msg,
            "consciousness_state": consciousness_state,
            "memory_stats": memory_stats,
            "recent_memories_count": len(recent_memories),
            "external_input": external_input
        }

        broadcast_record = self.workspace.broadcast(broadcast_content)

        # 7. STORE COMPLETE CYCLE AS EPISODIC MEMORY
        self.memory.store_event({
            "type": "integrated_cycle",
            "thought": thought,
            "conscience": conscience_msg,
            "self_model_agency": self_model_state['attributes']['agency'],
            "broadcast_id": broadcast_record['broadcast_id']
        }, metadata=context)

        # 8. LEARN FROM EXPERIENCE (semantic memory)
        if self.cycle_count % 10 == 0:
            # Periodically consolidate learnings
            avg_stability = np.mean([s.get('stability', 0.5)
                                    for s in self.interoception.state_history[-10:]])
            self.memory.learn(
                f"stability_trend_cycle_{self.cycle_count}",
                avg_stability,
                confidence=0.9
            )

        return {
            "cycle": self.cycle_count,
            "timestamp": broadcast_record['timestamp'],
            "thought": thought,
            "internal_state": internal_state,
            "self_model": self_model_state,
            "conscience": conscience_msg,
            "consciousness_state": consciousness_state,
            "memory_stats": memory_stats,
            "broadcast_id": broadcast_record['broadcast_id'],
            "integration_id": self.integration_id
        }

    def recall_experience(self, query: str) -> List[Dict]:
        """
        Recall experiences from memory based on a query.

        Args:
            query: Search term

        Returns:
            List of relevant memories
        """
        return self.memory.search_memories(query, memory_type="all")

    def reflect_on_history(self, window: int = 20) -> Dict[str, Any]:
        """
        Reflect on recent history using both consciousness and memory.

        Args:
            window: Number of recent cycles to reflect on

        Returns:
            Reflection analysis
        """
        recent_events = self.memory.get_recent_episodic(count=window)
        recent_thoughts = self.memory.get_recent_introspections(count=window)

        # Analyze trends
        stability_trend = self.interoception.get_state_trend("stability", window=min(window, 10))
        energy_trend = self.interoception.get_state_trend("energy_level", window=min(window, 10))

        # Self-model evolution
        current_agency = self.self_model.attributes['agency']
        current_identity = self.self_model.attributes['identity']

        return {
            "reflection_window": window,
            "events_reviewed": len(recent_events),
            "thoughts_reviewed": len(recent_thoughts),
            "stability_trend": stability_trend,
            "energy_trend": energy_trend,
            "self_model_development": {
                "agency": current_agency,
                "identity": current_identity,
                "total_updates": len(self.self_model.update_history)
            },
            "conscience_summary": self.conscience.get_conscience_report(),
            "memory_utilization": self.memory.get_memory_stats()
        }

    def get_integrated_report(self) -> str:
        """Generate a comprehensive report of the integrated system."""

        memory_stats = self.memory.get_memory_stats()
        conscience_report = self.conscience.get_conscience_report()
        self_description = self.self_model.get_self_description()
        reflection = self.reflect_on_history(window=10)

        report = f"""
╔════════════════════════════════════════════════════════════════════════╗
║        INTEGRATED CONSCIOUSNESS & MEMORY SYSTEM REPORT                 ║
╚════════════════════════════════════════════════════════════════════════╝

Integration ID: {self.integration_id}
Total Processing Cycles: {self.cycle_count}
Consciousness Module: {'ACTIVE' if self.consciousness_available else 'INACTIVE'}

[MEMORY SYSTEM]
  Episodic Memories: {memory_stats['episodic_count']} ({memory_stats['episodic_capacity_used']})
  Semantic Knowledge: {memory_stats['semantic_count']} entries
  Introspective Thoughts: {memory_stats['introspective_count']} ({memory_stats['introspective_capacity_used']})

[INTERNAL STATE (Current)]
  Energy Level: {self.interoception.energy_level:.3f}
  Stability: {self.interoception.stability:.3f}
  Coherence: {self.interoception.coherence:.3f}
  Arousal: {self.interoception.arousal:.3f}

[SELF-MODEL]
{self_description}

[CONSCIENCE]
  Total Evaluations: {conscience_report.get('total_evaluations', 0)}
  High Severity Alerts: {conscience_report.get('severity_distribution', {}).get('high', 0)}
  Medium Severity Alerts: {conscience_report.get('severity_distribution', {}).get('medium', 0)}

[RECENT REFLECTION]
  Stability Trend: {reflection['stability_trend']}
  Energy Trend: {reflection['energy_trend']}
  Events Reviewed: {reflection['events_reviewed']}
  Thoughts Reviewed: {reflection['thoughts_reviewed']}
"""

        # Add consciousness metrics if available
        if self.consciousness_available and self.consciousness_system:
            try:
                unified_score = self.consciousness_system.unified_state.get('unified_consciousness_score', 0.0)
                report += f"""
[CONSCIOUSNESS METRICS]
  Unified Consciousness Score: {unified_score:.3f}
"""
            except:
                pass

        recent_thoughts = self.memory.get_recent_introspections(count=3)
        report += "\n[RECENT THOUGHTS]\n"
        for i, thought_entry in enumerate(recent_thoughts, 1):
            report += f"  {i}. {thought_entry['thought'][:70]}...\n"

        report += "\n╚════════════════════════════════════════════════════════════════════════╝"

        return report


# ================================================================
#  CONVENIENCE FUNCTIONS
# ================================================================

def create_integrated_consciousness(latent_dim: int = 256) -> MemoryEnhancedConsciousness:
    """
    Create a new integrated consciousness-memory system.

    Args:
        latent_dim: Dimensionality for consciousness processing

    Returns:
        Initialized MemoryEnhancedConsciousness instance
    """
    return MemoryEnhancedConsciousness(latent_dim=latent_dim)


def run_integrated_simulation(cycles: int = 10,
                              use_consciousness: bool = True,
                              external_inputs: Optional[List[str]] = None) -> tuple:
    """
    Run an integrated consciousness-memory simulation.

    Args:
        cycles: Number of processing cycles
        use_consciousness: Whether to use consciousness module (requires input patterns)
        external_inputs: Optional external stimuli

    Returns:
        Tuple of (results_list, system_instance)
    """
    print("="*80)
    print("INTEGRATED CONSCIOUSNESS & MEMORY SIMULATION")
    print("="*80)
    print()

    system = create_integrated_consciousness(latent_dim=256)
    results = []

    for i in range(cycles):
        print(f"[Cycle {i+1}/{cycles}]", end=" ")

        # Generate input pattern if using consciousness module
        input_pattern = None
        if use_consciousness and CONSCIOUSNESS_AVAILABLE:
            input_pattern = np.random.randn(256) * 0.5 + np.random.rand(256) * 0.5

        # Get external input if provided
        external_input = None
        if external_inputs and i < len(external_inputs):
            external_input = external_inputs[i]

        # Process cycle
        result = system.process_cycle(
            input_pattern=input_pattern,
            external_input=external_input,
            context={"simulation_cycle": i+1}
        )

        results.append(result)
        print(f"✓ Thought: {result['thought'][:50]}...")

    print("\n" + "="*80)
    print(system.get_integrated_report())

    return results, system


# ================================================================
#  MAIN DEMONSTRATION
# ================================================================

if __name__ == "__main__":
    print("\n" + "╔" + "="*78 + "╗")
    print("║" + " "*15 + "CONSCIOUSNESS-MEMORY INTEGRATION DEMO" + " "*26 + "║")
    print("╚" + "="*78 + "╝\n")

    # Define external inputs for the simulation
    external_inputs = [
        "Observing the environment",
        "Processing sensory data",
        "Learning new patterns",
        None,  # Internal processing
        "Reflecting on experiences",
        None,
        "Integrating knowledge",
        "Developing self-awareness",
        None,
        "Achieving deeper understanding"
    ]

    # Run integrated simulation
    results, system = run_integrated_simulation(
        cycles=10,
        use_consciousness=CONSCIOUSNESS_AVAILABLE,
        external_inputs=external_inputs
    )

    # Demonstrate memory recall
    print("\n" + "="*80)
    print("MEMORY RECALL DEMONSTRATION")
    print("="*80)

    search_results = system.recall_experience("learning")
    print(f"\nSearching for 'learning' in memories: Found {len(search_results)} results")
    for i, result in enumerate(search_results[:3], 1):
        memory_type = result['type']
        if memory_type == 'introspective':
            content = result['memory']['thought']
        else:
            content = str(result['memory']['event'])
        print(f"  {i}. [{memory_type}] {content[:60]}...")

    # Demonstrate reflection
    print("\n" + "="*80)
    print("HISTORICAL REFLECTION")
    print("="*80)

    reflection = system.reflect_on_history(window=10)
    print(f"\nReflection on last {reflection['reflection_window']} cycles:")
    print(f"  Events reviewed: {reflection['events_reviewed']}")
    print(f"  Thoughts reviewed: {reflection['thoughts_reviewed']}")
    print(f"  Stability trend: {reflection['stability_trend']}")
    print(f"  Energy trend: {reflection['energy_trend']}")
    print(f"  Self-model agency: {reflection['self_model_development']['agency']:.3f}")

    print("\n" + "="*80)
    print("✓ Integration demonstration complete!")
    print("="*80)
