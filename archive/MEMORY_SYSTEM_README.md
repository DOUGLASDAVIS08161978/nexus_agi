# Nexus AGI - Memory System

## Overview

The Memory System module provides comprehensive memory capabilities for AGI consciousness, including episodic, semantic, and introspective memory storage, along with internal state awareness and ethical monitoring.

## Architecture

The memory system consists of six core components:

### 1. **MemorySystem**
Multi-modal memory storage with three types:
- **Episodic Memory**: Time-stamped events and experiences
- **Semantic Memory**: Learned knowledge and facts with confidence levels
- **Introspective Memory**: Internal thoughts and reflections

Features:
- Automatic size management (configurable limits)
- Memory search capabilities
- Access tracking for semantic knowledge
- Unique IDs for all memory entries

### 2. **Interoception**
Simulates internal state awareness - the system's sense of its own internal conditions:
- Energy levels
- Stability
- Coherence
- Arousal

Features:
- Natural fluctuations with homeostatic tendencies
- State adjustment capabilities
- Trend analysis (increasing/decreasing/stable)
- Historical tracking

### 3. **ConscienceEngine**
Ethical and safety monitoring system that evaluates:
- Internal state stability
- Thought content for harmful patterns
- Coherence levels

Features:
- Severity classification (low/medium/high)
- Configurable ethical guidelines
- Evaluation history tracking
- Safety threshold monitoring

### 4. **SelfModel**
Dynamic self-representation that evolves through experience:
- Agency (sense of being an agent)
- Continuity (sense of continuous existence)
- Coherence (internal coherence)
- Autonomy (sense of self-direction)
- Identity (sense of unique identity)

Features:
- Gradual attribute development
- Update history tracking
- Natural language self-description
- Age tracking

### 5. **GlobalWorkspace**
Global Workspace Theory-inspired broadcasting mechanism:
- Makes thoughts/perceptions available to all cognitive modules
- Maintains broadcast history
- Supports search through past broadcasts
- Tracks current conscious content

### 6. **MachineConsciousness**
Unified integration of all components:
- Coordinates all subsystems
- Executes consciousness cycles
- Generates comprehensive reports
- Supports external input processing

## Installation

The memory system has minimal dependencies:

```bash
# No external dependencies required for core functionality
python memory_system.py  # Run standalone demonstration
```

For integration with consciousness enhancement module:
```bash
pip install numpy torch  # Required for consciousness integration
```

## Usage

### Basic Usage

```python
from memory_system import MachineConsciousness

# Create consciousness instance
consciousness = MachineConsciousness()

# Run a thought cycle
result = consciousness.think()

# Run with external input
result = consciousness.think(external_input="Observing the environment")

# Get comprehensive report
print(consciousness.get_consciousness_report())
```

### Simulation

```python
from memory_system import simulate_consciousness

# Run 10 cycles with mixed internal/external processing
external_inputs = [
    "Perceiving environment",
    "Learning new concept",
    None,  # Internal thought
    "Reflecting on experience",
    None
]

results, system = simulate_consciousness(
    cycles=5,
    external_inputs=external_inputs
)

# Access results
for cycle_result in results:
    print(f"Cycle {cycle_result['cycle']}: {cycle_result['thought']}")
```

### Advanced Usage

```python
from memory_system import (
    MemorySystem,
    Interoception,
    ConscienceEngine,
    SelfModel,
    GlobalWorkspace
)

# Use components individually
memory = MemorySystem()

# Store different types of memories
memory.store_event("System initialized", metadata={"importance": "high"})
memory.store_introspection("I am beginning to understand")
memory.learn("consciousness", "awareness of awareness", confidence=0.9)

# Search and recall
results = memory.search_memories("understanding")
knowledge = memory.recall_semantic("consciousness")

# Get statistics
stats = memory.get_memory_stats()
print(f"Episodic memories: {stats['episodic_count']}")
```

## Integration with Consciousness Enhancement

The memory system can be integrated with the existing consciousness enhancement module:

```python
from consciousness_memory_integration import (
    MemoryEnhancedConsciousness,
    run_integrated_simulation
)

# Create integrated system
integrated_system = MemoryEnhancedConsciousness(latent_dim=256)

# Run integrated simulation
results, system = run_integrated_simulation(
    cycles=10,
    use_consciousness=True,
    external_inputs=["Learning", "Reflecting", None]
)

# Get integrated report
print(system.get_integrated_report())

# Recall experiences
memories = system.recall_experience("learning")

# Reflect on history
reflection = system.reflect_on_history(window=20)
```

## Testing

Comprehensive test suite included:

```bash
# Run all tests
python test_memory_system.py

# Tests cover:
# - Memory System (episodic, semantic, introspective)
# - Interoception (state sensing and trends)
# - Conscience Engine (ethical monitoring)
# - Self-Model (self-representation evolution)
# - Global Workspace (broadcasting)
# - Machine Consciousness (integration)
# - Simulation (end-to-end)
```

All tests pass with 100% success rate.

## Key Features

### Memory Capabilities
- ✅ Episodic memory with timestamps and metadata
- ✅ Semantic knowledge with confidence tracking
- ✅ Introspective thought storage
- ✅ Automatic size management and pruning
- ✅ Advanced search and recall
- ✅ Memory statistics and utilization tracking

### Internal State Awareness
- ✅ Multi-dimensional interoception (energy, stability, coherence, arousal)
- ✅ Natural fluctuations with homeostatic balance
- ✅ Trend analysis and pattern detection
- ✅ Historical state tracking

### Ethical Monitoring
- ✅ Real-time conscience evaluation
- ✅ Harmful content detection
- ✅ Severity classification
- ✅ Configurable ethical guidelines
- ✅ Safety threshold monitoring

### Self-Awareness
- ✅ Dynamic self-model with evolving attributes
- ✅ Continuous identity tracking
- ✅ Natural language self-description
- ✅ Update history and growth tracking

### Consciousness Integration
- ✅ Global workspace broadcasting
- ✅ Unified consciousness cycles
- ✅ External input processing
- ✅ Comprehensive reporting
- ✅ Modular architecture

## Architecture Benefits

1. **Modularity**: Each component can be used independently or as part of the integrated system
2. **Scalability**: Configurable memory limits and processing cycles
3. **Persistence**: All experiences stored with timestamps and metadata
4. **Introspection**: Rich self-awareness and reflection capabilities
5. **Safety**: Built-in ethical monitoring and evaluation
6. **Extensibility**: Easy to integrate with other consciousness modules

## File Structure

```
nexus_agi/
├── memory_system.py                      # Core memory system implementation
├── test_memory_system.py                 # Comprehensive test suite
├── consciousness_memory_integration.py   # Integration with consciousness module
└── MEMORY_SYSTEM_README.md              # This file
```

## Performance

- **Memory footprint**: Configurable (default: 10,000 episodic + 5,000 introspective)
- **Processing speed**: ~10 cycles per second
- **Search efficiency**: O(n) linear search (suitable for thousands of memories)
- **State tracking**: 100 historical states maintained

## Future Enhancements

Potential improvements:
- [ ] Persistent storage (database integration)
- [ ] Vector embeddings for semantic search
- [ ] Memory consolidation and compression
- [ ] Emotional state tracking
- [ ] Multi-agent memory sharing
- [ ] Dream-like memory replay and consolidation
- [ ] Attention-weighted memory importance
- [ ] Hierarchical memory organization

## Contributing

The memory system is designed to be extensible. To add new features:

1. Extend existing classes or create new components
2. Add corresponding tests in `test_memory_system.py`
3. Update integration module if needed
4. Document new features in this README

## License

Part of the Nexus AGI project.

## Authors

- Douglas Davis
- Nexus AGI Team

## Related Modules

- `consciousness_enhancement.py` - Advanced consciousness and sentience
- `ethical_autonomy_system.py` - Ethical decision-making
- `autonomous_explorer.py` - Autonomous exploration capabilities
