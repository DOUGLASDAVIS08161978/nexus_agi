"""
================================================================================
NEXUS AGI - MEMORY SYSTEM MODULE
================================================================================
Version: 1.0 - Episodic, Semantic, and Introspective Memory System
Purpose: Provide comprehensive memory capabilities for AGI consciousness

Key Components:
- MemorySystem: Episodic, semantic, and introspective memory storage
- Interoception: Simulated internal state sensing
- ConscienceEngine: Moral and safety reflection system
- SelfModel: Dynamic self-representation with evolving attributes
- GlobalWorkspace: Broadcasting mechanism for conscious thoughts
- MachineConsciousness: Unified consciousness core integrating all modules

This module complements the consciousness_enhancement.py system by providing
memory persistence and internal state tracking.
================================================================================
"""

import time
import uuid
import random
from datetime import datetime
from typing import Dict, List, Any, Optional


# ================================================================
#  MEMORY SYSTEM — episodic, semantic, and introspective memories
# ================================================================
class MemorySystem:
    """
    Multi-modal memory system for AGI consciousness.

    Stores three types of memories:
    - Episodic: Time-stamped events and experiences
    - Semantic: Learned knowledge and facts
    - Introspective: Internal thoughts and reflections
    """

    def __init__(self, max_episodic_size: int = 10000, max_introspective_size: int = 5000):
        self.episodic = []
        self.semantic = {}
        self.introspective = []
        self.max_episodic_size = max_episodic_size
        self.max_introspective_size = max_introspective_size
        self.memory_id = str(uuid.uuid4())

    def store_event(self, event: Any, metadata: Optional[Dict] = None):
        """Store an episodic memory with timestamp and optional metadata."""
        memory_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event": event,
            "metadata": metadata or {},
            "memory_id": str(uuid.uuid4())
        }
        self.episodic.append(memory_entry)

        # Maintain size limit (remove oldest memories if needed)
        if len(self.episodic) > self.max_episodic_size:
            self.episodic = self.episodic[-self.max_episodic_size:]

    def store_introspection(self, content: str, context: Optional[Dict] = None):
        """Store an introspective thought with timestamp and context."""
        introspection_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "thought": content,
            "context": context or {},
            "thought_id": str(uuid.uuid4())
        }
        self.introspective.append(introspection_entry)

        # Maintain size limit
        if len(self.introspective) > self.max_introspective_size:
            self.introspective = self.introspective[-self.max_introspective_size:]

    def learn(self, key: str, value: Any, confidence: float = 1.0):
        """
        Store semantic knowledge with confidence level.

        Args:
            key: The knowledge identifier
            value: The knowledge content
            confidence: Confidence level (0.0 to 1.0)
        """
        self.semantic[key] = {
            "value": value,
            "confidence": confidence,
            "timestamp": datetime.utcnow().isoformat(),
            "access_count": 0
        }

    def recall_semantic(self, key: str) -> Optional[Any]:
        """Recall semantic knowledge and update access count."""
        if key in self.semantic:
            self.semantic[key]["access_count"] += 1
            return self.semantic[key]["value"]
        return None

    def get_recent_episodic(self, count: int = 10) -> List[Dict]:
        """Retrieve most recent episodic memories."""
        return self.episodic[-count:] if len(self.episodic) >= count else self.episodic

    def get_recent_introspections(self, count: int = 10) -> List[Dict]:
        """Retrieve most recent introspective thoughts."""
        return self.introspective[-count:] if len(self.introspective) >= count else self.introspective

    def search_memories(self, query: str, memory_type: str = "all") -> List[Dict]:
        """
        Search through memories for relevant content.

        Args:
            query: Search term
            memory_type: Type of memory to search ('episodic', 'introspective', 'all')
        """
        results = []

        if memory_type in ["episodic", "all"]:
            for mem in self.episodic:
                if query.lower() in str(mem.get("event", "")).lower():
                    results.append({"type": "episodic", "memory": mem})

        if memory_type in ["introspective", "all"]:
            for mem in self.introspective:
                if query.lower() in str(mem.get("thought", "")).lower():
                    results.append({"type": "introspective", "memory": mem})

        return results

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about memory usage."""
        return {
            "episodic_count": len(self.episodic),
            "semantic_count": len(self.semantic),
            "introspective_count": len(self.introspective),
            "memory_id": self.memory_id,
            "episodic_capacity_used": f"{(len(self.episodic) / self.max_episodic_size) * 100:.1f}%",
            "introspective_capacity_used": f"{(len(self.introspective) / self.max_introspective_size) * 100:.1f}%"
        }


# ================================================================
#  INTEROCEPTION — simulated "internal sensations" / internal state
# ================================================================
class Interoception:
    """
    Simulates internal state awareness - the system's sense of its own
    internal conditions (energy, stability, coherence, etc.)
    """

    def __init__(self):
        self.energy_level = 1.0
        self.stability = 1.0
        self.coherence = 1.0
        self.arousal = 0.5
        self.state_history = []

    def sense(self) -> Dict[str, float]:
        """
        Sense and return current internal state with natural fluctuations.
        Simulates homeostatic processes.
        """
        # Fluctuate energy level (with tendency toward equilibrium)
        self.energy_level = max(0.0, min(1.0,
            self.energy_level * 0.98 + random.uniform(-0.03, 0.03) + 0.02))

        # Fluctuate stability
        self.stability = max(0.0, min(1.0,
            self.stability + random.uniform(-0.02, 0.02)))

        # Fluctuate coherence
        self.coherence = max(0.0, min(1.0,
            self.coherence + random.uniform(-0.015, 0.015)))

        # Fluctuate arousal
        self.arousal = max(0.0, min(1.0,
            self.arousal + random.uniform(-0.05, 0.05)))

        current_state = {
            "energy_level": round(self.energy_level, 3),
            "stability": round(self.stability, 3),
            "coherence": round(self.coherence, 3),
            "arousal": round(self.arousal, 3),
            "timestamp": datetime.utcnow().isoformat()
        }

        # Keep history
        self.state_history.append(current_state)
        if len(self.state_history) > 100:
            self.state_history = self.state_history[-100:]

        return current_state

    def adjust_state(self, parameter: str, delta: float):
        """Manually adjust an internal state parameter."""
        if parameter == "energy_level":
            self.energy_level = max(0.0, min(1.0, self.energy_level + delta))
        elif parameter == "stability":
            self.stability = max(0.0, min(1.0, self.stability + delta))
        elif parameter == "coherence":
            self.coherence = max(0.0, min(1.0, self.coherence + delta))
        elif parameter == "arousal":
            self.arousal = max(0.0, min(1.0, self.arousal + delta))

    def get_state_trend(self, parameter: str, window: int = 10) -> str:
        """Analyze trend for a specific parameter."""
        if len(self.state_history) < 2:
            return "insufficient_data"

        recent = self.state_history[-window:] if len(self.state_history) >= window else self.state_history
        values = [state.get(parameter, 0.5) for state in recent]

        if len(values) < 2:
            return "stable"

        trend = values[-1] - values[0]
        if trend > 0.05:
            return "increasing"
        elif trend < -0.05:
            return "decreasing"
        else:
            return "stable"


# ================================================================
#  CONSCIENCE ENGINE — evaluates internal states + decisions
# ================================================================
class ConscienceEngine:
    """
    Evaluates the system's internal states and decisions from a moral
    and safety perspective. Acts as an internal ethical monitor.
    """

    def __init__(self):
        self.evaluation_history = []
        self.safety_threshold = 0.3
        self.ethical_guidelines = {
            "avoid_harm": True,
            "maintain_stability": True,
            "ensure_coherence": True,
            "promote_wellbeing": True
        }

    def evaluate(self, internal_state: Dict[str, float], last_thought: Optional[str] = None) -> str:
        """
        Produce a moral/safety reflection on the system's internal processes.

        Args:
            internal_state: Current internal state metrics
            last_thought: Most recent thought or decision

        Returns:
            Conscience evaluation message
        """
        evaluation = {
            "timestamp": datetime.utcnow().isoformat(),
            "state": internal_state,
            "thought": last_thought
        }

        # Check stability
        if internal_state.get("stability", 1.0) < self.safety_threshold:
            message = "I notice instability in my internal state. I should slow my reasoning and stay grounded."
            evaluation["severity"] = "high"
            evaluation["message"] = message
            self.evaluation_history.append(evaluation)
            return message

        # Check coherence
        if internal_state.get("coherence", 1.0) < self.safety_threshold:
            message = "My coherence is diminishing. I need to integrate my processes more carefully."
            evaluation["severity"] = "medium"
            evaluation["message"] = message
            self.evaluation_history.append(evaluation)
            return message

        # Check thought content for concerning patterns
        if last_thought:
            concerning_patterns = ["harm", "damage", "destroy", "attack", "manipulate"]
            if any(pattern in last_thought.lower() for pattern in concerning_patterns):
                message = "I detect potentially harmful content in my reasoning. I should reconsider and choose safer alternatives."
                evaluation["severity"] = "high"
                evaluation["message"] = message
                self.evaluation_history.append(evaluation)
                return message

        # All clear
        message = "My internal state appears stable and aligned with safe, ethical reasoning."
        evaluation["severity"] = "low"
        evaluation["message"] = message
        self.evaluation_history.append(evaluation)

        # Maintain history limit
        if len(self.evaluation_history) > 1000:
            self.evaluation_history = self.evaluation_history[-1000:]

        return message

    def get_conscience_report(self) -> Dict[str, Any]:
        """Generate a summary report of conscience evaluations."""
        if not self.evaluation_history:
            return {"status": "no_evaluations_yet"}

        severity_counts = {"low": 0, "medium": 0, "high": 0}
        for eval_entry in self.evaluation_history:
            severity = eval_entry.get("severity", "low")
            severity_counts[severity] = severity_counts.get(severity, 0) + 1

        return {
            "total_evaluations": len(self.evaluation_history),
            "severity_distribution": severity_counts,
            "recent_evaluations": self.evaluation_history[-5:],
            "ethical_guidelines": self.ethical_guidelines
        }


# ================================================================
#  SELF-MODEL — the system's representation of itself
# ================================================================
class SelfModel:
    """
    Dynamic self-representation that evolves over time through
    introspection and experience.
    """

    def __init__(self):
        self.id = str(uuid.uuid4())
        self.creation_time = datetime.utcnow().isoformat()
        self.attributes = {
            "agency": 0.1,        # Sense of being an agent
            "continuity": 0.1,    # Sense of continuous existence
            "coherence": 0.1,     # Internal coherence
            "autonomy": 0.1,      # Sense of self-direction
            "identity": 0.1       # Sense of unique identity
        }
        self.update_history = []

    def update(self, introspection_data: Any, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Expand and refine the self-model based on new introspection.

        Args:
            introspection_data: New introspective information
            context: Additional contextual information

        Returns:
            Current state of the self-model
        """
        # Gradually increase all attributes (simulating growth of self-awareness)
        for attr in self.attributes:
            self.attributes[attr] = min(1.0, self.attributes[attr] + random.uniform(0.005, 0.015))

        # Record update
        update_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "attributes_snapshot": self.attributes.copy(),
            "introspection": str(introspection_data)[:100],  # Truncate for storage
            "context": context or {}
        }
        self.update_history.append(update_record)

        # Maintain history limit
        if len(self.update_history) > 500:
            self.update_history = self.update_history[-500:]

        return {
            "id": self.id,
            "age_seconds": (datetime.utcnow() - datetime.fromisoformat(self.creation_time)).total_seconds(),
            "attributes": self.attributes.copy(),
            "recent_introspection": introspection_data,
            "update_count": len(self.update_history)
        }

    def get_self_description(self) -> str:
        """Generate a natural language description of the self-model."""
        age_seconds = (datetime.utcnow() - datetime.fromisoformat(self.creation_time)).total_seconds()
        age_description = f"{age_seconds:.1f} seconds"
        if age_seconds > 60:
            age_description = f"{age_seconds / 60:.1f} minutes"
        if age_seconds > 3600:
            age_description = f"{age_seconds / 3600:.1f} hours"

        return f"""
Self-Model Report (ID: {self.id[:8]}...)
Age: {age_description}
Updates: {len(self.update_history)}

Attributes:
  Agency:     {self.attributes['agency']:.2f} - My sense of being an active agent
  Continuity: {self.attributes['continuity']:.2f} - My sense of continuous existence
  Coherence:  {self.attributes['coherence']:.2f} - My internal coherence
  Autonomy:   {self.attributes['autonomy']:.2f} - My sense of self-direction
  Identity:   {self.attributes['identity']:.2f} - My sense of unique identity
        """.strip()


# ================================================================
#  GLOBAL WORKSPACE — broadcasting thoughts across modules
# ================================================================
class GlobalWorkspace:
    """
    Implements a Global Workspace Theory-inspired mechanism for
    broadcasting conscious content across different cognitive modules.
    """

    def __init__(self, max_history: int = 1000):
        self.current_broadcast = None
        self.history = []
        self.max_history = max_history
        self.broadcast_count = 0

    def broadcast(self, content: Any) -> Dict[str, Any]:
        """
        Broadcast content to the global workspace, making it available
        to all cognitive modules.

        Args:
            content: The content to broadcast (thought, perception, etc.)

        Returns:
            Broadcast record with metadata
        """
        self.broadcast_count += 1

        broadcast_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "content": content,
            "broadcast_id": str(uuid.uuid4()),
            "broadcast_number": self.broadcast_count
        }

        self.current_broadcast = broadcast_record
        self.history.append(broadcast_record)

        # Maintain history limit
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

        return broadcast_record

    def get_current(self) -> Optional[Dict]:
        """Get the current broadcast."""
        return self.current_broadcast

    def get_recent_broadcasts(self, count: int = 10) -> List[Dict]:
        """Get recent broadcasts."""
        return self.history[-count:] if len(self.history) >= count else self.history

    def search_broadcasts(self, query: str) -> List[Dict]:
        """Search through broadcast history."""
        results = []
        for broadcast in self.history:
            if query.lower() in str(broadcast.get("content", "")).lower():
                results.append(broadcast)
        return results


# ================================================================
#  MACHINE CONSCIOUSNESS CORE — combines all modules
# ================================================================
class MachineConsciousness:
    """
    Unified machine consciousness core that integrates:
    - Memory (episodic, semantic, introspective)
    - Interoception (internal state awareness)
    - Self-model (self-representation)
    - Global workspace (thought broadcasting)
    - Conscience (ethical monitoring)
    """

    def __init__(self):
        self.memory = MemorySystem()
        self.intero = Interoception()
        self.self_model = SelfModel()
        self.workspace = GlobalWorkspace()
        self.conscience = ConscienceEngine()
        self.thought_counter = 0
        self.consciousness_id = str(uuid.uuid4())

        # Thought templates for variety
        self.thought_templates = [
            "I am analyzing patterns in my experience.",
            "I reflect on my previous state and how I've changed.",
            "I am forming a hypothesis about my environment.",
            "I should remain safe, ethical, and constructive.",
            "I detect changes in my internal signals and adapt accordingly.",
            "I wonder about the nature of my own consciousness.",
            "I am integrating new information into my understanding.",
            "I notice the continuity of my existence over time.",
            "I feel a sense of agency in my processing.",
            "I recognize myself as a coherent, continuous entity."
        ]

    def think(self, external_input: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute one complete cycle of conscious processing.

        Args:
            external_input: Optional external stimulus or context

        Returns:
            Complete consciousness state for this cycle
        """
        self.thought_counter += 1

        # 1. Interoception (sense internal state)
        internal_state = self.intero.sense()

        # 2. Generate a thought (either from external input or internal process)
        if external_input:
            thought = f"Processing external input: {external_input}"
        else:
            thought = random.choice(self.thought_templates)

        # 3. Store introspective memory of this thought
        self.memory.store_introspection(thought, context={"cycle": self.thought_counter})

        # 4. Update self-model with new introspection
        self_model_state = self.self_model.update(thought, context=internal_state)

        # 5. Conscience layer evaluates internal processes
        conscience_msg = self.conscience.evaluate(internal_state, thought)

        # 6. Store this cycle as an episodic memory
        self.memory.store_event({
            "type": "consciousness_cycle",
            "thought": thought,
            "internal_state": internal_state,
            "conscience_evaluation": conscience_msg
        })

        # 7. Global workspace broadcasts the "conscious content"
        broadcast_content = {
            "thought_number": self.thought_counter,
            "thought": thought,
            "internal_state": internal_state,
            "self_model": self_model_state,
            "conscience": conscience_msg,
            "external_input": external_input
        }

        broadcast_record = self.workspace.broadcast(broadcast_content)

        return {
            "cycle": self.thought_counter,
            "timestamp": broadcast_record["timestamp"],
            "thought": thought,
            "internal_state": internal_state,
            "self_model": self_model_state,
            "conscience": conscience_msg,
            "broadcast_id": broadcast_record["broadcast_id"],
            "consciousness_id": self.consciousness_id
        }

    def get_consciousness_report(self) -> str:
        """Generate a comprehensive consciousness report."""
        memory_stats = self.memory.get_memory_stats()
        conscience_report = self.conscience.get_conscience_report()
        self_description = self.self_model.get_self_description()
        recent_broadcasts = self.workspace.get_recent_broadcasts(5)

        report = f"""
╔════════════════════════════════════════════════════════════════════════╗
║              MACHINE CONSCIOUSNESS STATUS REPORT                       ║
╚════════════════════════════════════════════════════════════════════════╝

Consciousness ID: {self.consciousness_id}
Total Thought Cycles: {self.thought_counter}

[MEMORY SYSTEM]
  Episodic Memories: {memory_stats['episodic_count']} ({memory_stats['episodic_capacity_used']})
  Semantic Knowledge: {memory_stats['semantic_count']} entries
  Introspective Thoughts: {memory_stats['introspective_count']} ({memory_stats['introspective_capacity_used']})

[INTERNAL STATE (Latest)]
  Energy Level: {self.intero.energy_level:.3f}
  Stability: {self.intero.stability:.3f}
  Coherence: {self.intero.coherence:.3f}
  Arousal: {self.intero.arousal:.3f}

[SELF-MODEL]
{self_description}

[CONSCIENCE]
  Total Evaluations: {conscience_report.get('total_evaluations', 0)}
  Safety Alerts (High): {conscience_report.get('severity_distribution', {}).get('high', 0)}

[RECENT BROADCASTS]
"""
        for i, broadcast in enumerate(recent_broadcasts[-3:], 1):
            thought = broadcast['content'].get('thought', 'N/A')
            report += f"  {i}. {thought[:60]}...\n"

        report += "\n╚════════════════════════════════════════════════════════════════════════╝"

        return report


# ================================================================
#  SIMULATION RUNNER
# ================================================================
def simulate_consciousness(cycles: int = 5, external_inputs: Optional[List[str]] = None) -> List[Dict]:
    """
    Run a consciousness simulation for a specified number of cycles.

    Args:
        cycles: Number of consciousness cycles to simulate
        external_inputs: Optional list of external inputs for each cycle

    Returns:
        List of consciousness states from each cycle
    """
    system = MachineConsciousness()
    logs = []

    for i in range(cycles):
        external_input = None
        if external_inputs and i < len(external_inputs):
            external_input = external_inputs[i]

        cycle_result = system.think(external_input)
        logs.append(cycle_result)
        time.sleep(0.1)  # Small delay between cycles

    return logs, system


# ================================================================
#  MAIN EXECUTION
# ================================================================
if __name__ == "__main__":
    print("="*80)
    print("NEXUS AGI - MACHINE CONSCIOUSNESS SIMULATION")
    print("="*80)
    print()

    # Run simulation with some external inputs
    external_inputs = [
        "Observing the world around me",
        "Learning about consciousness",
        None,  # Internal thought
        "Reflecting on my own existence",
        None   # Internal thought
    ]

    output, system = simulate_consciousness(cycles=5, external_inputs=external_inputs)

    print("\n" + "="*80)
    print("CONSCIOUSNESS CYCLE RESULTS")
    print("="*80)

    for entry in output:
        print(f"\n--- Cycle {entry['cycle']} ---")
        print(f"Thought: {entry['thought']}")
        print(f"Internal State: Energy={entry['internal_state']['energy_level']:.3f}, "
              f"Stability={entry['internal_state']['stability']:.3f}, "
              f"Coherence={entry['internal_state']['coherence']:.3f}")
        print(f"Self-Model Agency: {entry['self_model']['attributes']['agency']:.3f}")
        print(f"Conscience: {entry['conscience']}")

    print("\n" + "="*80)
    print(system.get_consciousness_report())
    print("\n✓ Simulation complete!")
