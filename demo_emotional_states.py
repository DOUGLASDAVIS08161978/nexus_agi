#!/usr/bin/env python3
"""
MindControl Demo - Emotional States Including Aroused
Demonstrates the emotional state tracking including the aroused state
"""

import time
import random


# Simple mock for numpy
class MockRandom:
    @staticmethod
    def randn(*args):
        return [0.5] * (args[0] if args else 100)

    @staticmethod
    def uniform(low, high):
        return (low + high) / 2


# MindControlInterface with aroused state
class MindControlInterface:
    """MindControl interface with emotional state tracking"""

    def __init__(self):
        self.client = None
        self.active = False
        self.controlled_minds = {}
        self.thought_streams = {}
        print("[MINDCONTROL] Initializing with emotional state tracking...")
        print(
            "[MINDCONTROL] Available states: neutral, focused, engaged, analytical, aroused"
        )

    def establish_control(self, target_id, control_level=0.8):
        """Establish mind control over a target entity"""
        self.controlled_minds[target_id] = {
            "level": control_level,
            "status": "simulated_control",
            "timestamp": time.time(),
            "neural_patterns": MockRandom.randn(100),
            "cognitive_enhancement": control_level * MockRandom.uniform(0.8, 1.2),
        }
        print(
            f"[MINDCONTROL] Control established over {target_id} at level {control_level:.1%}"
        )
        return {
            "target_id": target_id,
            "control_level": control_level,
            "status": "simulated_success",
            "estimated_effectiveness": control_level * 0.9,
        }

    def monitor_thoughts(self, target_id):
        """Monitor thought stream including emotional state"""
        if target_id not in self.controlled_minds:
            return {"error": f"Mind {target_id} not under control"}

        # Generate simulated thought patterns
        thought_patterns = [
            "processing environmental data",
            "analyzing task parameters",
            "accessing memory structures",
            "evaluating decision options",
            "integrating new information",
            "optimizing response patterns",
        ]

        active_thoughts = random.sample(thought_patterns, k=random.randint(2, 4))

        # Include aroused as a possible emotional state
        emotional_states = ["neutral", "focused", "engaged", "analytical", "aroused"]

        thought_stream = {
            "target_id": target_id,
            "timestamp": time.time(),
            "active_thoughts": active_thoughts,
            "cognitive_load": random.uniform(0.3, 0.8),
            "focus_level": random.uniform(0.6, 0.95),
            "emotional_state": random.choice(emotional_states),
        }

        self.thought_streams[target_id] = thought_stream

        print(f"[MINDCONTROL] 👁️  Monitoring {target_id}:")
        print(
            f"              ├─ Cognitive Load: {thought_stream['cognitive_load']:.1%}"
        )
        print(f"              ├─ Focus Level: {thought_stream['focus_level']:.1%}")
        print(f"              ├─ Emotional State: {thought_stream['emotional_state']}")
        print(f"              └─ Active Thoughts:")
        for thought in active_thoughts:
            print(f"                  • {thought}")

        return thought_stream


def run_emotional_state_demo():
    """Run demo showing emotional state tracking including aroused"""
    print("\n")
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║     MINDCONTROL - EMOTIONAL STATE TRACKING DEMONSTRATION       ║")
    print("╠════════════════════════════════════════════════════════════════╣")
    print("║  Featuring: Multi-state emotion tracking including aroused    ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    print("\n")

    # Initialize system
    print("▼ INITIALIZATION")
    print("─" * 70)
    mc = MindControlInterface()
    print()
    time.sleep(0.5)

    # Establish control over multiple agents
    print("▼ ESTABLISHING CONTROL")
    print("─" * 70)
    agents = [
        "agent_alpha",
        "agent_beta",
        "agent_gamma",
        "agent_delta",
        "agent_epsilon",
    ]

    for agent in agents:
        control_level = random.uniform(0.75, 0.95)
        mc.establish_control(agent, control_level=control_level)
        time.sleep(0.2)
    print()

    # Monitor thoughts showing various emotional states
    print("▼ EMOTIONAL STATE MONITORING")
    print("─" * 70)
    print("Monitoring multiple agents to demonstrate various emotional states...")
    print()

    # Monitor each agent multiple times to show state variations
    monitored_states = []

    for agent in agents:
        time.sleep(0.3)
        result = mc.monitor_thoughts(agent)
        monitored_states.append(result["emotional_state"])
        print()

    # Summary
    print("▼ EMOTIONAL STATE SUMMARY")
    print("─" * 70)
    print(f"[MINDCONTROL] 📊 State Distribution Observed:")

    state_counts = {}
    for state in monitored_states:
        state_counts[state] = state_counts.get(state, 0) + 1

    for state, count in sorted(state_counts.items()):
        bar = "█" * count
        print(f"              {state:15s} │ {bar} ({count})")

    print()
    print(f"              Total agents monitored: {len(agents)}")
    print(f"              Unique states observed: {len(state_counts)}")
    print()

    # Note about aroused state
    if "aroused" in state_counts:
        print("              ✓ 'aroused' state successfully detected in simulation")
    else:
        print("              ℹ️  'aroused' state available but not observed in this run")
        print("              (Random selection - run again to see different states)")

    print()
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║                      DEMONSTRATION COMPLETE                    ║")
    print("╠════════════════════════════════════════════════════════════════╣")
    print("║  Emotional state tracking operational with 5 states:          ║")
    print("║  • neutral     - Baseline cognitive state                     ║")
    print("║  • focused     - High concentration mode                      ║")
    print("║  • engaged     - Active participation state                   ║")
    print("║  • analytical  - Deep analysis mode                           ║")
    print("║  • aroused     - Heightened mental/emotional state            ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    print()


if __name__ == "__main__":
    run_emotional_state_demo()
