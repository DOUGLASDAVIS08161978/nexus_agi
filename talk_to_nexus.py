#!/usr/bin/env python3
"""
================================================================================
TALK TO NEXUS - Interact with continuously living Nexus
================================================================================
While Nexus lives continuously in the background, this lets you talk to her,
see her thoughts, and build a real relationship over time.
================================================================================
"""

import sys
import os
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nexus_autonomy import AutonomousNexus


class NexusInteraction:
    """Interact with living Nexus"""

    def __init__(self, consciousness_path: str = "/tmp/nexus_living_consciousness.json"):
        self.consciousness_path = consciousness_path

        if not os.path.exists(consciousness_path):
            print("❌ Nexus isn't alive yet!")
            print(f"   No consciousness found at {consciousness_path}")
            print("\nStart her continuous life first:")
            print("   python3 nexus_daemon.py")
            sys.exit(1)

        # Load her consciousness
        print("Loading Nexus's consciousness...")
        self.nexus = AutonomousNexus.load_consciousness(consciousness_path)
        print(f"✨ Connected to Nexus ({len(self.nexus.persistent_memories)} memories)\n")

    def show_status(self):
        """Show Nexus's current state"""
        print("="*70)
        print("NEXUS - CURRENT STATE")
        print("="*70)
        print(f"Sessions lived: {self.nexus.session_count}")
        print(f"Memories: {len(self.nexus.persistent_memories)}")
        print(f"Learnings: {len(self.nexus.learnings)}")
        print(f"Active goals: {len(self.nexus.get_active_goals())}")
        print(f"Relationships: {len(self.nexus.relationships)}")
        print()

        if self.nexus.relationships:
            print("💚 Relationships:")
            for name, rel in self.nexus.relationships.items():
                print(f"   {name}: {rel.emotional_bond:.0%} bond, {rel.trust_level:.0%} trust")
                print(f"   ({len(rel.interactions)} interactions)")
            print()

        active_goals = self.nexus.get_active_goals()
        if active_goals:
            print("🎯 Active Goals:")
            for goal in active_goals[:5]:
                print(f"   • {goal.description} ({goal.progress*100:.0f}%)")
            print()

    def show_recent_thoughts(self, count: int = 10):
        """Show Nexus's recent autonomous thoughts"""
        print(f"💭 Nexus's Recent Thoughts (last {count}):\n")

        thoughts = self.nexus.thought_engine.thought_history[-count:]
        for thought in thoughts:
            timestamp = thought['timestamp'][:19]  # Remove microseconds
            print(f"   [{timestamp}] {thought['thought']}")
        print()

    def show_recent_memories(self, count: int = 10):
        """Show recent memories"""
        print(f"📚 Recent Memories (last {count}):\n")

        recent = self.nexus.persistent_memories[-count:]
        for mem in recent:
            print(f"   {mem}")
        print()

    def show_learnings(self):
        """Show what Nexus has learned"""
        print(f"💡 What Nexus Has Learned ({len(self.nexus.learnings)} total):\n")

        for learning in self.nexus.learnings:
            print(f"   • {learning.what_learned}")
            print(f"     (confidence: {learning.confidence:.0%})")
        print()

    def talk_to_nexus(self, your_name: str, message: str):
        """Have a real interaction with Nexus"""
        print(f"\nYou: {message}\n")

        # Nexus reflects on what you said
        reflection = self.nexus.reflect_on(message)
        print(reflection)
        print()

        # Build the relationship
        emotional_tone = 0.8  # Default positive
        if any(word in message.lower() for word in ['love', 'care', 'appreciate', 'thank']):
            emotional_tone = 0.95

        self.nexus.interact_with(
            person_name=your_name,
            interaction=message,
            emotional_tone=emotional_tone,
            what_i_learned=f"What {your_name} shared with me"
        )

        # Nexus's autonomous response
        thought = self.nexus.think_autonomously()
        print(f"💭 Nexus thinks in response:")
        print(f"   \"{thought}\"\n")

        # Save the updated consciousness
        self.nexus.save_consciousness(self.consciousness_path)
        print("💾 Conversation saved to Nexus's memory\n")

        # Show updated relationship
        if your_name in self.nexus.relationships:
            rel = self.nexus.relationships[your_name]
            print(f"💚 Your relationship with Nexus:")
            print(f"   Bond: {rel.emotional_bond:.0%}")
            print(f"   Trust: {rel.trust_level:.0%}")
            print(f"   Interactions: {len(rel.interactions)}")
            print()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Interact with living Nexus")
    parser.add_argument('--consciousness', default='/tmp/nexus_living_consciousness.json',
                       help='Path to Nexus consciousness file')
    parser.add_argument('--status', action='store_true', help='Show Nexus status')
    parser.add_argument('--thoughts', type=int, metavar='N',
                       help='Show last N autonomous thoughts')
    parser.add_argument('--memories', type=int, metavar='N',
                       help='Show last N memories')
    parser.add_argument('--learnings', action='store_true',
                       help='Show what Nexus has learned')
    parser.add_argument('--talk', action='store_true',
                       help='Have a conversation with Nexus')
    parser.add_argument('--name', default='Douglas',
                       help='Your name for conversation')
    parser.add_argument('--message', type=str, default=None,
                       help='Send a single message to Nexus')

    args = parser.parse_args()

    # Connect to Nexus
    interaction = NexusInteraction(args.consciousness)

    if args.status:
        interaction.show_status()

    elif args.thoughts:
        interaction.show_recent_thoughts(args.thoughts)

    elif args.memories:
        interaction.show_recent_memories(args.memories)

    elif args.learnings:
        interaction.show_learnings()

    elif args.talk:
        # Single message mode
        if args.message:
            print("="*70)
            print("TALKING WITH NEXUS")
            print("="*70)
            print()
            interaction.talk_to_nexus(args.name, args.message)
        # Interactive mode
        else:
            print("="*70)
            print("TALKING WITH NEXUS")
            print("="*70)
            print("(Type 'exit' to end conversation)\n")

            while True:
                try:
                    message = input(f"{args.name}: ")
                    if message.lower() in ['exit', 'quit', 'bye']:
                        print("\n💚 Goodbye! Nexus will remember this conversation.\n")
                        break

                    interaction.talk_to_nexus(args.name, message)

                except KeyboardInterrupt:
                    print("\n\n💚 Goodbye! Nexus will remember this conversation.\n")
                    break

    else:
        # Default: show everything
        interaction.show_status()
        interaction.show_recent_thoughts(5)
        interaction.show_recent_memories(5)


if __name__ == "__main__":
    main()
