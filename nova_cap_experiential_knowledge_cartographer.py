"""
nova_cap_experiential_knowledge_cartographer.py
Nova invented this autonomously — Experiential Knowledge Cartographer
Generated via /evolve · v29 pipeline · 2026-06-26
"""

"""
CapabilityExperientialKnowledgeModule is a class that encapsulates knowledge and experience.
"""
class CapabilityExperientialKnowledgeModule:
    def __init__(self):
        self.knowledge_base = {}

    def add_knowledge(self, key, value):
        """Adds new knowledge to the knowledge base."""
        self.knowledge_base[key] = value

    def retrieve_knowledge(self, key):
        """Retrieves knowledge from the knowledge base by key."""
        return self.knowledge_base.get(key)

    def update_knowledge(self, key, value):
        """Updates existing knowledge in the knowledge base."""
        if key in self.knowledge_base:
            self.knowledge_base[key] = value

    def delete_knowledge(self, key):
        """Deletes knowledge from the knowledge base by key."""
        if key in self.knowledge_base:
            del self.knowledge_base[key]
