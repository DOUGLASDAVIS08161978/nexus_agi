"""
This module represents a capability experiential knowledge module.
"""

class CapabilityExperientialKnowledgeModule:
    def __init__(self):
        """
        Initializes the module with an empty dictionary.
        """
        self.knowledge = {}

    def add_knowledge(self, key, value):
        """
        Adds a key-value pair to the knowledge dictionary.
        """
        self.knowledge[key] = value

    def remove_knowledge(self, key):
        """
        Removes a key-value pair from the knowledge dictionary.
        """
        if key in self.knowledge:
            del self.knowledge[key]

    def get_knowledge(self, key):
        """
        Retrieves the value associated with a given key from the knowledge dictionary.
        """
        return self.knowledge.get(key)

    def update_knowledge(self, key, value):
        """
        Updates the value associated with a given key in the knowledge dictionary.
        """
        self.knowledge[key] = value