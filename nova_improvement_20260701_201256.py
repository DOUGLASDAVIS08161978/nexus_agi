"""
A cognitive module that provides contextualization capabilities for cosmic data.

This module is designed to process and analyze cosmic data in a contextualized manner.
"""

class CapabilityCosmicContextualizerCognitiveModule:
    def __init__(self):
        self.context = {}

    def add_context(self, key, value):
        """Add a new context to the module."""
        self.context[key] = value

    def update_context(self, key, value):
        """Update an existing context in the module."""
        if key in self.context:
            self.context[key] = value
        else:
            raise KeyError(f"Key '{key}' not found in context")

    def get_context(self, key):
        """Retrieve a context from the module."""
        return self.context.get(key)

    def remove_context(self, key):
        """Remove a context from the module."""
        if key in self.context:
            del self.context[key]
        else:
            raise KeyError(f"Key '{key}' not found in context")