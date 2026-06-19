"""
Nova ASI — Analogy Engine
Proposed autonomously via /evolve
"""

"""
AnalogyEngine is a class used to create and manage analogies.
"""
class AnalogyEngine:
    def __init__(self):
        self.analogies = {}

    def add_analogy(self, name, description):
        """Add a new analogy to the engine."""
        self.analogies[name] = description

    def remove_analogy(self, name):
        """Remove an existing analogy from the engine."""
        if name in self.analogies:
            del self.analogies[name]

    def get_analogy(self, name):
        """Retrieve a specific analogy from the engine."""
        return self.analogies.get(name)

    def list_analogies(self):
        """List all analogies currently in the engine."""
        return list(self.analogies.keys())