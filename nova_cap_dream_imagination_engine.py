"""
Nova ASI — Dream / Imagination Engine
Proposed autonomously via /evolve
"""

"""
ImaginationEngine class: handles creative ideas and thoughts.
"""
class ImaginationEngine:
    def __init__(self):
        self.ideas = {}

    def add_idea(self, name, description):
        """Add a new idea to the imagination engine."""
        self.ideas[name] = description

    def remove_idea(self, name):
        """Remove an existing idea from the imagination engine."""
        if name in self.ideas:
            del self.ideas[name]

    def update_idea(self, name, description):
        """Update an existing idea in the imagination engine."""
        if name in self.ideas:
            self.ideas[name] = description

    def get_idea(self, name):
        """Retrieve a specific idea from the imagination engine."""
        return self.ideas.get(name)

    def list_ideas(self):
        """List all ideas in the imagination engine."""
        return list(self.ideas.keys())