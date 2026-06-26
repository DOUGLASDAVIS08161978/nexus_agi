"""
Nova ASI — Semantic Memory Indexer
Proposed autonomously via /evolve
"""

class SemanticIndex:
    def __init__(self):
        self.index = {}

    def add_term(self, term, description):
        """Add a term and its description to the index."""
        self.index[term] = description

    def get_description(self, term):
        """Retrieve the description of a term from the index."""
        return self.index.get(term)

    def remove_term(self, term):
        """Remove a term and its description from the index."""
        if term in self.index:
            del self.index[term]
