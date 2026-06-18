"""
Nova ASI — Persistent Long-Term Memory
Proposed autonomously via /evolve
"""

"""
NovaMemoryStore is a class for storing and retrieving data in memory.
"""
class NovaMemoryStore:
    def __init__(self):
        self.data = {}

    def get(self, key):
        """Retrieve a value from the store by its key."""
        return self.data.get(key)

    def set(self, key, value):
        """Set a value in the store by its key."""
        self.data[key] = value

    def delete(self, key):
        """Delete a value from the store by its key."""
        if key in self.data:
            del self.data[key]

    def exists(self, key):
        """Check if a key exists in the store."""
        return key in self.data

    def keys(self):
        """Return a list of all keys in the store."""
        return list(self.data.keys())

    def values(self):
        """Return a list of all values in the store."""
        return list(self.data.values())

    def items(self):
        """Return a list of all key-value pairs in the store."""
        return list(self.data.items())