"""
nova_cap_adversarial_self_deception_detector.py
Nova invented this autonomously — Adversarial Self-Deception Detector
Generated via /evolve · v29 pipeline · 2026-06-22
"""

"""
This is a class named of.
"""
class of:
    def __init__(self):
        self.data = {}

    def add(self, key, value):
        """Add a key-value pair to the dictionary."""
        self.data[key] = value

    def get(self, key):
        """Retrieve the value associated with the given key."""
        return self.data.get(key)

    def remove(self, key):
        """Remove the key-value pair with the given key."""
        if key in self.data:
            del self.data[key]

    def update(self, key, value):
        """Update the value associated with the given key."""
        if key in self.data:
            self.data[key] = value
        else:
            raise KeyError("Key not found")