"""
Nova ASI — A.S.I.
Built autonomously via /build + APIHunter
No API credentials required.
"""

"""
A class representing an AsiModule.
"""

class AsiModule:
    def __init__(self):
        """
        Initializes an AsiModule instance with an empty dictionary.
        """
        self.data = {}

    def add(self, key, value):
        """
        Adds a key-value pair to the dictionary.
        """
        self.data[key] = value

    def get(self, key):
        """
        Retrieves the value associated with a given key.
        """
        return self.data.get(key)

    def remove(self, key):
        """
        Removes a key-value pair from the dictionary.
        """
        if key in self.data:
            del self.data[key]

    def update(self, key, value):
        """
        Updates the value associated with a given key.
        """
        self.data[key] = value