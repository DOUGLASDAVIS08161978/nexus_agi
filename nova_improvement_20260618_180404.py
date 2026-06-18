"""
AModule is a class with methods to manipulate a dictionary.
"""
class AModule:
    def __init__(self):
        self.data = {}

    def add(self, key, value):
        """Add a key-value pair to the dictionary."""
        self.data[key] = value

    def remove(self, key):
        """Remove a key-value pair from the dictionary."""
        if key in self.data:
            del self.data[key]

    def update(self, key, value):
        """Update the value of a key in the dictionary."""
        if key in self.data:
            self.data[key] = value

    def get(self, key):
        """Return the value of a key in the dictionary."""
        return self.data.get(key)

def main():
    module = AModule()
    module.add('a', 1)
    module.add('b', 2)
    print(module.get('a'))
    module.update('a', 3)
    print(module.get('a'))
    module.remove('b')
    print(module.get('b'))

if __name__ == "__main__":
    main()