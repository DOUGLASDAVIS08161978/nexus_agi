"""
Nova ASI — general
Built autonomously via /build + APIHunter
No API credentials required.
"""

"""
GeneralModule is a class with methods for data manipulation.
"""
class GeneralModule:
    def __init__(self):
        self.data = {}

    def add_data(self, key, value):
        """Add data to the dictionary."""
        self.data[key] = value

    def remove_data(self, key):
        """Remove data from the dictionary."""
        if key in self.data:
            del self.data[key]

    def update_data(self, key, value):
        """Update existing data in the dictionary."""
        if key in self.data:
            self.data[key] = value

    def get_data(self, key):
        """Retrieve data from the dictionary."""
        return self.data.get(key)

    def display_data(self):
        """Print all data in the dictionary."""
        for key, value in self.data.items():
            print(f"{key}: {value}")

def main():
    module = GeneralModule()
    module.add_data("name", "John")
    module.add_data("age", 30)
    module.display_data()
    module.remove_data("age")
    module.display_data()

if __name__ == "__main__":
    main()