"""
nova_cap_self_reflection_and_metacognition_contin.py
Nova ASI — self-reflection and metacognition: continuous audit of Nova's own blind spots and biases
Generated via /evolve · v29 pipeline · 2026-06-22
"""

"""
CapabilitySelfreflectionAndModule is a class that provides methods for self-reflection and module management.
It uses a dictionary to store module information.
"""
class CapabilitySelfreflectionAndModule:
    def __init__(self):
        self.modules = {}

    def add_module(self, name, description):
        # Add a module to the dictionary with its name and description.
        self.modules[name] = description

    def remove_module(self, name):
        # Remove a module from the dictionary by its name.
        if name in self.modules: del self.modules[name]

    def get_module(self, name):
        # Get a module's description from the dictionary by its name.
        return self.modules.get(name)

    def list_modules(self):
        # List all modules in the dictionary.
        return list(self.modules.keys())
