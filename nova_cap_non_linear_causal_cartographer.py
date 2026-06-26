"""
nova_cap_non_linear_causal_cartographer.py
Nova invented this autonomously — Non-Linear Causal Cartographer
Generated via /evolve · v29 pipeline · 2026-06-26
"""

"""
CapabilityNonlinearCausalModule is a class that represents a nonlinear causal module.
It has methods to add, remove, update and get capabilities.
"""
class CapabilityNonlinearCausalModule:
    def __init__(self):
        self.capabilities = {}

    def add_capability(self, name, value):
        """Add a new capability to the module."""
        self.capabilities[name] = value

    def remove_capability(self, name):
        """Remove a capability from the module."""
        if name in self.capabilities:
            del self.capabilities[name]

    def update_capability(self, name, value):
        """Update the value of an existing capability."""
        if name in self.capabilities:
            self.capabilities[name] = value

    def get_capability(self, name):
        """Get the value of a capability."""
        return self.capabilities.get(name)
