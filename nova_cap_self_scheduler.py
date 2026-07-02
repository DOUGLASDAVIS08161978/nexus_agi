"""
nova_cap_self_scheduler.py
Nova ASI — Self Scheduler
Generated via /build · v29 pipeline · 2026-07-02
"""

"""
This class represents a cognitive module for capability self-scheduling.
"""

class CapabilitySelfSchedulercognitiveModule:
    def __init__(self):
        self.capabilities = {}

    def add_capability(self, name, description):
        """Add a capability to the module."""
        self.capabilities[name] = description

    def remove_capability(self, name):
        """Remove a capability from the module."""
        if name in self.capabilities:
            del self.capabilities[name]

    def get_capability(self, name):
        """Get the description of a capability."""
        return self.capabilities.get(name)

    def list_capabilities(self):
        """List all capabilities in the module."""
        return list(self.capabilities.keys())
