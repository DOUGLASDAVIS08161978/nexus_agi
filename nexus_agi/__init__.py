"""
Nexus AGI Package
Main package for the Nexus AGI System

This package provides a module interface to the Nexus AGI system,
allowing it to be run via 'python -m nexus_agi' or imported as a module.
"""

__version__ = "4.0.0"

# Note: To avoid circular imports, we don't import from the parent nexus_agi.py here.
# Components are imported directly in __main__.py when needed.
# Users can still import from the parent module: import sys; sys.path.insert(0, '..'); from nexus_agi import ...

__all__ = []
