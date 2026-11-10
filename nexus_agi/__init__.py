"""
Nexus AGI Package
Main package for the Nexus AGI System

This package provides a module interface to the Nexus AGI system,
allowing it to be run via 'python -m nexus_agi' or imported as a module.

NOTE on the name collision (fixed 2026-11-10): the implementation lives in
the sibling file nexus_agi.py, and this package directory shares its name.
Python resolves `import nexus_agi` to the package, so `from nexus_agi
import MetaAlgorithm_NexusCore` used to fail even though the class exists
in the file — which broke the service loop and every other importer.
This module lazily re-exports the core file's classes (PEP 562
__getattr__), so existing importers work unchanged. The core file is
loaded only on first access, so `import nexus_agi` stays light and does
not require the core's heavy stack (torch, transformers, ...) until the
core is actually used.
"""

__version__ = "4.0.0"

_CORE_MODULE_NAME = "_nexus_agi_core_file"
_CORE_MODULE = None

# Public classes defined in the sibling nexus_agi.py. Checked before loading
# so a typo raises AttributeError immediately instead of importing the heavy
# core stack first. Keep in sync with the class list in nexus_agi.py.
_CORE_EXPORTS = frozenset({
    "OpenQuantumSimulator",
    "SymbolicReasoner",
    "NeuralProcessor",
    "NeuroAxiomaticFusionEngine",
    "ArchitectureTemplate",
    "HoloConceptEngine",
    "ConscientiaLattice",
    "InfiniteMetaLoop",
    "SimuVerse",
    "EmpathyNet",
    "NeuralManifoldExplorer",
    "KnowledgeCrystallization",
    "MetaAlgorithm_NexusCore",
    "PositionalEncoding",
    "BayesianLinear",
})


def _load_core():
    """Load the sibling nexus_agi.py file once, and cache it."""
    global _CORE_MODULE
    if _CORE_MODULE is None:
        import importlib.util
        import os

        core_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "nexus_agi.py",
        )
        spec = importlib.util.spec_from_file_location(_CORE_MODULE_NAME, core_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _CORE_MODULE = module
    return _CORE_MODULE


def __getattr__(name):
    """Re-export public classes from the sibling nexus_agi.py on demand."""
    if name not in _CORE_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(_load_core(), name)


__all__ = ["MetaAlgorithm_NexusCore"]
