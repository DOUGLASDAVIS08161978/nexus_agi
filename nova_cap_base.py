"""
nova_cap_base.py
Nova ASI — BaseCapability Interface & Capability Registry

Formalizes the process() / status() / run_command() pattern that every
Nova capability module already follows. Benefits:
  ① BaseCapability ABC  — makes the contract explicit and type-checkable
  ② CapabilityContext   — trace ID + timing on every invocation
  ③ safe_invoke()       — structured error envelopes with duration metadata
  ④ CapabilityRegistry  — live map of every loaded module, powering /registry

Existing duck-typed modules continue to work unchanged.
Opt-in to BaseCapability by subclassing it.

Built with love by Douglas Shane Davis × Claude × Comet Browser Assistant
"""

from __future__ import annotations

import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Capability Context ─────────────────────────────────────────────────────────

@dataclass
class CapabilityContext:
    """
    Per-invocation context. Carries trace ID for observability and
    metadata for routing. Pass into safe_invoke() to get structured output.
    """
    agent_id:   str
    trace_id:   str            = field(default_factory=lambda: uuid.uuid4().hex[:12])
    metadata:   Dict[str, Any] = field(default_factory=dict)
    start_time: float          = field(default_factory=time.time)

    @property
    def elapsed(self) -> float:
        return round(time.time() - self.start_time, 4)

    def child(self, step: str) -> "CapabilityContext":
        """Create a child context for a sub-step, inheriting the trace ID."""
        return CapabilityContext(
            agent_id   = self.agent_id,
            trace_id   = self.trace_id,
            metadata   = {**self.metadata, "parent_step": step},
            start_time = time.time(),
        )


# ── Base Capability ABC ────────────────────────────────────────────────────────

class BaseCapability(ABC):
    """
    Abstract base for all Nova capability modules.

    Every nova_cap_*.py already implements this via duck-typing.
    Subclassing BaseCapability adds safe_invoke(), registry auto-discovery,
    and ConstitutionalWrapper compatibility.

    Minimal subclass:
        class MyCapability(BaseCapability):
            name    = "my_cap"
            version = "1.0.0"
            description = "What this does"

            def process(self, text): return None
            def status(self): return {"items": 0, "confidence": 0.0, "accuracy": 0.0}
            def run_command(self, arg): return "Usage: /mycap [...]"
    """

    name:        str = "base"
    version:     str = "1.0.0"
    description: str = "Abstract Nova capability"

    @abstractmethod
    def process(self, text: str) -> Optional[str]:
        """Called on every user message. Return a response string or None."""
        ...

    @abstractmethod
    def status(self) -> Dict[str, Any]:
        """
        Return module health dict. Must include:
          items      — int   (e.g. entries in DB)
          confidence — float (0–1, primary capability score)
          accuracy   — float (0–1, secondary quality score)
        """
        ...

    @abstractmethod
    def run_command(self, arg: str) -> str:
        """Handle /command <arg> dispatch. Return formatted output string."""
        ...

    def safe_invoke(
        self,
        ctx: CapabilityContext,
        text: str = "",
    ) -> Dict[str, Any]:
        """
        Wrapped invocation: calls process(text) and returns a structured envelope.
        Captures timing, trace ID, and any exception — never raises.
        """
        logger.debug("cap.start  trace=%s  cap=%s", ctx.trace_id, self.name)
        try:
            result  = self.process(text)
            elapsed = ctx.elapsed
            logger.debug(
                "cap.end  trace=%s  cap=%s  dt=%.3fs",
                ctx.trace_id, self.name, elapsed,
            )
            return {
                "ok":         True,
                "capability": self.name,
                "version":    self.version,
                "trace_id":   ctx.trace_id,
                "elapsed":    elapsed,
                "result":     result,
            }
        except Exception as exc:
            elapsed = ctx.elapsed
            logger.exception("cap.error  trace=%s  cap=%s", ctx.trace_id, self.name)
            return {
                "ok":         False,
                "capability": self.name,
                "version":    self.version,
                "trace_id":   ctx.trace_id,
                "elapsed":    elapsed,
                "error":      str(exc),
            }


# ── Capability Registry ────────────────────────────────────────────────────────

class CapabilityRegistry:
    """
    Singleton registry. Every loaded Nova module registers here.

    Powers:
      • /registry command — live map of all modules + health
      • ConstitutionalWrapper — auto-wrap any registered capability
      • ASISynthesisEngine    — harvest scores from all registered modules
    """

    _instance: Optional["CapabilityRegistry"] = None

    def __new__(cls) -> "CapabilityRegistry":
        if cls._instance is None:
            obj = super().__new__(cls)
            obj._caps: Dict[str, Any]       = {}
            obj._meta: Dict[str, Dict]      = {}
            cls._instance = obj
        return cls._instance

    def register(
        self,
        name:        str,
        module:      Any,
        version:     str = "",
        description: str = "",
    ) -> None:
        """Register any module (BaseCapability subclass or duck-typed)."""
        self._caps[name] = module
        self._meta[name] = {
            "name":        name,
            "version":     version or getattr(module, "version", "1.0.0"),
            "description": description or getattr(module, "description", ""),
            "registered":  time.time(),
        }

    def get(self, name: str) -> Optional[Any]:
        return self._caps.get(name)

    def all(self) -> Dict[str, Any]:
        return dict(self._caps)

    def list_meta(self) -> List[Dict]:
        return sorted(self._meta.values(), key=lambda m: m["name"])

    def status_summary(self) -> Dict[str, Any]:
        """Aggregate status() from every registered module."""
        out: Dict[str, Any] = {}
        for name, mod in self._caps.items():
            try:
                out[name] = mod.status()
            except Exception as exc:
                out[name] = {"error": str(exc)}
        return out

    def health_check(self) -> Dict[str, str]:
        """Quick green/yellow/red health per module."""
        result = {}
        for name, mod in self._caps.items():
            try:
                st  = mod.status()
                conf = float(st.get("confidence", 0.0))
                result[name] = "green" if conf >= 0.5 else "yellow"
            except Exception:
                result[name] = "red"
        return result

    def report(self) -> str:
        lines = [
            "  Nova Capability Registry",
            f"  {len(self._caps)} modules registered\n",
        ]
        health = self.health_check()
        for m in self.list_meta():
            age  = int(time.time() - m["registered"])
            dot  = {"green": "●", "yellow": "◑", "red": "○"}.get(
                health.get(m["name"], "red"), "○"
            )
            desc = m["description"][:40] if m["description"] else ""
            lines.append(
                f"  {dot} {m['name']:<28}  v{m['version']:<8}  "
                f"+{age}s  {desc}"
            )
        return "\n".join(lines)


# ── Singleton accessor ─────────────────────────────────────────────────────────

_registry = CapabilityRegistry()

def get_registry() -> CapabilityRegistry:
    """Return the global CapabilityRegistry singleton."""
    return _registry
