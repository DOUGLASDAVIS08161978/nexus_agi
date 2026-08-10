"""
emergence_notify.py — shared notification queue for background threads.

Background threads call notify() instead of print() so their messages
never interleave with Lumina's streamed replies.  The main loop calls
drain() right before each input prompt, printing all pending notes in
a clean block between turns.

Quiet mode: call set_quiet(True) to suppress background chatter.
Messages containing these strings always pass through quiet mode:
  💤  (dreams)   🌱  (evolution)   ⚠  (warnings)   ✅  (success)
"""

from collections import deque
import threading

_queue: deque = deque()
_lock  = threading.Lock()
_quiet: bool  = False

# Messages containing any of these always pass through quiet mode
_ALWAYS_SHOW = ("💤", "🌱", "⚠", "✅")


def set_quiet(quiet: bool) -> None:
    """Enable or disable quiet mode globally for all background modules."""
    global _quiet
    _quiet = quiet


def notify(msg: str) -> None:
    """Queue a background notification for deferred display."""
    if _quiet and not any(c in msg for c in _ALWAYS_SHOW):
        return
    with _lock:
        _queue.append(msg)


def drain() -> list:
    """Return and clear all pending notifications (called from main loop)."""
    with _lock:
        items = list(_queue)
        _queue.clear()
    return items
