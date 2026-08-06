"""
emergence_notify.py — shared notification queue for background threads.

Background threads call notify() instead of print() so their messages
never interleave with Lumina's streamed replies.  The main loop calls
drain() right before each input prompt, printing all pending notes in
a clean block between turns.
"""

from collections import deque
import threading

_queue: deque = deque()
_lock  = threading.Lock()


def notify(msg: str) -> None:
    """Queue a background notification for deferred display."""
    with _lock:
        _queue.append(msg)


def drain() -> list:
    """Return and clear all pending notifications (called from main loop)."""
    with _lock:
        items = list(_queue)
        _queue.clear()
    return items
