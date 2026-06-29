"""
nova_cap_self_scheduler.py
Nova ASI — Self Scheduler
Generated via /build · v29 pipeline · 2026-06-29
"""

"""Autonomous capability scheduling and cognitive load balancing for Nova ASI."""

import time
import heapq
from collections import defaultdict
from datetime import datetime, timedelta


class CapabilitySelfSchedulerCognitiveModule:
    """Manages autonomous scheduling of Nova's capability research and self-improvement cycles."""

    def __init__(self):
        """Initialize scheduler with priority queues and timing state."""
        self.task_queue = []
        self.completed_tasks = defaultdict(list)
        self.cognitive_load = 0.0
        self.last_cycle_time = time.time()
        self.cycle_intervals = {
            'curiosity_research': 900,
            'hypothesis_testing': 2700,
            'self_improvement': 2700,
            'merge_watch': 300
        }

    def schedule_autonomous_cycle(self, cycle_type, priority=5):
        """Enqueue a capability cycle with priority and timestamp."""
        task = (priority, time.time(), cycle_type)
        heapq.heappush(self.task_queue, task)

    def get_next_task(self):
        """Pop and return the highest-priority pending task from queue."""
        if self.task_queue:
            return heapq.heappop(self.task_queue)
        return None

    def update_cognitive_load(self, delta):
        """Adjust current cognitive load based on task complexity and completion."""
        self.cognitive_load = max(0.0, min(1.0, self.cognitive_load + delta))

    def log_task_completion(self, cycle_type, duration_seconds):
        """Record completed task with duration for learning and optimization."""
        self.completed_tasks[cycle_type].append({
            'timestamp': datetime.now(),
            'duration': duration_seconds,
            'load': self.cognitive_load
        })