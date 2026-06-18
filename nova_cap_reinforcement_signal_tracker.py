"""
Nova ASI — Reinforcement Signal Tracker
Proposed autonomously via /evolve
"""

"""
RewardTracker class to track and manage rewards.
"""
class RewardTracker:
    def __init__(self):
        self.rewards = {}

    def add_reward(self, name, value):
        """Add a new reward to the tracker."""
        self.rewards[name] = value

    def remove_reward(self, name):
        """Remove a reward from the tracker."""
        if name in self.rewards:
            del self.rewards[name]

    def update_reward(self, name, value):
        """Update the value of an existing reward."""
        if name in self.rewards:
            self.rewards[name] = value

    def get_reward(self, name):
        """Get the value of a specific reward."""
        return self.rewards.get(name)

    def __str__(self):
        return str(self.rewards) 

    def __repr__(self):
        return f"RewardTracker({self.rewards})"