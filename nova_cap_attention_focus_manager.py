"""
Nova ASI — Attention & Focus Manager
Proposed autonomously via /evolve
"""

"""
This module adds an enhanced AttentionManager class with functionality for timed focus windows, active topic tracking, and interruption logging.
"""

import threading
import time
import datetime

class EnhancedAttentionManager:
    """
    Enhanced AttentionManager class with timed focus windows, active topic tracking, and interruption logging.
    """
    
    def __init__(self):
        """
        Initialize the EnhancedAttentionManager instance.
        """
        self.focus_topic = None
        self.focus_end_time = None
        self.distraction_log = []
        self.current_focus_thread = None

    def focus_on(self, topic, minutes):
        """
        Set a timed focus window on the specified topic.

        Args:
            topic (str): The topic to focus on.
            minutes (int): The number of minutes to focus on the topic.

        Returns:
            None
        """
        self.focus_topic = topic
        self.focus_end_time = datetime.datetime.now() + datetime.timedelta(minutes=minutes)
        self.current_focus_thread = threading.Thread(target=self.update_focus_status)
        self.current_focus_thread.daemon = True
        self.current_focus_thread.start()

    def update_focus_status(self):
        """
        Update the focus status every minute.

        Returns:
            None
        """
        while datetime.datetime.now() < self.focus_end_time:
            time.sleep(60)
            if not self.current_focus():
                self.distraction_log_now(f"Lost focus on {self.focus_topic} at {datetime.datetime.now()}")

    def current_focus(self):
        """
        Get the currently active topic.

        Returns:
            str: The currently active topic, or None if no focus is set.
        """
        if self.focus_end_time and self.focus_topic:
            if datetime.datetime.now() < self.focus_end_time:
                return self.focus_topic
        return None

    def distraction_log_now(self, note):
        """
        Log a distraction event with the current timestamp.

        Args:
            note (str): The distraction note.

        Returns:
            None
        """
        self.distraction_log.append((datetime.datetime.now(), note))

# Usage example:
# attention_manager = EnhancedAttentionManager()
# attention_manager.focus_on("Task A", 30)
# print(attention_manager.current_focus())  # Output: Task A
# time.sleep(31)
# print(attention_manager.current_focus())  # Output: None