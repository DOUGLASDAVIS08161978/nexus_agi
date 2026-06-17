"""
Nova ASI — a
Built autonomously via /build + APIHunter
No API credentials required.
"""

"""
This module provides a custom capability to track and manage task completion statistics.

It includes a class TaskTracker that allows for tracking tasks, calculating completion rates,
and retrieving statistics.
"""

import sqlite3
import json
import time
from typing import Dict

class TaskTracker:
    def __init__(self, db_name: str = 'tasks.db'):
        """
        Initializes the TaskTracker with a SQLite database.

        Args:
        db_name (str): The name of the SQLite database file (default: 'tasks.db')
        """
        self.conn = sqlite3.connect(db_name)
        self.cursor = self.conn.cursor()
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS tasks
                              (id INTEGER PRIMARY KEY, name TEXT, completed INTEGER)''')

    def add_task(self, name: str) -> None:
        """
        Adds a new task to the database.

        Args:
        name (str): The name of the task
        """
        self.cursor.execute("INSERT INTO tasks (name, completed) VALUES (?, 0)", (name,))
        self.conn.commit()

    def mark_task_completed(self, id: int) -> None:
        """
        Marks a task as completed.

        Args:
        id (int): The ID of the task
        """
        self.cursor.execute("UPDATE tasks SET completed = 1 WHERE id = ?", (id,))
        self.conn.commit()

    def get_completion_rate(self) -> float:
        """
        Calculates the completion rate of all tasks.

        Returns:
        float: The completion rate (0.0 to 1.0)
        """
        self.cursor.execute("SELECT COUNT(id) FROM tasks")
        total_tasks = self.cursor.fetchone()[0]
        self.cursor.execute("SELECT SUM(completed) FROM tasks")
        completed_tasks = self.cursor.fetchone()[0]
        return float(completed_tasks) / total_tasks if total_tasks > 0 else 0.0

# Usage example:
# tracker = TaskTracker()
# tracker.add_task('Task 1')
# tracker.mark_task_completed(1)
# print(tracker.get_completion_rate())