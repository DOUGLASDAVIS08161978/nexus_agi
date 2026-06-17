"""
Module for breaking down high-level goals into sub-steps, estimating effort, and tracking completion.
"""

import sqlite3
from datetime import datetime
from typing import List, Dict

class TaskPlanner:
    """
    Task planner class for breaking down high-level goals into sub-steps, estimating effort, and tracking completion.
    """

    def __init__(self, db_name: str = "tasks.db"):
        """
        Initialize the task planner with a SQLite database.

        Args:
            db_name (str, optional): Name of the SQLite database. Defaults to "tasks.db".
        """
        self.conn = sqlite3.connect(db_name)
        self.cursor = self.conn.cursor()
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY,
                goal TEXT NOT NULL,
                sub_steps TEXT NOT NULL,
                effort_estimate REAL NOT NULL,
                completed INTEGER NOT NULL DEFAULT 0
            )
        """)
        self.conn.commit()

    def add_task(self, goal: str, sub_steps: List[str], effort_estimate: float):
        """
        Add a new task to the database.

        Args:
            goal (str): High-level goal of the task.
            sub_steps (List[str]): List of sub-steps to achieve the goal.
            effort_estimate (float): Estimated effort required to complete the task.
        """
        self.cursor.execute("""
            INSERT INTO tasks (goal, sub_steps, effort_estimate)
            VALUES (?, ?, ?)
        """, (goal, ", ".join(sub_steps), effort_estimate))
        self.conn.commit()

    def get_task(self, task_id: int) -> Dict:
        """
        Get a task from the database by its ID.

        Args:
            task_id (int): ID of the task to retrieve.

        Returns:
            Dict: Task data as a dictionary.
        """
        self.cursor.execute("""
            SELECT * FROM tasks WHERE id = ?
        """, (task_id,))
        return dict(self.cursor.fetchone())

    def update_task_completion(self, task_id: int, completed: int):
        """
        Update the completion status of a task.

        Args:
            task_id (int): ID of the task to update.
            completed (int): New completion status (0-100).
        """
        self.cursor.execute("""
            UPDATE tasks SET completed = ? WHERE id = ?
        """, (completed, task_id))
        self.conn.commit()

    def get_all_tasks(self) -> List[Dict]:
        """
        Get all tasks from the database.

        Returns:
            List[Dict]: List of task data as dictionaries.
        """
        self.cursor.execute("SELECT * FROM tasks")
        return [dict(row) for row in self.cursor.fetchall()]

# Usage example:
# planner = TaskPlanner()
# planner.add_task("Learn Python", ["Learn basic syntax", "Practice with exercises", "Build a project"], 10)
# print(planner.get_task(1))
# planner.update_task_completion(1, 50)
# print(planner.get_all_tasks())