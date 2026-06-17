"""
Nova ASI — a
Built autonomously via /build + APIHunter
No API credentials required.
"""

"""
This module adds a custom capability to Nova, allowing it to track and analyze user interactions.
"""

import sqlite3
from datetime import datetime
from typing import Dict, List

class InteractionTracker:
    def __init__(self, db_file: str):
        """
        Initializes the InteractionTracker with a SQLite database file.

        Args:
            db_file (str): The path to the SQLite database file.
        """
        self.conn = sqlite3.connect(db_file)
        self.cursor = self.conn.cursor()
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY,
                user_id TEXT,
                timestamp DATETIME,
                action TEXT,
                context TEXT
            )
        """)

    def log_interaction(self, user_id: str, action: str, context: Dict) -> None:
        """
        Logs an interaction with the specified user, action, and context.

        Args:
            user_id (str): The ID of the user.
            action (str): The action performed.
            context (Dict): Additional context about the interaction.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.cursor.execute("""
            INSERT INTO interactions (user_id, timestamp, action, context)
            VALUES (?, ?, ?, ?)
        """, (user_id, timestamp, action, json.dumps(context)))
        self.conn.commit()

    def get_user_interactions(self, user_id: str) -> List[Dict]:
        """
        Retrieves all interactions for the specified user.

        Args:
            user_id (str): The ID of the user.

        Returns:
            List[Dict]: A list of interaction dictionaries.
        """
        self.cursor.execute("""
            SELECT * FROM interactions WHERE user_id = ?
        """, (user_id,))
        interactions = self.cursor.fetchall()
        return [{'id': i[0], 'timestamp': i[1], 'action': i[2], 'context': json.loads(i[3])} for i in interactions]

# Usage example:
# tracker = InteractionTracker('interactions.db')
# tracker.log_interaction('user1', 'clicked_button', {'button_id': 'submit'})
# print(tracker.get_user_interactions('user1'))