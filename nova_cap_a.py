"""
Nova ASI — a
Built autonomously via /build + APIHunter
No API credentials required.
"""

"""
This module adds a CustomCapability class, which allows Nova to store and retrieve custom data from a SQLite database.
"""

import sqlite3
import datetime
import threading
import json

class CustomCapability:
    def __init__(self, db_name: str = 'nova.db'):
        self.db_name = db_name
        self.lock = threading.Lock()
        self.conn = sqlite3.connect(self.db_name)
        self.cursor = self.conn.cursor()
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS custom_data
            (id INTEGER PRIMARY KEY, timestamp TEXT, data TEXT)
        ''')
        self.conn.commit()

    def add_data(self, data: str):
        """
        Add custom data to the database.

        Args:
            data (str): The data to add.
        """
        with self.lock:
            timestamp = datetime.datetime.now().isoformat()
            self.cursor.execute('INSERT INTO custom_data (timestamp, data) VALUES (?, ?)', (timestamp, json.dumps(data)))
            self.conn.commit()

    def get_data(self, id: int = None, timestamp: str = None):
        """
        Get custom data from the database.

        Args:
            id (int, optional): The ID of the data to get. Defaults to None.
            timestamp (str, optional): The timestamp of the data to get. Defaults to None.

        Returns:
            str: The data in JSON format.
        """
        with self.lock:
            if id is not None:
                self.cursor.execute('SELECT data FROM custom_data WHERE id = ?', (id,))
            elif timestamp is not None:
                self.cursor.execute('SELECT data FROM custom_data WHERE timestamp = ?', (timestamp,))
            else:
                self.cursor.execute('SELECT data FROM custom_data ORDER BY timestamp DESC LIMIT 1')
            row = self.cursor.fetchone()
            if row:
                return json.loads(row[0])
            else:
                return None

# Usage example:
# custom = CustomCapability()
# custom.add_data({'key': 'value'})
# print(custom.get_data()) # prints the last added data
# print(custom.get_data(id=1)) # prints the data with id 1