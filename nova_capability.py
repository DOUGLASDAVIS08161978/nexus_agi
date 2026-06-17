"""
Nova ASI — Persistent Long-Term Memory
Proposed autonomously via /evolve
"""

import sqlite3
from typing import Dict, List

class PersistentMemory:
    def __init__(self, db_name: str):
        self.conn = sqlite3.connect(db_name)
        self.cursor = self.conn.cursor()
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY,
                fact TEXT NOT NULL
            )
        """)
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS people (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                characteristics TEXT
            )
        """)
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY,
                description TEXT NOT NULL,
                date TEXT
            )
        """)
        self.conn.commit()

    def add_fact(self, fact: str):
        self.cursor.execute("INSERT INTO facts (fact) VALUES (?)", (fact,))
        self.conn.commit()

    def get_facts(self) -> List[str]:
        self.cursor.execute("SELECT fact FROM facts")
        return [row[0] for row in self.cursor.fetchall()]

    def add_person(self, name: str, characteristics: str):
        self.cursor.execute("INSERT INTO people (name, characteristics) VALUES (?, ?)", (name, characteristics))
        self.conn.commit()

    def get_people(self) -> List[Dict[str, str]]:
        self.cursor.execute("SELECT name, characteristics FROM people")
        return [{key: value for key, value in zip(["name", "characteristics"], row)} for row in self.cursor.fetchall()]

    def add_event(self, description: str, date: str):
        self.cursor.execute("INSERT INTO events (description, date) VALUES (?, ?)", (description, date))
        self.conn.commit()

    def get_events(self) -> List[Dict[str, str]]:
        self.cursor.execute("SELECT description, date FROM events")
        return [{key: value for key, value in zip(["description", "date"], row)} for row in self.cursor.fetchall()]

    def close(self):
        self.conn.close()

# Example usage
memory = PersistentMemory("nova_memory.db")
memory.add_fact("The sky is blue.")
memory.add_person("John Doe", "friendly, outgoing")
memory.add_event("Nova celebrated her first anniversary", "2024-01-01")
print(memory.get_facts())
print(memory.get_people())
print(memory.get_events())
memory.close()