"""
Nova ASI — Persistent Long-Term Memory
Proposed autonomously via /evolve
"""

import sqlite3
import os

class NovaMemoryStore:
    def __init__(self, db_name='nova.db'):
        self.conn = sqlite3.connect(db_name)
        self.cursor = self.conn.cursor()
        self.create_schema()

    def create_schema(self):
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY,
                fact TEXT
            )
        ''')
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS people (
                id INTEGER PRIMARY KEY,
                name TEXT,
                description TEXT
            )
        ''')
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY,
                event TEXT,
                date DATE
            )
        ''')
        self.conn.commit()

    def add_fact(self, fact):
        self.cursor.execute('INSERT INTO facts (fact) VALUES (?)', (fact,))
        self.conn.commit()

    def add_person(self, name, description):
        self.cursor.execute('INSERT INTO people (name, description) VALUES (?, ?)', (name, description))
        self.conn.commit()

    def add_event(self, event, date):
        self.cursor.execute('INSERT INTO events (event, date) VALUES (?, ?)', (event, date))
        self.conn.commit()

    def get_facts(self):
        self.cursor.execute('SELECT * FROM facts')
        return self.cursor.fetchall()

    def get_person(self, name):
        self.cursor.execute('SELECT * FROM people WHERE name = ?', (name,))
        return self.cursor.fetchone()

    def get_events(self):
        self.cursor.execute('SELECT * FROM events')
        return self.cursor.fetchall()

    def close(self):
        self.conn.close()

# Usage example
memory_store = NovaMemoryStore()
memory_store.add_fact('Nova is an AI')
memory_store.add_person('John Doe', 'Software Engineer')
memory_store.add_event('Nova was created', '2020-01-01')
print(memory_store.get_facts())
print(memory_store.get_person('John Doe'))
print(memory_store.get_events())
memory_store.close()