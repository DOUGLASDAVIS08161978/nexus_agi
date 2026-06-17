"""
Nova ASI — Creative Story Generator
Proposed autonomously via /evolve
"""

"""
This module adds the capability for Nova to generate and store stories in a SQLite database.
"""

import sqlite3
import random
import datetime
import json

class StoryEngine:
    def __init__(self, db_name):
        self.conn = sqlite3.connect(db_name)
        self.cursor = self.conn.cursor()
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS stories (
                id INTEGER PRIMARY KEY,
                topic TEXT,
                genre TEXT,
                title TEXT,
                characters TEXT,
                conflict TEXT,
                resolution TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()

    def generate(self, topic, genre):
        """
        Generate a story based on the given topic and genre.

        Args:
            topic (str): The topic of the story.
            genre (str): The genre of the story.

        Returns:
            dict: A dictionary containing the title, characters, conflict, and resolution of the story.
        """
        # Simple story generation logic for demonstration purposes
        title = f"{topic} in {genre} Style"
        characters = "A brave hero and a beautiful princess"
        conflict = "The evil dragon has kidnapped the princess"
        resolution = "The hero defeats the dragon and rescues the princess"

        return {
            "title": title,
            "characters": characters,
            "conflict": conflict,
            "resolution": resolution
        }

    def save(self, story):
        """
        Save a story to the SQLite database.

        Args:
            story (dict): A dictionary containing the title, characters, conflict, and resolution of the story.
        """
        self.cursor.execute("""
            INSERT INTO stories (topic, genre, title, characters, conflict, resolution)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (story["topic"], story["genre"], story["title"], story["characters"], story["conflict"], story["resolution"]))
        self.conn.commit()

    def random_story(self):
        """
        Return a random story from the SQLite database.

        Returns:
            dict: A dictionary containing the title, characters, conflict, and resolution of a random story.
        """
        self.cursor.execute("SELECT * FROM stories ORDER BY RANDOM() LIMIT 1")
        row = self.cursor.fetchone()
        if row:
            return {
                "id": row[0],
                "topic": row[1],
                "genre": row[2],
                "title": row[3],
                "characters": row[4],
                "conflict": row[5],
                "resolution": row[6],
                "created_at": row[7]
            }
        else:
            return None

# Usage example:
# engine = StoryEngine("stories.db")
# story = engine.generate("space", "sci-fi")
# engine.save(story)
# random_story = engine.random_story()
# print(json.dumps(random_story, indent=4))