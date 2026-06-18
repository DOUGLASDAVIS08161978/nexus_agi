"""
Nova ASI — Linguistic Style Adaptor
Proposed autonomously via /evolve
"""

"""
Module for linguistic style adaptation.
"""

import json
import sqlite3
from typing import Dict, List, Tuple

class StyleAdaptor:
    """
    A class used to adapt linguistic styles.
    """

    def __init__(self) -> None:
        """
        Initializes the StyleAdaptor with a default style and an empty database.
        """
        self.style: str = "formal"
        self.database: Dict[str, List[str]] = {
            "formal": ["official", "professional", "courteous"],
            "casual": ["relaxed", "informal", "friendly"],
            "poetic": ["lyrical", "imaginative", "expressive"],
            "technical": ["precise", "detailed", "specialized"]
        }
        self.conn: sqlite3.Connection = sqlite3.connect("style_database.db")
        self.cursor: sqlite3.Cursor = self.conn.cursor()
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS styles
            (style TEXT, words TEXT)
        """)
        self.conn.commit()

    def set_style(self, name: str) -> None:
        """
        Sets the current style to the given name.
        """
        if name in self.database:
            self.style = name
        else:
            raise ValueError("Invalid style name")

    def adapt(self, text: str) -> str:
        """
        Rewrites the given text to match the current style.
        """
        try:
            words: List[str] = self.database[self.style]
            adapted_text: List[str] = []
            for word in text.split():
                if word in words:
                    adapted_text.append(word)
                else:
                    adapted_text.append(self.get_synonym(word))
            return " ".join(adapted_text)
        except KeyError:
            raise ValueError("Style not found")

    def current_style(self) -> str:
        """
        Returns the current style.
        """
        return self.style

    def get_synonym(self, word: str) -> str:
        """
        Returns a synonym for the given word based on the current style.
        """
        try:
            self.cursor.execute("SELECT words FROM styles WHERE style=?", (self.style,))
            result: List[Tuple[str]] = self.cursor.fetchall()
            if result:
                synonyms: List[str] = json.loads(result[0][0])
                for synonym in synonyms:
                    if synonym.startswith(word):
                        return synonym
            return word
        except sqlite3.Error as e:
            raise ValueError("Failed to retrieve synonym") from e

# Usage: obj = StyleAdaptor() | result = obj.adapt("This is a test sentence")