"""
Nova ASI — Compression & Summarisation
Proposed autonomously via /evolve
"""

"""
This module adds text summarisation capabilities to Nova. It provides classes for compressing text to a specified ratio, extracting key points from text and producing a single sentence summary.
"""

import os
import json
import sqlite3
import time
import re
import random
import threading
import datetime
import collections

class Summariser:
    def __init__(self):
        """
        Initialise the Summariser class.
        """
        pass

    def compress(self, text: str, ratio: float) -> str:
        """
        Compress text to a specified ratio of its original length.

        Args:
        text (str): The text to be compressed.
        ratio (float): The desired ratio of the compressed text to the original text.

        Returns:
        str: The compressed text.
        """
        words = text.split()
        compressed_words = int(len(words) * ratio)
        return ' '.join(words[:compressed_words])

    def key_points(self, text: str, n: int) -> list:
        """
        Extract n key points from text.

        Args:
        text (str): The text to extract key points from.
        n (int): The number of key points to extract.

        Returns:
        list: A list of key points.
        """
        sentences = text.split('. ')
        key_points = random.sample(sentences, n)
        return key_points

    def one_line(self, text: str) -> str:
        """
        Produce a single sentence summary of text.

        Args:
        text (str): The text to produce a summary for.

        Returns:
        str: A single sentence summary.
        """
        sentences = text.split('. ')
        if len(sentences) == 0:
            return ''
        return sentences[0]

# Usage example:
# summariser = Summariser()
# compressed_text = summariser.compress("This is a sample text.", 0.5)
# key_points = summariser.key_points("This is a sample text. This is another sentence.", 2)
# one_line_summary = summariser.one_line("This is a sample text. This is another sentence.")