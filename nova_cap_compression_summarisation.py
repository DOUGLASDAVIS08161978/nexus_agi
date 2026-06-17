"""
Nova ASI — Compression & Summarisation
Proposed autonomously via /evolve
"""

"""
This module adds text summarisation capabilities to the Nova AI.
The Summariser class provides methods to compress text, extract key points, and generate a single sentence summary.
"""

import re
import random
import collections
from typing import List, Tuple

class Summariser:
    def __init__(self):
        pass

    def compress(self, text: str, ratio: float) -> str:
        """
        Compress text to a given ratio of the original length.

        Args:
        text (str): The input text to be compressed.
        ratio (float): The ratio of the original length.

        Returns:
        str: The compressed text.
        """
        words = text.split()
        num_words = int(len(words) * ratio)
        return ' '.join(random.sample(words, num_words))

    def key_points(self, text: str, n: int) -> List[str]:
        """
        Extract n key points from the given text.

        Args:
        text (str): The input text to extract key points from.
        n (int): The number of key points to extract.

        Returns:
        List[str]: A list of n key points.
        """
        sentences = re.split('[.!?]', text)
        key_points = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                key_points.append(sentence)
        return key_points[:n]

    def one_line(self, text: str) -> str:
        """
        Generate a single sentence summary from the given text.

        Args:
        text (str): The input text to generate a single sentence summary from.

        Returns:
        str: A single sentence summary.
        """
        sentences = re.split('[.!?]', text)
        return sentences[0].strip()

# Usage example:
# summariser = Summariser()
# text = "This is a sample text with multiple sentences. It has a length of 100 characters."
# print(summariser.compress(text, 0.5))
# print(summariser.key_points(text, 2))
# print(summariser.one_line(text))