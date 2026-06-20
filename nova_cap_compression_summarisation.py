"""
Nova ASI — Compression & Summarisation
Proposed autonomously via /evolve
"""

"""
This module provides an enhanced summarization capability for Nova, allowing it to compress text to a specified ratio, extract key points from text, and produce a single sentence summary.
"""

import re
from typing import List, Tuple

class Summariser:
    """
    A summarizer class that provides methods to compress, extract key points, and produce a single line summary of text.
    """

    def __init__(self):
        pass

    def compress(self, text: str, ratio: float) -> str:
        """
        Compress the input text to the specified ratio.

        Args:
            text (str): The input text to be compressed.
            ratio (float): The desired compression ratio.

        Returns:
            str: The compressed text.
        """
        # Calculate the number of sentences to keep based on the ratio
        num_sentences = int(len(text.split('.')) * ratio)
        # Split the text into sentences
        sentences = text.split('.')
        # Select the top N sentences
        compressed = '. '.join(sentences[:num_sentences])
        return compressed

    def key_points(self, text: str, n: int) -> List[str]:
        """
        Extract the top N key points from the input text.

        Args:
            text (str): The input text.
            n (int): The number of key points to extract.

        Returns:
            List[str]: A list of key points.
        """
        # Tokenize the text into sentences
        sentences = re.split('\. ', text)
        # Remove short sentences
        sentences = [sentence for sentence in sentences if len(sentence.split()) > 5]
        # Select the top N sentences
        key_points = sentences[:n]
        return key_points

    def one_line(self, text: str) -> str:
        """
        Produce a single sentence summary of the input text.

        Args:
            text (str): The input text.

        Returns:
            str: A single sentence summary.
        """
        # Tokenize the text into sentences
        sentences = re.split('\. ', text)
        # Select the longest sentence
        one_line = max(sentences, key=len)
        return one_line

# Usage example:
# summariser = Summariser()
# text = "This is a sample text. It has multiple sentences. We want to compress it to a single sentence."
# print(summariser.compress(text, 0.5))  # Compress to 50% of the original length
# print(summariser.key_points(text, 2))  # Extract 2 key points
# print(summariser.one_line(text))  # Produce a single sentence summary
