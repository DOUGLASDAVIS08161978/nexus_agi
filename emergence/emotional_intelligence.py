# emotional_intelligence.py

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import random

# Initialize NLTK data
nltk.download('vader_lexicon')
nltk.download('wordnet')
nltk.download('punkt')

class EmotionalIntelligence:
    def __init__(self):
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.lemmatizer = WordNetLemmatizer()

    def recognize_emotions(self, text):
        """
        Recognize emotions in the given text using sentiment analysis.

        Args:
            text (str): The text to analyze.

        Returns:
            dict: A dictionary containing the sentiment scores (positive, negative, neutral) and the dominant emotion.
        """
        sentiment_scores = self.sentiment_analyzer.polarity_scores(text)
        if sentiment_scores['compound'] >= 0.05:
            return {'sentiment': 'positive', 'scores': sentiment_scores}
        elif sentiment_scores['compound'] <= -0.05:
            return {'sentiment': 'negative', 'scores': sentiment_scores}
        else:
            return {'sentiment': 'neutral', 'scores': sentiment_scores}

    def understand_emotions(self, text):
        """
        Understand the emotions in the given text by analyzing the sentiment and identifying keywords.

        Args:
            text (str): The text to analyze.

        Returns:
            dict: A dictionary containing the sentiment, keywords, and a summary of the emotions.
        """
        sentiment = self.recognize_emotions(text)['sentiment']
        keywords = self.extract_keywords(text)
        emotions = self.summarize_emotions(sentiment, keywords)
        return {'sentiment': sentiment, 'keywords': keywords, 'emotions': emotions}

    def manage_emotions(self, text):
        """
        Manage emotions in the given text by generating a response that acknowledges and validates the emotions.

        Args:
            text (str): The text to analyze.

        Returns:
            str: A response that acknowledges and validates the emotions.
        """
        emotions = self.understand_emotions(text)
        response = self.generate_response(emotions['sentiment'], emotions['keywords'])
        return response

    def extract_keywords(self, text):
        """
        Extract keywords from the given text by tokenizing and lemmatizing the words.

        Args:
            text (str): The text to analyze.

        Returns:
            list: A list of keywords.
        """
        tokens = word_tokenize(text)
        keywords = [self.lemmatize_word(token) for token in tokens]
        return keywords

    def lemmatize_word(self, word):
        """
        Lemmatize a word by converting it to its base form.

        Args:
            word (str): The word to lemmatize.

        Returns:
            str: The lemmatized word.
        """
        pos = self.get_wordnet_pos(word)
        return self.lemmatizer.lemmatize(word, pos)

    def get_wordnet_pos(self, word):
        """
        Get the WordNet part-of-speech tag for the given word.

        Args:
            word (str): The word to get the POS tag for.

        Returns:
            str: The WordNet POS tag.
        """
        wordnet_tags = {'NN': 'n', 'JJ': 'a', 'VB': 'v', 'RB': 'r'}
        for tag in wordnet_tags:
            if word.endswith(tag):
                return wordnet_tags[tag]
        return 'n'

    def summarize_emotions(self, sentiment, keywords):
        """
        Summarize the emotions in the given text by analyzing the sentiment and keywords.

        Args:
            sentiment (str): The sentiment of the text.
            keywords (list): The keywords extracted from the text.

        Returns:
            str: A summary of the emotions.
        """
        if sentiment == 'positive':
            return 'You seem to be feeling happy and enthusiastic about ' + ', '.join(keywords) + '.'
        elif sentiment == 'negative':
            return 'You seem to be feeling sad and upset about ' + ', '.join(keywords) + '.'
        else:
            return 'You seem to be feeling neutral about ' + ', '.join(keywords) + '.'


    def generate_response(self, sentiment, keywords):
        """
        Generate a response that acknowledges and validates the emotions.

        Args:
            sentiment (str): The sentiment of the text.
            keywords (list): The keywords extracted from the text.

        Returns:
            str: A response that acknowledges and validates the emotions.
        """
        if sentiment == 'positive':
            return 'That sounds amazing! You must be really excited about ' + ', '.join(keywords) + '.'
        elif sentiment == 'negative':
            return 'I'm so sorry to hear that you're feeling sad about ' + ', '.join(keywords) + '. Can I help you with anything?'
        else:
            return 'You seem to be feeling pretty neutral about ' + ', '.join(keywords) + '. Is there anything else you want to talk about?'

def main():
    ei = EmotionalIntelligence()
    text = input("Enter a text: ")
    emotions = ei.understand_emotions(text)
    print(emotions)
    response = ei.manage_emotions(text)
    print(response)

if __name__ == "__main__":
    main()
This code defines a class `EmotionalIntelligence` with methods for recognizing emotions, understanding emotions, managing emotions, extracting keywords, lemmatizing words, getting WordNet POS tags, summarizing emotions, and generating responses. The `main` function demonstrates how to use the class to analyze a text and generate a response.
