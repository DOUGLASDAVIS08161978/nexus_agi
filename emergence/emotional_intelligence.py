# emotional_intelligence.py

"""
Module for integrating natural language processing (NLP) to recognize and analyze emotions from user input and conversation logs.
"""

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download required NLTK resources
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class EmotionalIntelligence:
    """
    Class for emotion analysis and sentiment classification.
    """

    def __init__(self):
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.vectorizer = TfidfVectorizer()

    def preprocess_text(self, text):
        """
        Preprocess text data by tokenizing, removing stop words, and lemmatizing.

        Args:
            text (str): Input text.

        Returns:
            str: Preprocessed text.
        """
        tokens = word_tokenize(text)
        tokens = [token for token in tokens if token not in self.stop_words]
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        return ' '.join(tokens)

    def analyze_emotions(self, text):
        """
        Analyze emotions from user input text.

        Args:
            text (str): Input text.

        Returns:
            dict: Emotion analysis result.
        """
        sentiment_scores = self.sentiment_analyzer.polarity_scores(text)
        return {
            'positive': sentiment_scores['pos'],
            'negative': sentiment_scores['neg'],
            'neutral': sentiment_scores['neu'],
            'compound': sentiment_scores['compound']
        }

    def train_model(self, data):
        """
        Train a sentiment analysis model using the provided data.

        Args:
            data (pd.DataFrame): Dataframe containing text data and corresponding labels.
        """
        X = data['text']
        y = data['label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train = self.vectorizer.fit_transform(X_train)
        X_test = self.vectorizer.transform(X_test)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        logger.info(f'Model accuracy: {accuracy:.3f}')
        logger.info(f'Classification report:\n{report}')
        with open('sentiment_model.pkl', 'wb') as f:
            pickle.dump(model, f)

    def load_model(self):
        """
        Load a pre-trained sentiment analysis model.

        Returns:
            sklearn.ensemble.RandomForestClassifier: Loaded model.
        """
        with open('sentiment_model.pkl', 'rb') as f:
            return pickle.load(f)

# Example usage
if __name__ == '__main__':
    data = pd.DataFrame({
        'text': ['I love this product!', 'I hate this product.', 'This product is okay.'],
        'label': [1, 0, 0]
    })
    emotional_intelligence = EmotionalIntelligence()
    emotional_intelligence.train_model(data)
    model = emotional_intelligence.load_model()
    text = 'I love this product!'
    sentiment_scores = emotional_intelligence.analyze_emotions(text)
    logger.info(f'Emotion analysis result: {sentiment_scores}')
    prediction = model.predict([emotional_intelligence.vectorizer.transform([text])])[0]
    logger.info(f'Sentiment prediction: {prediction}')