# dialogue_manager.py
# Created by Lumina

from nltk.sentiment import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()
    def analyze_sentiment(self, text):
        return sia.polarity_scores(text)
