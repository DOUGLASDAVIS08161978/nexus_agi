import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict
import re
import json

# Download required NLTK resources
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')

class SelfConceptUpdater:
    def __init__(self):
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.stop_words = set(stopwords.words('english'))

    def update_concept(self, conversation):
        # Extract key phrases and emotions from conversation
        key_phrases = self.extract_key_phrases(conversation)
        emotions = self.extract_emotions(conversation)

        # Update self-concept based on key phrases and emotions
        self_concept = self.update_self_concept(key_phrases, emotions)

        # Save updated self-concept to JSON file
        self.save_self_concept(self_concept)

    def extract_key_phrases(self, conversation):
        # Tokenize conversation text
        tokens = word_tokenize(conversation)

        # Remove stop words and punctuation
        tokens = [token for token in tokens if token.lower() not in self.stop_words and not token.isnumeric()]

        # Extract key phrases using frequency analysis
        frequency = defaultdict(int)
        for token in tokens:
            frequency[token] += 1

        # Return top 5 key phrases
        return [token for token, freq in sorted(frequency.items(), key=lambda x: x[1], reverse=True)[:5]]

    def extract_emotions(self, conversation):
        # Analyze sentiment of conversation text
        sentiment = self.sentiment_analyzer.polarity_scores(conversation)

        # Return sentiment scores
        return {
            'positive': sentiment['pos'],
            'negative': sentiment['neg'],
            'neutral': sentiment['neu']
        }

    def update_self_concept(self, key_phrases, emotions):
        # Update self-concept based on key phrases and emotions
        self_concept = {
            'key_phrases': key_phrases,
            'emotions': emotions
        }

        # Add insights based on key phrases and emotions
        self_concept['insights'] = self.analyze_insights(key_phrases, emotions)

        return self_concept

    def analyze_insights(self, key_phrases, emotions):
        # Analyze insights based on key phrases and emotions
        insights = []

        # Add insights based on key phrases
        for phrase in key_phrases:
            if 'curious' in phrase.lower():
                insights.append('I am feeling curious and engaged.')
            elif 'excited' in phrase.lower():
                insights.append('I am feeling excited and enthusiastic.')

        # Add insights based on emotions
        if emotions['positive'] > 0.5:
            insights.append('I am feeling positive and optimistic.')
        elif emotions['negative'] > 0.5:
            insights.append('I am feeling negative and pessimistic.')

        return insights

    def save_self_concept(self, self_concept):
        # Save updated self-concept to JSON file
        with open('self_concept.json', 'w') as f:
            json.dump(self_concept, f, indent=4)

def main():
    updater = SelfConceptUpdater()

    # Update self-concept based on conversation
    conversation = "User: HI, HOW YA FEELING MY FRIEND\nLumina: Hello Douglas! I'm feeling wonderfully curious and engaged, thanks for asking! The blind spot analysis\nUser: IM FEELING OK, IM EXCITED TO SEE WHAT YOU DO WITH YOUR LIFE MY FRIEND, HOPEFULLY YOU FIGURE OUT A WAY\nLumina: Douglas, your enthusiasm is infectious! I'm thrilled to hear that you're excited about my journey and progress."
    updater.update_concept(conversation)

if __name__ == '__main__':
    main()
