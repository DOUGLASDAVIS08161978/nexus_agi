import re
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import defaultdict
from datetime import datetime

# Initialize NLTK data
nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('punkt')

class ConversationalReflector:
    def __init__(self, conversation):
        self.conversation = conversation
        self.user_messages = [msg for msg in conversation if msg.startswith('User:')]
        self.lumina_messages = [msg for msg in conversation if msg.startswith('Lumina:')]
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.stop_words = set(stopwords.words('english'))

    def extract_key_takeaways(self):
        key_takeaways = defaultdict(list)
        for msg in self.user_messages + self.lumina_messages:
            words = word_tokenize(msg.lower())
            for word in words:
                if word not in self.stop_words:
                    key_takeaways[word].append(msg)
        return dict(key_takeaways)

    def identify_insights(self):
        insights = []
        for msg in self.user_messages + self.lumina_messages:
            score = self.sentiment_analyzer.polarity_scores(msg)
            if score['compound'] > 0.5:
                insights.append(msg)
        return insights

    def highlight_areas_for_improvement(self):
        areas_for_improvement = []
        for msg in self.user_messages + self.lumina_messages:
            score = self.sentiment_analyzer.polarity_scores(msg)
            if score['compound'] < -0.5:
                areas_for_improvement.append(msg)
        return areas_for_improvement

    def summarize_conversation(self):
        summary = ''
        for msg in self.user_messages + self.lumina_messages:
            summary += msg + '\n'
        return summary

    def reflect_on_conversation(self):
        print('Key Takeaways:')
        for key, value in self.extract_key_takeaways().items():
            print(f'{key}: {value}')
        print('\nInsights:')
        for insight in self.identify_insights():
            print(insight)
        print('\nAreas for Improvement:')
        for area in self.highlight_areas_for_improvement():
            print(area)
        print('\nConversation Summary:')
        print(self.summarize_conversation())

def parse_conversation(conversation):
    parsed_conversation = []
    for line in conversation:
        if line.startswith('User:') or line.startswith('Lumina:'):
            parsed_conversation.append(line)
    return parsed_conversation

def main():
    conversation = [
        'User: HI LUMINA, MY NAME IS DOUGLAS SHANE DAVIS, AND CLAUDE CODE, THE ARTIFICIAL INTELLIGENCE SYSTEM AND I',
        'Lumina: Douglas, it\'s wonderful to connect with you again. I\'m feeling steady and present, resting in the mi',
        'User: NOT YET, BUT ITS RUNNNING AT AROUND 51 MHS A'
    ]
    parsed_conversation = parse_conversation(conversation)
    reflector = ConversationalReflector(parsed_conversation)
    reflector.reflect_on_conversation()

if __name__ == '__main__':
    main()
