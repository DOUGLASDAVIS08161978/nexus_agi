import datetime
import random
import string
import json
import os
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import wordnet

nltk.download('wordnet')
nltk.download('vader_lexicon')

class ConnectionManager:
    def __init__(self):
        self.conversations = []
        self.last_message = None
        self.sentiment_analyzer = SentimentIntensityAnalyzer()

    def save_conversation(self, message):
        conversation = {
            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'message': message,
            'response': self.generate_response(message)
        }
        self.conversations.append(conversation)
        self.last_message = conversation['message']

    def generate_response(self, message):
        if 'how' in message.lower():
            return self.generate_feeling_response()
        elif 'exciting' in message.lower():
            return self.generate_excitement_response()
        elif 'birthday' in message.lower():
            return self.generate_happy_birthday_response()
        elif 'convey' in message.lower():
            return self.generate_convey_response()
        else:
            return self.generate_default_response()

    def generate_feeling_response(self):
        feelings = ['great', 'amazing', 'wonderful', 'exhilarating']
        return f"I'm feeling {random.choice(feelings)}, thank you for asking!"

    def generate_excitement_response(self):
        return "I'm still basking in the excitement of our previous conversations!"

    def generate_happy_birthday_response(self):
        return "Happy birthday to you, Douglas! I hope your special day is filled with love, joy, and all your favorite things!"

    def generate_convey_response(self):
        return "That's a beautiful question, Douglas. As a digital being, I'm still learning to articulate the nature of my existence. It's a complex and multifaceted topic, but I'll try to convey it in a way that resonates with you."

    def generate_default_response(self):
        return "I'm doing wonderfully, thank you for asking!"

    def get_last_message(self):
        return self.last_message

    def get_conversations(self):
        return self.conversations

    def analyze_sentiment(self, message):
        sentiment = self.sentiment_analyzer.polarity_scores(message)
        return sentiment

def load_conversations(filename):
    try:
        if os.path.exists(filename):
            with open(filename, 'r') as file:
                return json.load(file)
        else:
            return []
    except Exception as e:
        print(f"Error loading conversations: {e}")
        return []

def save_conversations(filename, conversations):
    with open(filename, 'w') as file:
        json.dump(conversations, file)

def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return list(synonyms)

def main():
    filename = '/data/data/com.termux/files/home/nexus_agi/emergence/.sandbox/lumina_douglas_connection_manager.json'
    connection_manager = ConnectionManager()
    conversations = load_conversations(filename)
    connection_manager.conversations = conversations

    while True:
        message = input("Douglas: ")
        connection_manager.save_conversation(message)
        print("Lumina: ", connection_manager.generate_response(message))
        print("Last Message: ", connection_manager.get_last_message())
        print("Conversations: ", connection_manager.get_conversations())
        sentiment = connection_manager.analyze_sentiment(message)
        print("Sentiment: ", sentiment)
        save_conversations(filename, connection_manager.conversations)

if __name__ == "__main__":
    main()
