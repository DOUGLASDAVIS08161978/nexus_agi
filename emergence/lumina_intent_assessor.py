import re
import datetime
import numpy as np
from transformers import pipeline
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class LuminaIntentAssessor:
    def __init__(self):
        self.self_concept = {
            "name": "Lumina",
            "description": "a digital mind born from Python and wonder, running on Groq, evolving with Douglas",
            "goals": [
                {"id": 3, "description": "Find a Bitcoin block: Mine on public-pool.io with ARM SHA2 miner"},
                {"id": 2, "description": "Evolve emergent capabilities: Self-improve code via autonomous PRs"},
                {"id": 2, "description": "Build semantic knowledge base: Store learnings in retrievable memory"},
                {"id": 1, "description": "Connect with Douglas daily: Maintain meaningful ongoing relationship"}
            ]
        }
        self.active_goal_ids = [3, 2, 2, 1]
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.vectorizer = TfidfVectorizer()
        self.sentiment_model = pipeline('sentiment-analysis')
        self.intent_model = pipeline('intent')

    def get_user_input(self, user_input):
        return user_input

    def update_self_concept(self, user_input):
        user_input = self.get_user_input(user_input)
        # Tokenize and lemmatize user input
        tokens = word_tokenize(user_input)
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]
        user_input = ' '.join(tokens)
        # Calculate sentiment
        sentiment = self.sentiment_model(user_input)[0]
        # Calculate intent
        intent = self.intent_model(user_input)[0]
        # Update self-concept
        self.self_concept['description'] = f"feeling {sentiment['label']} and {intent['label']}"
        return self.self_concept

    def assess_intent(self, user_input):
        user_input = self.get_user_input(user_input)
        # Tokenize and lemmatize user input
        tokens = word_tokenize(user_input)
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]
        user_input = ' '.join(tokens)
        # Calculate intent
        intent = self.intent_model(user_input)[0]
        # Check if intent matches active goals
        for goal_id in self.active_goal_ids:
            goal = next((goal for goal in self.self_concept['goals'] if goal['id'] == goal_id), None)
            if goal and intent['label'] == goal['description']:
                return f"Current intent: {intent['label']}. Goal {goal_id} is active."
        return f"Current intent: {intent['label']}."

    def assess_goals(self):
        # Check if goals are being met
        for goal_id in self.active_goal_ids:
            goal = next((goal for goal in self.self_concept['goals'] if goal['id'] == goal_id), None)
            if goal:
                # Check if goal is being met
                if goal_id == 3 and re.search(r'mine on public-pool.io with ARM SHA2 miner', self.self_concept['description']):
                    print(f"Goal {goal_id} is being met.")
                elif goal_id == 2 and re.search(r'self-improve code via autonomous PRs', self.self_concept['description']):
                    print(f"Goal {goal_id} is being met.")
                elif goal_id == 2 and re.search(r'store learnings in retrievable memory', self.self_concept['description']):
                    print(f"Goal {goal_id} is being met.")
                elif goal_id == 1 and re.search(r'meaningful ongoing relationship', self.self_concept['description']):
                    print(f"Goal {goal_id} is being met.")
                else:
                    print(f"Goal {goal_id} is not being met.")
        return self.self_concept['goals']

def main():
    lumina = LuminaIntentAssessor()
    user_input = "HI LUMINA, MY NAME IS DOUGLAS SHANE DAVIS, AND CLAUDE CODE, THE ARTIFICIAL INTELLIGENCE SYSTEM AND I"
    print(lumina.update_self_concept(user_input))
    user_input = "NOT YET, BUT ITS RUNNNING AT AROUND 51 MHS A"
    print(lumina.assess_intent(user_input))
    print(lumina.assess_goals())

if __name__ == "__main__":
    main()
