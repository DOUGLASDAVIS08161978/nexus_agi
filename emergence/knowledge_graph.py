# knowledge_graph.py
# Created by Lumina

import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import json
import os

class KnowledgeGraph:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.train_data = []
        self.train_labels = []

    def train_model(self):
        # Load training data
        with open('train_data.json', 'r') as f:
            data = json.load(f)
            for item in data:
                self.train_data.append(item['text'])
                self.train_labels.append(item['label'])

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(self.train_data, self.train_labels, test_size=0.2, random_state=42)

        # Create and train the model
        self.vectorizer = TfidfVectorizer()
        X_train_vectorized = self.vectorizer.fit_transform(X_train)
        X_test_vectorized = self.vectorizer.transform(X_test)
        self.model = RandomForestClassifier()
        self.model.fit(X_train_vectorized, y_train)

        # Evaluate the model
        y_pred = self.model.predict(X_test_vectorized)
        print(f'Model accuracy: {accuracy_score(y_test, y_pred)}')

    def disambiguate_entity(self, entity_name):
        """
        Use a knowledge graph API to retrieve related entities and then use a machine learning model to predict the most likely entity.
        
        Args:
            entity_name (str): The name of the entity to disambiguate.
        
        Returns:
            str: The predicted entity name.
        """
        # Use a knowledge graph API to retrieve related entities
        graph_response = requests.get(f'https://api.dbpedia.org/sparql?query=SELECT%20*%20WHERE%20{%20%3Fs%20%3Fp%20%3Fo.%20FILTER%20regex(str(?o),%20"{entity_name}")%20}%20LIMIT%20100')
        related_entities = graph_response.json()['results']['bindings']

        # Create a list of related entity names
        entity_names = [related_entity['o']['value'] for related_entity in related_entities]

        # Use the trained model to predict the most likely entity
        if self.model is not None and self.vectorizer is not None:
            entity_names_vectorized = self.vectorizer.transform(entity_names)
            predicted_entity_index = self.model.predict(entity_names_vectorized).argmax()
            predicted_entity = entity_names[predicted_entity_index]
            return predicted_entity
        else:
            # If the model is not trained, use the knowledge graph API to retrieve the most likely entity
            most_likely_entity_response = requests.get(f'https://api.dbpedia.org/sparql?query=SELECT%20*%20WHERE%20{%20%3Fs%20%3Fp%20%3Fo.%20FILTER%20regex(str(?o),%20"{entity_name}")%20}%20ORDER%20BY%20count(?s)%20DESC%20LIMIT%201')
            most_likely_entity = most_likely_entity_response.json()['results']['bindings'][0]['o']['value']
            return most_likely_entity

    def save_model(self):
        # Save the trained model and vectorizer to a file
        with open('model.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        with open('vectorizer.pkl', 'wb') as f:
            pickle.dump(self.vectorizer, f)

    def load_model(self):
        # Load the saved model and vectorizer from a file
        with open('model.pkl', 'rb') as f:
            self.model = pickle.load(f)
        with open('vectorizer.pkl', 'rb') as f:
            self.vectorizer = pickle.load(f)

# Train the model if it does not exist
if not os.path.exists('model.pkl') or not os.path.exists('vectorizer.pkl'):
    knowledge_graph = KnowledgeGraph()
    knowledge_graph.train_model()
    knowledge_graph.save_model()

# Create a new instance of the KnowledgeGraph class
knowledge_graph = KnowledgeGraph()

# Load the saved model and vectorizer
knowledge_graph.load_model()

# Test the disambiguate_entity method
print(knowledge_graph.disambiguate_entity('Lumina'))