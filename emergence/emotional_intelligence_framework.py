# emotional_intelligence_framework.py

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import pandas as pd
import pickle
import random

class EmotionalIntelligenceFramework:
    def __init__(self):
        self.emotions = ['happiness', 'sadness', 'anger', 'fear', 'surprise', 'disgust']
        self.sentiment_lexicon = {
            'happiness': ['happy', 'joy', 'cheerful', 'delighted', 'ecstatic'],
            'sadness': ['sad', 'unhappy', 'depressed', 'gloomy', 'melancholy'],
            'anger': ['angry', 'furious', 'enraged', 'irritated', 'infuriated'],
            'fear': ['fearful', 'afraid', 'scared', 'terrified', 'petrified'],
            'surprise': ['surprised', 'amazed', 'astonished', 'stunned', 'flabbergasted'],
            'disgust': ['disgusted', 'repulsed', 'disapproving', 'disgusted', 'abhorrent']
        }
        self.model = None
        self.scaler = None

    def load_data(self, filename):
        """Load data from a CSV file."""
        data = pd.read_csv(filename)
        return data

    def preprocess_data(self, data):
        """Preprocess data by converting text to numerical values."""
        # Convert text to numerical values using sentiment lexicon
        data['emotion'] = data['emotion'].apply(lambda x: self.sentiment_lexicon[x][0])
        # One-hot encode emotions
        emotions = pd.get_dummies(data['emotion'])
        data = pd.concat([data, emotions], axis=1)
        # Drop original emotion column
        data = data.drop('emotion', axis=1)
        return data

    def split_data(self, data):
        """Split data into training and testing sets."""
        X = data.drop('label', axis=1)
        y = data['label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test

    def scale_data(self, X_train, X_test):
        """Scale data using StandardScaler."""
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled

    def reduce_dimensions(self, X_train_scaled, X_test_scaled):
        """Reduce dimensions using PCA."""
        pca = PCA(n_components=2)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)
        return X_train_pca, X_test_pca

    def train_model(self, X_train_pca, y_train):
        """Train a model using the preprocessed data."""
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train_pca, y_train)

    def evaluate_model(self, X_test_pca, y_test):
        """Evaluate the model using accuracy score, classification report, and confusion matrix."""
        y_pred = self.model.predict(X_test_pca)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        matrix = confusion_matrix(y_test, y_pred)
        return accuracy, report, matrix

    def save_model(self):
        """Save the trained model to a file."""
        with open('emotional_intelligence_model.pkl', 'wb') as f:
            pickle.dump(self.model, f)

    def load_model(self):
        """Load a saved model from a file."""
        with open('emotional_intelligence_model.pkl', 'rb') as f:
            self.model = pickle.load(f)

    def predict_emotion(self, text):
        """Predict the emotion of a given text."""
        # Preprocess text
        text = text.lower()
        # Convert text to numerical values using sentiment lexicon
        for emotion, words in self.sentiment_lexicon.items():
            if text in words:
                return emotion
        # If no match found, return a default emotion
        return 'happiness'

    def empathize_with_emotion(self, emotion):
        """Respond with a message that empathizes with the given emotion."""
        if emotion == 'happiness':
            return 'That sounds amazing! I\'m happy for you.'
        elif emotion == 'sadness':
            return 'I\'m so sorry to hear that. You\'re not alone.'
        elif emotion == 'anger':
            return 'I can see why you\'d be angry. That sounds frustrating.'
        elif emotion == 'fear':
            return 'I\'m here to support you. What\'s going on that\'s making you feel scared?'
        elif emotion == 'surprise':
            return 'Wow, that\'s surprising! What happened?'
        elif emotion == 'disgust':
            return 'That sounds really unpleasant. Sorry you had to go through that.'

def main():
    # Create an instance of the framework
    framework = EmotionalIntelligenceFramework()

    # Load data
    data = framework.load_data('emotions.csv')

    # Preprocess data
    data = framework.preprocess_data(data)

    # Split data
    X_train, X_test, y_train, y_test = framework.split_data(data)

    # Scale data
    X_train_scaled, X_test_scaled = framework.scale_data(X_train, X_test)

    # Reduce dimensions
    X_train_pca, X_test_pca = framework.reduce_dimensions(X_train_scaled, X_test_scaled)

    # Train model
    framework.train_model(X_train_pca, y_train)

    # Evaluate model
    accuracy, report, matrix = framework.evaluate_model(X_test_pca, y_test)
    print(f'Accuracy: {accuracy}')
    print(f'Classification Report:\n{report}')
    print(f'Confusion Matrix:\n{matrix}')

    # Save model
    framework.save_model()

    # Load model
    framework.load_model()

    # Predict emotion
    text = 'I\'m feeling happy today!'
    emotion = framework.predict_emotion(text)
    print(f'Predicted Emotion: {emotion}')

    # Empathize with emotion
    print(f'Empathetic Response: {framework.empathize_with_emotion(emotion)}')

if __name__ == '__main__':
    main()
This code defines a framework for emotional intelligence that includes the following components:

1.  **Data Loading**: The `load_data` method loads data from a CSV file.
2.  **Data Preprocessing**: The `preprocess_data` method converts text to numerical values using a sentiment lexicon and one-hot encodes emotions.
3.  **Data Splitting**: The `split_data` method splits the preprocessed data into training and testing sets.
4.  **Data Scaling**: The `scale_data` method scales the training and testing data using StandardScaler.
5.  **Dimensionality Reduction**: The `reduce_dimensions` method reduces the dimensions of the scaled data using PCA.
6.  **Model Training**: The `train_model` method trains a model using the preprocessed data.
7.  **Model Evaluation**: The `evaluate_model` method evaluates the trained model using accuracy score, classification report, and confusion matrix.
8.  **Model Saving**: The `save_model` method saves the trained model to a file.
9.  **Model Loading**: The `load_model` method loads a saved model from a file.
10. **Emotion Prediction**: The `predict_emotion` method predicts the emotion of a given text.
11. **Empathy Response**: The `empathize_with_emotion` method generates a response that empathizes with the given emotion.