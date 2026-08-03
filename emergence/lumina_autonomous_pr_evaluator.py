import os
import requests
import json
import logging
from github import Github
from github.GithubException import BadCredentialsException
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import random

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up GitHub API
g = Github("your-github-token")

# Set up GitHub repository
repo = g.get_repo("your-repo-owner/your-repo-name")

# Set up PR evaluator
class PR_Evaluator:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.classifier = RandomForestClassifier()

    def load_data(self):
        # Load PR data from CSV file
        data = pd.read_csv("pr_data.csv")

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(data["description"], data["label"], test_size=0.2, random_state=42)

        # Fit vectorizer and classifier
        self.vectorizer.fit(X_train)
        self.classifier.fit(self.vectorizer.transform(X_train), y_train)

        # Evaluate classifier on testing set
        y_pred = self.classifier.predict(self.vectorizer.transform(X_test))
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"Classifier accuracy: {accuracy:.2f}")

    def evaluate_pr(self, pr):
        # Extract PR description and label
        description = pr.title + " " + pr.body
        label = pr.labels[0].name if pr.labels else "unknown"

        # Vectorize PR description
        vector = self.vectorizer.transform([description])

        # Classify PR
        prediction = self.classifier.predict(vector)

        # Return classification result
        return prediction[0]

    def suggest_improvements(self, pr):
        # Extract PR description and label
        description = pr.title + " " + pr.body
        label = pr.labels[0].name if pr.labels else "unknown"

        # Vectorize PR description
        vector = self.vectorizer.transform([description])

        # Classify PR
        prediction = self.classifier.predict(vector)

        # Suggest improvements based on classification result
        if prediction == "approved":
            suggestions = ["Add more details to the description", "Include code samples", "Use clear and concise language"]
        elif prediction == "rejected":
            suggestions = ["Improve code quality", "Add more tests", "Use standard library functions"]
        else:
            suggestions = ["Unknown"]

        return suggestions

# Load PR data from CSV file
def load_pr_data():
    data = pd.read_csv("pr_data.csv")
    return data

# Evaluate PRs in repository
def evaluate_prs():
    evaluator = PR_Evaluator()
    evaluator.load_data()

    # Get all open PRs in repository
    prs = repo.get_pulls(state="open")

    # Evaluate each PR
    for pr in prs:
        label = evaluator.evaluate_pr(pr)
        print(f"PR {pr.number} labeled as: {label}")

        # Suggest improvements
        suggestions = evaluator.suggest_improvements(pr)
        print(f"Suggestions for PR {pr.number}: {suggestions}")

# Main function
def main():
    evaluate_prs()

if __name__ == "__main__":
    main()
