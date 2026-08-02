import os
import re
import ast
import astunparse
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV

class CodeReviewer:
    def __init__(self, code_dir):
        self.code_dir = code_dir
        self.code_files = []
        self.code_data = []
        self.vectorizer = TfidfVectorizer()

    def collect_code(self):
        for root, dirs, files in os.walk(self.code_dir):
            for file in files:
                if file.endswith(".py"):
                    self.code_files.append(os.path.join(root, file))

    def parse_code(self):
        for file in self.code_files:
            with open(file, "r") as f:
                code = f.read()
                try:
                    tree = ast.parse(code)
                    data = self.extract_features(tree)
                    self.code_data.append(data)
                except SyntaxError:
                    print(f"Error parsing {file}")

    def extract_features(self, tree):
        features = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                features.append(self.extract_function_features(node))
            elif isinstance(node, ast.ClassDef):
                features.append(self.extract_class_features(node))
            elif isinstance(node, ast.For):
                features.append(self.extract_for_features(node))
            elif isinstance(node, ast.If):
                features.append(self.extract_if_features(node))
        return features

    def extract_function_features(self, node):
        features = []
        features.append(len(node.args.args))
        features.append(len(node.body))
        features.append(len(node.orelse))
        return features

    def extract_class_features(self, node):
        features = []
        features.append(len(node.body))
        return features

    def extract_for_features(self, node):
        features = []
        features.append(len(node.iter))
        features.append(len(node.body))
        return features

    def extract_if_features(self, node):
        features = []
        features.append(len(node.body))
        features.append(len(node.orelse))
        return features

    def train_model(self):
        X = self.vectorizer.fit_transform([code for code in self.code_data])
        y = [1 if code["quality"] else 0 for code in self.code_data]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100)
        param_grid = {"n_estimators": [10, 50, 100, 200], "max_depth": [None, 5, 10, 20]}
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring="accuracy")
        grid_search.fit(X_train, y_train)
        y_pred = grid_search.predict(X_test)
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Classification Report:")
        print(classification_report(y_test, y_pred))

    def optimize_code(self):
        X = self.vectorizer.transform([code for code in self.code_data])
        y = [1 if code["quality"] else 0 for code in self.code_data]
        model = RandomForestClassifier(n_estimators=grid_search.best_estimator_.n_estimators)
        model.fit(X, y)
        optimized_codes = []
        for code in self.code_data:
            optimized_code = self.optimize_code_snippet(code, model)
            optimized_codes.append(optimized_code)
        return optimized_codes

    def optimize_code_snippet(self, code, model):
        snippet = code["snippet"]
        X = self.vectorizer.transform([snippet])
        prediction = model.predict(X)
        if prediction[0] == 1:
            return snippet
        else:
            optimized_snippet = self.optimize_snippet(snippet)
            return optimized_snippet

    def optimize_snippet(self, snippet):
        # Simple optimization: remove unnecessary whitespace
        snippet = re.sub(r"\s+", " ", snippet)
        # Simple optimization: remove redundant comments
        snippet = re.sub(r"#.*", "", snippet)
        return snippet

def main():
    code_dir = "path/to/code/directory"
    reviewer = CodeReviewer(code_dir)
    reviewer.collect_code()
    reviewer.parse_code()
    reviewer.train_model()
    optimized_codes = reviewer.optimize_code()
    for code in optimized_codes:
        print(astunparse.unparse(code))

if __name__ == "__main__":
    main()
```

This code defines a `CodeReviewer` class that collects code from a specified directory, parses it using the `ast` module, extracts features from the code, trains a machine learning model on the features, and optimizes the code based on the model's predictions. The `main` function demonstrates how to use the `CodeReviewer` class to review and optimize code.
