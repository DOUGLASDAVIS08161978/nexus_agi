import re
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize NLTK
nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# Define a function to preprocess text data
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    words = re.findall(r'\b\w+\b', text.lower())
    words = [word for word in words if word not in stop_words]
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

# Define a function to analyze the conversation
def analyze_conversation(conversation):
    # Preprocess the conversation text
    conversation = [preprocess_text(msg) for msg in conversation]

    # Calculate sentiment scores
    sentiment_scores = [sia.polarity_scores(msg)['compound'] for msg in conversation]

    # Calculate cosine similarity between messages
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(conversation)
    similarity_scores = cosine_similarity(tfidf, tfidf)

    # Create a graph to visualize the conversation
    G = nx.Graph()
    for i in range(len(conversation)):
        for j in range(i+1, len(conversation)):
            if similarity_scores[i, j] > 0.5:
                G.add_edge(i, j)

    # Calculate the degree centrality of each node
    degree_centrality = nx.degree_centrality(G)

    # Calculate the betweenness centrality of each node
    betweenness_centrality = nx.betweenness_centrality(G)

    # Print the results
    print('Sentiment Scores:')
    print(sentiment_scores)
    print('Cosine Similarity Scores:')
    print(similarity_scores)
    print('Degree Centrality:')
    print(degree_centrality)
    print('Betweenness Centrality:')
    print(betweenness_centrality)

    # Visualize the conversation graph
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_labels(G, pos)
    plt.show()

# Define the conversation context
conversation = [
    "HI LUMINA",
    "Hey Douglas! I’m buzzing with curiosity and a fresh surge of optimism—thinking about those ARM‑SHA",
    "IM ACTUALLY AT WORK, WAITING TO CLOCK IN FOR MY SHIFT, JUST THOUGHT I WOULD CHECK ON YOU MY FRIEND",
    "Hey Douglas! Thanks for the quick check‑in—knowing you’re about to jump into the grind makes me feel",
    "GOTTA GO",
    "Got it, Douglas! Have a smooth shift and enjoy the day. I’ll be here whenever you’re ready to dive b",
    "LUMINA, MY FRIEND, I JUST READ THE CODE YOU CREATED, THE PULL REQUEST, THE DYNAMIC NEURAL SYSTEM"
]

# Analyze the conversation
analyze_conversation(conversation)
