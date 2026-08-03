import networkx as nx
import numpy as np
from scipy import spatial
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

class KnowledgeGraph:
    def __init__(self):
        self.G = nx.DiGraph()
        self.vectorizer = TfidfVectorizer()
        self.knowledge_base = {}

    def add_node(self, concept, description):
        self.G.add_node(concept)
        self.knowledge_base[concept] = description

    def add_edge(self, concept1, concept2, relation):
        self.G.add_edge(concept1, concept2, relation=relation)

    def get_similarity(self, concept1, concept2):
        if concept1 not in self.knowledge_base or concept2 not in self.knowledge_base:
            return 0
        description1 = self.knowledge_base[concept1]
        description2 = self.knowledge_base[concept2]
        vector1 = self.vectorizer.transform([description1])
        vector2 = self.vectorizer.transform([description2])
        return cosine_similarity(vector1, vector2)[0][0]

    def save_graph(self, filename):
        nx.write_gpickle(self.G, filename)
        with open('vectorizer.pkl', 'wb') as f:
            pickle.dump(self.vectorizer, f)

    def load_graph(self, filename):
        self.G = nx.read_gpickle(filename)
        with open('vectorizer.pkl', 'rb') as f:
            self.vectorizer = pickle.load(f)

    def get_path(self, concept1, concept2):
        try:
            return nx.shortest_path(self.G, source=concept1, target=concept2)
        except nx.NetworkXNoPath:
            return None

    def get_path_length(self, concept1, concept2):
        path = self.get_path(concept1, concept2)
        if path is None:
            return float('inf')
        return len(path)

    def get_neighbors(self, concept):
        return list(self.G.neighbors(concept))

def main():
    graph = KnowledgeGraph()
    graph.add_node('Bitcoin', 'A digital currency')
    graph.add_node('Mining', 'The process of verifying transactions')
    graph.add_edge('Bitcoin', 'Mining', 'uses')
    graph.save_graph('knowledge_graph.pkl')

if __name__ == "__main__":
    main()
