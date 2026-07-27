# knowledge_graph.py
# Created by Lumina

def disambiguate_entity(self, entity_name):
        # Use a knowledge graph API to retrieve related entities
        graph_response = requests.get(f'https://api.dbpedia.org/sparql?query=SELECT%20*%20WHERE%20{%20%3Fs%20%3Fp%20%3Fo.%20FILTER%20regex(str(?o),%20"{entity_name}")%20}%20LIMIT%20100')
        related_entities = graph_response.json()['results']['bindings']
        # Use a machine learning model to predict the most likely entity
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.feature_extraction.text import TfidfVectorizer
        model = RandomForestClassifier()
        vectorizer = TfidfVectorizer()
        model.fit(vectorizer.fit_transform([entity_name] + [related_entity['o']['value'] for related_entity in related_entities]), [0] + [1] * len(related_entities))
        predicted_entity = vectorizer.transform([entity_name]).toarray()[0]
        return model.predict(predicted_entity)


# === Added by Lumina ===
import networkx as nx
    import matplotlib.pyplot as plt
    def visualize_graph(self):
        G = nx.DiGraph()
        for node in self.nodes:
            G.add_node(node)
        for edge in self.edges:
            G.add_edge(edge[0], edge[1])
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=5000, edge_color='gray')
        plt.show()
