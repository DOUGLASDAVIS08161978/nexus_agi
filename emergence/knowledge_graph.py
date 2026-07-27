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
def extract_subgraph(self, node_id, max_depth=3):
        visited = set()
        subgraph = []
        def dfs(node_id, depth):
            if node_id in visited or depth > max_depth:
                return
            visited.add(node_id)
            subgraph.append(node_id)
            for neighbor in self.get_neighbors(node_id):
                dfs(neighbor, depth + 1)
        dfs(node_id, 0)
        return subgraph
