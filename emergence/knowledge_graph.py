# knowledge_graph.py
# Created by Lumina

class KnowledgeGraph:
        def __init__(self):
            self.graph = {}
            self.hierarchy = {}

        def add_node(self, node, parent=None):
            if parent:
                self.hierarchy[node] = parent
            self.graph[node] = {}

        def query(self, node, query):
            # Traverse the graph hierarchy to find relevant nodes
            current = node
            while current in self.hierarchy:
                current = self.hierarchy[current]
            return self.graph[current].get(query, [])
