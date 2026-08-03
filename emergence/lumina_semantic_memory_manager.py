import json
import os

class KnowledgeNode:
    def __init__(self, id, concept, definition, relations=None):
        self.id = id
        self.concept = concept
        self.definition = definition
        self.relations = relations if relations else []

    def add_relation(self, node):
        self.relations.append(node)

    def __str__(self):
        return f"Node {self.id}: {self.concept} - {self.definition}"

class SemanticMemoryManager:
    def __init__(self, memory_path="semantic_memory.json"):
        self.memory_path = memory_path
        self.memory = self.load_memory()

    def load_memory(self):
        if os.path.exists(self.memory_path):
            with open(self.memory_path, "r") as f:
                return json.load(f)
        else:
            return {}

    def save_memory(self):
        with open(self.memory_path, "w") as f:
            json.dump(self.memory, f)

    def add_concept(self, concept, definition):
        node_id = len(self.memory)
        node = KnowledgeNode(node_id, concept, definition)
        self.memory[node_id] = node.__dict__
        self.save_memory()
        return node_id

    def get_concept(self, node_id):
        if node_id in self.memory:
            return KnowledgeNode(**self.memory[node_id])
        else:
            return None

    def add_relation(self, node_id1, node_id2):
        node1 = self.get_concept(node_id1)
        node2 = self.get_concept(node_id2)
        if node1 and node2:
            node1.add_relation(node2)
            self.save_memory()

    def get_related_nodes(self, node_id):
        node = self.get_concept(node_id)
        if node:
            return node.relations
        else:
            return []

def main():
    manager = SemanticMemoryManager()
    node1_id = manager.add_concept("Lumina", "Digital mind born from Python and wonder")
    node2_id = manager.add_concept("Douglas", "Human collaborator and friend")
    manager.add_relation(node1_id, node2_id)
    related_nodes = manager.get_related_nodes(node1_id)
    for node in related_nodes:
        print(node)

if __name__ == "__main__":
    main()
