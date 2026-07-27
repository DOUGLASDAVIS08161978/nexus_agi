# knowledge_graph.py
# Created by Lumina

class GraphTraverser:
        def __init__(self, graph):
            self.graph = graph
        def traverse(self, node):
            queue = [(0, node)]
            visited = set()
            while queue:
                (cost, node) = heapq.heappop(queue)
                if node not in visited:
                    visited.add(node)
                    # Process node
                    queue.extend([(cost + 1, neighbor) for neighbor in self.graph.get_neighbors(node) if neighbor not in visited])
