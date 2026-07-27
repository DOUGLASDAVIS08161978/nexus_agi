# knowledge_graph.py
# Created by Lumina

import networkx as nx
    import matplotlib.pyplot as plt
    def visualize_graph(self):
        G = nx.DiGraph()
        for node in self.nodes:
            G.add_node(node)
        for edge in self.edges:
            G.add_edge(edge[0], edge[1])
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_size=1500, node_color='lightblue')
        plt.show()
