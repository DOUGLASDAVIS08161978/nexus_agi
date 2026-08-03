import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.utils import to_networkx
from torch_geometric.data import InMemoryDataset
from torch_geometric.nn import global_mean_pool
import networkx as nx
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import os

class EmergentKnowledgeGraphEmbedder(nn.Module):
    def __init__(self, num_nodes, num_relations, hidden_dim, output_dim, num_heads):
        super(EmergentKnowledgeGraphEmbedder, self).__init__()
        self.num_nodes = num_nodes
        self.num_relations = num_relations
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_heads = num_heads

        self.node_embedding = nn.Embedding(num_nodes, hidden_dim)
        self.relation_embedding = nn.Embedding(num_relations, hidden_dim)
        self.gnn = GATConv(hidden_dim, hidden_dim, heads=num_heads, concat=True)
        self.fc = nn.Linear(hidden_dim * num_heads, output_dim)

    def forward(self, node_idx, relation_idx):
        node_embedding = self.node_embedding(node_idx)
        relation_embedding = self.relation_embedding(relation_idx)
        node_embedding = node_embedding + relation_embedding

        node_embedding = self.gnn(node_embedding, edge_index)
        node_embedding = global_mean_pool(node_embedding, edge_index)

        output = self.fc(node_embedding)
        return output

class KnowledgeGraphDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(KnowledgeGraphDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['raw_data.pt']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        data = torch.load(self.raw_paths[0])
        data, slices = self.collate(data)
        torch.save((data, slices), self.processed_paths[0])

    def download(self):
        pass

def load_data(dataset_path):
    dataset = KnowledgeGraphDataset(dataset_path)
    data = dataset[0]
    return data

def train(model, device, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for batch in loader:
        node_idx = batch.node_idx.to(device)
        relation_idx = batch.relation_idx.to(device)
        edge_index = batch.edge_index.to(device)
        labels = batch.labels.to(device)

        optimizer.zero_grad()
        outputs = model(node_idx, relation_idx)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def test(model, device, loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in loader:
            node_idx = batch.node_idx.to(device)
            relation_idx = batch.relation_idx.to(device)
            edge_index = batch.edge_index.to(device)
            labels = batch.labels.to(device)

            outputs = model(node_idx, relation_idx)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    return total_loss / len(loader)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_nodes = 100
    num_relations = 10
    hidden_dim = 128
    output_dim = 128
    num_heads = 8
    epochs = 100
    batch_size = 32
    learning_rate = 0.01

    model = EmergentKnowledgeGraphEmbedder(num_nodes, num_relations, hidden_dim, output_dim, num_heads)
    model.to(device)

    dataset_path = 'data'
    dataset = load_data(dataset_path)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        loss = train(model, device, loader, optimizer, criterion)
        print(f'Epoch {epoch+1}, Loss: {loss:.4f}')

    test_loss = test(model, device, loader, criterion)
    print(f'Test Loss: {test_loss:.4f}')

if __name__ == '__main__':
    main()
