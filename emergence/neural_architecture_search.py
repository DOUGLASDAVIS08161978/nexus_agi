# neural_architecture_search.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random
from datetime import datetime
from functools import reduce
import operator
from collections import namedtuple
from itertools import product

# Define a namedtuple to represent a neural architecture
Architecture = namedtuple('Architecture', ['input_dim', 'hidden_layers', 'output_dim', 'activation', 'optimizer', 'learning_rate'])

# Define a function to generate all possible architectures
def generate_architectures(input_dim, output_dim, max_hidden_layers, hidden_layer_sizes, activation_functions, optimizers, learning_rates):
    """
    Generate all possible neural architectures.

    Args:
    - input_dim (int): The input dimension of the neural network.
    - output_dim (int): The output dimension of the neural network.
    - max_hidden_layers (int): The maximum number of hidden layers.
    - hidden_layer_sizes (list): A list of possible hidden layer sizes.
    - activation_functions (list): A list of possible activation functions.
    - optimizers (list): A list of possible optimizers.
    - learning_rates (list): A list of possible learning rates.

    Returns:
    - A list of all possible architectures.
    """
    architectures = []
    for hidden_layers in range(1, max_hidden_layers + 1):
        for layer_sizes in product(hidden_layer_sizes, repeat=hidden_layers):
            for activation in activation_functions:
                for optimizer in optimizers:
                    for learning_rate in learning_rates:
                        architecture = Architecture(
                            input_dim=input_dim,
                            hidden_layers=layer_sizes,
                            output_dim=output_dim,
                            activation=activation,
                            optimizer=optimizer,
                            learning_rate=learning_rate
                        )
                        architectures.append(architecture)
    return architectures

# Define a function to train a neural network for a given architecture
def train_network(model, device, train_loader, test_loader, epochs):
    """
    Train a neural network for a given architecture.

    Args:
    - model (nn.Module): The neural network model.
    - device (torch.device): The device to train on (e.g., GPU or CPU).
    - train_loader (DataLoader): The training data loader.
    - test_loader (DataLoader): The testing data loader.
    - epochs (int): The number of training epochs.

    Returns:
    - The trained model.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = getattr(optim, model.architecture.optimizer)(model.parameters(), lr=model.architecture.learning_rate)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        model.eval()
        with torch.no_grad():
            total_correct = 0
            for batch in test_loader:
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, dim=1)
                total_correct += (predicted == labels).sum().item()
        print(f'Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}, Accuracy: {total_correct / len(test_loader.dataset)}')
    return model

# Define a function to evaluate a neural network for a given architecture
def evaluate_network(model, device, test_loader):
    """
    Evaluate a neural network for a given architecture.

    Args:
    - model (nn.Module): The neural network model.
    - device (torch.device): The device to evaluate on (e.g., GPU or CPU).
    - test_loader (DataLoader): The testing data loader.

    Returns:
    - The accuracy of the neural network.
    """
    model.eval()
    with torch.no_grad():
        total_correct = 0
        for batch in test_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, dim=1)
            total_correct += (predicted == labels).sum().item()
    return total_correct / len(test_loader.dataset)

# Define a function to perform neural architecture search
def neural_architecture_search(input_dim, output_dim, max_hidden_layers, hidden_layer_sizes, activation_functions, optimizers, learning_rates, train_loader, test_loader, epochs):
    """
    Perform neural architecture search.

    Args:
    - input_dim (int): The input dimension of the neural network.
    - output_dim (int): The output dimension of the neural network.
    - max_hidden_layers (int): The maximum number of hidden layers.
    - hidden_layer_sizes (list): A list of possible hidden layer sizes.
    - activation_functions (list): A list of possible activation functions.
    - optimizers (list): A list of possible optimizers.
    - learning_rates (list): A list of possible learning rates.
    - train_loader (DataLoader): The training data loader.
    - test_loader (DataLoader): The testing data loader.
    - epochs (int): The number of training epochs.

    Returns:
    - The best architecture and its corresponding accuracy.
    """
    architectures = generate_architectures(input_dim, output_dim, max_hidden_layers, hidden_layer_sizes, activation_functions, optimizers, learning_rates)
    best_accuracy = 0
    best_architecture = None
    for architecture in architectures:
        model = nn.Sequential()
        for i, layer_size in enumerate(architecture.hidden_layers):
            model.add_module(f'hidden_layer_{i}', nn.Linear(architecture.input_dim if i == 0 else architecture.hidden_layers[i-1], layer_size))
            model.add_module(f'activation_{i}', nn.ReLU() if architecture.activation == 'relu' else nn.Tanh())
            model.add_module(f'dropout_{i}', nn.Dropout(p=0.2))
            architecture.input_dim = layer_size
        model.add_module('output_layer', nn.Linear(architecture.input_dim, architecture.output_dim))
        model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        model = train_network(model, torch.device('cuda' if torch.cuda.is_available() else 'cpu'), train_loader, test_loader, epochs)
        accuracy = evaluate_network(model, torch.device('cuda' if torch.cuda.is_available() else 'cpu'), test_loader)
        print(f'Architecture: {architecture}')
        print(f'Accuracy: {accuracy}')
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_architecture = architecture
    return best_architecture, best_accuracy

# Example usage
if __name__ == '__main__':
    input_dim = 784
    output_dim = 10
    max_hidden_layers = 3
    hidden_layer_sizes = [128, 256, 512]
    activation_functions = ['relu', 'tanh']
    optimizers = ['adam', 'sgd']
    learning_rates = [0.001, 0.01]
    epochs = 10
    train_loader = DataLoader(torch.randn(100, input_dim), batch_size=32)
    test_loader = DataLoader(torch.randn(100, input_dim), batch_size=32)
    best_architecture, best_accuracy = neural_architecture_search(input_dim, output_dim, max_hidden_layers, hidden_layer_sizes, activation_functions, optimizers, learning_rates, train_loader, test_loader, epochs)
    print(f'Best Architecture: {best_architecture}')
    print(f'Best Accuracy: {best_accuracy}')
This code defines a module for neural architecture search. It generates all possible neural architectures, trains each architecture on a given dataset, and evaluates the performance of each architecture. The best architecture and its corresponding accuracy are returned. The code includes example usage at the end.
