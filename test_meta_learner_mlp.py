"""
Example demonstrating the evolutionary meta-learning system with SimpleMLP.

This script shows how to use the MetaLearner to evolve a population of neural networks
for a simple classification task using evolutionary algorithms.
"""

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from meta_learner_mlp import SimpleMLP, MetaLearner


def create_synthetic_dataset(n_samples=1000, n_features=10, n_classes=3):
    """Create a simple synthetic classification dataset."""
    torch.manual_seed(42)
    X = torch.randn(n_samples, n_features)
    # Create class labels based on simple linear combination
    weights = torch.randn(n_features, n_classes)
    logits = X @ weights
    y = torch.argmax(logits, dim=1)
    return X, y


def accuracy_reward(outputs, targets, model, meta_state):
    """Reward function based on classification accuracy."""
    predictions = torch.argmax(outputs, dim=1)
    accuracy = (predictions == targets).float().mean()
    return accuracy


def main():
    # Configuration
    N_FEATURES = 10
    N_CLASSES = 3
    HIDDEN_DIM = 32
    POPULATION_SIZE = 10
    N_GENERATIONS = 20
    BATCH_SIZE = 64
    ADAPT_STEPS = 3

    print("=" * 60)
    print("Evolutionary Meta-Learning with SimpleMLP")
    print("=" * 60)

    # Create synthetic dataset
    print(f"\nCreating synthetic dataset...")
    X, y = create_synthetic_dataset(n_samples=1000, n_features=N_FEATURES, n_classes=N_CLASSES)
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    print(f"Dataset: {len(X)} samples, {N_FEATURES} features, {N_CLASSES} classes")

    # Define model constructor
    def model_ctor():
        return SimpleMLP(in_dim=N_FEATURES, hidden_dim=HIDDEN_DIM, out_dim=N_CLASSES)

    # Initialize meta-learner
    print(f"\nInitializing MetaLearner with population size {POPULATION_SIZE}")
    meta_learner = MetaLearner(
        model_ctor=model_ctor,
        population_size=POPULATION_SIZE,
        reward_fn=accuracy_reward,
        device="cpu"
    )

    # Evolve the population
    print(f"\nStarting evolution for {N_GENERATIONS} generations...")
    print("-" * 60)

    meta_learner.evolve(
        dataloader=dataloader,
        n_generations=N_GENERATIONS,
        adapt_steps=ADAPT_STEPS,
        elite_fraction=0.2,
        mutation_std=0.01,
        lr_mutation_std=0.1,
        loss_fn=nn.CrossEntropyLoss(),
        max_batches_per_gen=5
    )

    print("-" * 60)

    # Get best model
    best = meta_learner.best_model()
    print(f"\nBest model fitness: {best.fitness:.4f}")
    print(f"Best model learning rate: {best.lr:.6f}")

    # Test best model on full dataset
    print("\nEvaluating best model on full dataset...")
    best.model.eval()
    with torch.no_grad():
        outputs = best.model(X)
        predictions = torch.argmax(outputs, dim=1)
        accuracy = (predictions == y).float().mean()
        print(f"Final accuracy: {accuracy:.4f}")

    print("\n" + "=" * 60)
    print("Evolution complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
