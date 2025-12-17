import copy
import math
import random
from typing import Callable, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class SimpleMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class EvolvableModel:
    """Wrapper holding a PyTorch model plus meta info."""

    def __init__(self, model: nn.Module, lr: float = 1e-3):
        self.model = model
        self.lr = lr
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.fitness = None

    def clone(self) -> "EvolvableModel":
        new = EvolvableModel(
            model=copy.deepcopy(self.model),
            lr=self.lr,
        )
        return new

    def train_step(self, x, y, loss_fn):
        self.model.train()
        self.opt.zero_grad()
        logits = self.model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        self.opt.step()
        return logits.detach(), loss.detach()

    def evaluate(self, x, y, loss_fn):
        self.model.eval()
        with torch.no_grad():
            logits = self.model(x)
            loss = loss_fn(logits, y)
        return logits, loss

    def mutate(self, weight_noise_std: float = 0.01, lr_noise_std: float = 0.1):
        # Add Gaussian noise to parameters
        with torch.no_grad():
            for p in self.model.parameters():
                p.add_(torch.randn_like(p) * weight_noise_std)
        # Perturb learning rate
        factor = math.exp(random.gauss(0.0, lr_noise_std))
        self.lr *= factor
        for g in self.opt.param_groups:
            g["lr"] = self.lr


class MetaLearner:
    """
    Evolutionary meta-learner over a population of models for supervised tasks.
    reward_fn: callable(outputs, targets, model, meta_state) -> float tensor (higher is better)
    """

    def __init__(
        self,
        model_ctor: Callable[[], nn.Module],
        population_size: int,
        reward_fn: Callable[[torch.Tensor, torch.Tensor, nn.Module, Dict], torch.Tensor],
        device: str = "cpu",
    ):
        self.device = device
        self.population: List[EvolvableModel] = [
            EvolvableModel(model_ctor().to(device)) for _ in range(population_size)
        ]
        self.reward_fn = reward_fn
        self.meta_state: Dict = {}

    def _evaluate_population(
        self,
        batch_x: torch.Tensor,
        batch_y: torch.Tensor,
        loss_fn,
        adapt_steps: int,
    ):
        rewards = []
        for indiv in self.population:
            # Fast adaptation
            for _ in range(adapt_steps):
                _logits, _loss = indiv.train_step(batch_x, batch_y, loss_fn)

            # Post-adaptation evaluation for reward
            logits, loss = indiv.evaluate(batch_x, batch_y, loss_fn)
            reward = self.reward_fn(logits, batch_y, indiv.model, self.meta_state)
            indiv.fitness = reward.item()
            rewards.append(reward)
        return torch.stack(rewards)

    def _select_and_reproduce(
        self,
        elite_fraction: float,
        mutation_std: float,
        lr_mutation_std: float,
    ):
        # Sort by fitness descending
        self.population.sort(key=lambda m: m.fitness if m.fitness is not None else -1e9, reverse=True)
        n_elite = max(1, int(len(self.population) * elite_fraction))
        elites = self.population[:n_elite]

        new_population: List[EvolvableModel] = []
        # Keep elites
        for e in elites:
            new_population.append(e)

        # Fill rest with mutated clones
        while len(new_population) < len(self.population):
            parent = random.choice(elites)
            child = parent.clone()
            child.mutate(weight_noise_std=mutation_std, lr_noise_std=lr_mutation_std)
            new_population.append(child)

        self.population = new_population

    def evolve(
        self,
        dataloader: DataLoader,
        n_generations: int,
        adapt_steps: int = 1,
        elite_fraction: float = 0.2,
        mutation_std: float = 0.01,
        lr_mutation_std: float = 0.1,
        loss_fn: Callable = None,
        max_batches_per_gen: int = 1,
    ):
        if loss_fn is None:
            loss_fn = nn.CrossEntropyLoss()

        for gen in range(n_generations):
            for i, (x, y) in enumerate(dataloader):
                if i >= max_batches_per_gen:
                    break
                x = x.to(self.device)
                y = y.to(self.device)

                rewards = self._evaluate_population(x, y, loss_fn, adapt_steps)
                self.meta_state["last_gen"] = gen
                self.meta_state["last_rewards"] = rewards.detach().cpu()

            self._select_and_reproduce(
                elite_fraction=elite_fraction,
                mutation_std=mutation_std,
                lr_mutation_std=lr_mutation_std,
            )
            best_fitness = max(m.fitness for m in self.population if m.fitness is not None)
            print(f"Generation {gen}: best fitness = {best_fitness:.4f}")

    def best_model(self) -> EvolvableModel:
        return max(self.population, key=lambda m: m.fitness if m.fitness is not None else -1e9)
