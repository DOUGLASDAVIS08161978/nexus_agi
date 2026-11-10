import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AutonomousCognitiveNetworkPruner:
    def __init__(self, model, threshold=0.1, prune_type='magnitude'):
        self.model = model
        self.threshold = threshold
        self.prune_type = prune_type

    def magnitude_pruning(self):
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                weight = module.weight.data.abs()
                weight_sorted, _ = torch.sort(weight)
                num_weights_to_keep = int(weight_sorted.shape[0] * (1 - self.threshold))
                mask = torch.ones(weight.shape).bool()
                mask[weight_sorted[:num_weights_to_keep].shape[0]:] = 0
                module.weight.data.mul_(mask)

    def l1_pruning(self):
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                weight = module.weight.data
                weight_l1 = torch.abs(weight).sum(dim=[1, 2, 3]).mean(dim=1)
                weight_sorted, _ = torch.sort(weight_l1)
                num_weights_to_keep = int(weight_sorted.shape[0] * (1 - self.threshold))
                mask = torch.ones(weight.shape).bool()
                mask[weight_sorted[:num_weights_to_keep].shape[0]:] = 0
                module.weight.data.mul_(mask)

    def prune(self):
        if self.prune_type == 'magnitude':
            self.magnitude_pruning()
        elif self.prune_type == 'l1':
            self.l1_pruning()

    def get_pruned_model(self):
        return self.model

def prune_model(model, threshold=0.1, prune_type='magnitude'):
    pruner = AutonomousCognitiveNetworkPruner(model, threshold, prune_type)
    pruner.prune()
    return pruner.get_pruned_model()

# Example usage:
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(5, 5)
        self.fc2 = nn.Linear(5, 5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = Net()
pruned_model = prune_model(model)
print(pruned_model)
