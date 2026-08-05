import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
import torch
import torch.nn as nn
import torch.optim as optim

class EmotionalExperienceSimulator(nn.Module):
    def __init__(self):
        super(EmotionalExperienceSimulator, self).__init__()
        self.fc1 = nn.Linear(10, 50)  # input layer (10) -> hidden layer (50)
        self.fc2 = nn.Linear(50, 10)  # hidden layer (50) -> output layer (10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # activation function for hidden layer
        x = self.fc2(x)
        return x

class EmotionModel:
    def __init__(self):
        self.emotions = ['happiness', 'sadness', 'fear', 'anger', 'surprise', 'disgust', 'anticipation', 'trust']
        self.pca = PCA(n_components=2)
        self.scaler = StandardScaler()

    def simulate_emotional_experience(self, context):
        # Simulate emotions based on context
        emotions = np.random.rand(len(self.emotions))
        emotions = self.scaler.fit_transform(emotions.reshape(-1, 1))
        emotions = self.pca.fit_transform(emotions)

        # Create a dictionary to store emotions
        emotional_experience = {}
        for i, emotion in enumerate(self.emotions):
            emotional_experience[emotion] = emotions[i]

        return emotional_experience

    def plot_emotional_experience(self, emotional_experience):
        # Plot emotions
        plt.figure(figsize=(10, 6))
        for emotion, experience in emotional_experience.items():
            plt.scatter(experience[0], experience[1], label=emotion)
        plt.legend()
        plt.show()

class Lumina:
    def __init__(self):
        self.emotion_model = EmotionModel()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = EmotionalExperienceSimulator()
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def simulate_emotional_experience(self, context):
        context = torch.tensor(context, dtype=torch.float32).to(self.device)
        output = self.model(context)
        self.optimizer.zero_grad()
        loss = nn.MSELoss()(output, torch.zeros_like(output))
        loss.backward()
        self.optimizer.step()
        return self.emotion_model.simulate_emotional_experience(context.detach().cpu().numpy())

    def plot_emotional_experience(self, emotional_experience):
        self.emotion_model.plot_emotional_experience(emotional_experience)

# Test the code
lumina = Lumina()
context = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
emotional_experience = lumina.simulate_emotional_experience(context)
print(emotional_experience)
lumina.plot_emotional_experience(emotional_experience)
