import gym
import torch
import random
import numpy as np
from collections import deque
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# Définition du réseau de neurones pour l'estimation de la valeur de l'état
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


# Définition de l'algorithme Double DQN
class DoubleDQNAgent:
    def __init__(
        self,
        state_size,
        action_size,
        batch_size,
        gamma,
        epsilon,
        epsilon_decay,
        epsilon_min,
        is_test=False,
        model_path=None,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.memory = deque(maxlen=4000)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters())
        self.is_test = is_test
        self.model_path = model_path

        if self.is_test:
            self.epsilon = 0

        if self.model_path:
            self.model.load_state_dict(torch.load(self.model_path))
            self.update_target_model()

    # Ajout de l'expérience à la mémoire tampon de rejeu
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # Sélection d'actions en utilisant une politique epsilon-greedy
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_values = self.model(torch.Tensor(state))
            return torch.argmax(q_values).item()

    # Entraînement de l'agent à partir de la mémoire tampon de replay
    def train(self):
        # Si la mémoire tampon est insuffisante pour l'apprentissage, on sort de la fonction
        if len(self.memory) < self.batch_size:
            return

        # Échantillonnage d'un mini-batch à partir de la mémoire tampon de rejeu
        batch = random.sample(self.memory, self.batch_size)

        # Préparation de l'entrée pour le modèle d'estimation de la valeur
        states = []
        q_targets = []
        for state, action, reward, next_state, done in batch:
            state_tensor = torch.Tensor(state)
            next_state_tensor = torch.Tensor(next_state)
            q_values = self.model(state_tensor)

            # Calcul des cibles de la valeur TD
            if done:
                target = reward
            else:
                next_q_values = self.model(next_state_tensor)
                best_action = torch.argmax(next_q_values).item()
                target = (
                    reward
                    + self.gamma
                    * self.target_model(next_state_tensor)[best_action].item()
                )

            q_values[action] = target

            # Ajout de l'entrée et de la cible à la liste à utiliser pour l'apprentissage
            states.append(state)
            q_targets.append(q_values.detach().numpy())

        # Conversion de la liste en tenseurs PyTorch
        states_tensor = torch.Tensor(states)
        q_targets_tensor = torch.Tensor(q_targets)

        # Calcul de la fonction de perte et mise à jour du modèle
        loss = F.mse_loss(self.model(states_tensor), q_targets_tensor)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Mise à jour du modèle cible
        self.update_target_model()

    # Mise à jour du modèle cible
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    # Réduction de l'exploration au fil du temps
    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


