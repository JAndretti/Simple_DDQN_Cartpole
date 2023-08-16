import gym
import torch
import random
import numpy as np
from collections import deque
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from agent import DoubleDQNAgent
import matplotlib.pyplot as plt

game = 'CartPole-v1'

# Instanciation de l'environnement et de l'agent
env = gym.make(game)
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = DoubleDQNAgent(state_size, action_size, batch_size=32, gamma=0.99, epsilon=0, epsilon_decay=0, epsilon_min=0, is_test=True)
model_path = f"{game}_ddqn.pth"

agent.model.load_state_dict(torch.load(model_path))

# Boucle de jeu
for i_episode in range(5):
    state = env.reset()[0]
    total_reward = 0
    for t in range(200):
        # Affichage du rendu graphique de l'environnement
        env.render()
        
        action = agent.act(state)

        next_state, reward, done, trunc, info = env.step(action)
        total_reward += reward
        
        state = next_state
        
        
        if t == 199:
            print("finish with reward : {}".format(total_reward))
            break
        elif done:
            print("Episode finished after {} timesteps. Total reward: {}".format(t+1, total_reward))
            break


# Fermeture de l'environnement
env.close()
