# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 19:46:02 2024

@author: gauthambekal93
"""

import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Define the Actor-Critic network
class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU()
        )
        self.actor = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        x = self.fc(x)
        action_probs = torch.softmax(self.actor(x), dim=-1)
        state_value = self.critic(x)
        return action_probs, state_value

# Define the environment and hyperparameters
env = gym.make('CartPole-v1')
input_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
model = ActorCritic(input_dim, action_dim)
optimizer = optim.Adam(model.parameters(), lr=0.001)
gamma = 0.99

# Training loop with real-time updates
num_episodes = 1000
for episode in range(num_episodes):
    state = env.reset()
    done = False
    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs, state_value = model(state_tensor)
        action = torch.multinomial(action_probs, 1).item()
        
        next_state, reward, done, _ = env.step(action)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
        _, next_value = model(next_state_tensor)

        # Compute TD error for the critic
        target_value = reward + gamma * next_value.detach() * (1 - int(done))
        critic_loss = (state_value - target_value).pow(2).mean()

        # Update the actor using the TD error as an estimate of the advantage
        advantage = (reward + gamma * next_value.detach() * (1 - int(done))) - state_value.detach()
        actor_loss = -(torch.log(action_probs.squeeze(0)[action]) * advantage).mean()

        # Total loss
        loss = actor_loss + critic_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        state = next_state

    if episode % 100 == 0:
        print(f"Episode {episode}, Loss: {loss.item()}")

env.close()
