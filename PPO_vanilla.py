import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from torch.distributions import Categorical

import random
import numpy as np
import time

import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"

class Memory:
    def __init__(self):
        self.states = []
        self.old_action_probs = []
        self.actions = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.states[:]
        del self.old_action_probs[:]
        del self.actions[:]
        del self.rewards[:]
        del self.is_terminals[:]



class PPO_Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size):
        super(PPO_Actor, self).__init__()
        self.state_dim = state_dim

        self.l1 = nn.Linear(state_dim, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, action_dim)

    def forward(self, state):
        state = state.reshape(-1,self.state_dim)
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        action_values = self.l3(x)

        action_probs = F.softmax(action_values,dim=-1)

        dist = Categorical(action_probs)
        action = dist.sample()

        return action_probs[:,action], action

    def policy(self, state, action):
        action = action.flatten()
        state = state.reshape(-1,self.state_dim)
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        action_values = self.l3(x)

        action_probs = F.softmax(action_values,dim=-1)

        return action_probs[np.arange(0,action_probs.shape[0]),action]


class PPO_Critic(nn.Module):
    def __init__(self, state_dim, hidden_size):
        super(PPO_Critic, self).__init__()

        self.l1 = nn.Linear(state_dim, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, 1)

    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        state_value = self.l3(x)
        return state_value



class PPO:
    def __init__(self, state_dim, action_dim=1, hidden_size=32, lr_actor=0.001, lr_critic=0.001, weight_decay=0.1, gamma=0.99, K=10, eps_clip=0.1, epsilon=0.05, epsilon_decay=0.999, beta=1, use_gae=True, delta=1, mode="CLIP"):

        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.weight_decay = weight_decay
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K = K
        self.beta = beta
        self.delta = delta
        self.mode = mode
        self.action_dim = action_dim
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.use_gae = use_gae

        self.memory = Memory()

        self.actor = PPO_Actor(state_dim, action_dim, hidden_size).to(device)
        self.critic = PPO_Critic(state_dim, hidden_size).to(device)

        self.actor_optimizer = torch.optim.RMSprop(self.actor.parameters(), lr=self.lr_actor, weight_decay=weight_decay)
        self.critic_optimizer = torch.optim.RMSprop(self.critic.parameters(), lr=self.lr_critic, weight_decay=weight_decay)

        self.old_actor = PPO_Actor(state_dim, action_dim, hidden_size).to(device)
        self.old_actor.load_state_dict(self.actor.state_dict())

        self.MSELoss = nn.MSELoss()
        self.KLDivLoss = nn.KLDivLoss(reduction="batchmean")

    def apply_noise_decay(self):
        self.epsilon = self.epsilon * self.epsilon_decay

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return self.random_action()

        return self.old_actor(state)

    def rewards_to_go(self, rewards, is_terminals):
        rewards_to_go = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(rewards.tolist()), reversed(is_terminals.tolist())):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards_to_go.insert(0, discounted_reward)

        rewards_to_go = torch.tensor(rewards_to_go).to(device)

        return rewards_to_go


    def update(self):

        states = torch.stack(self.memory.states).to(device).detach()
        actions = torch.LongTensor(self.memory.actions).to(device).flatten().detach()
        rewards = torch.Tensor(self.memory.rewards).to(device).flatten().detach()
        is_terminals = torch.Tensor(self.memory.is_terminals).to(device).flatten().detach()
        old_action_probs = torch.stack(self.memory.old_action_probs).to(device).flatten().detach()

        rewards_to_go = self.rewards_to_go(rewards, is_terminals)

        for _ in range(self.K):

            action_probs = self.actor.policy(states, actions)
            state_values = self.critic(torch.Tensor(states))

            advantages = rewards_to_go - state_values.detach()

            if self.mode == "CLIP":
                ratios = action_probs / old_action_probs
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
                loss_actor = - torch.min(surr1,surr2).mean()

            elif self.mode == "KL":
                kl = self.KLDivLoss(old_action_probs,action_probs)
                if kl>1.5*self.delta:
                    self.beta *= 2
                else:
                    self.beta /= 2
                loss_actor = - (torch.log(action_probs) * advantages).mean() - self.beta*kl
            else :
                loss_actor = - (torch.log(action_probs) * advantages).mean()

            loss_critic =  self.MSELoss(state_values.flatten(), rewards_to_go.flatten())

            self.actor_optimizer.zero_grad()
            loss_actor.backward()
            self.actor_optimizer.step()

            self.critic_optimizer.zero_grad()
            loss_critic.backward()
            self.critic_optimizer.step()

        self.memory.clear_memory()
        self.old_actor.load_state_dict(self.actor.state_dict())

    def random_action(self):
        actions_probs = np.random.rand(self.action_dim)
        actions_probs = torch.Tensor(actions_probs / np.sum(actions_probs))

        dist = Categorical(actions_probs)

        action = dist.sample()
        action = action

        return dist.probs[action].view(1, 1), action
