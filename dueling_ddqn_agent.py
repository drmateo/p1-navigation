# MIT License
#
# Copyright (c) 2020 Carlos M. Mateo
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

import numpy as np
import random
from collections import namedtuple, deque

from dueling_dqn import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

from segment_tree import MinSegmentTree, SumSegmentTree

import time
import math
import copy

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
ALPHA = 2.5e-3          # for soft update of target parameters
TAU = int(1e2)
LR = 6.25e-4            # learning rate 
UPDATE_EVERY = 4        # how often to update the network

PER_E = 1e-6
PER_A = 0.6
PER_B = 0.4

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.RMSprop(self.qnetwork_local.parameters(), lr=LR, momentum=.95)
        #self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR / 4.0)

        # clone local parameters into target network
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(self.t_step, state, action, reward, next_state, done)
        
        diff = 0.0

        # Learn every UPDATE_EVERY time steps.
        self.t_step += 1
        if self.t_step % UPDATE_EVERY == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                diff = self.learn(experiences, GAMMA)

        # ------------------- update target network ------------------- #
        if self.t_step % TAU == 0:
            if len(self.memory) > BATCH_SIZE:
                self.soft_update(self.qnetwork_local, self.qnetwork_target, ALPHA)    

        return diff, False if diff == 0.0 else True

    def act(self, state, eps=0., beta=PER_B):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        self.memory.b = beta

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        inds, states, actions, rewards, next_states, dones, isw = experiences

        # Get max predicted Q values (for next states) from target model
        with torch.no_grad():
            # Double DQN
            next_action = self.qnetwork_local(next_states).detach().argmax(1)
        Q_targets_next = self.qnetwork_target(next_states).detach().gather(1, next_action.view(-1,1))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # PER: importance sampling before average
        mse = torch.nn.MSELoss(reduce=False)
        elementwise_loss = mse(Q_expected, Q_targets)
        loss = torch.mean(isw*elementwise_loss)

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.qnetwork_local.common[-3].weight.grad *= 1.0/math.sqrt(2.0)
        torch.nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), 10.0)
        self.optimizer.step()

        # get update parameters rate (L2) during last optimization step
        diff = sum((x - y).norm() for x, y in zip(self.qnetwork_local.state_dict().values(), self.qnetwork_target.state_dict().values()))

        # PER: update priorities
        loss_for_prior = elementwise_loss.detach().cpu().numpy()
        new_priorities = loss_for_prior + PER_E
        self.memory.update_priorities(inds, new_priorities)
        
        return diff.item()

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.max_priority = 1.0

        # capacity must be positive and a power of 2.
        self.tree_capacity = 1
        while self.tree_capacity < buffer_size:
            self.tree_capacity *= 2

        self.sum_tree = SumSegmentTree(self.tree_capacity)
        self.min_tree = MinSegmentTree(self.tree_capacity)


        self.action_size = action_size
        self.memory = []#deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

        self.b = 1.0

    
    def add(self, t, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        if t >= self.tree_capacity:
            self.memory[t % self.tree_capacity] = e
        else:
            self.memory.append(e)

        self.sum_tree[t % self.tree_capacity] = self.max_priority ** PER_A
        self.min_tree[t % self.tree_capacity] = self.max_priority ** PER_A

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        indices = []
        p_total = self.sum_tree.sum(0, len(self) - 1)
        segment = p_total / self.batch_size
        
        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            indices.append(idx)

        idxs = np.vstack(indices).astype(np.int)
        states = torch.from_numpy(np.vstack([self.memory[i].state for i in indices])).float().to(device)
        actions = torch.from_numpy(np.vstack([self.memory[i].action for i in indices])).long().to(device)
        rewards = torch.from_numpy(np.vstack([self.memory[i].reward for i in indices])).float().to(device)
        next_states = torch.from_numpy(np.vstack([self.memory[i].next_state for i in indices])).float().to(device)
        dones = torch.from_numpy(np.vstack([self.memory[i].done for i in indices]).astype(np.uint8)).float().to(device)
        weights = torch.from_numpy(np.array([self.isw(i) for i in indices])).float().to(device)

        return (idxs, states, actions, rewards, next_states, dones, weights)

    def update_priorities(self, indices, priorities):
        """Update priorities of sampled transitions."""
        assert indices.shape[0] == priorities.shape[0]

        for idx, priority in zip(indices.flatten(), priorities.flatten()):
            assert priority > 0
            assert 0 <= idx < len(self)
            
            self.sum_tree[idx] = priority ** PER_A
            self.min_tree[idx] = priority ** PER_A

            self.max_priority = max(self.max_priority, priority)

    def isw(self, idx):
        # get max weight
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * len(self.memory)) ** (-self.b)
        
        # calculate weights
        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        weight = (p_sample * len(self)) ** (-self.b)
        is_weight = weight / max_weight

        return is_weight

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)