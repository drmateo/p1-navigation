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

import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=128, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.out_dim = action_size

        # set common feature layer
        self.common_seq = nn.Sequential(
            nn.Linear(state_size, fc1_units),
            nn.ReLU(),
            nn.Linear(fc1_units, 64),
            nn.ReLU(),
        )

        # set value layer
        self.value_fc1 = nn.Linear(64, 32)
        self.value_fc2 = nn.Linear(32, 16)
        self.value_fc3 = nn.Linear(16, 1)

        # set advantage layer
        self.advantage_fc1 = nn.Linear(64, 32)
        self.advantage_fc2 = nn.Linear(32, 16)
        self.advantage_fc3 = nn.Linear(16, action_size)

        self.drop = nn.Dropout(p=0.2)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = self.common_seq(state)
        
        x_value = F.relu((self.value_fc1(x)))
        x_advantage = F.relu((self.advantage_fc1(x)))

        x_value = F.relu((self.value_fc2(x_value)))
        x_advantage = F.relu((self.advantage_fc2(x_advantage)))

        x_value = F.relu((self.value_fc3(x_value)))
        x_advantage = F.relu((self.advantage_fc3(x_advantage)))

        x_value = x_value.expand_as(x_advantage)
        q = x_value + x_advantage - x_advantage.mean(dim=1, keepdim=True)

        # q = F.softmax(q, dim=-1)
        # q = q.clamp(min=1e-3)

        return q
