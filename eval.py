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

import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

from unityagents import UnityEnvironment
import numpy as np

from dqn_agent import Agent


if __name__ == "__main__":

    env = UnityEnvironment(file_name="Banana.x86_64", seed=0)

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    agent = Agent(state_size=37, action_size=4, seed=0)
    agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))
    
    for i in range(3):
        score = 0
        env_info = env.reset(train_mode=False)[brain_name] # reset the environment
        state = env_info.vector_observations[0]            # get the current state
        while True:
            action = agent.act(state)
            env_info = env.step(action)[brain_name]        # send the action to the environment
            next_state = env_info.vector_observations[0]   # get the next state
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0]                  # see if episode has finished
            state = next_state                             # roll over the state to next time step
            score += reward                                # update the score
            if done:                                       # exit loop if episode finished
                break

        print('\rEpisode {}\t Cumulative reward: {:.2f}'.format(i, score))
                
    env.close()