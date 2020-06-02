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

import time

from dueling_ddqn_agent import Agent


if __name__ == "__main__":

    env = UnityEnvironment(file_name="Banana.x86_64", seed=int(time.time()*1e6) % int(2**18))

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]
    action_size = brain.vector_action_space_size        # number of actions
    state_size = len(env_info.vector_observations[0])   # observation size
    k=2
    eps=0.001
    n_evaluations=100

    agent = Agent(state_size=state_size*k, action_size=action_size)
    agent.qnetwork_local.load_state_dict(torch.load('model.pt'))

    
    scores = []
    mean_score = 0
    for t in range(1, n_evaluations+1):
        score = 0
        env_info = env.reset(train_mode=False)[brain_name] # reset the environment
        state = env_info.vector_observations[0]            # get the current state

        state_stack = deque(maxlen=k)
        for i in range(1,k):
            state_stack.append(state)

        done = False
        while not done:
            state_stack.append(state)
            s = np.array(list(state_stack)).flatten()
            action = agent.act(s, eps=eps)

            env_info = env.step(action)[brain_name]             # send the action to the environment
            reward = env_info.rewards[0]                        # get the reward
            done = env_info.local_done[0]                       # see if episode has finished
            state = env_info.vector_observations[0]             # get the next state
            score += reward                                     # update the score
            
            print('\rEpisode {}\tCumulative Reward: {}, \tavg(CR): {:.2f}'.format(t, int(score), mean_score), end="")
        
        scores.append(score)
        mean_score = np.mean(scores)
        if t%10==0:
            print('\rEpisode {}\tCumulative Reward: {}, \tavg(CR): {:.2f}'.format(t, int(score), mean_score))
                
    env.close()