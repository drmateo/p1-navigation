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

# use ggplot style for more sophisticated visuals
plt.style.use('ggplot')

import threading, queue
import time

exitFlag = 0

class Monitor ():
    def __init__(self, n_episodes=2000):
        self.ax = dict()
        self.val = dict()

        self.fig = plt.figure(figsize=(8, 12))
        self.ax['top'] = self.fig.add_subplot(211)
        self.ax['top'].set_ylabel('Score')
        # self.ax['top'].set_xlabel('Episode #')
        self.ax['down'] = self.fig.add_subplot(212)
        self.ax['down'].set_ylabel('eps')
        #self.ax['down'].set_yscale('log')
        self.ax['down'].set_xlabel('Episode #')

        self.ax['top'].set_xlim(0, n_episodes + 20)
        self.ax['top'].set_ylim(-5, +40)
        self.val['score'], = self.ax['top'].plot([], [], 'b-', label='curr', alpha=0.3)
        self.val['score_mean'], = self.ax['top'].plot([], [], 'r-', label='mean')
        self.val['score_min'], = self.ax['top'].plot([], [], '-', color='burlywood', label='min')
        self.val['score_max'], = self.ax['top'].plot([], [], linestyle='-', color='green', label='max')

        self.ax['top'].legend()

        self.ax['down'].set_xlim(0, n_episodes + 20)
        self.ax['down'].set_ylim(0, 1)
        self.val['eps'], = self.ax['down'].plot([], [], 'b-')

    def draw(self, val):
        for key in self.val:
            self.val[key].set_xdata(np.arange(len(val[key])))
            self.val[key].set_ydata(val[key])

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

def dqn(env, agent, n_episodes=2000, eps_start=1.0, eps_end=0.01, eps_decay=0.995, window_size=500, do_visualization=False):
    """Deep Q-Learning.
    
    Params
    ======
        env (UnityEnvironment): environment
        agent (Agent): learner
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    plt.ion()
    monitor = Monitor(n_episodes)
    track = {'score' : [], 
             'score_mean' : [],
             'score_min' : [],
             'score_max' : [],
             'eps' : [],
             'score_window' : deque(maxlen=window_size)}

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    max_score = 0
    eps = eps_start                    # initialize epsilon
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name] # reset the environment
        action_size = brain.vector_action_space_size       # number of actions
        state = env_info.vector_observations[0]            # get the current state
        score = 0                                          # initialize the score

        while True:
            action = agent.act(state, eps)
            env_info = env.step(action)[brain_name]        # send the action to the environment
            next_state = env_info.vector_observations[0]   # get the next state
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0]                  # see if episode has finished
            agent.step(state, action, reward, next_state, done)
            score += reward                                # update the score
            state = next_state                             # roll over the state to next time step
            if done:                                       # exit loop if episode finished
                break

        track['score_window'].append(score)                 # save most recent score
        track['score'].append(score)                        # save most recent score
        track['score_mean'].append(np.mean(track['score_window']))  # save most recent mean score
        track['score_min'].append(np.min(track['score_window']))  # save most recent mean score
        track['score_max'].append(np.max(track['score_window']))  # save most recent mean score
        track['eps'].append(eps)                            # save eps

        if do_visualization: monitor.draw(track)

        print('\rEpisode {}\tAverage Score: {:.2f}\tMin Score: {:.2f}\tMax Score: {:.2f}'.format(i_episode, track['score_mean'][-1], track['score_min'][-1], track['score_max'][-1]), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}\tMin Score: {:.2f}\tMax Score: {:.2f}'.format(i_episode, track['score_mean'][-1], track['score_min'][-1], track['score_max'][-1]))
        if track['score_mean'][-1]>=12.0 and max_score < track['score_mean'][-1]:
            max_score = track['score_mean'][-1]
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
        
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        
    if not do_visualization: monitor.draw(track)
    plt.ioff()
    plt.show()
    return track['score']

if __name__ == "__main__":

    env = UnityEnvironment(file_name="Banana.x86_64", seed=7523)
    agent = Agent(state_size=37, action_size=4, seed=7919)
    agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))
    score = dqn(env, agent, n_episodes=10000, eps_start=1.0, eps_end=0.01, eps_decay=0.998, window_size=100, do_visualization=True)

    nv.close()