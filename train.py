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

from dueling_ddqn_agent import Agent, PER_B, UPDATE_EVERY

# use ggplot style for more sophisticated visuals
plt.style.use('ggplot')

import threading, queue
import time
import math

exitFlag = 0

class Monitor ():
    def __init__(self, n_episodes=2000):
        self.ax = dict()
        self.val = dict()
        self.n_episodes = n_episodes

        self.fig = plt.figure(figsize=(8, 12))
        self.ax['top'] = self.fig.add_subplot(311)
        self.ax['top'].set_ylabel('Score')
        self.ax['midd'] = self.fig.add_subplot(312)
        self.ax['midd'].set_ylabel('MSE')
        self.ax['down'] = self.fig.add_subplot(313)
        self.ax['down'].set_ylabel('eps')
        #self.ax['down'].set_yscale('log')
        self.ax['down'].set_xlabel('Episode #')

        self.ax['top'].set_xlim(0, n_episodes)
        self.ax['top'].set_ylim(-3, +30)
        self.val['train_score'], = self.ax['top'].plot([], [], 'r-', alpha=0.3, label='train')
        self.val['train_score_mean'], = self.ax['top'].plot([], [], 'r-', label='train(mean)')
        self.val['valid_score'], = self.ax['top'].plot([], [], 'b-', alpha=0.3, label='valid')
        self.val['valid_score_mean'], = self.ax['top'].plot([], [], 'b-', label='valid(mean)')

        self.ax['top'].legend()

        self.ax['midd'].set_xlim(0, n_episodes)
        self.ax['midd'].set_ylim(0, 1.0)
        self.val['mse'], = self.ax['midd'].plot([], [], '-', color='burlywood')

        self.ax['down'].set_xlim(0, n_episodes)
        self.ax['down'].set_ylim(0, 1.01)
        self.val['eps'], = self.ax['down'].plot([], [], 'b-')
        self.val['beta'], = self.ax['down'].plot([], [], 'g-')

    def draw(self, val, new_size=0):
        for key in self.val:
            self.val[key].set_xdata(np.arange(len(val[key])))
            self.val[key].set_ydata(val[key])

        self.ax['top'].set_xlim(0, max(self.n_episodes, new_size))
        self.ax['midd'].set_xlim(0, max(self.n_episodes, new_size))
        self.ax['down'].set_xlim(0, max(self.n_episodes, new_size))

        self.ax['midd'].set_ylim(0, max(1, max(val['mse']))+0.1)
        self.ax['top'].set_ylim(min(min(val['train_score']), min(val['valid_score'])) - 1, max(max(val['train_score']), max(val['valid_score'])) + 1)
        

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def flush(self):
        self.fig.canvas.flush_events()

def dqn(env, agent, n_episodes=2000, n_validations=50, eps_start=1.0, eps_end=0.01, 
        eps_decay=0.995, window_size=500, do_visualization=False, info_update_rate=100):
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

    track = {'train_score' : [], 
             'train_score_mean' : [],
             'valid_score' : [],
             'valid_score_mean' : [],
             'eps' : [],
             'beta' : [],
             'se' : [],
             'mse' : [],
             'train_score_window' : deque(maxlen=window_size),
             'valid_score_window' : deque(maxlen=window_size)}

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    eps = eps_start                    # initialize epsilon
    beta = PER_B
    beta_inc = (1.0 - beta) / n_episodes
    eps_decay = (eps_start - eps_end) / n_episodes

    k = 2
    frame_skiped = 1


    t = 0
    while True:
        env_info = env.reset(train_mode=True)[brain_name] # reset the environment
        action_size = brain.vector_action_space_size       # number of actions
        state = env_info.vector_observations[0]            # get the current state

        state_stack = deque(maxlen=k)
        next_state_stack = deque(maxlen=k)
        for i in range(1,k):
            state_stack.append(state)
            next_state_stack.append(state)

        while True:
            if t%frame_skiped == 0:
                state_stack.append(state)
                s = np.array(list(state_stack)).flatten()
                action = random.choice(np.arange(4))

            env_info = env.step(action)[brain_name]             # send the action to the environment
            reward = env_info.rewards[0]                        # get the reward
            done = env_info.local_done[0]                       # see if episode has finished

            if t%frame_skiped == 0:
                next_state = env_info.vector_observations[0]    # get the next state
                next_state_stack.append(next_state)

                next_s = np.array(list(next_state_stack)).flatten()

                agent.memory.add(agent.t_step, s, action, reward, next_s, done)
                agent.t_step += 1

                state = next_state                              # roll over the state to next time step
            
            if do_visualization: monitor.flush()
            if done:                                            # exit loop if episode finished
                break

            print('\rRandom action {}'.format(t), end="")

            t += 1

        if t > 5000:
            break

    print('\rRandom action {}'.format(t))

    i_episode = 0
    while True:
        i_episode += 1

        env_info = env.reset(train_mode=True)[brain_name] # reset the environment
        action_size = brain.vector_action_space_size       # number of actions
        state = env_info.vector_observations[0]            # get the current state
        score = 0                                          # initialize the score
        mse = []

        state_stack = deque(maxlen=k)
        next_state_stack = deque(maxlen=k)
        for i in range(1,k):
            state_stack.append(state)
            next_state_stack.append(state)
        
        t = 0
        e = 0.0
        # Train agent
        while True:
            if t%frame_skiped == 0:
                state_stack.append(state)
                s = np.array(list(state_stack)).flatten()
                action = agent.act(s, eps, beta)

            env_info = env.step(action)[brain_name]             # send the action to the environment
            reward = env_info.rewards[0]                        # get the reward
            done = env_info.local_done[0]                       # see if episode has finished

            if t%k == 0: r = 0
            r += reward

            if t%frame_skiped == 0:
                next_state = env_info.vector_observations[0]    # get the next state
                next_state_stack.append(next_state)

                next_s = np.array(list(next_state_stack)).flatten()

                e, update = agent.step(s, action, r, next_s, done)
                if update: mse.append(e)

                state = next_state                              # roll over the state to next time step
            
            score += reward                                     # update the score
            
            if do_visualization: monitor.flush()
            if done:                                            # exit loop if episode finished
                break

            t += 1
            
        track['train_score_window'].append(score)                 # save most recent score
        track['train_score'].append(score)                        # save most recent score
        track['train_score_mean'].append(np.mean(track['train_score_window']))  # save most recent mean score

        track['eps'].append(eps)                            # save eps
        track['beta'].append(beta)                          # save eps
        track['mse'].append(np.sum(mse) / len(mse))

        # eps = max(eps_end, eps_decay*eps) # decrease epsilon
        eps = max(eps_end, eps - eps_decay)
        beta = min(beta + beta_inc, 1.0)

        # Validate agent
        valid_score = []
        for i in range(n_validations):
            env_info = env.reset(train_mode=True)[brain_name] # reset the environment
            action_size = brain.vector_action_space_size       # number of actions
            state = env_info.vector_observations[0]            # get the current state
            score = 0                                          # initialize the score

            state_stack = deque(maxlen=k)
            for i in range(1,k):
                state_stack.append(state)

            t = 0
            while True:
                if t%frame_skiped == 0:
                    state_stack.append(state)
                    s = np.array(list(state_stack)).flatten()
                    action = agent.act(s)

                env_info = env.step(action)[brain_name]        # send the action to the environment
                reward = env_info.rewards[0]                   # get the reward
                done = env_info.local_done[0]                  # see if episode has finished
                state = env_info.vector_observations[0]   # get the next state
                score += reward                                # update the score
                
                if do_visualization: monitor.flush()
                if done:                                       # exit loop if episode finished
                    break
                    
                t += 1
            valid_score.append(score)

        track['valid_score_window'].append(np.mean(np.array(valid_score)))                 # save most recent score
        track['valid_score'].append(np.mean(np.array(valid_score)))                        # save most recent score
        track['valid_score_mean'].append(np.mean(track['valid_score_window']))  # save most recent mean score

        print('\rEpisode {}\tAverage Score(train): {:.2f}\tAverage Score(valid): {:.2f}  '
                '\tMSE: {:.2E}'.format(i_episode, track['train_score_mean'][-1],
                                       track['valid_score_mean'][-1], track['mse'][-1]), end="")

        if i_episode % info_update_rate == 0:
            if do_visualization: monitor.draw(track, new_size=len(track['valid_score_mean']))

            print('\rEpisode {}\tAverage Score(train): {:.2f}\tAverage Score(valid): {:.2f}  '
                    '\tMSE: {:.2E}'.format(i_episode, track['train_score_mean'][-1],
                                            track['valid_score_mean'][-1], track['mse'][-1]))

            if track['valid_score_mean'][-1]>=3.0:
                torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')

            # Stop condition
            # if track['mse'][-1] < 1e-3:
            if do_visualization: monitor.draw(track, new_size=len(track['valid_score_mean']))
            #     break
        
        
    if not do_visualization: monitor.draw(track)
    plt.ioff()
    plt.show()
    return track['score']

if __name__ == "__main__":

    env = UnityEnvironment(file_name="Banana.x86_64", seed=0) #'''int(time.time()*1e6)%int(2^18)'''
    agent = Agent(state_size=37 * 2, action_size=4, seed=0) # '''int(time.time()*1e6)'''
    # agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))
    score = dqn(env, agent, n_episodes=int(1e5), n_validations=1, eps_start=1.0, eps_end=0.1,
        eps_decay=0.998, window_size=1000, do_visualization=True, info_update_rate=100)

    env.close()