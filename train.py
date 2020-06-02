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

import time
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

from unityagents import UnityEnvironment
import numpy as np

from dueling_ddqn_agent import Agent

# use ggplot style for more sophisticated visuals
plt.style.use('ggplot')

class Monitor ():
    """Interactive plot figure (window)."""

    def __init__(self, n_episodes=2000):
        """Initialize plot figure.

        Params
        ======
            n_episode (int): minimum number of episodes
        """
        self.ax = dict()
        self.val = dict()
        self.n_episodes = n_episodes

        self.fig = plt.figure(figsize=(8, 12))
        self.ax['top'] = self.fig.add_subplot(311)
        self.ax['top'].set_ylabel('Score')
        self.ax['midd'] = self.fig.add_subplot(312)
        self.ax['midd'].set_ylabel('MSE')
        self.ax['midd'].set_yscale('log')
        self.ax['down'] = self.fig.add_subplot(313)
        self.ax['down'].set_xlabel('Episode #')

        self.ax['top'].set_xlim(0, n_episodes)
        self.ax['top'].set_ylim(-3, +30)
        self.val['train_score'], = self.ax['top'].plot([], [], 'r-', alpha=0.3, label='train')
        self.val['train_score_mean'], = self.ax['top'].plot([], [], 'r-', label='train(mean)')
        self.val['valid_score'], = self.ax['top'].plot([], [], 'b-', alpha=0.3, label='valid')
        self.val['valid_score_mean'], = self.ax['top'].plot([], [], 'b-', label='valid(mean)')
        self.ax['top'].legend()

        self.ax['midd'].set_xlim(0, n_episodes)
        self.ax['midd'].set_ylim(1e-4, 1.0)
        self.val['mse'], = self.ax['midd'].plot([], [], '-', color='burlywood')

        self.ax['down'].set_xlim(0, n_episodes)
        self.ax['down'].set_ylim(0, 1.01)
        self.val['eps'], = self.ax['down'].plot([], [], 'b-', label='ε')
        self.val['beta'], = self.ax['down'].plot([], [], 'g-', label='β')
        self.ax['down'].legend()

        self.wasClosed = False
        self.fig.canvas.mpl_connect('close_event', self.handle_close)

    def draw(self, val, new_size=1):
        """Initialize plot figure.

        Params
        ======
            val (dic): dictionary with values to plot
            new_size (int): new x-axis sizes
        """
        for key in self.val:
            self.val[key].set_xdata(np.arange(len(val[key])))
            self.val[key].set_ydata(val[key])

        self.ax['top'].set_xlim(-25, max(self.n_episodes, new_size))
        self.ax['midd'].set_xlim(-25, max(self.n_episodes, new_size))
        self.ax['down'].set_xlim(-25, max(self.n_episodes, new_size))

        self.ax['midd'].set_ylim(min(val['mse']) / 10.0 / 2.0, max(val['mse']) * 10.0 / 2.0)
        self.ax['top'].set_ylim(min(min(val['train_score']), min(val['valid_score'])) - 1, max(max(val['train_score']), max(val['valid_score'])) + 1)
        

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def flush(self):
        """Flush events."""
        if not self.wasClosed:
            self.fig.canvas.flush_events()

    def handle_close(self, evt):
        self.wasClosed = True


def random_episodes(env, brain_name, agent, action_size, monitor, k=1, n_samples=5000,
                    do_visualization=False):
    """Generate sample experiences from random actions."""
    t = 0
    while t < n_samples:
        env_info = env.reset(train_mode=True)[brain_name]   # reset the environment
        state = env_info.vector_observations[0]             # get the current state

        state_stack = deque(maxlen=k)
        next_state_stack = deque(maxlen=k)
        r = deque(maxlen=k)
        for i in range(1,k):
            state_stack.append(state)
            next_state_stack.append(state)

        done = False
        while not done:
            state_stack.append(state)
            s = np.array(list(state_stack)).flatten()
            action = random.choice(np.arange(action_size))

            env_info = env.step(action)[brain_name]         # send the action to the environment
            reward = env_info.rewards[0]                    # get the reward
            done = env_info.local_done[0]                   # see if episode has finished

            r.append(reward)
            next_state = env_info.vector_observations[0]    # get the next state
            next_state_stack.append(next_state)

            next_s = np.array(list(next_state_stack)).flatten()

            agent.memory.add(agent.t_step, s, action, np.sum(list(r)), next_s, done)
            agent.t_step += 1

            state = next_state                              # roll over the state to next time step
            
            if do_visualization: monitor.flush()

            print('\rRandom action {}'.format(t), end="")

            t += 1

    print('\rRandom action {}'.format(t))

def generate_episodes(env, brain_name, agent, action_size, monitor, k=1, eps=0.001, 
                    n_samples=5000, do_visualization=False):
    """Generate sample episodes using ε-greedy policy."""
    t = 0
    while t < n_samples:
        env_info = env.reset(train_mode=True)[brain_name]       # reset the environment
        state = env_info.vector_observations[0]                 # get the current state
        score = 0                                               # initialize the score

        state_stack = deque(maxlen=k)
        next_state_stack = deque(maxlen=k)
        r = deque(maxlen=k)
        for i in range(1,k):
            state_stack.append(state)
            next_state_stack.append(state)

        done = False
        while not done:
            state_stack.append(state)
            s = np.array(list(state_stack)).flatten()
            action = agent.act(s, eps=eps)

            env_info = env.step(action)[brain_name]         # send the action to the environment
            reward = env_info.rewards[0]                    # get the reward
            done = env_info.local_done[0]                   # see if episode has finished

            r.append(reward)
            next_state = env_info.vector_observations[0]    # get the next state
            next_state_stack.append(next_state)

            next_s = np.array(list(next_state_stack)).flatten()

            agent.memory.add(agent.t_step, s, action, np.sum(list(r)), next_s, done)
            agent.t_step += 1

            state = next_state                              # roll over the state to next time step
            
            if do_visualization: monitor.flush()

            print('\rGenerate ε-greedy action {}'.format(t), end="")

            t += 1

    print('\rGenerate ε-greedy action {}'.format(t))


def train(env, brain_name, agent, action_size, monitor, eps, beta, k=1,
          do_visualization=False):
    """Train dq-learning agent."""
    env_info = env.reset(train_mode=True)[brain_name]       # reset the environment
    state = env_info.vector_observations[0]                 # get the current state

    score = 0                                               # initialize the score
    mse = []

    state_stack = deque(maxlen=k)
    next_state_stack = deque(maxlen=k)
    r = deque(maxlen=k)
    for i in range(1,k):
        state_stack.append(state)
        next_state_stack.append(state)
    
    t = 0
    done = False
    while not done:
        state_stack.append(state)
        s = np.array(list(state_stack)).flatten()
        action = agent.act(s, eps)

        env_info = env.step(action)[brain_name]             # send the action to the environment
        reward = env_info.rewards[0]                        # get the reward
        done = env_info.local_done[0]                       # see if episode has finished

        r.append(reward)
        next_state = env_info.vector_observations[0]        # get the next state
        next_state_stack.append(next_state)

        next_s = np.array(list(next_state_stack)).flatten()

        e, update = agent.step(s, action, np.sum(list(r)), next_s, done, beta=beta)
        if update: mse.append(e)

        state = next_state                                  # roll over the state to next time step
        score += reward                                     # update the score
        
        if do_visualization: monitor.flush()
        t += 1

    return score, np.sum(mse) / len(mse)


def valid(env, brain_name, agent, action_size, monitor, k=1, do_visualization=False):
    """Validate dq-learning agent, using a greedy policy."""
    env_info = env.reset(train_mode=True)[brain_name]       # reset the environment
    state = env_info.vector_observations[0]                 # get the current state
    score = 0                                               # initialize the score

    state_stack = deque(maxlen=k)
    for i in range(1,k):
        state_stack.append(state)

    t = 0
    done = False
    while not done:
        state_stack.append(state)
        s = np.array(list(state_stack)).flatten()
        action = agent.act(s)

        env_info = env.step(action)[brain_name]             # send the action to the environment
        reward = env_info.rewards[0]                        # get the reward
        done = env_info.local_done[0]                       # see if episode has finished
        state = env_info.vector_observations[0]             # get the next state
        score += reward                                     # update the score
        
        if do_visualization: monitor.flush()
        t += 1
    
    return score


def update_info(monitor, track, i_episode, train_score, valid_score, eps, beta, error, 
                do_visualization=False, info_update_rate=100):
    """Update information to log and plot."""
    track['train_score_window'].append(train_score)         # save most recent score
    track['train_score'].append(train_score)                # save most recent score
    track['train_score_mean'].append(np.mean(track['train_score_window']))  # save most recent mean score

    track['eps'].append(eps)                                # save eps
    track['beta'].append(beta)                              # save beta
    track['mse'].append(error)

    track['valid_score_window'].append(valid_score)         # save most recent score
    track['valid_score'].append(valid_score)                # save most recent score
    track['valid_score_mean'].append(np.mean(track['valid_score_window']))  # save most recent mean score

    print('\rEpisode {}\tAverage Score(train): {:.2f}\tAverage Score(valid): {:.2f}  '
            '\tMSE: {:.3E}'.format(i_episode, track['train_score_mean'][-1],
                                    track['valid_score_mean'][-1], track['mse'][-1]), end="")

    if i_episode % info_update_rate == 0:
        if do_visualization: monitor.draw(track, new_size=len(track['valid_score_mean']))

        print('\rEpisode {}\tAverage Score(train): {:.2f}\tAverage Score(valid): {:.2f}  '
                '\tMSE: {:.3E}'.format(i_episode, track['train_score_mean'][-1],
                                        track['valid_score_mean'][-1], track['mse'][-1]))

        if do_visualization: monitor.draw(track, new_size=len(track['valid_score_mean']))


def dqn(env, brain_name, agent, action_size, state_size, k=1, eps_start=1.0, eps_end=0.01,
        eps_decay=0.995, beta_start=0.4, stop_crit=1e-3, n_episodes=2000, n_samples=5000,
        n_validations=1, window_size=100, do_visualization=True, info_update_rate=50, 
        pretrain=None, goal_score=3.0, early_stop=False):
    """Deep Q-Learning.
    
    Params
    ======
        env (UnityEnvironment): environment
        brain_name (string): brain name
        agent (Agent): learner
        action_size (int): Number of agent actions
        state_size (int): Dimension of observations (state)
        k(int): stack of observations used
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
        beta_start (int): initial beta value for priority experience replay
        n_episodes (int): minimum number of training episodes
        n_samples (int): minimum number of samples
        n_validations (int): number of validations per training
        window_size (int): sliding window size
        do_visualization (int): interative plot visualization
        info_update_rate (int): update information each info_update_rate
        pretrain (string): path to previous trained q-nets (default: None)
        goal_score (float): goal score obtain by validation agent to save q-network weights
    """
    plt.ion()
    monitor = Monitor(n_episodes)

    track = {'train_score' : [], 
             'train_score_mean' : [],
             'valid_score' : [],
             'valid_score_mean' : [],
             'eps' : [],
             'beta' : [],
             'mse' : [],
             'train_score_window' : deque(maxlen=window_size),
             'valid_score_window' : deque(maxlen=window_size)}

    if pretrain is None:
        # generate random experience samples
        random_episodes(env, brain_name, agent, action_size, monitor, k, n_samples, do_visualization)
    else:
        # generate ε-greedy experience samples
        #agent.qnetwork_local.load_state_dict(torch.load(pretrain))
        agent.qnetwork_target.load_state_dict(torch.load(pretrain))
        generate_episodes(env, brain_name, agent, action_size, monitor, k, 0.001, n_samples, do_visualization)

    # initialize eps and beta values
    eps = eps_start
    #eps_decay = (eps_start - eps_end) / n_episodes
    beta = beta_start
    beta_inc = (1.0 - beta) / n_episodes
    
    i_episode = 0
    while i_episode < n_episodes or track['mse'][-1] > stop_crit:
        # train q-network using current episode
        train_score, error = train(env, brain_name, agent, action_size, monitor, eps, beta, k, do_visualization)

        # validate current q-network parameters using greedy policy
        valid_score = []
        for i in range(n_validations):
            score = valid(env, brain_name, agent, action_size, monitor, k, do_visualization)        
            valid_score.append(score)

        # update metrics
        update_info(monitor, track, i_episode+1, train_score, np.mean(np.array(valid_score)), 
                eps, beta, error, do_visualization, info_update_rate)

        # save qnetwork if procced
        if i_episode%info_update_rate == 0 and (track['valid_score_mean'][-1]>=goal_score or track['train_score_mean'][-1]>=goal_score):
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint_%d_%d.pth'%(i_episode, track['valid_score_mean'][-1]))
            if early_stop:
                break

        # step forward
        eps = max(eps_end, eps_decay*eps)                   # decrease exponentialy epsilon
        #eps = max(eps_end, eps - eps_decay)
        beta = min(beta + beta_inc, 1.0)
        i_episode += 1

        if monitor.wasClosed:
            print('')
            break

    if not monitor.wasClosed:    
        if not do_visualization: monitor.draw(track)
        plt.ioff()
        plt.show()


if __name__ == "__main__":

    # start Unity Environment
    env = UnityEnvironment(file_name="Banana.x86_64", seed=int(time.time()*1e6)%int(2**18))

    # get the default brain and action/state sizes, plus k definition
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]
    action_size = brain.vector_action_space_size        # number of actions
    state_size = len(env_info.vector_observations[0])   # observation size
    k = 2
    
    # start agent
    agent = Agent(state_size=state_size * k, action_size=action_size)

    # train deep q-network
    dqn(env, brain_name, agent, action_size, state_size, k=k, eps_start=1.0, eps_end=0.01,
            eps_decay=0.998, beta_start=0.4, stop_crit=1e-3, n_episodes=int(1.5e3), n_samples=int(5e3),
            n_validations=1, window_size=100, do_visualization=True, info_update_rate=50,
            goal_score=13.0)

    # close environment
    env.close()