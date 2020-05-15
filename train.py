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

def dqn(env, agent, n_episodes=2000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
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

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    fig = plt.figure(figsize=(8, 12))
    axes1 = fig.add_subplot(211)
    axes1.set_ylabel('Score')
    # axes1.set_xlabel('Episode #')
    axes2 = fig.add_subplot(212)
    axes2.set_ylabel('eps')
    #axes2.set_yscale('log')
    axes2.set_xlabel('Episode #')

    axes1.set_xlim(0, n_episodes + 20)
    axes1.set_ylim(-5, +40)
    line1, = axes1.plot([], [], 'b-')
    line2, = axes1.plot([], [], 'r-')

    axes2.set_xlim(0, n_episodes + 20)
    axes2.set_ylim(0, 1)
    line3, = axes2.plot([], [], 'b-')

    scores = []                        # list containing scores from each episode
    scores_mean = []
    epss = []
    scores_window = deque(maxlen=100)  # last 100 scores
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

        scores_window.append(score)                 # save most recent score
        scores.append(score)                        # save most recent score
        scores_mean.append(np.mean(scores_window))  # save most recent mean score
        epss.append(eps)                            # save eps

        line1.set_xdata(np.arange(len(scores)))
        line1.set_ydata(scores)
        
        line2.set_xdata(np.arange(len(scores_mean)))
        line2.set_ydata(scores_mean)

        line3.set_xdata(np.arange(len(epss)))
        line3.set_ydata(epss)

        fig.canvas.draw()

        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, scores_mean[-1]), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, scores_mean[-1]))
        if scores_mean[-1]>=12.0 and max_score < scores_mean[-1]:
            max_score = scores_mean[-1]
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')

    return scores

if __name__ == "__main__":
    plt.ion()

    env = UnityEnvironment(file_name="Banana.x86_64", seed=0)
    agent = Agent(state_size=37, action_size=4, seed=0)
    agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))
    scores = dqn(env, agent, n_episodes=4000, eps_start=0.04, eps_end=0.01, eps_decay=0.995)

    env.close()