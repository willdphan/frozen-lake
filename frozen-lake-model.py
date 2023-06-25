import random
import time

import numpy as np
import gym

from IPython.display import clear_output

env = gym.make('FrozenLake-v1')
action_space_size = env.action_space.n
state_space_size = env.observation_space.n

q_table = np.zeros((state_space_size, action_space_size))
q_table

num_episodes = int(1e4)
max_steps_per_episode = 100

lr = 1e-1
dr = 0.99

exploration_rate = 1
max_exp_rate = 1
min_exp_rate = 1e-3
exp_decay_rate = 1e-3

# building a reward list for all the episodes
rewards_all_episodes = []

# Q-learning algorithm
for episode in range(num_episodes):
    # reset the movements in the env
    state = env.reset()
    # check if the agent reaches the target
    done = False
    
    # variable for expected return G_t
    rewards_current_episode = 0
    
    # for loop for each step for the agent
    for step in range(max_steps_per_episode):
        
        # apply epsilon greedy stategy
        random_number = random.uniform(0, 1)
        # Exploration Vs. Exploitation trade-off
        if random_number > exploration_rate:
            # start exploitation ---> getting the maximum Q-value from the possible movements of his current state.
            action = np.argmax(q_table[state, :])
        else:
            # start exploration ---> select any random action to explore a random state.
            action = env.action_space.sample()
        
        # after taking the action, we're going to update our agent with the new info, rewards, state, and if he reaches the end or not!
        new_state, reward, done, info = env.step(action)
        
        # Update our Q-table for Q(s, a) using Bellman Equation
                                            # Old Q-value
        q_table[state, action] = (1 - lr) * q_table[state, action] + \
                                 lr * (reward + dr*(np.max(q_table[new_state, :])))
                                            # learned value

        # transition to the next state
        state = new_state
        rewards_current_episode += reward
        
        # check to see if our last action ended the episode for us,
        # meaning, did our agent step in a hole or reach the goal?
        if done:
            break
        # If the action did end the episode, then we jump out of this loop and move on to the next episode.
        # Otherwise, we transition to the next time-step.
    
    # Exploration Rate Decay
    # https://en.wikipedia.org/wiki/Exponential_decay
    exploration_rate = min_exp_rate + \
                      (max_exp_rate - min_exp_rate) * np.exp(-exp_decay_rate * episode)
    
    # append the current rewards in the list of rewards
    rewards_all_episodes.append(rewards_current_episode)

    # Calculate the average reward per thousand episodes
rewards_per_thousand_episodes = np.split(np.array(rewards_all_episodes), num_episodes/1000)
count = 1000

print("Average rewards per thousand episodes".center(100, '*'))
for reward in rewards_per_thousand_episodes:
    print(f'Count No. {count:,}: {sum(reward/1000)}')
    count += 1000

print("Q-Table".center(100, '*'))
print()

for row in q_table:
    print(' '* 25, row)

for episode in range(5):
    state = env.reset()
    done = False
    print(f'Episode: {episode+1}'.center(50, '='))
    time.sleep(1)
    
    for step in range(max_steps_per_episode):
        # for clearning the board
        clear_output(wait=True)
        # allows you to check the agent's environment
        env.render()
        time.sleep(0.4)
        
        # invoke the action with the highest Q-value from the Q-Table for the current state
        action = np.argmax(q_table[state, :])
        
        # take the action and move to the new state
        new_state, reward, done, info = env.step(action)
        
        # acting condition
        if done:
            clear_output(wait=True)
            env.render()
            if reward == 1:
                print('You reach the goal!'.center(50, '*'))
                time.sleep(3)
            else:
                print('You fall through a hole!'.center(50, '-'))
                time.sleep(3)
                clear_output(wait=True)
            break

        # select the new state based on the agent action
        state = new_state

# close the environment
env.close()