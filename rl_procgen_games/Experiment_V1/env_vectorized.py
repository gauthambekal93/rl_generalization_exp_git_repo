# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 21:29:11 2024

@author: gauthambekal93
"""
'''
from procgen import ProcgenGym3Env
import numpy as np

# Create the Procgen vectorized environment
num_envs = 4  # Number of parallel environments
env = ProcgenGym3Env(num=num_envs, env_name="coinrun")

# Number of steps to run
num_steps = 10

# Get initial observations
obs, rew, done = env.observe()

for step in range(num_steps):
    # Example actions (random actions for demonstration)
    actions = np.random.randint(0, env.ac_space.eltype.n, size=(num_envs, 1))

    # Take a step in the environment
    env.act(actions)

    # Observe the next state, reward, and done flag
    obs, rew, done = env.observe()

    print(f"Step {step + 1}:")
    for i in range(num_envs):
        print(f"Environment {i + 1}:")
        print(f"Observation: {obs[i]}")
        print(f"Reward: {rew[i]}")
        print(f"Done: {done[i]}")

    # Check if any episode is done
    if np.any(done):
        print("At least one episode finished!")
        break



'''

'''
import gym3
from procgen import ProcgenGym3Env
env = ProcgenGym3Env(num=2, env_name="coinrun")
step = 0
while True:
    env.act( gym3.types_np.sample(env.ac_space, bshape=(env.num,)) )
    rew, obs, first = env.observe()
    print(f"step {step} reward {rew} first {first}")
    step += 1
    
    
'''    
    
    
import numpy as np
import time
import gym3
from procgen import ProcgenGym3Env

env = ProcgenGym3Env(num=10, env_name="coinrun", render_mode="rgb_array")
env = gym3.ViewerWrapper(env, info_key="rgb")
step = 0
x=  np.array([2,4,0, 1, 0,2,3,5,6,9])
while True:
    #env.act(gym3.types_np.sample(env.ac_space, bshape=(env.num,)) )
    env.act( x    )
    rew, obs, first = env.observe()
    print(f"step {step} reward {rew} first {first}")
    step += 1
    time.sleep(0.5)
    break


#we will have a global action variable which is empty numpy array
#unless the array is completely filled we donot carry out vector based action step
#we basically wait for this action vector to be completly filled.
#we update the global network based on individual worker network 
#each network will have its own memory module







    
    
    
    
    
    
    
    