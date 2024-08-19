# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 21:29:11 2024

@author: gauthambekal93
"""

import numpy as np
import time
import gym3
from procgen import ProcgenGym3Env
import gym 



num_envs = 20
envs = ProcgenGym3Env(num= num_envs, env_name="coinrun")
step = 0
max_step = 20000
start = time.time()

while step<=max_step:
    envs.act ( np.random.randint(1, envs.ac_space.eltype.n, num_envs) )
    #envs.act( gym3.types_np.sample(envs.ac_space, bshape=(envs.num,)) )
    rew, obs, first = envs.observe()
  
    step += num_envs
    
    
print("Time taken for vectorized env", time.time() -start)    

    



env = gym.make("procgen:procgen-coinrun-v0", start_level=0, num_levels=1, distribution_mode="easy")
state =  env.reset()
 
step = 0
start = time.time()
max_step = 20000 

start = time.time()



while step<=max_step:
    action = np.random.randint(1, env.action_space.n)
    new_state, reward, done, info = env.step(action) 
    step +=1
 

print("Time taken for NON vectorized env", time.time() -start)    

    
    
    
    
    
    
    
    