# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 11:14:53 2024

@author: gauthambekal93
"""
import os
os.chdir(r"C:/Users/gauthambekal93/Research/rl_generalization_exps/rl_generalization_exp_git_repo/rl_procgen_games/Experiment_V6")

model_path =r"C:/Users/gauthambekal93/Research/rl_generalization_exps/rl_generalization_exp_git_repo/rl_procgen_games/Experiment_V6/Models"

result_path =r"C:/Users/gauthambekal93/Research/rl_generalization_exps/rl_generalization_exp_git_repo/rl_procgen_games/Experiment_V6/Results"

import torch
import torch.nn as nn
import numpy as np
import gym3
from torch import optim
from tqdm import tqdm
from PIL import Image
from collections import deque
import gym3
import gym
from procgen import ProcgenGym3Env
import random
import time 
import csv 
import pandas as pd
from test_script import test_model
import json
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)




if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        

def preprocess_image_rgb(images):
    return np.stack([ np.array(Image.fromarray(image).convert('L')) for image in images])  # 'L' mode is for grayscale
     

def stack_frames(frames, states, num_envs, is_new_episode):  #frame shape (numenv, x dim , y dim)
    
    for i in range(num_envs):
        if is_new_episode:
            frames.append( deque([np.zeros((64, 64), dtype=np.uint8) for _ in range(4)], maxlen=4) )
            
            for _ in range(4):
                frames[i].append(states[i,:,:] )
        else:
            frames[i].append(states[i,:,:])

    
    stacked_frames = np.stack(frames, axis=0)
    
    return stacked_frames, frames



total_timesteps = 100 #8000000 #was 2000000
# environment hyperparams
num_envs = 20 #was 20 #10 #was 20 #worked with one or 2 envs till now
num_levels = 10000# 5000 #was 1  this shows number of unique levels
#n_updates = int( total_timesteps / num_envs) #50000   #was  100000
n_steps_per_update = 256 #128
#randomize_domain = False

# agent hyperparams
gamma = 0.999
lam = 0.95  # hyperparameter for GAE
ent_coef = 0.01  # coefficient for the entropy bonus (to encourage exploration)
#NEED TO CHECK IF THE LEARNING RATES WHICH ARE OPTIMAL FOR BELOW 3 NETWORKS
conv_lr = 1e-5 #1e-4 #was 0.001
actor_lr = 1e-5 #1e-4 # was 0.001
critic_lr = 1e-4  #5e-4 # was 0.005

logging_rate = n_steps_per_update* num_envs #10000
        

envs = ProcgenGym3Env(num= num_envs, 
                      env_name="coinrun", 
                      render_mode="rgb_array",
                      num_levels=num_levels,
                      start_level=0,
                      distribution_mode="hard",  #easy
                      use_sequential_levels =False
                      )
envs = gym3.ViewerWrapper(envs, info_key="rgb")

time_step =0

while time_step<= total_timesteps:
    
    rewards, states, terminated  = envs.observe()
    
    envs.act ( np.random.randint(1, envs.ac_space.eltype.n, num_envs) )
    
    time_step +=1
    
    if time_step == 5:
        print("stop")
        print("stop")
        states2 = envs.callmethod("get_state")
        
    print("Time step ", time_step)




# Custom JSON encoder that converts NumPy arrays to lists
class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
    
    
with open('my_dict.json', 'w') as file:
    json.dump(states, file, cls=NumpyArrayEncoder,  indent=4) 



time_step2 = 0
total_timesteps2 = 100

while time_step2<= total_timesteps2:
    envs.callmethod("set_state", states2)
    time_step2 += 1
    envs.observe()

#    -------------------------------------------------------------------

import os
os.chdir(r"C:/Users/gauthambekal93/Research/rl_generalization_exps/rl_generalization_exp_git_repo/rl_procgen_games/Experiment_V4")

model_path =r"C:/Users/gauthambekal93/Research/rl_generalization_exps/rl_generalization_exp_git_repo/rl_procgen_games/Experiment_V4/Models"

result_path =r"C:/Users/gauthambekal93/Research/rl_generalization_exps/rl_generalization_exp_git_repo/rl_procgen_games/Experiment_V4/Results"

import torch
import torch.nn as nn
import numpy as np
import gym3
from torch import optim
from tqdm import tqdm
from PIL import Image
from collections import deque
import gym3
import gym
from procgen import ProcgenGym3Env
import random
import time 
import csv 
import pandas as pd
from test_script import test_model
import json
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


total_timesteps = 100 #8000000 #was 2000000
# environment hyperparams
num_envs = 1 #was 20 #10 #was 20 #worked with one or 2 envs till now
num_levels = 10000# 5000 #was 1  this shows number of unique levels
#n_updates = int( total_timesteps / num_envs) #50000   #was  100000
n_steps_per_update = 256 #128
#randomize_domain = False

# agent hyperparams
gamma = 0.999
lam = 0.95  # hyperparameter for GAE
ent_coef = 0.01  # coefficient for the entropy bonus (to encourage exploration)
#NEED TO CHECK IF THE LEARNING RATES WHICH ARE OPTIMAL FOR BELOW 3 NETWORKS
conv_lr = 1e-5 #1e-4 #was 0.001
actor_lr = 1e-5 #1e-4 # was 0.001
critic_lr = 1e-4  #5e-4 # was 0.005

# Create the environment
envs = ProcgenGym3Env(num= num_envs, 
                      env_name="coinrun", 
                      render_mode="rgb_array",
                      num_levels=num_levels,
                      start_level=0,
                      distribution_mode="hard",  #easy
                      use_sequential_levels =False
                      )
envs = gym3.ViewerWrapper(envs, info_key="rgb")


# Save the current state of the environment


# Perform some actions
for i in range(1000):
    rewards, states, terminated  = envs.observe()
    
    envs.act ( np.random.randint(1, envs.ac_space.eltype.n, num_envs) )
    #envs.act ( np. array([4]) )
    
    if i ==200:
        states2 = envs.callmethod("get_state")
        
    print(i)
# Now, restore the environment to the previously saved state
envs.callmethod("set_state", states2)

# Perform different actions from the restored state
for j in range(30):
    rewards, states, terminated  = envs.observe()
    
    envs.act ( np.random.randint(1, envs.ac_space.eltype.n, num_envs) )
    
    print(j)
# Close the environment







