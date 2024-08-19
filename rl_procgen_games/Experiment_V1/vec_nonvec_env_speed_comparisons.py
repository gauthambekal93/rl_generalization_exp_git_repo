# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 21:50:31 2024

@author: gauthambekal93
"""

import time
import gym
from procgen import ProcgenGym3Env
import numpy as np

import os
os.chdir(r"C:/Users/gauthambekal93/Research/rl_generalization_exps/rl_procgen_games")

#import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from PIL import Image
from collections import deque
import gym3
from procgen import ProcgenGym3Env
import random
import time 
import csv 
import pandas as pd
import gym
from torch.cuda.amp import autocast
#from test_script import test_model
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
import pandas as pd

if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

total_duration = 0
total_duration7 = 0
total_duration8 = 0


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
    '''    
    if is_new_episode:
        frames = deque([np.zeros((64, 64), dtype=np.uint8) for _ in range(4)], maxlen=4)
        for _ in range(4):
            frames.append(frame)
    else:
        frames.append(frame)
    '''
    
    stacked_frames = np.stack(frames, axis=0)
    
    return stacked_frames, frames


class SharedConv(nn.Module):
    
    def __init__( self, obs_shape, device):
        super(SharedConv,self).__init__()
        
        #self.conv1  = nn.Conv2d(in_channels = obs_shape[1], out_channels=32, kernel_size=8, stride=4)
        #self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        #self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        
        self.conv = nn.Sequential(*[
                                    nn.Conv2d(in_channels = obs_shape[1], out_channels=32, kernel_size=8, stride=4), 
                                    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2), 
                                    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
                                    ] )
    
    #def get_feature_size(self):
        #return self.conv3(self.conv2(self.conv1( torch.zeros(1, *obs_shape[1:] ) ) ) ).view(-1).shape[0]
        #return self.conv (torch.zeros(1, *obs_shape[1:] ) ) .view(-1).shape[0] 
    
    def forward(self, x):
        return self.conv(x)
    


class Actor(nn.Module):
    
    def __init__(self, shared_conv, in_features, out_features, n_actions, device):
        super(Actor, self).__init__()
        self.device = device
        
        actor_layers = [
            nn.Linear(in_features, out_features),
            nn.ReLU(),
            nn.Linear(out_features, out_features),
            nn.ReLU(),
            nn.Linear(
                out_features, n_actions
            ),  # estimate action logits (will be fed into a softmax later)
        ]
        
        self.shared_conv = shared_conv
        
        self.actor = nn.Sequential(*actor_layers).to(self.device)
        
        #self.actor_optim = optim.RMSprop(self.actor.parameters(), lr=actor_lr)
            
    def forward(self, x):
        global total_duration
        
        start = time.time()
        x = torch.Tensor(x).to(self.device)
        tmp = time.time() - start
        print("D1 ", tmp)
        total_duration = total_duration + tmp
        
        start = time.time()
        x = self.shared_conv(x)  
        tmp = time.time() - start
        print("D2 ", tmp )
        total_duration = total_duration + tmp
        
        start = time.time()
        x = x.view(x.size(0), -1)  # Flatten
        tmp = time.time() - start
        print("D3 ", tmp)
        total_duration = total_duration + tmp
        
        start = time.time()
        action_logits_vec = self.actor(x)   
        tmp = time.time() - start
        print("D4 ", tmp)
        total_duration = total_duration + tmp
        
        return action_logits_vec




class Critic(nn.Module):
    
    def __init__(self, shared_conv, in_features, out_features, n_actions, device):
        super(Critic, self).__init__()
        self.device = device
        
        critic_layers = [
            nn.Linear(in_features, out_features),
            nn.ReLU(),
            nn.Linear(out_features, out_features),
            nn.ReLU(),
            nn.Linear(out_features, 1),  # estimate V(s)
        ]
        
        self.shared_conv= shared_conv
        
        self.critic = nn.Sequential(*critic_layers)
    
    def forward(self, x):
        
        x = torch.Tensor(x).to(self.device)
        x=  self.shared_conv(x)
        x = x.view(x.size(0), -1)  # Flatten
        state_values = self.critic(x)
        return state_values
        


def select_action(x, actor, critic):
    global total_duration7, total_duration8
    """
    Returns a tuple of the chosen actions and the log-probs of those actions.

    Args:
        x: A batched vector of states.

    Returns:
        actions: A tensor with the actions, with shape [n_steps_per_update, n_envs].
        action_log_probs: A tensor with the log-probs of the actions, with shape [n_steps_per_update, n_envs].
        state_values: A tensor with the state values, with shape [n_steps_per_update, n_envs].
    """
    action_logits = actor(x)
    
    
    with autocast():
        start = time.time()
        action_pd = torch.softmax(action_logits, dim=-1) #torch.distributions.Categorical(logits=action_logits)
        tmp = time.time() - start
        print("Tmp7 ", tmp)
        total_duration7= total_duration7+ tmp
        
        start = time.time()
        actions = torch.multinomial(action_pd, num_samples=1) #action_pd.sample()
        print("Tmp8 ", tmp)
        total_duration8 = total_duration+ tmp
        
      
    return (actions, action_pd)    
        



num_timesteps = 20000 #was 2000000
# environment hyperparams
num_envs = 1 #50 #was 20 #10 #was 20 #worked with one or 2 envs till now
num_levels = 5000 #was 1  this shows number of unique levels
n_updates = int(  num_timesteps / num_envs) #50000   #was  100000
n_steps_per_update = 128
#randomize_domain = False

# agent hyperparams
gamma = 0.999
lam = 0.95  # hyperparameter for GAE
ent_coef = 0.01  # coefficient for the entropy bonus (to encourage exploration)
#NEED TO CHECK IF THE LEARNING RATES WHICH ARE OPTIMAL FOR BELOW 3 NETWORKS
conv_lr = 1e-4 #was 0.001
actor_lr = 1e-4 # was 0.001
critic_lr = 5e-4 # was 0.005
        
        
'''
# Vectorized environment with 10 parallel environments
num_envs = 20
envs = ProcgenGym3Env(num=num_envs, env_name="coinrun", start_level=0, num_levels=1)

num_timesteps = 20000
total_timesteps = 0

start_time = time.time()

while total_timesteps < num_timesteps:
    actions = np.random.randint(0, 15 , size=num_envs)   #env.action_space.sample()  # Sample random actions for all environments
    envs.act(actions)  # Perform actions in all environments
    rewards, states, dones = envs.observe()  # Get the observations
    total_timesteps += num_envs  # Since we have 10 environments running in parallel
    print("Time steps ", total_timesteps)
vectorized_time = time.time() - start_time
print(f"Vectorized environment completed {num_timesteps} timesteps in {vectorized_time:.2f} seconds.")
'''



#-----------------------------------------------------------------------

import time
import gym
import numpy as np
# Non-vectorized environment with only 1 environment
env = gym.make("procgen:procgen-coinrun-v0", start_level=0, num_levels=1)

num_timesteps = 20000
total_timesteps = 0

state =  env.reset()

        

start_time = time.time()

while total_timesteps < num_timesteps:
    action =  np.random.randint(0, 15 , size=1)  
    #action = env.action_space.sample()  # Sample a random action
    state, reward, done, info = env.step(action[0])  # Perform the action in the environment
    if done:
        env.reset()  # Reset the environment if done
    total_timesteps += 1

non_vectorized_time = time.time() - start_time
print(f"Non-vectorized environment completed {num_timesteps} timesteps in {non_vectorized_time:.2f} seconds.")







