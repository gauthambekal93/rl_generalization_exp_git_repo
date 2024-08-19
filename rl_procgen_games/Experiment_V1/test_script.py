# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 23:24:37 2024

@author: gauthambekal93
"""

from procgen import ProcgenGym3Env
import gym3
import numpy as np
from PIL import Image
from collections import deque
import torch
import torch.nn as nn
import csv

num_envs = 20


total_test_duration = 5000


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


def select_action(x, actor, critic):
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
    
    state_values = critic(x)
    
    action_pd = torch.distributions.Categorical( logits=action_logits )  # implicitly uses softmax
    
    actions = action_pd.sample()
    
    action_log_probs = action_pd.log_prob(actions)
    
    entropy = action_pd.entropy()
    
    return (actions, action_log_probs, state_values, entropy) 




def test_model(actor, critic, time_elasped, envs_test  ):
    
    total_reward = 0
    
    time_step = 0
    
    rewards, states, terminated = envs_test.observe()  #envs_wrapper.reset(seed=42)

    states = states['rgb']
    
    states = preprocess_image_rgb(states)
    
    frames = []
    
    states, frames = stack_frames(frames, states, num_envs, is_new_episode = True)
    
    #log_rewards ={}
    
    #for step in range(total_test_duration):
        
    while time_step <= total_test_duration:
        
        # select an action A_{t} using S_{t} as input for the agent
        actions, action_log_probs, state_value_preds, entropy = select_action( states, actor, critic )

        # perform the action A_{t} in the environment to get S_{t+1} and R_{t+1}    
        actions = np.array(actions.cpu().detach())
        
        envs_test.act( actions  )
        
        rewards, states, terminated = envs_test.observe()
        
        total_reward = total_reward + rewards.sum() #mean() 
        
        #time_step = time_step + 1
            
        
        #if time_step % 1000 == 0 :
            
            #log_rewards[ time_step  ] =  total_reward / time_step
            
           
        states = states['rgb']
        
        states = preprocess_image_rgb(states)
        
        states, frames = stack_frames(frames, states, num_envs, is_new_episode = False)
        
        time_step = time_step + num_envs
        
    #print(" Reward rate : {0} ".format( total_reward / total_test_duration  ) )
    with open("Results.csv",'a',newline='') as file:
        writer = csv.writer(file)
        writer.writerows( [[ "Test - Datasize: {0}".format(total_test_duration), str(time_elasped) , str(total_reward), str(total_reward / total_test_duration) ]]  )  
        file.close()
       
        
   # [ "Train - Datasize: {0} ".format(logging_rate) , str(time_step) , str(total_reward), str(total_reward / logging_rate) ]
    
      
'''

import time
time_step =0

start = time.time()
while time_step <= 5000:
     
     # select an action A_{t} using S_{t} as input for the agent
     actions, action_log_probs, state_value_preds, entropy = select_action( states, actor, critic )

     # perform the action A_{t} in the environment to get S_{t+1} and R_{t+1}    
     actions = np.array(actions.cpu().detach())
     
     envs_test.act( actions  )
     
     rewards, states, terminated = envs_test.observe()        
     
     time_step = time_step + num_envs
     
     states = states['rgb']
     
     states = preprocess_image_rgb(states)
     
     states, frames = stack_frames(frames, states, num_envs, is_new_episode = False)
     
print("Time taken ",time.time() - start)
'''