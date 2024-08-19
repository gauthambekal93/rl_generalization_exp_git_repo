# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 09:28:57 2024

@author: gauthambekal93
"""

import multiprocessing
import os
os.chdir(r"C:/Users/gauthambekal93/Research/rl_generalization_exps/rl_procgen_games/Experiment_V2")

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

#from test_script import test_model
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
import pandas as pd

import logging

def setup_logger(worker):
    logging.basicConfig(level=logging.INFO, format=f'%(asctime)s - Process {worker} - %(message)s')


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
        
        x = torch.Tensor(x).to(self.device)
        x = self.shared_conv(x)     
        x = x.view(x.size(0), -1)  # Flatten
        action_logits_vec = self.actor(x)     
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
        
        
def get_losses( rewards, action_log_probs, value_preds, entropy, masks, gamma, lam, ent_coef, device, n_envs) :
    """
    Computes the loss of a minibatch (transitions collected in one sampling phase) for actor and critic
    using Generalized Advantage Estimation (GAE) to compute the advantages (https://arxiv.org/abs/1506.02438).

    Args:
        rewards: A tensor with the rewards for each time step in the episode, with shape [n_steps_per_update, n_envs].
        action_log_probs: A tensor with the log-probs of the actions taken at each time step in the episode, with shape [n_steps_per_update, n_envs].
        value_preds: A tensor with the state value predictions for each time step in the episode, with shape [n_steps_per_update, n_envs].
        masks: A tensor with the masks for each time step in the episode, with shape [n_steps_per_update, n_envs].
        gamma: The discount factor.
        lam: The GAE hyperparameter. (lam=1 corresponds to Monte-Carlo sampling with high variance and no bias,
                                      and lam=0 corresponds to normal TD-Learning that has a low variance but is biased
                                      because the estimates are generated by a Neural Net).
        device: The device to run the computations on (e.g. CPU or GPU).

    Returns:
        critic_loss: The critic loss for the minibatch.
        actor_loss: The actor loss for the minibatch.
    """
    T = len(rewards)
    advantages = torch.zeros(T, n_envs, device=device)

    # compute the advantages using GAE
    gae = 0.0
    for t in reversed(range(T - 1)):
        td_error = (
            rewards[t] + gamma * masks[t] * value_preds[t + 1] - value_preds[t]
        )
        gae = td_error + gamma * lam * masks[t] * gae
        advantages[t] = gae

    # calculate the loss of the minibatch for actor and critic
    critic_loss = advantages.pow(2).mean()

    # give a bonus for higher entropy to encourage exploration
    actor_loss =  -(advantages.detach() * action_log_probs).mean() - ent_coef * entropy.mean()  
    
    return (critic_loss, actor_loss)


def update_parameters( optimizer,  actor_loss, critic_loss) :
    """
    Updates the parameters of the actor and critic networks.

    Args:
        critic_loss: The critic loss.
        actor_loss: The actor loss.
    """
    optimizer.zero_grad()
    
    critic_loss.backward()

    actor_loss.backward()
    
    optimizer.step()

    
global_var = multiprocessing.Value('i', 0)  # 'i' for integer  
total_timesteps = 20000 #8000000 #was 2000000


# environment hyperparams
num_envs = 1 #50 #was 20 #10 #was 20 #worked with one or 2 envs till now
num_levels = 5000 #was 1  this shows number of unique levels
#n_updates = int( total_timesteps / num_envs) #50000   #was  100000
#n_steps_per_update = 128
#randomize_domain = False

# agent hyperparams
gamma = 0.999
lam = 0.95  # hyperparameter for GAE
ent_coef = 0.01  # coefficient for the entropy bonus (to encourage exploration)
#NEED TO CHECK IF THE LEARNING RATES WHICH ARE OPTIMAL FOR BELOW 3 NETWORKS
conv_lr = 1e-4 #was 0.001
actor_lr = 1e-4 # was 0.001
critic_lr = 5e-4 # was 0.005


def create_env(worker):

   print("Create Env ", worker, flush=True)
   env = gym.make("procgen:procgen-coinrun-v0", start_level=0, num_levels=1)
   
   state =  env.reset()
   
   state = np.expand_dims(state, axis=0)

   num_actions = env.action_space.n #env.ac_space.eltype.n

   state = preprocess_image_rgb(state)

   frames = []

   state, frames = stack_frames(frames, state, num_envs, is_new_episode=True)

   obs_shape = (state.shape[0], state.shape[1], state.shape[2], state.shape[3]) #env_num, history of frames, x dim, y dim
   
   return env, state, frames, obs_shape,  num_actions


def create_model(env, state, frames, obs_shape, num_actions):
    # set the device
    use_cuda = True #False
    
    if use_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")


    shared_conv = SharedConv(obs_shape, device).to(device)

    in_features = shared_conv( torch.zeros(1, *obs_shape[1:] ).to(device) ).view(-1).shape[0]  #shared_conv.get_feature_size()  #1024

    out_features = 32

    actor =  Actor(shared_conv, in_features, out_features, num_actions, device).to(device)

    critic =  Critic(shared_conv, in_features, out_features, num_actions, device).to(device)
    
    optimizer = optim.Adam([
        {'params': shared_conv.parameters(), 'lr': conv_lr},
        {'params': actor.actor.parameters(), 'lr': actor_lr},
        {'params': critic.critic.parameters(), 'lr': critic_lr}
    ])
    
    return actor, critic, optimizer


def run_simulation(env, actor, critic, state, frames, worker):
    global global_var
    done = False
    #time_step = 0
    start = time.time()
    
    while global_var.value <=total_timesteps:
    
        print("Process",worker,"Done ",done,"step ", global_var.value, flush=True)    
        
        action, action_log_probs, state_value_preds, entropy = select_action( state, actor, critic )  #np.array([9]).reshape(-1)
        
        action = np.array( action.detach().cpu())[0]
        
        new_state, reward, done, info = env.step(action)
        
        new_state = np.expand_dims(new_state, axis=0)
        
        new_state = preprocess_image_rgb(new_state)
        
        state, frames = stack_frames(frames, new_state, num_envs, is_new_episode=False)
    
        #time_step +=1
        
        global_var.value += 1
    
    print("Process",worker,"Ended!!!","Total time steps ", global_var.value,"Time take ", time.time()- start, flush=True )    
    filename = 'empty_file.txt'
    
    with open(filename, 'w') as file:
      file.write("Process completed "+ str(worker))  # 'pass' does nothing, just ensures the file is created


def save_model(actor, critic, optimizer):
    print("----SAVE THE MODEL---")
    torch.save(actor.state_dict(), 'actor.pth')
    torch.save(critic.state_dict(), 'critic.pth')
    torch.save(optimizer.state_dict(), 'optimizer.pth')

def load_model(actor, critic, optimizer):
    print("----LOAD THE MODEL---")
    actor.load_state_dict(torch.load('C:/Users/gauthambekal93/Research/rl_generalization_exps/rl_procgen_games/Experiment_V1/actor.pth'))
    critic.load_state_dict(torch.load('C:/Users/gauthambekal93/Research/rl_generalization_exps/rl_procgen_games/Experiment_V1/critic.pth'))
    optimizer.load_state_dict(torch.load('C:/Users/gauthambekal93/Research/rl_generalization_exps/rl_procgen_games/Experiment_V1/optimizer.pth'))





def worker(worker):
    setup_logger(worker)
    #logging.info("is running")
    
    print(f"Process {worker} is running", flush=True)
    
    env, state, frames, obs_shape, num_actions = create_env(worker)
    
    actor, critic, optimizer = create_model(env, state, frames, obs_shape, num_actions)
    
    run_simulation(env, actor, critic, state, frames, worker)
    #update_local_model()
    #update_global_model()
    #evaluvate()




if __name__ == "__main__":
    processes = []
    num_workers = 5 #multiprocessing.cpu_count()
    for i in range(num_workers):
        p = multiprocessing.Process(target=worker, args=(i,))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()  # Wait for all processes to finish
