# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 18:04:21 2024

@author: gauthambekal93
"""


import os
os.chdir(r"C:/Users/gauthambekal93/Research/rl_generalization_exps/rl_generalization_exp_git_repo/rl_procgen_games/Experiment_V5")

model_path =r"C:/Users/gauthambekal93/Research/rl_generalization_exps/rl_generalization_exp_git_repo/rl_procgen_games/Experiment_V5/Models/Curiosity"

#model_path =r"C:/Users/gauthambekal93/Research/rl_generalization_exps/rl_generalization_exp_git_repo/rl_procgen_games/Experiment_V4/Models/Curiosity"

#result_path =r"C:/Users/gauthambekal93/Research/rl_generalization_exps/rl_generalization_exp_git_repo/rl_procgen_games/Experiment_V4/Results/Curiosity"

import numpy as np
import torch
import torch.nn as nn
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
import torch.nn.functional as F
from test_script import test_model


seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        


class Curiosity(nn.Module):
    
    def __init__(self, obs_shape, n_actions, device):
        super().__init__()
        
        #in_channels = obs_shape[1] #+ n_actions
        self.obs_shape = obs_shape
        
        self.device = device
        
        self.n_actions = n_actions
        
        self.weight_forward = 1.0
        
        self.weight_inverse = 1.0 
        
        self.conv = nn.Sequential(*[
                                    nn.Conv2d(in_channels = obs_shape[1] , out_channels=32, kernel_size=8, stride=4), 
                                    #nn.BatchNorm2d(32),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2), 
                                    #nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
                                    #nn.BatchNorm2d(64),
                                    nn.ReLU()
                                    ] )#.to(self.device)
        
        
        x = torch.zeros(*( 1, obs_shape[1] , obs_shape[2], obs_shape[3]) )
        
        in_features = self.conv(x).flatten().shape[0] 
        
        hidden_features_1 = in_features // 2
        
        self.reduce =  nn.Sequential( *[ nn.Linear(in_features, hidden_features_1 ),
                                        nn.ReLU() ] )
        
        hidden_features_2 = hidden_features_1 // 16
        
        out_features =  hidden_features_1 #obs_shape[1] * obs_shape[2] * obs_shape[3]
        
        self.forward_dynamics =nn.Sequential(* [
                    nn.Linear(hidden_features_1 +  self.n_actions , hidden_features_2),
                    nn.ReLU(),
                    nn.Linear(hidden_features_2, hidden_features_2),
                    nn.ReLU(),
                    nn.Linear(
                    hidden_features_2, out_features
                    ),  
                ])
                
        
        self.inverse_dynamics = nn.Sequential(* [
                    nn.Linear(hidden_features_1 * 2 , hidden_features_2),
                    nn.ReLU(),
                    nn.Linear(hidden_features_2, hidden_features_2),
                    nn.ReLU(),
                    nn.Linear(
                    hidden_features_2, n_actions
                    ), 
                ])
                
    
    def forward(self, states, next_states, action):
        
        #convert to torch tensor and to device
        states = torch.tensor(states,  dtype=torch.float).to(self.device)
        
        next_states = torch.tensor(next_states,  dtype=torch.float).to(self.device)
        
        action = torch.tensor(action).to(self.device)
        
        #pass input to convolution nn and flatten them
        feature_states = self.conv(states)
        
        feature_states = feature_states.view(feature_states.shape[0], -1)
        
        feature_states = self.reduce(feature_states) 
        
        feature_next_states = self.conv(next_states)
        
        feature_next_states = feature_next_states.view(feature_next_states.shape[0], -1)
        
        feature_next_states = self.reduce(feature_next_states)
        #one hot encode the actions
        
        action_one_hot = torch.nn.functional.one_hot(action, num_classes=self.n_actions).float().to(self.device)
        
        #create inputs for forward and inverse dynamics
        forward_ip =  torch.cat([feature_states, action_one_hot], dim=1)
        
        inverse_ip = torch.cat([feature_states, feature_next_states], dim=1)
        
        #predict from forward and inverse dynamics
        
        predicted_next_state = self.forward_dynamics(forward_ip)
        
        #predicted_next_state = predicted_next_state.view(-1, *(self.obs_shape[1], self.obs_shape[2], self.obs_shape[3] ) )
        
        predicted_action = self.inverse_dynamics(inverse_ip)
        
        #calculate the metrics
        #loss, intrinsic_reward = self.metric_calculations( next_states, predicted_next_state, action, predicted_action)
        loss,  forward_loss, inverse_loss , intrinsic_reward = self.metric_calculations( feature_next_states, predicted_next_state, action, predicted_action)
        
        return loss,  forward_loss, inverse_loss , intrinsic_reward
        
        
    def metric_calculations( self, feature_next_states, predicted_next_state, action, predicted_action):
        
        #squared_error = F.mse_loss(predicted_next_state, next_state, reduction='none')
        squared_error = F.mse_loss(predicted_next_state, feature_next_states, reduction='none')
        
        #forward_loss  = squared_error.mean(dim=(1, 2, 3))
        forward_loss  = squared_error.mean(dim=(1))
        
        #forward_loss = F.mse_loss(predicted_next_state, next_state)
        
        inverse_loss = F.cross_entropy(predicted_action, action, reduction='none') 
        
        losses = self.weight_forward * forward_loss + self.weight_inverse * inverse_loss
        
        intrinsic_rewards = forward_loss.detach().cpu()
        
        
        return losses,  forward_loss, inverse_loss , intrinsic_rewards
    

    def update_curiosity_params(self, curiosity_loss, curiosity_optimizer):
        
        curiosity_optimizer.zero_grad()
        
        curiosity_loss.backward()
        
        curiosity_optimizer.step()
        
    

class SharedConv(nn.Module):
    
    def __init__( self, obs_shape, device ):
        super(SharedConv,self).__init__()
        
        self.conv = nn.Sequential(*[
                                    nn.Conv2d(in_channels = obs_shape[1], out_channels=32, kernel_size=8, stride=4), 
                                    #nn.BatchNorm2d(32),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2), 
                                    #nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
                                    #nn.BatchNorm2d(64),
                                    nn.ReLU()
                                    ] )
        
        self.device = device
    
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
        
        self.actor = nn.Sequential(*actor_layers)#.to(self.device)
        
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
    
    action_logits = actor(x)
    
    state_values = critic(x)
    
    action_pd = torch.distributions.Categorical( logits=action_logits )  
    
    actions = action_pd.sample()
    
    action_log_probs = action_pd.log_prob(actions)
    
    entropy = action_pd.entropy()
    
    return (actions, action_log_probs, state_values, entropy)    




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
'''

def preprocess_image_rgb(image):
    image = Image.fromarray(image).convert('L')  # 'L' mode is for grayscale
    return np.array(image)


def stack_frames(frames, state, is_new_episode):
    if is_new_episode:
        frames = deque([np.zeros((64, 64), dtype=np.uint8) for _ in range(4)], maxlen=4)
        for _ in range(4):
            frames.append(state)
    else:
        frames.append(state)
    
    stacked_frames = np.stack(frames, axis=0)
    return stacked_frames, frames

'''
   
def normalize_states(states):
    
    pixel_min, pixel_max  = states.min(), states.max()
    
    states = (states - pixel_min) / (pixel_max - pixel_min)
    
    return states


parallel_env = False
total_timesteps =10000 # 1000000 
num_models_saved = 2
# environment hyperparams
num_envs = 1 #was 20 #10 #was 20 #worked with one or 2 envs till now 
num_levels = 100000 #100000 #was 10000
start_level = 100000 #10000
num_test_levels = 10  #was 200
#n_updates = int( total_timesteps / num_envs) #50000   #was  100000
n_steps_per_update = 256 #128
#randomize_domain = False

# agent hyperparams
gamma = 0.999
lam = 0.95  # hyperparameter for GAE
ent_coef = 0.01  # coefficient for the entropy bonus (to encourage exploration)

curiosity_lr = 1e-5 #was 1e-4  upto 1000000
#NEED TO CHECK IF THE LEARNING RATES WHICH ARE OPTIMAL FOR BELOW 3 NETWORKS
conv_lr = 1e-5 #1e-4 #was 0.001
actor_lr = 1e-5 #1e-4 # was 0.001
critic_lr = 1e-4  #5e-4 # was 0.005

logging_rate = n_steps_per_update* num_envs #10000
        

if parallel_env:
    env = gym.make("procgen:procgen-coinrun-v0", start_level=start_level, num_levels=num_levels, distribution_mode="easy", use_sequential_levels=True)
else:
        
    env = ProcgenGym3Env(num= num_envs, 
                          env_name="coinrun", 
                          render_mode="rgb_array",
                          num_levels = num_levels, 
                          start_level = start_level,
                          distribution_mode="hard",  #easy
                          use_sequential_levels =False #False #we keep it as True in order to make it easy to obtain levels where we obtain the goals
                          )
    env = gym3.ViewerWrapper(env, info_key="rgb")

if parallel_env:
    state =  env.reset()

if parallel_env: 
    num_actions =  env.action_space.n
else:
    num_actions = env.ac_space.eltype.n   

while True:
    if parallel_env:
       states, _, _, _ = env.step( np.random.randint(1, env.action_space.n)  )
    else:
        env.act( np.random.randint(0, num_actions , size=num_envs)  )
        _, states, done = env.observe()
        states = states['rgb']
    break

if parallel_env:
    states = np.expand_dims(states, axis=0)

states = preprocess_image_rgb(states)

frames = []

states, frames = stack_frames(frames, states, num_envs, is_new_episode = True)

states = normalize_states(states)

obs_shape = (states.shape[0], states.shape[1], states.shape[2], states.shape[3]) #env_num, history of frames, x dim, y dim

 

# set the device
use_cuda = True #False
if use_cuda:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")

curiosity_model = Curiosity(obs_shape, num_actions, device).to(device)

curiosity_optimizer = optim.Adam([ {'params': curiosity_model.parameters(), 'lr': curiosity_lr}  ])



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



def load_model():
    print("----LOAD THE MODELS---")
    
    last_model_time = [ int(m.name.split("_")[1].replace(".pth","")) for m in os.scandir(model_path) if 'actor' in m.name ]
    
    if last_model_time:
        time_step  = max(last_model_time)
        
        actor.load_state_dict(torch.load( os.path.join(model_path,'actor_'+str(time_step)+'.pth' )))
        critic.load_state_dict(torch.load( os.path.join(model_path, 'critic_'+str(time_step)+'.pth' )))
        optimizer.load_state_dict(torch.load( os.path.join(model_path,'optimizer_'+str(time_step)+'.pth' ) ))
        
        curiosity_model.load_state_dict(torch.load( os.path.join(model_path, 'curiosity_'+str(time_step)+'.pth' )))
        curiosity_optimizer.load_state_dict(torch.load( os.path.join(model_path,'curiosity_optimizer_'+str(time_step)+'.pth' ) ))
    
    
    
load_model()

state_trajectory = [] #np.zeros((1000, 2))
score_states = []
    
time_step = 0
    
thresh = 17

while time_step <= total_timesteps:   
      
      action, action_log_probs, state_value_preds, entropy = select_action( states, actor, critic )

      action = np.array(action.cpu().detach())
      
      if parallel_env:
          next_states, reward, done, info = env.step(action[0]) 
      else:
          env.act( action  )
          _, next_states, terminated = env.observe()
          
      if terminated:
          print("stop")
          print("stop")
          
      if parallel_env:
          next_states = np.expand_dims(next_states, axis=0)
      else:
         next_states = next_states['rgb']
      
      
      next_states = preprocess_image_rgb(next_states)
      
      next_states, frames = stack_frames(frames, next_states, num_envs, is_new_episode = False)
      
      next_states = normalize_states(states)
      
      curiosity_loss,  forward_loss, inverse_loss, rewards = curiosity_model( states, next_states, action)
      
      print("Time step ", time_step, "Terminated ", terminated,"Rewards ", curiosity_loss.item()) 
       
      state_trajectory.append( states)
      
      score_states.append(curiosity_loss.item()) 
      
      if score_states[-1]>= thresh:
          print("Surprise state")
          
          
      states = next_states
    
      time_step +=1
      
      
    
a = np.stack(state_trajectory)    
        
    
    
    


