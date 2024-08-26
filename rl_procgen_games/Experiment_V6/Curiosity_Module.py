# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 08:00:57 2024

@author: gauthambekal93
"""

import os
os.chdir(r"C:/Users/gauthambekal93/Research/rl_generalization_exps/rl_generalization_exp_git_repo/rl_procgen_games/Experiment_V6")

model_path =r"C:/Users/gauthambekal93/Research/rl_generalization_exps/rl_generalization_exp_git_repo/rl_procgen_games/Experiment_V6/Models/Curiosity"

result_path =r"C:/Users/gauthambekal93/Research/rl_generalization_exps/rl_generalization_exp_git_repo/rl_procgen_games/Experiment_V6/Results/Curiosity"

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
                                    nn.BatchNorm2d(32),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2), 
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU()
                                    ] )#.to(self.device)
        
        
        x = torch.zeros(*( 1, obs_shape[1] , obs_shape[2], obs_shape[3]) )
        
        in_features = self.conv(x).flatten().shape[0] 
        
        hidden_features_1 = in_features // 2
        
        self.reduce =  nn.Sequential( *[ nn.Linear(in_features, hidden_features_1 ),
                                        nn.ReLU() ] )
        
        hidden_features_2 = hidden_features_1 // 4
        
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
                                    nn.BatchNorm2d(32),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2), 
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
                                    nn.BatchNorm2d(64),
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
    
    
        
        
def get_losses( rewards, action_log_probs, value_preds, entropy, masks, gamma, lam, ent_coef, device, n_envs):
    
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

   
def normalize_states(states):
    
    #pixel_min, pixel_max  = states.min(), states.max()
    
    #states = (states - pixel_min) / (pixel_max - pixel_min)
    
    return states
    
total_timesteps = 50000000#6000000 #was 1000000 
num_models_saved = 5 #was 2
# environment hyperparams
num_envs = 20 #20 
num_train_levels = 10000 #was 50000
num_test_levels = 10  #was 200
#n_updates = int( total_timesteps / num_envs) #50000   #was  100000
n_steps_per_update = 256 #128 #12
#randomize_domain = False

# agent hyperparams
gamma = 0.999
lam = 0.95  # hyperparameter for GAE
ent_coef = 0.01  # coefficient for the entropy bonus (to encourage exploration)

curiosity_lr = 1e-4 #was 1e-4
#NEED TO CHECK IF THE LEARNING RATES WHICH ARE OPTIMAL FOR BELOW 3 NETWORKS
conv_lr = 1e-5 #1e-4 #was 0.001
actor_lr = 1e-5 #1e-4 # was 0.001
critic_lr = 1e-4  #5e-4 # was 0.005

logging_rate = n_steps_per_update* num_envs #10000
        
envs = ProcgenGym3Env(num= num_envs, 
                      env_name="coinrun", 
                      #render_mode="rgb_array",
                      num_levels = num_train_levels, 
                      start_level=0,
                      distribution_mode="hard",  #easy
                      use_sequential_levels =False #False #we keep it as True in order to make it easy to obtain levels where we obtain the goals
                      )
#envs = gym3.ViewerWrapper(envs, info_key="rgb")

'''
envs_test = ProcgenGym3Env(num= num_envs, 
                      env_name="coinrun", 
                      render_mode="rgb_array",
                      num_levels = num_test_levels, #was 200
                      start_level=15000,
                      distribution_mode="easy",  #easy
                      use_sequential_levels=True
                      )
#envs_test = gym3.ViewerWrapper(envs_test, info_key="rgb")
'''

num_actions = envs.ac_space.eltype.n        


while True:
    envs.act( np.random.randint(0, num_actions , size=num_envs)  )
    _, states, done = envs.observe()
    states = states['rgb']
    break



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

out_features = 512  #was 32   #THIS seems very low !!!

actor =  Actor(shared_conv, in_features, out_features, num_actions, device).to(device)

critic =  Critic(shared_conv, in_features, out_features, num_actions, device).to(device)



optimizer = optim.Adam([
    {'params': shared_conv.parameters(), 'lr': conv_lr},
    {'params': actor.actor.parameters(), 'lr': actor_lr},
    {'params': critic.critic.parameters(), 'lr': critic_lr}
])




def save_model(actor, critic, optimizer, time_step, curiosity_model, curiosity_optimizer):
    print("----SAVE THE MODELS---")
    
    torch.save(actor.state_dict(), os.path.join(model_path,'actor_'+str(time_step)+'.pth'))
    torch.save(critic.state_dict(), os.path.join(model_path,'critic_'+str(time_step)+'.pth'))
    torch.save(optimizer.state_dict(), os.path.join(model_path,'optimizer_'+str(time_step)+'.pth'))
    
    torch.save(curiosity_model.state_dict(), os.path.join(model_path,'curiosity_'+str(time_step)+'.pth'))
    torch.save(curiosity_optimizer.state_dict(), os.path.join(model_path,'curiosity_optimizer_'+str(time_step)+'.pth'))
    


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
    


#time_step = 0
#current_reward_rate = 0
#best_reward_rate = 0




data = pd.read_csv ( os.path.join(result_path,"Curiosity_Results.csv") )


if len(data)<=1: 
    time_step = 0
    model_save_step = 0
    #best_reward_rate = 0
else:
    time_step = int( data.iloc[-1]['Time_Steps'] )
    model_save_step = time_step + (total_timesteps / num_models_saved )
   # best_reward_rate =  data.loc[data['Type']=='Test']['Reward_Rate'].max() 
    
   # best_time_step = data.loc[ (data["Reward_Rate"] == best_reward_rate) & (data['Type']=='Test') ]['Time_Steps'].iloc[0]


load_model()
 

#time_step =0

while time_step <= total_timesteps:   
     
     print("Time step ", time_step)
     #parameters for curiosity module loss caluclations
     #batch_intrinsic_reward = 0
     
     batch_extrinsic_reward = 0
     
     #batch_complete_reward = 0
     
     #parameters for actor-critic and curiosity loss calculations
     
     ep_curiosity_loss = torch.zeros(n_steps_per_update, num_envs, device=device)
     
     ep_forward_loss = torch.zeros(n_steps_per_update, num_envs, device=device)
     
     ep_inverse_loss = torch.zeros(n_steps_per_update, num_envs, device=device)
     
     ep_value_preds = torch.zeros(n_steps_per_update, num_envs, device=device)
     
     ep_rewards = torch.zeros(n_steps_per_update, num_envs, device=device)
     
     ep_action_log_probs = torch.zeros(n_steps_per_update, num_envs, device=device)
     
     masks = torch.zeros(n_steps_per_update, num_envs, device=device)
     
     ongoing_masks = torch.ones(num_envs, device=device)
     
     _, states, terminated = envs.observe()  #envs_wrapper.reset(seed=42)
 
     states = states['rgb']
     
     states = preprocess_image_rgb(states)
     
     states, frames = stack_frames(frames, states, num_envs, is_new_episode = False)
     
     states = normalize_states(states)
     
     start = time.time() 
     # play n steps in our parallel environments to collect data
     for step in range(n_steps_per_update):

         actions, action_log_probs, state_value_preds, entropy = select_action( states, actor, critic )
 
         actions = np.array(actions.cpu().detach())
         
         envs.act( actions  )
         
         #print("Step ", step, "Action ", actions[0])
         
         extrinsic_rewards, next_states, terminated = envs.observe()
         
         batch_extrinsic_reward = batch_extrinsic_reward + extrinsic_rewards.sum()
         
         #extrinsic_rewards =  torch.tensor(extrinsic_rewards, dtype=float) #*100
         
         next_states = next_states['rgb']
         
         next_states = preprocess_image_rgb(next_states)
         
         next_states, frames = stack_frames(frames, next_states, num_envs, is_new_episode = False)
         
         next_states = normalize_states(next_states)
         
         curiosity_loss,  forward_loss, inverse_loss, _ = curiosity_model( states, next_states, actions)
         
         #complete_rewards = extrinsic_rewards + intrinsic_rewards
         
         ep_curiosity_loss[step] =  torch.squeeze ( curiosity_loss ) 
         
         ep_forward_loss[step] =  torch.squeeze ( forward_loss ) 
         
         ep_inverse_loss[step] =  torch.squeeze ( inverse_loss ) 
         
         ep_value_preds[step] = torch.squeeze(state_value_preds)
         
         ep_rewards[step] = torch.tensor(extrinsic_rewards)
         
         ep_action_log_probs[step] = action_log_probs
 
         # Update ongoing_masks to ensure terminated episodes remain terminated
         ongoing_masks *= torch.tensor([not term for term in terminated], device=device)
         
         masks[step] =  ongoing_masks 
         
        # batch_intrinsic_reward = batch_intrinsic_reward + intrinsic_rewards.sum()
         
         
         
        # batch_complete_reward = batch_complete_reward + complete_rewards.sum() #mean()
         
         time_step = time_step + num_envs
         
         states = next_states
     
        # calculate the losses for actor and critic
     critic_loss, actor_loss = get_losses(
         ep_rewards,
         ep_action_log_probs,
         ep_value_preds,
         entropy,
         masks,
         gamma,
         lam,
         ent_coef,
         device,
         num_envs
     )
 
    
     # update the actor and critic networks
     update_parameters(optimizer, critic_loss, actor_loss)
     
     critic_loss = critic_loss.detach().cpu().numpy()
     
     actor_loss =  actor_loss.detach().cpu().numpy()
     
     
     curiosity_loss = torch.mean(ep_curiosity_loss)
     
     curiosity_model.update_curiosity_params( curiosity_loss, curiosity_optimizer)
     
     curiosity_loss = curiosity_loss.detach().cpu().numpy()
     
     forward_loss, inverse_loss = torch.mean(ep_forward_loss).detach().cpu().numpy(),  torch.mean(ep_inverse_loss).detach().cpu().numpy() 
     
     print("Train: ", "Time Step: ",time_step , "Extrinsic Reward: ", batch_extrinsic_reward, "Extrinsic Rate:", batch_extrinsic_reward/logging_rate, "Critic Loss: ", critic_loss, "Actor Loss: ", actor_loss, "Curiosity Loss: ",  curiosity_loss, "Forward Loss: ", forward_loss, "Inverse Loss: ", inverse_loss )
     
     
     with open(os.path.join(result_path, "Curiosity_Results.csv"), 'a', newline='') as file:
         
         writer = csv.writer(file)
         
         writer.writerow([ "Train" , time_step ,batch_extrinsic_reward.item(), batch_extrinsic_reward.item()/logging_rate, critic_loss, actor_loss, curiosity_loss, forward_loss,inverse_loss  ]  )  
         
         file.close()
         
     
     if time_step>= model_save_step:
         save_model(actor, critic, optimizer, time_step, curiosity_model, curiosity_optimizer)  #save the best model only
         model_save_step = model_save_step + (total_timesteps / num_models_saved )
     
     
     
     '''
     current_reward_rate = test_model(actor, critic, time_step, envs_test, result_path, logging_rate )
     
     if best_reward_rate < current_reward_rate:
         save_model(actor, critic, optimizer, time_step)  #save the best model only
         best_reward_rate = current_reward_rate
         
     '''    
   
     
     print("Duration per batch data collection", time.time()- start)  
     
     
save_model(actor, critic, optimizer, time_step, curiosity_model, curiosity_optimizer) 































