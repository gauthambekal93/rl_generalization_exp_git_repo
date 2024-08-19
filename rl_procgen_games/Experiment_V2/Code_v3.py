# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 19:58:07 2024

@author: gauthambekal93
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 10:21:29 2024

@author: gauthambekal93
"""


import os
os.chdir(r"C:/Users/gauthambekal93/Research/rl_generalization_exps/rl_procgen_games/Experiment_V2")

model_path =r"C:/Users/gauthambekal93/Research/rl_generalization_exps/rl_procgen_games/Experiment_V2/Models"

result_path =r"C:/Users/gauthambekal93/Research/rl_generalization_exps/rl_procgen_games/Experiment_V2/Results"

#import matplotlib.pyplot as plt
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
from test_script import test_model
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)



if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def main():
    
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
        
            
            
        total_timesteps = 8000000 #was 2000000
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
                              #render_mode="rgb_array",
                              num_levels=num_levels,
                              start_level=0,
                              distribution_mode="hard",  #easy
                              use_sequential_levels =False
                              )
        #envs = gym3.ViewerWrapper(envs, info_key="rgb")
        
        
        envs_test = ProcgenGym3Env(num= num_envs, 
                              env_name="coinrun", 
                              #render_mode="rgb_array",
                              num_levels=200,
                              start_level=15000,
                              distribution_mode="hard",  #easy
                              use_sequential_levels=True
                              )
        #envs_test = gym3.ViewerWrapper(envs_test, info_key="rgb")
        
        num_actions = envs.ac_space.eltype.n
        
       
        
        
        while True:
            envs.act( np.random.randint(0, num_actions , size=num_envs)  )
            rewards, states, done = envs.observe()
            states = states['rgb']
            break
        
        
        
        states = preprocess_image_rgb(states)
        
        frames = []
        states, frames = stack_frames(frames, states, num_envs, is_new_episode = True)
        
        obs_shape = (states.shape[0], states.shape[1], states.shape[2], states.shape[3]) #env_num, history of frames, x dim, y dim
        
        
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
        
        
        #critic_losses = []
        #actor_losses = []
        #entropies = []
        
        optimizer = optim.Adam([
            {'params': shared_conv.parameters(), 'lr': conv_lr},
            {'params': actor.actor.parameters(), 'lr': actor_lr},
            {'params': critic.critic.parameters(), 'lr': critic_lr}
        ])
        
        
        def save_model(actor, critic, optimizer, time_step):
            print("----SAVE THE MODEL---")
            torch.save(actor.state_dict(), os.path.join(model_path,'actor_'+str(time_step)+'.pth'))
            torch.save(critic.state_dict(), os.path.join(model_path,'critic_'+str(time_step)+'.pth'))
            torch.save(optimizer.state_dict(), os.path.join(model_path,'optimizer_'+str(time_step)+'.pth'))
        
        
        def load_model(best_time_step):
            print("----LOAD THE MODEL---")
            actor.load_state_dict(torch.load( os.path.join(model_path,'actor_'+str(best_time_step)+'.pth' )))
            critic.load_state_dict(torch.load( os.path.join(model_path, 'critic_'+str(best_time_step)+'.pth' )))
            optimizer.load_state_dict(torch.load( os.path.join(model_path,'optimizer_'+str(best_time_step)+'.pth' ) ))
        
        
        total_reward = 0
        
        data = pd.read_csv ( os.path.join(result_path,"Results.csv") )
        
        if len(data)<=1: 
            time_step = 0
            best_reward_rate = 0
        else:
            time_step = data.iloc[-1]['Time_Steps']
            
            best_reward_rate =  data.loc[data['Type']=='Test']['Reward_Rate'].max() 
            
            best_time_step = data.loc[ (data["Reward_Rate"] == best_reward_rate) & (data['Type']=='Test') ]['Time_Steps'].iloc[0]
            
            load_model(best_time_step )
            
        #if os.path.exists( os.path.join(model_path,'actor.pth' ) ) and os.path.exists( os.path.join(model_path,'critic.pth' ) ) and os.path.exists( os.path.join(model_path,'optimizer.pth' ) ): 
            
            

        print("Time step ", time_step)
        start = time.time()
        current_reward_rate = test_model(actor, critic, time_step, envs_test, result_path, logging_rate )
        print(time.time() - start)
        while time_step <= total_timesteps:    
            
                
            # we don't have to reset the envs, they just continue playing
            # until the episode is over and then reset automatically
            
            # reset lists that collect experiences of an episode (sample phase)
            ep_value_preds = torch.zeros(n_steps_per_update, num_envs, device=device)
            
            ep_rewards = torch.zeros(n_steps_per_update, num_envs, device=device)
            
            ep_action_log_probs = torch.zeros(n_steps_per_update, num_envs, device=device)
            
            masks = torch.zeros(n_steps_per_update, num_envs, device=device)
            
            ongoing_masks = torch.ones(num_envs, device=device)
            
                
            rewards, states, terminated = envs.observe()  #envs_wrapper.reset(seed=42)
        
            states = states['rgb']
            
            states = preprocess_image_rgb(states)
            
            states, frames = stack_frames(frames, states, num_envs, is_new_episode = False)
            
            start = time.time() 
            # play n steps in our parallel environments to collect data
            for step in range(n_steps_per_update):
            #step = 0    
            #while not terminated:
                # select an action A_{t} using S_{t} as input for the agent
                actions, action_log_probs, state_value_preds, entropy = select_action( states, actor, critic )
        
                # perform the action A_{t} in the environment to get S_{t+1} and R_{t+1}    
                actions = np.array(actions.cpu().detach())
                
                envs.act( actions  )
                
                rewards, states, terminated = envs.observe()
                
                total_reward = total_reward + rewards.sum() #mean() 
                
                time_step = time_step + num_envs
                
                states = states['rgb']
                
                states = preprocess_image_rgb(states)
                
                states, frames = stack_frames(frames, states, num_envs, is_new_episode = False)
            
                ep_value_preds[step] = torch.squeeze(state_value_preds)
                
                ep_rewards[step] = torch.tensor(rewards, device=device)
                
                ep_action_log_probs[step] = action_log_probs
                
                # Update ongoing_masks to ensure terminated episodes remain terminated
                ongoing_masks *= torch.tensor([not term for term in terminated], device=device)
                
                # add a mask (for the return calculation later);
                # for each env the mask is 1 if the episode is ongoing and 0 if it is terminated (not by truncation!)
                masks[step] =  ongoing_masks 
              
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
            
            print("Train: ", "Time Step: ", str(time_step) , "Reward: ", str(total_reward), "Reward Rate: ",str(current_reward_rate), "Critic Loss: ", critic_loss, "Actor Loss: ", actor_loss, "Total Loss: ",  actor_loss + critic_loss )
            
            with open(os.path.join(result_path, "Results.csv"), 'a', newline='') as file:
                
                writer = csv.writer(file)
                
                writer.writerow([ "Train" , str(time_step) , str(total_reward), str(total_reward / logging_rate), critic_loss, actor_loss, actor_loss + critic_loss  ]  )  
                
                file.close()
                
            total_reward = 0
            
            current_reward_rate = test_model(actor, critic, time_step, envs_test, result_path, logging_rate )
            
            if best_reward_rate < current_reward_rate:
                save_model(actor, critic, optimizer, time_step)  #save the best model only
                best_reward_rate = current_reward_rate
                
                
            # log the losses and entropy
            #critic_losses.append(critic_loss)
            #actor_losses.append(actor_loss)
            #entropies.append(entropy.detach().mean().cpu().numpy())
            
            
            print("Duration per batch data collection", time.time()- start)  
        
        

    

if __name__ == "__main__":
    main()



     
