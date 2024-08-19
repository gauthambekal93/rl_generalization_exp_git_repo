# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 16:49:48 2024

@author: gauthambekal93
"""
import os
os.chdir(r"C:/Users/gauthambekal93/Research/rl_generalization_exps/rl_procgen_games")
import gym
import time
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from PIL import Image
import numpy as np
from collections import deque
from memory import Memory
import random
import csv

# Set the seed for PyTorch
seed = 42
torch.manual_seed(seed)
# Set the seed for NumPy
np.random.seed(seed)
# Set the seed for Python random module
random.seed(seed)

if torch.cuda.is_available():
    print("CUDA is available. PyTorch can use the GPU.")
    print("CUDA version:", torch.version.cuda)
    print("Number of GPUs:", torch.cuda.device_count())
    print("GPU name:", torch.cuda.get_device_name(0))
    device = torch.device("cuda") 
else:
    print("CUDA is not available. PyTorch is using the CPU.")
    

def preprocess_image_rgb(image):
    image = Image.fromarray(image).convert('L')  # 'L' mode is for grayscale
    return np.array(image)

def stack_frames(frames, frame, is_new_episode):
    if is_new_episode:
        frames = deque([np.zeros((64, 64), dtype=np.uint8) for _ in range(4)], maxlen=4)
        for _ in range(4):
            frames.append(frame)
    else:
        frames.append(frame)
    
    stacked_frames = np.stack(frames, axis=0)
    return stacked_frames, frames




class ActorCritic(nn.Module):

    def __init__(self, input_dims, n_actions, gamma=0.99, tau=0.98, size =1000):
       super(ActorCritic, self).__init__()
       self.gamma = gamma
       self.tau = tau
       
       self.matrix = torch.zeros(size, size)
       alpha = self.gamma*self.tau
        # Fill the matrix according to the given pattern
       for i in range(size):
            for j in range(i + 1):
                self.matrix[i, j] = alpha ** (i - j)
                
       self.conv1 = nn.Conv2d(in_channels=input_dims[0], out_channels=32, kernel_size=8, stride=4)
       self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
       self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
      
       in_features = self.conv3(self.conv2(self.conv1(torch.zeros(1, *input_dims)))).view(-1).shape[0]
       out_features = 512
        
       self.fc1 = nn.Linear(in_features =  in_features , out_features = out_features )
       
       self.pi = nn.Linear(out_features, n_actions)
       
       self.v = nn.Linear(out_features, 1)
       
       
    def forward(self, x):
      
       x = torch.tensor(x, dtype = torch.float32) #.to(device)   # takes 0.10  #
      
       x = F.relu(self.conv1(x))
       x = F.relu(self.conv2(x))
       x = F.relu(self.conv3(x))
       x= x.reshape(-1)
       #x = x.reshape(x.size(0), -1)  #x.size(0) will have the batch size
       
       x = F.relu(self.fc1(x))
       
       pi = F.relu(self.pi(x))
       
       #value=  F.relu(self.v(x))   #if the env can have negetive rewards, will we have F.relu?
       
       value = F.leaky_relu(self.v(x), negative_slope=0.01)  #since in some env the reward can be negetive, so we want both positive and negetive rewards

       probs = torch.softmax(pi, dim=0)
       
       dist = Categorical(probs)
       
       action = dist.sample()
       
       log_prob = dist.log_prob(action)
       
       return action.numpy(), value, log_prob
   
    
    def calc_R(self, done, rewards, values):
       values = torch.cat(values).squeeze()
       if len(values.size()) == 1:  # batch of states
           R = values[-1] * (1-int(done))
       elif len(values.size()) == 0:  # single state
           R = values*(1-int(done))

       batch_return = []
       for reward in rewards[::-1]:
           R = reward + self.gamma * R
           batch_return.append(R)
       batch_return.reverse()
       batch_return = torch.tensor(batch_return, dtype=torch.float).reshape(values.size())
       return batch_return

    def calc_loss(self, new_state, done, rewards, values, log_probs):
       
       returns = self.calc_R(done, rewards, values)
      
       next_v = torch.zeros(1) if done else self.forward(torch.tensor([new_state], dtype=torch.float))[1]  #this line gives warning of being extremely slow
       
       
       values.append(next_v.detach())
     
       
       values = torch.cat(values).squeeze()
      
       log_probs = torch.stack(log_probs)  #syntax error
       rewards = torch.tensor(rewards)   
      
       
       delta_t = rewards + self.gamma*values[1:] - values[:-1]
       '''
       n_steps = len(delta_t)
       
       gae = np.zeros(n_steps)   
       
       for t in range(n_steps):    #loop is extremely slow !!! 5.132314920425415
           for k in range(0, n_steps-t):
               temp = (self.gamma*self.tau)**k*delta_t[t+k]
               gae[t] += temp
               
       gae = torch.tensor(gae, dtype=torch.float) 
       '''
       gae =  torch.matmul( delta_t.reshape(1, -1 ) , self.matrix).reshape(-1)
      
       actor_loss = -(log_probs*gae).sum()
       entropy_loss = (-log_probs*torch.exp(log_probs)).sum()
       
       critic_loss = F.mse_loss(values[:-1].squeeze(), returns)

       total_loss = actor_loss + critic_loss - 0.01*entropy_loss
      
       return total_loss



env = gym.make("procgen:procgen-coinrun-v0", start_level=0, num_levels=1)

state =  env.reset()

state = preprocess_image_rgb(state)

frames = None

state, frames = stack_frames(frames, state, is_new_episode=True)

#state = state.reshape(1, -1, -1, -1)  #batch size of 1

state = torch.tensor(state, dtype = torch.float32)
 
max_size = 1000 #was20

model = ActorCritic(input_dims = state.shape , n_actions = env.action_space.n, size = max_size)

learning_rate = 0.001

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

memory = Memory()


episodes = 20000

score = {}

with open('Results_1.csv', mode='w', newline='') as file:
    
    writer = csv.writer(file)
    
    writer.writerows(["Episde", "Score"])
    
    for episode in range(1, episodes):
        start_1 = time.time()
        
        
        state = env.reset()
        
        state = preprocess_image_rgb(state)
        
        start = time.time()
        state, frames = stack_frames(frames, state, is_new_episode = True)
        print("Time0 ",time.time() - start) 
        
        done = False
        
        steps = 0
        
        while not done:
              
              start = time.time() 
              action, value, log_prob = model(state)   #0.015 for a single line, but this called 1000 times per episode hence 2.2secs!!
              print("Time1 ",time.time() - start) 
              
              new_state, reward, done, info = env.step(action)
              
              start = time.time() 
              new_state = preprocess_image_rgb(new_state)     #0.000549
              print("Time2 ",time.time() - start) 
              
              start = time.time()
              new_state, frames = stack_frames(frames, new_state, is_new_episode = False)   #0.0000
              print("Time3 ",time.time() - start) 
              
              start = time.time()
              memory.remember(state, action, reward, new_state, value, log_prob)
              print("Time4 ",time.time() - start) 
              state = new_state
              
              #start = time.time()
              
              if memory.size() == max_size:
                  
                  states, actions, rewards, new_states, values, log_probs = memory.get_data()  #0.0
                  
                  start = time.time()
                  loss = model.calc_loss( new_state, done, rewards, values, log_probs)  #0.15
                  print("Time5 ",time.time() - start) 
                  
                  optimizer.zero_grad()  #0.0
                  
                  start = time.time()
                  loss.backward()     #0.23
                  print("Time6 ",time.time() - start) 
                  
                  #start = time.time()
                  optimizer.step()   #0.0002
                  #print(time.time() - start)
                  memory.clear_memory()
              
              
              steps = steps + 1
              
              score[episode] =  score.get(episode, 0 ) + reward
              
              #writer.writerows([ str(episode),  str(score[episode]) ])
        
        print("Final Time ",time.time() - start_1)      
        
        print("Episode ", episode, "Total time steps ",steps, "Score ", score.get(episode, 0 )  )  
        
        #if score[episode]>0:
        #    print("stop")
        #    print("stop")
            
          
          

          
          
         
