# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 19:29:29 2024

@author: gauthambekal93
"""
class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.new_states = []
        self.values = []
        self.log_probs = []

    def remember(self, state, action, reward, new_state, value, log_p):
        self.actions.append(action)
        self.rewards.append(reward)
        self.states.append(state)
        self.new_states.append(new_state)
        self.log_probs.append(log_p)
        self.values.append(value)

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.new_states = []
        self.values = []
        self.log_probs = []

    def get_data(self):
        return self.states, self.actions, self.rewards, self.new_states,\
               self.values, self.log_probs
   
    def size(self):
        return len(self.states)