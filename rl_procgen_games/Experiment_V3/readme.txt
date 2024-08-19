This is experiment is easy mode with 80k levels for train and test starts at 90Learning rate
We will use 50 parallel environments
Train for 8 million time steps
We will update at end of episode ie return is calculated at end of  level for each env
  conv_lr =1e-4
  actor_lr = 1e-4 # was 0.001
  critic_lr = 5e-4 # was 0.005k level