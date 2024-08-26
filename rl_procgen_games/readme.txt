https://gymnasium.farama.org/tutorials/gymnasium_basics/vector_envs_tutorial/


Code_V1 is only a basic skeleton to run procgen with single environment

A2C_with_advantage_and_bootstrapping is a reference code for working of A2C

reference_code is the code obtained from (https://gymnasium.farama.org/tutorials/gymnasium_basics/vector_envs_tutorial/) and is for vector environemnts. We have modified it for cartpole

env_vectorized is just a sample vectorized environment for procgen code


Experiment V1
==> Just some sanity checks and other stuff, no real experiments.

Experiment_V2
==> We choose hard level for both train and test here.
We choose 10000 levels for train and test starts after 15000


Experiment_V3
==> This forlder is primarily around performance measures

Experiment_V4
==> we used easy levels, from 0 to 50000
==> we found that model does not explore very well, and sates such as goal states were given very low reward.
==> one possibility is our model is too complex and predicts even slightly novel states very well.
==> To prevent this we may reduce the hidden layer size of curiosity module so that, it has hard time predicting even slighlt novel states thus increasing error and reward, which forces agent to explore further.


Experiment_V5
==> Uses a combination of extrinsic and intrinsic rewards

Experiment_V6