import torch
import torch.nn as nn
import torch.optim as optim

class SharedConv(nn.Module):
    def __init__(self, conv_layers):
        super(SharedConv, self).__init__()
        self.conv = nn.Sequential(*conv_layers)

    def forward(self, x):
        return self.conv(x)

class ActorNetwork(nn.Module):
    def __init__(self, shared_conv, actor_layers):
        super(ActorNetwork, self).__init__()
        self.shared_conv = shared_conv
        self.actor = nn.Sequential(*actor_layers)

    def forward(self, x):
        x = self.shared_conv(x)
        x = x.view(x.size(0), -1)  # Flatten
        return self.actor(x)

class CriticNetwork(nn.Module):
    def __init__(self, shared_conv, critic_layers):
        super(CriticNetwork, self).__init__()
        self.shared_conv = shared_conv
        self.critic = nn.Sequential(*critic_layers)

    def forward(self, x):
        x = self.shared_conv(x)
        x = x.view(x.size(0), -1)  # Flatten
        return self.critic(x)

# Example usage
conv_layers = [
    nn.Conv2d(in_channels=3, out_channels=32, kernel_size=8, stride=4),
    nn.ReLU(),
    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
    nn.ReLU(),
    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
    nn.ReLU()
]

actor_layers = [
    nn.Linear(64 * 7 * 7, 512),  # Adjust the dimensions based on the output of shared_conv
    nn.ReLU(),
    nn.Linear(512, 4)  # Example number of actions
]

critic_layers = [
    nn.Linear(64 * 7 * 7, 512),  # Adjust the dimensions based on the output of shared_conv
    nn.ReLU(),
    nn.Linear(512, 1)
]

# Create shared convolutional layers
shared_conv = SharedConv(conv_layers)

# Create actor and critic networks
actor = ActorNetwork(shared_conv, actor_layers)
critic = CriticNetwork(shared_conv, critic_layers)

# Separate the parameters
shared_params = list(shared_conv.parameters())
actor_params = list(actor.actor.parameters())  # Only actor-specific parameters
critic_params = list(critic.critic.parameters())  # Only critic-specific parameters

# Define different learning rates
optimizer = optim.Adam([
    {'params': shared_params, 'lr': 1e-4},  # Shared parameters
    {'params': actor_params, 'lr': 1e-4},   # Actor-specific parameters
    {'params': critic_params, 'lr': 1e-3}   # Critic-specific parameters
])

# Example forward pass
example_input = torch.randn(1, 3, 84, 84)  # Batch size of 1
actor_output = actor(example_input)
critic_output = critic(example_input)

print("Actor output:", actor_output)
print("Critic output:", critic_output)

# Check if the parameters of the shared convolution layers are the same
for p1, p2 in zip(actor.shared_conv.parameters(), critic.shared_conv.parameters()):
    assert p1.data_ptr() == p2.data_ptr(), "Parameters are not shared!"
print("Parameters are shared between actor and critic.")
